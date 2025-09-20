# trainer_potential.py
import os
import os.path as osp
import sys
import time
import math
import json
import shutil
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torchvision.utils import make_grid

from .aux import sample_z, TrainingStatTracker, update_progress, update_stdout, sec2dhms
from .aux import CosineScheduleWithWarmup, build_adamw, ImageLogger, ImageViz, _pack_BK,_per_k_grad_norms,module_grad_norm

import torch
from torch.utils.data import TensorDataset, DataLoader
from typing import Optional

DTYPE = torch.float32


class DataParallelPassthrough(nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super(DataParallelPassthrough, self).__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


class TrainerPotential(object):
    def __init__(self, params=None, exp_dir=None, device=torch.device('cpu'), use_cuda=False, use_mps=False, multi_gpu=False):
        if params is None:
            raise ValueError("Cannot build a Trainer instance with empty params: params={}".format(params))
        self.params = params
        self.device = device
        self.use_cuda = use_cuda
        self.use_mps = use_mps
        self.multi_gpu = multi_gpu

        # TensorBoard
        self.tensorboard = self.params.tensorboard
        self.wip_dir = osp.join("experiments", "wip", exp_dir)
        self.complete_dir = osp.join("experiments", "complete", exp_dir)

        self.stats_json = osp.join(self.wip_dir, 'stats.json')
        os.makedirs(self.wip_dir, exist_ok=True)
        if not osp.isfile(self.stats_json):
            with open(self.stats_json, 'w') as out:
                json.dump({}, out)

        self.models_dir = osp.join(self.wip_dir, 'models')
        os.makedirs(self.models_dir, exist_ok=True)
        self.checkpoint = osp.join(self.models_dir, 'checkpoint.pt')

        self.tb_writer = None
        if self.tensorboard:
            from tensorboard import program
            from torch.utils.tensorboard import SummaryWriter
            exp_dir, run_name = exp_dir.split("__")
            self.tb_dir = os.path.join("experiments", "tensorboard", "wip", exp_dir)

            os.makedirs(os.path.join(self.tb_dir, run_name), exist_ok=True)
            self.tb = program.TensorBoard()
            self.tb.configure(argv=[None, '--logdir', self.tb_dir])
            self.tb_url = self.tb.launch()
            print(f"#. Start TensorBoard at {self.tb_url} (run: {run_name})")
            self.tb_writer = SummaryWriter(log_dir=os.path.join(self.tb_dir, run_name))

        # Loss bits
        self.ce_label_smoothing = float(getattr(self.params, "ce_label_smoothing", 0.00))
        self.conf_penalty_weight = float(getattr(self.params, "conf_penalty_weight", 0.0))
        self.cross_entropy = nn.CrossEntropyLoss(label_smoothing=self.ce_label_smoothing)

        # Stat tracker
        self.stat_tracker = TrainingStatTracker(
            ema_decay=getattr(self.params, "ema_decay", 0.9),
            ema_max_history=getattr(self.params, "ema_max_history", 200),
        )

        self.K = None


    # ------------------------ helpers for TB visuals ------------------------
    def _write_stats_json(self):
        # Update training statistics json file (optimizer-step keyed)
        with open(self.stats_json, 'w') as out:
            json.dump(self.stat_tracker.stats_by_step, out)

    def log_progress(self, step_idx, mean_step_time, elapsed_time, eta):
        if step_idx >1:
            update_stdout(10)
        stats = self.stat_tracker.stats_by_step.get(int(step_idx), {})
        total_opt_steps = math.ceil(self.params.max_iter / max(1, int(getattr(self.params, "accumulate_grad_steps", 1))))
        update_progress(
            "\\__.Training [bs: {}] [opt-step: {:06d}/{:06d}] ".format(
                self.params.batch_size, step_idx, total_opt_steps
            ),
            total_opt_steps,
            step_idx + 1,
        )
        # Stdout progress (now on optimizer-step cadence)
        print()
        print("   \\__Batch accuracy Index      : {:.03f}".format(stats.get('accuracy_index', 0.0)))
        print("   \\__Classification loss       : {:.08f}".format(stats.get('classification_loss', 0.0)))
        print("   \\__Wave loss (PDE-JVP combo) : {:.08f}".format(stats.get('wave_loss', 0.0)))
        print("   \\__Total loss                : {:.08f}".format(stats.get('total_loss', 0.0)))
        print("      ==============================================================")
        print("   \\__Opt-step time  : {:.3f} sec".format(mean_step_time))
        print("   \\__Elapsed time   : {}".format(sec2dhms(elapsed_time)[:-6]))
        print("   \\__ETA            : {}".format(sec2dhms(eta)[:-6]))
        print("      ==============================================================")
        

    def loss_allK(self, support_sets, generator, reconstructor, z, t_index, dt, acc_denominator: int):
        """
        One forward for all K in parallel. Ensures classifier gradients reach ψ/f via img2.
        """
        # PDE/OT forward
        potential_preds, latent1_bk, latent2_bk, pde_loss, dt = support_sets(z, t_index, dt=dt, direction=dt)
        B, K, D = latent1_bk.shape



        # ---- pack latents for generator/classifier ----
        lat1_flat, targets, (b_idx, k_idx), (B, K) = _pack_BK(latent1_bk)
        lat2_flat, _,      _,                  _   = _pack_BK(latent2_bk)  # mapping identical by shape
        
        # Add initial latent
        z0 = z.clone().unsqueeze(1).expand(B, K, D)
        lat0_flat, _,      _,                  _   = _pack_BK(z0)

        # Images
        img0 = generator(lat0_flat)
        img1 = generator(lat1_flat)
        img2 = generator(lat2_flat)

        # Detach img1 so CE must flow via img2 → latent2 → ψ/f
        img1_det = img1.detach()

        # Classify
        logits, magnitudes = reconstructor(img1, img2)  # [B*K, K]
        logits0, magnitudes0 = reconstructor(img0, img1)
        mse_loss = nn.MSELoss()(magnitudes.reshape(B, K, 2), potential_preds[:,:,-2:])
        mse_loss = (mse_loss + nn.MSELoss()(magnitudes0.reshape(B, K, 2), potential_preds[:,:,[0,-2]]))/2
        cls_loss = (self.cross_entropy(logits, targets) + self.cross_entropy(logits0, targets))/2



        # Total loss
        loss = (
            self.params.lambda_cls * cls_loss
            + self.params.lambda_reg * (mse_loss)
            + self.params.lambda_pde * pde_loss
        )
        loss = loss / max(1, int(acc_denominator))
        d2 = (latent2_bk - latent1_bk).norm(dim=-1).min()
        # Asymmetric penalty: penalize d2 approaching 0 with a 1/x type penalty
        # d2_penalty = 1.0 / (d2 + 1e-6)  # add epsilon for stability
        loss = loss #+ 1.0 * d2_penalty  # 0.1 is a weighting factor; adjust as needed

        loss.backward()


        # Logging views
        with torch.no_grad():
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)  # [B*K]
            entropy = -(probs * (probs.clamp_min(1e-8).log())).sum(dim=1).mean()

            z_bk = z.unsqueeze(1).expand(B, K, D)
            d1 = (latent1_bk - z_bk).norm(dim=-1).min()

            # reshape images back for visuals
            img1_bk = img1.detach().contiguous().view(B, K, *img1.shape[1:])
            img2_bk = img2.detach().contiguous().view(B, K, *img2.shape[1:])

        return {
            "total_loss": float(loss.detach()),
            "classification_loss": float(cls_loss.detach()),
            "pde_loss": float(pde_loss.detach()),
            "entropy": float(entropy.item()),
            "step1_norm": float(d1.item()),
            "step2_norm": float(d2.item()),"mse_loss": float(mse_loss.detach())
    }, logits.detach(), targets, latent1_bk.detach(), latent2_bk.detach(), img1_bk, img2_bk, potential_preds.detach()


    # ------------------------ checkpoint utils ------------------------
    def get_starting_iteration(self, support_sets, reconstructor, support_opt=None, recon_opt=None, support_sched=None, recon_sched=None):
        def safe_load_state_dict(obj, ckpt, name, strict=True):
            if obj is not None and name in ckpt:
                try:
                    obj.load_state_dict(ckpt[name])
                except Exception as e:
                    print(f"Error loading state_dict for {name}: {e}")
        start_iter = 1
        if osp.isfile(self.checkpoint):
            ckpt = torch.load(self.checkpoint, map_location=self.device)
            start_iter = int(ckpt.get('iter', 1))
            safe_load_state_dict(support_sets, ckpt, 'support_sets', strict=False)
            safe_load_state_dict(reconstructor, ckpt, 'reconstructor', strict=False)
            # Add noise to all parameters after loading
            def add_noise_to_params(module, std=1e-3):
                for p in module.parameters():
                    if p.requires_grad:
                        noise = torch.randn_like(p) * std
                        p.data.add_(noise)
            if support_sets is not None:
                add_noise_to_params(support_sets)
            if reconstructor is not None:
                add_noise_to_params(reconstructor)
            safe_load_state_dict(support_opt, ckpt, 'support_opt')
            safe_load_state_dict(recon_opt, ckpt, 'recon_opt')
            safe_load_state_dict(support_sched, ckpt, 'support_sched')
            safe_load_state_dict(recon_sched, ckpt, 'recon_sched')
            self.stat_tracker.set_opt_step(start_iter)
        return start_iter

    # ------------------------ optim/sched ------------------------
    def init_optimizers(self, support_sets, reconstructor, acc_steps: int):
        support_set_wd = float(getattr(self.params, "support_set_wd", 0.01))
        reconstructor_wd = float(getattr(self.params, "reconstructor_wd", 0.001))
        betas = tuple(getattr(self.params, "adam_betas", (0.9, 0.999)))
        eps = float(getattr(self.params, "adam_eps", 1e-8))

        reset_lr = bool(getattr(self.params, "reset_lr", True))
        reset_weight_decay = bool(getattr(self.params, "reset_weight_decay", False))
        reset_schedulers = bool(getattr(self.params, "reset_schedulers", False))
        reset_start_iter = bool(getattr(self.params, "reset_start_iter", False))

        # Optimizers: treat PSI and F jointly (two groups if you want WD split)
        support_sets_optim = build_adamw(
            [
                {"params": support_sets.PSI.parameters(), "weight_decay": support_set_wd, "lr": self.params.support_set_lr},
                {"params": support_sets.F.parameters(), "weight_decay": 0.5, "lr": self.params.support_set_lr},
                {"params": [support_sets.c], "weight_decay": 0.0, "lr": self.params.support_set_lr},
            ],
            lr=self.params.support_set_lr,
            weight_decay=0.0,  # per-group above
            extra_no_decay_names=(),
            betas=betas,
            eps=eps,
        )

        reconstructor_optim = build_adamw(
            reconstructor,
            lr=self.params.reconstructor_lr,
            weight_decay=reconstructor_wd,
            extra_no_decay_names=(),
            betas=betas,
            eps=eps,
        )

        total_opt_steps = max(1, math.ceil(self.params.max_iter / max(1, acc_steps)))
        warmup_steps = math.ceil(float(getattr(self.params, "warmup_fraction", 0.0)) * total_opt_steps)

        sched_support = CosineScheduleWithWarmup(
            support_sets_optim, num_warmup_steps=warmup_steps, num_training_steps=total_opt_steps, last_epoch=-1
        )
        sched_recon = CosineScheduleWithWarmup(
            reconstructor_optim, num_warmup_steps=warmup_steps, num_training_steps=total_opt_steps, last_epoch=-1
        )

        starting_opt_step = self.get_starting_iteration(
            support_sets, reconstructor,
            support_opt=support_sets_optim, recon_opt=reconstructor_optim,
            support_sched=sched_support, recon_sched=sched_recon,
        )

        if reset_lr:
            for g in support_sets_optim.param_groups:
                g["lr"] = float(self.params.support_set_lr)
            for g in reconstructor_optim.param_groups:
                g["lr"] = float(self.params.reconstructor_lr)
            if hasattr(sched_support, "base_lrs"):
                sched_support.base_lrs = [g["lr"] for g in support_sets_optim.param_groups]
            if hasattr(sched_recon, "base_lrs"):
                sched_recon.base_lrs = [g["lr"] for g in reconstructor_optim.param_groups]

        if reset_schedulers:
            sched_support = CosineScheduleWithWarmup(support_sets_optim, num_warmup_steps=warmup_steps, num_training_steps=total_opt_steps, last_epoch=-1)
            sched_recon = CosineScheduleWithWarmup(reconstructor_optim, num_warmup_steps=warmup_steps, num_training_steps=total_opt_steps, last_epoch=-1)
            if reset_weight_decay and not reset_start_iter:
                starting_opt_step = 0

        if reset_start_iter:
            starting_micro = 1
            opt_step_idx = 0
            self.stat_tracker.set_opt_step(0)
        else:
            starting_micro = starting_opt_step * acc_steps + 1
            opt_step_idx = starting_opt_step

        support_sets_optim.zero_grad(set_to_none=True)
        reconstructor_optim.zero_grad(set_to_none=True)
        return starting_micro, opt_step_idx, support_sets_optim, reconstructor_optim, sched_support, sched_recon
 
    # ------------------------ train ------------------------
    def train(self, generator, support_sets, reconstructor):
        histograms = True
        save_images = True
        save_checkpoints = True
        analytics = True

        # One-time save of initial support_sets
        if not osp.isfile(self.checkpoint):
            torch.save(support_sets.state_dict(), osp.join(self.models_dir, 'support_sets_init.pt'))
        else:
            print("#. checkpoint found, skipping contrastive pretraining.")

        # Modes / devices
        generator = generator.to(self.device).eval()
        generator.requires_grad_(False)
        support_sets = support_sets.to(self.device).train()
        reconstructor = reconstructor.to(self.device).train()

        if self.multi_gpu:
            print(f"#. Parallelize G, R over {torch.cuda.device_count()} GPUs...")
            generator = DataParallelPassthrough(generator)
            reconstructor = DataParallelPassthrough(reconstructor)
            cudnn.benchmark = True

        # Bookkeeping
        self.K = int(self.params.num_support_sets)
        self.stat_tracker.init_per_k(self.K)
        half_range = self.K // 2
        init_truncation = float(getattr(self.params, "z_truncation", 1.0))


        acc_steps = max(1, int(getattr(self.params, "accumulate_grad_steps", 1)))
        VIS_IMAGE_K = min(32, self.K)


        (starting_micro, opt_step_idx,
        support_sets_optim, reconstructor_optim,
        sched_support, sched_recon) = self.init_optimizers(support_sets, reconstructor, acc_steps)


        # === Image logger (rotating) ===
        img_keep_last = int(getattr(self.params, "image_keep_last", 10))

        if self.tensorboard and save_images:
            img_logger = ImageLogger(
                writer=self.tb_writer,
                keep_last_images=img_keep_last,
                downscale=1/reconstructor.max_pool_size if hasattr(reconstructor, 'max_pool_size') else 1,
            )

        t0 = time.time()

        # Early exit if already done
        if starting_micro > self.params.max_iter:
            print("#. This experiment has already been completed and can be found @ {}".format(self.wip_dir))
            print("#. Copy {} to {}...".format(self.wip_dir, self.complete_dir))
            shutil.copytree(src=self.wip_dir, dst=self.complete_dir, ignore=shutil.ignore_patterns('checkpoint.pt'))
            sys.exit()

        print(f"#. Start training from micro-step {starting_micro}")
        print(f"#. Training loop: {starting_micro} to {self.params.max_iter}")

        # === main loop: NO per-k loop; one pass covers all K ===
        for micro_idx, iteration in enumerate(range(starting_micro, self.params.max_iter + 1), start=1):
            iter_t0 = time.time()

            z = sample_z(self.params.batch_size, generator, self.params, self.device)

            # Random step index per sample
            dt = torch.randint(2_500, 7_00, (1, 1), device=self.device)/5_000
            dt = dt.repeat(self.params.batch_size, 1)
            t_idx = torch.randint(1, max(1, half_range - 1), (self.params.batch_size, 1), device=self.device)

            (loss_dict, logits, targets,
            latent1_bk, latent2_bk, img1_bk, img2_bk, potential_preds) = self.loss_allK(
                support_sets, generator, reconstructor, z, t_idx, dt, acc_denominator=acc_steps)

            # ===== analytics & stats =====
            with torch.no_grad():
                B = self.params.batch_size
                K = self.K
                preds = torch.argmax(logits, dim=1)  # [B*K]
                preds_2d = preds.view(B, K)
                true_2d = torch.arange(K, device=self.device).unsqueeze(0).expand(B, K)
                batch_acc = (preds_2d == true_2d).float().mean().item()

                per_k_acc = (preds_2d == true_2d).float().mean(dim=0).cpu().numpy()  # [K]
                per_k_gn = _per_k_grad_norms(support_sets) if analytics else None

                wave_dict = support_sets.get_losses()
                wave_dict['potential_std'] = potential_preds.std().item()
                wave_dict['xf_now'] = wave_dict['xf_now'].norm(dim=-1).mean().item()

                # Stat tracker accumulates raw window metrics
                self.stat_tracker.add_micro(
                    acc=batch_acc,
                    **loss_dict,
                    **{k: float(v.item()) if torch.is_tensor(v) else float(v) for k, v in wave_dict.items()},
                )

                # Update per-k EMA stats
                if analytics:
                    for k in range(K):
                        self.stat_tracker.update_per_k_after_micro(
                            true_k=int(k),
                            preds=preds_2d[:, k].detach().cpu().numpy(),
                            batch_size=int(B),
                            grad_norm_selected_mlp=(per_k_gn[k] if per_k_gn is not None else None),
                        )

            # ===== perform optimizer step at accumulation boundary =====
            is_boundary = (micro_idx % acc_steps == 0) or (iteration == self.params.max_iter)
            if is_boundary:
                # Grad norm snapshots (to scalars writer)
                if analytics and self.tensorboard:
                    gn_support = module_grad_norm(support_sets.PSI) + module_grad_norm(support_sets.F)
                    gn_recon = module_grad_norm(reconstructor)
                    self.tb_writer.add_scalar("train/grad_norm/support_sets", gn_support, self.stat_tracker.global_opt_step)
                    self.tb_writer.add_scalar("train/grad_norm/reconstructor", gn_recon, self.stat_tracker.global_opt_step)

                torch.nn.utils.clip_grad_norm_(support_sets.parameters(), max_norm=3.0)
                torch.nn.utils.clip_grad_norm_(reconstructor.parameters(), max_norm=3.0)

                support_sets_optim.step(); reconstructor_optim.step()
                support_sets_optim.zero_grad(set_to_none=True); reconstructor_optim.zero_grad(set_to_none=True)
                sched_support.step(); sched_recon.step()
                self.params.z_truncation = init_truncation * 0.95 + 0.05 * 1.0


                self.stat_tracker.set_lrs(sched_support.get_last_lr()[0], sched_recon.get_last_lr()[0])

                win_means = self.stat_tracker.close_window()


                # -------- TensorBoard logging (scalars + light figures only) --------
                if self.tensorboard:
                    
                    for key, value in win_means.items():
                        self.tb_writer.add_scalar(f"train/{key}", float(value), self.stat_tracker.global_opt_step)
                    self.tb_writer.add_scalar("train/support_sets_lr", self.stat_tracker.last_support_lr, self.stat_tracker.global_opt_step)
                    self.tb_writer.add_scalar("train/reconstructor_lr", self.stat_tracker.last_recon_lr, self.stat_tracker.global_opt_step)

                    if analytics:
                        if (self.stat_tracker.global_opt_step % self.params.log_freq) == 0:
                            self.stat_tracker.snapshot_per_k_history(self.stat_tracker.global_opt_step)
                        if histograms:
                            self.tb_writer.add_histogram("train/logits", logits.detach().cpu().numpy(), self.stat_tracker.global_opt_step)

                    # Heavy image grids go to rotating image writer
                    if save_images and ((self.stat_tracker.global_opt_step) % self.params.log_freq) == 0 and img_logger is not None:
                        # First sample imagery across K potentials
                        first_img = generator(z[:1])  # [1,C,H,W]
                        img_logger.log_triplet(
                            tag_prefix="images",
                            x0=img1_bk, x1=img2_bk, x2=first_img,
                            step=self.stat_tracker.global_opt_step,
                            n_vis=VIS_IMAGE_K,
                        )

                    # Light figures stay in normal TB (small, infrequent)
                    if len(self.stat_tracker.ema_history) >= 2 and self.stat_tracker.global_opt_step % self.params.log_freq == 0:
                        hist_mat = np.stack(self.stat_tracker.ema_history, axis=1)[:, -30:]
                        fig = ImageViz.plot_heatmap(hist_mat, K=self.K,
                                                    title="Per-MLP EMA Accuracy over Time",
                                                    xlabel="optimizer step snapshot", ylabel="MLP index k")
                        self.tb_writer.add_figure("per_mlp/accuracy_heatmap", fig, global_step=self.stat_tracker.global_opt_step)
                        plt.close(fig)

                        fig_c = ImageViz.plot_confusion(self.stat_tracker.confusion, K=self.K)
                        self.tb_writer.add_figure("classifier/confusion_matrix", fig_c, global_step=self.stat_tracker.global_opt_step)
                        plt.close(fig_c)
                        for k in range(K):
                            self.tb_writer.add_histogram(f"potential_distribution/{k}",
                                                    potential_preds[:, k].reshape(-1).cpu(), self.stat_tracker.global_opt_step)
                # --------------------------- /TensorBoard logging ---------------------------

                # Timing, progress, persist
                step_dt = time.time() - iter_t0
                self.stat_tracker.push_step_time(step_dt)
                elapsed_time = time.time() - t0
                mean_step_time = self.stat_tracker.mean_step_time()
                total_opt_steps = math.ceil(self.params.max_iter / acc_steps)
                eta = (total_opt_steps - self.stat_tracker.global_opt_step) * mean_step_time

                self.stat_tracker.finalize_step(
                    step_idx=self.stat_tracker.global_opt_step,
                    window_means=win_means,
                    elapsed_from_start=elapsed_time,
                    mean_step_time=mean_step_time,
                    eta_seconds=eta,
                )
                self.log_progress(self.stat_tracker.global_opt_step, mean_step_time, elapsed_time, eta)

                if self.stat_tracker.global_opt_step % self.params.log_freq == 0:
                    self._write_stats_json()

            if save_checkpoints and (self.stat_tracker.global_opt_step % self.params.ckp_freq == 0):
                checkpoint_dict = {
                    'iter': self.stat_tracker.global_opt_step,
                    'support_sets': support_sets.state_dict(),
                    'reconstructor': reconstructor.state_dict(),
                    'support_opt': support_sets_optim.state_dict(),
                    'recon_opt': reconstructor_optim.state_dict(),
                    'support_sched': sched_support.state_dict(),
                    'recon_sched': sched_recon.state_dict(),
                }
                torch.save(checkpoint_dict, self.checkpoint)

        # === end loop ===
        torch.save(support_sets.state_dict(), osp.join(self.models_dir, 'support_sets.pt'))
        reconstructor_model_filename = osp.join(self.models_dir, 'reconstructor.pt')
        torch.save(reconstructor.module.state_dict() if self.multi_gpu else reconstructor.state_dict(),
                reconstructor_model_filename)

        # Close rotating image writer cleanly
        if img_logger is not None:
            img_logger.close()

        print("\n" * 10)
        print("#.Training completed -- Total elapsed time: {}.".format(sec2dhms(elapsed_time)))
        print("#. Copy {} to {}...".format(self.wip_dir, self.complete_dir))
        try:
            shutil.copytree(src=self.wip_dir, dst=self.complete_dir, ignore=shutil.ignore_patterns('checkpoint.pt'))
            print(" \\__Done!")
        except IOError as e:
            print(" \\__Already exists -- {}".format(e))