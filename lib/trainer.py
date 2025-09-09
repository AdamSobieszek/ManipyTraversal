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
from .aux import CosineScheduleWithWarmup, build_adamw
from .validity_loss import KLPath


class DataParallelPassthrough(nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super(DataParallelPassthrough, self).__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


class Trainer(object):
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
            run_name = f"run_{int(time.time())}"
            self.tb_dir = osp.join(self.wip_dir, 'tensorboard', run_name)
            os.makedirs(self.tb_dir, exist_ok=True)
            self.tb = program.TensorBoard()
            self.tb.configure(argv=[None, '--logdir', osp.join(self.wip_dir, 'tensorboard')])
            self.tb_url = self.tb.launch()
            print(f"#. Start TensorBoard at {self.tb_url} (run: {run_name})")
            self.tb_writer = SummaryWriter(log_dir=self.tb_dir)

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

    def _to_uint8_images(self, x):
        """
        Normalize tensor images from [-1, 1] or [0,1] to [0,1] and clamp.
        x: (B, C, H, W)
        """
        x = x.detach().cpu()
        if x.min() < 0.0:
            x = (x + 1.0) / 2.0
        x = x.clamp(0.0, 1.0)
        return x

    def _plot_potential_distribution(self, writer, tag_prefix, potential_preds, iteration):
        """
        potential_preds: [B,K,1]
        """
        for k in range(self.K):
            potential_preds = potential_preds.detach().cpu()
            writer.add_histogram(f"{tag_prefix}/{k}/potential_distribution", potential_preds[:, k].reshape(-1), iteration)

    def _log_image_triplet(self, writer, tag_prefix, x0, x1, x2, iteration, n_vis=8):
        """
        Works with the new train() call:
            _log_image_triplet(writer, "images", img1_bk, img2_bk, first_img, step, n_vis)

        Inputs:
        - x0: step-1 images, shape [B, K, C, H, W] or [B, C, H, W]
        - x1: step-2 images, shape [B, K, C, H, W] or [B, C, H, W]
        - x2: reference/original image(s), shape [1, C, H, W] or [B, C, H, W] or [B, K, C, H, W]

        Behavior:
        - Visualizes the first batch element, across up to n_vis support sets (K columns).
        - Logs two images:
            * "<tag_prefix>/triplet": rows = [orig, step1, step2]
            * "<tag_prefix>/diff_triplet_abs": rows = [|step1-orig|, |step2-orig|]
        """
        # Map to semantic names
        step1_src, step2_src, ref_src = x0, x1, x2

        def pick_first_batch_and_K(t, k_vis):
            # Accept [B,K,C,H,W], [B,C,H,W], or [C,H,W]
            if t.ndim == 5:            # [B,K,C,H,W]
                return t[0, :k_vis]    # [k_vis,C,H,W]
            elif t.ndim == 4:          # [B,C,H,W]
                return t[:k_vis]       # [min(B,k_vis),C,H,W]
            elif t.ndim == 3:          # [C,H,W]
                return t.unsqueeze(0).repeat(k_vis, 1, 1, 1)
            else:
                raise ValueError(f"Unexpected tensor ndim={t.ndim} for visualization")

        # Determine how many K columns we can show
        if step1_src.ndim == 5:
            K = step1_src.shape[1]
            k_vis = min(int(n_vis), K)
        elif step1_src.ndim == 4:
            # No K axis; fall back to batch columns
            k_vis = min(int(n_vis), step1_src.shape[0])
        else:
            k_vis = int(n_vis)

        # Slice/select to [k_vis, C, H, W]
        step1 = pick_first_batch_and_K(step1_src, k_vis)
        step2 = pick_first_batch_and_K(step2_src, k_vis)

        # Reference/original: repeat to match K columns if needed
        if ref_src.ndim == 5:
            orig = ref_src[0, :k_vis]  # [k_vis,C,H,W]
        elif ref_src.ndim == 4:
            if ref_src.shape[0] == 1:
                orig = ref_src.repeat(k_vis, 1, 1, 1)
            else:
                orig = ref_src[:k_vis]
        elif ref_src.ndim == 3:
            orig = ref_src.unsqueeze(0).repeat(k_vis, 1, 1, 1)
        else:
            raise ValueError(f"Unexpected reference tensor ndim={ref_src.ndim}")

        # Normalize to [0,1] for grid
        orig_n = self._to_uint8_images(orig)
        s1_n   = self._to_uint8_images(step1)
        s2_n   = self._to_uint8_images(step2)

        # Grids: K columns, rows stacked vertically
        grid_o  = make_grid(orig_n, nrow=k_vis)
        grid_s1 = make_grid(s1_n,   nrow=k_vis)
        grid_s2 = make_grid(s2_n,   nrow=k_vis)
        stacked_grid = torch.cat([grid_o, grid_s1, grid_s2], dim=1)  # stack along height

        # Absolute differences vs original
        diff1 = (s1_n - orig_n).abs()
        diff2 = (s2_n - orig_n).abs()
        grid_d1 = make_grid(diff1, nrow=k_vis)
        grid_d2 = make_grid(diff2, nrow=k_vis)
        stacked_diff_grid = torch.cat([grid_d1, grid_d2], dim=1)

        # Write to TensorBoard
        writer.add_image(f"{tag_prefix}/triplet", stacked_grid, iteration)
        writer.add_image(f"{tag_prefix}/diff_triplet_abs", stacked_diff_grid, iteration)

    def _plot_heatmap(self, mat_t_by_k, title, xlabel, ylabel):
        """
        mat_t_by_k: np.array of shape [K, T_hist] (preferred) or [T_hist, K]
        Returns a matplotlib Figure.
        """
        arr = np.array(mat_t_by_k)
        if arr.ndim == 2 and arr.shape[0] != self.K:
            arr = arr.T  # make it [K, T_hist]
        fig, ax = plt.subplots(figsize=(max(6, arr.shape[1] * 0.15), max(4, self.K * 0.15)))
        im = ax.imshow(arr, aspect='auto', origin='lower', interpolation='nearest')
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_yticks(np.arange(self.K))
        ax.set_yticklabels([str(i) for i in range(self.K)])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        return fig

    def _plot_confusion(self, conf_mat_nd):
        """
        conf_mat_nd: numpy.ndarray [K, K], rows = true k, cols = predicted k
        """
        cm = torch.tensor(conf_mat_nd, dtype=torch.float32)
        row_sums = cm.sum(dim=1, keepdim=True).clamp(min=1.0)
        cm_norm = (cm / row_sums).cpu().numpy()
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm_norm, interpolation='nearest', aspect='auto', origin='lower')
        ax.set_title("Classifier Confusion (row=true k, col=pred k)")
        ax.set_xlabel("predicted k")
        ax.set_ylabel("true k")
        ax.set_xticks(np.arange(self.K))
        ax.set_yticks(np.arange(self.K))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        return fig

    # -----------------------------------------------------------------------
    def _pack_BK(self, x_bk: torch.Tensor):
        """
        Flatten [B,K,...] → [B*K,...] in a *known* order and return the mapping + targets.
        Order: (b=0,k=0..K-1), (b=1,k=0..K-1), ...
        """
        assert x_bk.dim() >= 2, f"Expected [B,K,...], got {tuple(x_bk.shape)}"
        B, K = x_bk.shape[:2]

        # Ensure [B,K,...] with K as dim=1 (guard against permuted tensors)
        if x_bk.stride(1) == 1 and x_bk.is_contiguous(memory_format=torch.contiguous_format):
            x_bk_c = x_bk
        else:
            x_bk_c = x_bk.contiguous()

        flat = x_bk_c.view(B * K, *x_bk_c.shape[2:])

        # Mapping & targets (same device as input)
        dev = x_bk.device
        b_idx = torch.arange(B, device=dev).repeat_interleave(K)     # [0,0,...,1,1,...]
        k_idx = torch.arange(K, device=dev).repeat(B)                # [0..K-1, 0..K-1, ...]
        targets = k_idx                                              # class k for row (b,k)

        return flat, targets, (b_idx, k_idx), (B, K)
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
        

    # ------------------------ per-k grad norms (stacked) ------------------------
    @torch.no_grad()
    def _per_k_grad_norms(self, support_sets) -> np.ndarray:
        K = support_sets.num_support_sets
        g2 = torch.zeros(K, dtype=torch.float32)

        for p in support_sets.parameters():
            g = p.grad
            if g is None:
                continue
            g = g.detach().float()
            if g.ndim == 0:
                continue
                
            # find which axis corresponds to K (don’t assume it’s dim 0)
            axes_with_K = [ax for ax, sz in enumerate(g.shape) if sz == K]
            if not axes_with_K:
                continue
            k_ax = axes_with_K[0]
            if k_ax != 0:
                g = g.movedim(k_ax, 0)  # put K in front
            g2 += g.float().reshape(K, -1).pow(2).sum(dim=1)
        return torch.sqrt(torch.clamp(g2, min=1e-12)).cpu().numpy()

    def loss_allK(self, support_sets, generator, reconstructor, z, t_index, acc_denominator: int):
        """
        One forward for all K in parallel. Ensures classifier gradients reach ψ/f via img2.
        """
        # PDE/OT forward
        potential_preds, latent1_bk, latent2_bk, loss_wave = support_sets(z, t_index, direction=+1)
        B, K, D = latent1_bk.shape


        # ---- pack latents for generator/classifier ----
        lat1_flat, targets, (b_idx, k_idx), (B, K) = self._pack_BK(latent1_bk)
        lat2_flat, _,      _,                  _   = self._pack_BK(latent2_bk)  # mapping identical by shape

        # Images
        img1 = generator(lat1_flat)
        img2 = generator(lat2_flat)

        # Detach img1 so CE must flow via img2 → latent2 → ψ/f
        img1_det = img1.detach()

        # Classify
        logits, _ = reconstructor(img1_det, img2)  # [B*K, K]
        cls_loss = self.cross_entropy(logits, targets)


        # KL (optional)
        kl_space = getattr(self.params, "kl_space", "latent")
        lambda_kl = float(getattr(self.params, "lambda_kl", 0.0))
        kl_bandwidth = getattr(self.params, "kl_bandwidth", None)
        kl_detach_ref = bool(getattr(self.params, "kl_detach_reference", True))

        if lambda_kl > 0.0:
            if kl_space == "latent":
                ref, manip = lat1_flat, lat2_flat
            elif kl_space == "image":
                ref, manip = img1_det, img2   # use detached ref consistently
            else:
                raise ValueError(f"Unknown kl_space={kl_space}")
            if kl_detach_ref:
                ref = ref.detach()
            kl_loss = self.kl_loss_fn(ref, manip, bandwidth=kl_bandwidth)
        else:
            kl_loss = torch.tensor(0.0, device=z.device, dtype=z.dtype)

        # Total loss
        loss = (
            self.params.lambda_cls * cls_loss
            + self.params.lambda_pde * loss_wave
            + lambda_kl * kl_loss
        )
        loss = loss / max(1, int(acc_denominator))


        loss.backward()


        # Logging views
        with torch.no_grad():
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)  # [B*K]
            entropy = -(probs * (probs.clamp_min(1e-8).log())).sum(dim=1).mean()

            z_bk = z.unsqueeze(1).expand(B, K, D)
            d1 = (latent1_bk - z_bk).norm(dim=2).mean()
            d2 = (latent2_bk - latent1_bk).norm(dim=2).mean()

            # reshape images back for visuals
            img1_bk = img1.detach().contiguous().view(B, K, *img1.shape[1:])
            img2_bk = img2.detach().contiguous().view(B, K, *img2.shape[1:])

        return (
            loss.detach(), cls_loss.detach(), loss_wave.detach(), kl_loss.detach(),
            logits.detach(), targets,
            latent1_bk.detach(), latent2_bk.detach(),
            img1_bk, img2_bk,
            float(entropy.item()), float(d1.item()), float(d2.item()),
            potential_preds.detach(),
        )

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
            safe_load_state_dict(support_opt, ckpt, 'support_opt')
            safe_load_state_dict(recon_opt, ckpt, 'recon_opt')
            safe_load_state_dict(support_sched, ckpt, 'support_sched')
            safe_load_state_dict(recon_sched, ckpt, 'recon_sched')
        return start_iter

    # ------------------------ optim/sched ------------------------
    def init_optimizers(self, support_sets, reconstructor, acc_steps: int):
        support_set_wd = float(getattr(self.params, "support_set_wd", 0.05))
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
                {"params": support_sets.F.parameters(), "weight_decay": 0.1, "lr": self.params.support_set_lr},
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

        if reset_weight_decay:
            # Rebuild optimizers (drop moments)
            support_sets_optim = build_adamw(
                [
                    {"params": support_sets.PSI.parameters(), "weight_decay": support_set_wd, "lr": self.params.support_set_lr},
                    {"params": support_sets.F.parameters(), "weight_decay": 0.1, "lr": self.params.support_set_lr},
                    {"params": [support_sets.c], "weight_decay": 0.0, "lr": self.params.support_set_lr},
                ],
                lr=self.params.support_set_lr, weight_decay=0.0, extra_no_decay_names=(), betas=betas, eps=eps
            )
            reconstructor_optim = build_adamw(
                reconstructor, lr=self.params.reconstructor_lr, weight_decay=reconstructor_wd, extra_no_decay_names=(), betas=betas, eps=eps
            )
            sched_support = CosineScheduleWithWarmup(support_sets_optim, num_warmup_steps=warmup_steps, num_training_steps=total_opt_steps, last_epoch=-1)
            sched_recon = CosineScheduleWithWarmup(reconstructor_optim, num_warmup_steps=warmup_steps, num_training_steps=total_opt_steps, last_epoch=-1)

        if reset_schedulers:
            sched_support = CosineScheduleWithWarmup(support_sets_optim, num_warmup_steps=warmup_steps, num_training_steps=total_opt_steps, last_epoch=-1)
            sched_recon = CosineScheduleWithWarmup(reconstructor_optim, num_warmup_steps=warmup_steps, num_training_steps=total_opt_steps, last_epoch=-1)
            if reset_weight_decay and not reset_start_iter:
                starting_opt_step = 0

        if reset_start_iter:
            starting_micro = 1
            opt_step_idx = 0
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

        acc_steps = max(1, int(getattr(self.params, "accumulate_grad_steps", 1)))
        VIS_IMAGE_K = min(16, self.K)

        (starting_micro, opt_step_idx,
         support_sets_optim, reconstructor_optim,
         sched_support, sched_recon) = self.init_optimizers(support_sets, reconstructor, acc_steps)

        # KL path config
        kl_mode = getattr(self.params, "kl_mode", "gaussian")
        kl_symmetric = bool(getattr(self.params, "kl_symmetric", True))
        kl_bandwidth = getattr(self.params, "kl_bandwidth", None)
        kl_detach_ref = bool(getattr(self.params, "kl_detach_reference", True))
        self.kl_loss_fn = KLPath(mode=kl_mode, symmetric=kl_symmetric, bandwidth=kl_bandwidth, detach_reference=kl_detach_ref)

        half_range = self.params.num_support_timesteps // 2

        t0 = time.time()

        # Early exit if already done
        if starting_micro > self.params.max_iter:
            print("#. This experiment has already been completed and can be found @ {}".format(self.wip_dir))
            print("#. Copy {} to {}...".format(self.wip_dir, self.complete_dir))
            try:
                shutil.copytree(src=self.wip_dir, dst=self.complete_dir, ignore=shutil.ignore_patterns('checkpoint.pt'))
                print("  \\__Done!")
            except IOError as e:
                print("  \\__Already exists -- {}".format(e))
            sys.exit()

        print(f"#. Start training from micro-step {starting_micro}")
        print(f"#. Training loop: {starting_micro} to {self.params.max_iter}")

        # === main loop: NO per-k loop; one pass covers all K ===
        for micro_idx, iteration in enumerate(range(starting_micro, self.params.max_iter + 1), start=1):
            iter_t0 = time.time()

            # Sample new z every micro step (decoupled from grad accumulation)
            z = sample_z(
                batch_size=self.params.batch_size,
                dim_z=generator.dim_z,
                truncation=self.params.z_truncation,
            )
            if self.use_cuda:
                z = z.cuda(non_blocking=True)
            elif self.use_mps:
                z = z.to(self.device)
            if getattr(generator, "shift_in_w_space", False):
                z = generator.get_w(z)

            # Random step index per sample
            t_idx = torch.randint(0, max(1, half_range - 1), (self.params.batch_size, 1), device=self.device)

            (loss, cls_loss, loss_wave, kl_loss, logits, targets,
             latent1_bk, latent2_bk, img1_bk, img2_bk, entropy, step1_norm, step2_norm,
             potential_preds) = self.loss_allK(
                support_sets, generator, reconstructor, z, t_idx, acc_denominator=acc_steps
            )

            # ===== analytics & stats =====
            with torch.no_grad():
                B = self.params.batch_size
                K = self.K
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)  # [B*K]
                preds_2d = preds.view(B, K)
                true_2d = torch.arange(K, device=self.device).unsqueeze(0).expand(B, K)
                batch_acc = (preds_2d == true_2d).float().mean().item()

                # Per-k accuracy and grad norm snapshots
                per_k_acc = (preds_2d == true_2d).float().mean(dim=0).cpu().numpy()  # [K]
                per_k_gn = self._per_k_grad_norms(support_sets) if analytics else None

                wave_dict = support_sets.get_losses()
                wave_dict['potential_std'] = potential_preds.std().item()
                wave_dict['xf_now'] = wave_dict['xf_now'].norm(dim=-1).mean().item()

                # Stat tracker accumulates raw window metrics
                self.stat_tracker.add_micro(
                    acc=batch_acc,
                    classification_loss=float(cls_loss.item()),
                    wave_loss=float(loss_wave.item()),
                    kl_loss=float(kl_loss.item()),
                    total_loss=float(loss.item()),
                    entropy=float(entropy),
                    step1_norm=float(step1_norm),
                    step2_norm=float(step2_norm),
                    **{k: float(v.item()) if torch.is_tensor(v) else float(v) for k, v in wave_dict.items() if k not in ("potential_preds")},
                )

                # Update per-k EMA stats by iterating k
                if analytics:
                    for k in range(K):
                        self.stat_tracker.update_per_k_after_micro(
                            true_k=int(k),
                            preds=preds_2d[:, k].detach().cpu().numpy(),
                            batch_size=int(B),
                            grad_norm_selected_mlp=(per_k_gn[k] if per_k_gn is not None else None),
                        )

            # ===== perform optimizer step at accumulation boundary =====
            if (micro_idx % acc_steps == 0) or (iteration == self.params.max_iter):
                # self.log_progress(self.stat_tracker.global_opt_step, mean_step_time, elapsed_time, eta)
                if analytics and self.tensorboard:
                    def module_grad_norm(mod):
                        total_sq = 0.0
                        for p in mod.parameters():
                            if p.grad is not None:
                                total_sq += float(p.grad.detach().to('cpu').pow(2).sum().item())
                        return math.sqrt(total_sq)
                    gn_support = module_grad_norm(support_sets.PSI) + module_grad_norm(support_sets.F)
                    gn_recon = module_grad_norm(reconstructor)
                    self.tb_writer.add_scalar("train/grad_norm/support_sets", gn_support, self.stat_tracker.global_opt_step)
                    self.tb_writer.add_scalar("train/grad_norm/reconstructor", gn_recon, self.stat_tracker.global_opt_step)

                torch.nn.utils.clip_grad_norm_(support_sets.parameters(), max_norm=3.0)
                torch.nn.utils.clip_grad_norm_(reconstructor.parameters(), max_norm=3.0)

                support_sets_optim.step(); reconstructor_optim.step()
                support_sets_optim.zero_grad(set_to_none=True); reconstructor_optim.zero_grad(set_to_none=True)
                sched_support.step(); sched_recon.step()

                # Snapshot LRs
                self.stat_tracker.set_lrs(sched_support.get_last_lr()[0], sched_recon.get_last_lr()[0])

                win_means = self.stat_tracker.close_window()

                # TensorBoard logging
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

                    if save_images and ((self.stat_tracker.global_opt_step) % self.params.log_freq) == 0:
                        # First sample imagery across K potentials
                        first_img = generator(z[:1])  # [1,C,H,W]
                        self._log_image_triplet(self.tb_writer, "images", img1_bk, img2_bk, first_img, self.stat_tracker.global_opt_step, n_vis=VIS_IMAGE_K)
                    # Heatmap of per-MLP EMA accuracy over time 
                    if len(self.stat_tracker.ema_history) >= 2: 
                        hist_mat = np.stack(self.stat_tracker.ema_history, axis=1)[:, -30:] 
                        fig = self._plot_heatmap(hist_mat, title="Per-MLP EMA Accuracy over Time", xlabel="optimizer step snapshot", ylabel="MLP index k") 
                        self.tb_writer.add_figure("per_mlp/accuracy_heatmap", fig, global_step=self.stat_tracker.global_opt_step) 
                        plt.close(fig) 
                        # Confusion matrix (normalized rows) 
                        fig_c = self._plot_confusion(self.stat_tracker.confusion) 
                        self.tb_writer.add_figure("classifier/confusion_matrix", fig_c, global_step=self.stat_tracker.global_opt_step) 
                        plt.close(fig_c) 
                        self._plot_potential_distribution(self.tb_writer, "potential_distribution", potential_preds, self.stat_tracker.global_opt_step)
                # ============================ /TensorBoard logging ===========================

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
                self._write_stats_json()

                if self.stat_tracker.global_opt_step % self.params.log_freq == 0:
                    self.log_progress(self.stat_tracker.global_opt_step, mean_step_time, elapsed_time, eta)

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

        elapsed_time = time.time() - t0
        support_sets_model_filename = osp.join(self.models_dir, 'support_sets.pt')
        torch.save(support_sets.state_dict(), support_sets_model_filename)
        reconstructor_model_filename = osp.join(self.models_dir, 'reconstructor.pt')
        torch.save(reconstructor.module.state_dict() if self.multi_gpu else reconstructor.state_dict(), reconstructor_model_filename)

        print("\n" * 10)
        print("#.Training completed -- Total elapsed time: {}.".format(sec2dhms(elapsed_time)))
        print("#. Copy {} to {}...".format(self.wip_dir, self.complete_dir))
        try:            
            shutil.copytree(src=self.wip_dir, dst=self.complete_dir, ignore=shutil.ignore_patterns('checkpoint.pt'))
            print(" \\__Done!")
        except IOError as e:
            print(" \\__Already exists -- {}".format(e))