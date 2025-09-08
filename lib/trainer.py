import sys
import os
import os.path as osp
import json
import torch
from torch import nn
import torch.backends.cudnn as cudnn
import numpy as np
import time
import shutil
import math
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from tensorboard import program
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
        else:
            self.params = params
        self.device = device
        self.use_cuda = use_cuda
        self.use_mps = use_mps
        self.multi_gpu = multi_gpu

        # Use TensorBoard
        self.tensorboard = self.params.tensorboard

        # Set output directory for current experiment (wip)
        self.wip_dir = osp.join("experiments", "wip", exp_dir)

        # Set directory for completed experiment
        self.complete_dir = osp.join("experiments", "complete", exp_dir)

        # Create log sub-directory and define stat.json file
        self.stats_json = osp.join(self.wip_dir, 'stats.json')
        if not osp.isfile(self.stats_json):
            with open(self.stats_json, 'w') as out:
                json.dump({}, out)

        # Create models sub-directory
        self.models_dir = osp.join(self.wip_dir, 'models')
        os.makedirs(self.models_dir, exist_ok=True)
        # Define checkpoint model file
        self.checkpoint = osp.join(self.models_dir, 'checkpoint.pt')

        # Setup TensorBoard
        self.tb_writer = None
        if self.tensorboard:
            import time
            # Create tensorboard sub-directory with time-based run name
            run_name = f"run_{int(time.time())}"
            print(f"#. TensorBoard run name: {run_name}")
            self.tb_dir = osp.join(self.wip_dir, 'tensorboard', run_name)
            os.makedirs(self.tb_dir, exist_ok=True)

            self.tb = program.TensorBoard()
            self.tb.configure(argv=[None, '--logdir', osp.join(self.wip_dir, 'tensorboard')])
            self.tb_url = self.tb.launch()
            print(f"#. Start TensorBoard at {self.tb_url} (run: {run_name})")

            self.tb_writer = SummaryWriter(log_dir=self.tb_dir)
        # --- Anti-overfitting knobs (defaults) ---
        self.ce_label_smoothing = float(getattr(self.params, "ce_label_smoothing", 0.00))
        self.conf_penalty_weight = float(getattr(self.params, "conf_penalty_weight", 0.0))

            # Define cross entropy loss (with optional label smoothing)
        self.cross_entropy = nn.CrossEntropyLoss(label_smoothing=self.ce_label_smoothing)

        # Statistics tracker (keeps rolling means printed to stdout)
        # (Now also owns per-MLP analytics, step timing, LR snapshots, and JSON-ready step records.)
        self.stat_tracker = TrainingStatTracker(
            ema_decay=getattr(self.params, "ema_decay", 0.9),
            ema_max_history=getattr(self.params, "ema_max_history", 200),
        )

        # ========= Enhanced logging state (set later in train() once K is known) =========
        self.K = None  # just for plotting helpers in this class (heatmap/confusion figs)


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

    def _log_image_triplet(self, writer, tag_prefix, x0, x1, x2, iteration, n_vis=8):
        """
        Logs two images: 
        - One grid with original, step1, step2 stacked vertically (each row is a batch grid).
        - One grid with abs differences (step1-orig, step2-orig) stacked vertically.
        """
        b = min(n_vis, x0.size(0))
        x0n = self._to_uint8_images(x0[:b])
        x1n = self._to_uint8_images(x1[:b])
        x2n = self._to_uint8_images(x2[:b])

        # Make horizontal grids for each
        grid0 = make_grid(x0n, nrow=b)
        grid1 = make_grid(x1n, nrow=b)
        grid2 = make_grid(x2n, nrow=b)

        # Stack vertically: shape [3*C, H, W] if C=channels
        stacked_grid = torch.cat([grid0, grid1, grid2], dim=1)  # stack along height

        # abs diffs vs original
        diff1 = (x1n - x0n).abs()
        diff2 = (x2n - x1n).abs()
        grid_d1 = make_grid(diff1, nrow=b)
        grid_d2 = make_grid(diff2, nrow=b)
        stacked_diff_grid = torch.cat([grid_d1, grid_d2], dim=1)  # stack along height

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

    def get_starting_iteration(
        self,
        support_sets,
        reconstructor,
        support_opt=None,
        recon_opt=None,
        support_sched=None,
        recon_sched=None,
    ):
        """
        Loads all available states from checkpoint:
        - models (support_sets, reconstructor)
        - optimizers (support_opt, recon_opt)   [if provided]
        - schedulers (support_sched, recon_sched) [if provided]
        Returns the stored optimizer-step index ('iter') or 1 if no checkpoint.
        """
        def safe_load_state_dict(obj, ckpt, name, strict=True):
            if obj is not None and name in ckpt:
                try:
                    # if strict is an available argument for load_state_dict, use it
                    if hasattr(obj, 'load_state_dict') and hasattr(obj.load_state_dict, 'strict'):
                        incompatibilities = obj.load_state_dict(ckpt[name], strict=strict)
                    else:
                        incompatibilities = obj.load_state_dict(ckpt[name])
                    if incompatibilities:
                        if str(incompatibilities) != "<All keys matched successfully>":
                            print(f"Warning: {name} loaded state_dict with non-strict mode")
                            print(f"Incompatibilities: {incompatibilities}")
                    
                except Exception as e:
                    print(f"Error loading state_dict for {name}: {e}")

        start_iter = 1
        if osp.isfile(self.checkpoint):
            ckpt = torch.load(self.checkpoint, map_location=self.device)
            start_iter = int(ckpt.get('iter', 1))

            # Model weights (allow non-strict to be robust to minor changes)
            safe_load_state_dict(support_sets, ckpt, 'support_sets', strict=False)
            safe_load_state_dict(reconstructor, ckpt, 'reconstructor', strict=False)

            # Optimizers (if both objects and states exist)
            safe_load_state_dict(support_opt, ckpt, 'support_opt')
            safe_load_state_dict(recon_opt, ckpt, 'recon_opt')

            # Schedulers (if both objects and states exist)
            safe_load_state_dict(support_sched, ckpt, 'support_sched')
            safe_load_state_dict(recon_sched, ckpt, 'recon_sched')

        return start_iter
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
        

    def loss(self, support_sets, generator, reconstructor, index, z, time_step, acc_steps):
        """
        Computes the total loss for one semantic index.
        - Keeps gradients to support_sets (the PDE potentials).
        - Keeps gradients from classifier through generator to latent1.
        - Uses latent2 as a target frame (with grads to potentials via classifier).
        - Optional KL regularizer can be computed in latent or image space; the 'reference'
        side can be detached via kl_detach_reference.
        """
        # --- Config for optional KL ---
        kl_space        = getattr(self.params, "kl_space", "latent")               # "latent" | "image"
        lambda_kl       = float(getattr(self.params, "lambda_kl", 0.0))            # main weight
        kl_bandwidth    = getattr(self.params, "kl_bandwidth", None)               # (passed inside self.kl_loss_fn if used)
        kl_detach_ref   = True
        
        # --- Ensure proper shapes/types for time ---
        if time_step.ndim == 1:
            time_step = time_step.unsqueeze(1)  # [B,1]
        time_step = time_step.to(z.dtype).to(z.device)

        # === PDE/OT step (potentials forward) ===
        # NOTE: WavePDE.forward returns: potential_preds, latent1 (with grads), latent2 (with grads), loss_wave (= total PDE loss)
        potential_preds, latent1, latent2, loss_wave = support_sets(index.item(), z, time_step, direction=+1)

        # === Generate images (generator is frozen but we allow grads through inputs) ===
        img_step1 = generator(latent1)        # grads flow back to latent1 (and thus to potentials)
        img_step2 = generator(latent2)       

        # === Optional KL path loss ===
        if lambda_kl > 0.0:
            if kl_space == "latent":
                initial_samples     = latent1
                manipulated_samples = latent2  
            elif kl_space == "image":
                initial_samples     = img_step1
                manipulated_samples = img_step2 
            else:
                raise ValueError(f"Unknown kl_space={kl_space}, expected 'latent' or 'image'.")

            if kl_detach_ref:
                initial_samples = initial_samples.clone().detach()

            kl_loss = self.kl_loss_fn(initial_samples, manipulated_samples, bandwidth=kl_bandwidth)
        else:
            kl_loss = torch.tensor(0.0, device=self.device, dtype=z.dtype)

        # === Semantic classifier (main discriminability signal) ===
        # logits shape: [B, K]; 'target' is [B] with the semantic index
        predicted_support_sets_indices, _ = reconstructor(img_step1, img_step2)
        target = index.repeat(self.params.batch_size)
        classification_loss = self.cross_entropy(predicted_support_sets_indices, target)

        # === Total loss ===
        loss = (
            self.params.lambda_cls * classification_loss
            + self.params.lambda_pde * loss_wave
            + lambda_kl * kl_loss
        )
        # gradient accumulation normalize
        loss = loss / max(1, int(acc_steps))
        loss.backward()

        return (
            loss.detach(),
            classification_loss.detach(),
            loss_wave.detach(),
            kl_loss.detach(),
            predicted_support_sets_indices.detach(),
            target,
            latent1,   
            latent2,    
            img_step1,
            img_step2,
        )


    def train(self, generator, support_sets, reconstructor):
        histograms = True
        save_images = True
        save_checkpoints = True
        analytics = True

        if not osp.isfile(self.checkpoint):
            # Save initial `support_sets` model as `support_sets_init.pt`
            torch.save(support_sets.state_dict(), osp.join(self.models_dir, 'support_sets_init.pt'))
        else:
            print("#. checkpoint found, skipping contrastive pretraining.")

        # ===================== Modes / devices =====================
        generator     = generator.to(self.device).eval()   # frozen, but keep graph through inputs
        generator.requires_grad_(False)                    # no grads to generator params

        support_sets  = support_sets.to(self.device).train()    # potentials (ψ, f) learnable
        reconstructor = reconstructor.to(self.device).train()   # classifier learnable

        # ===================== Bookkeeping =====================
        self.K = int(self.params.num_support_sets)
        self.stat_tracker.init_per_k(self.K)

        # Gradient accumulation sanity check:
        acc_steps = max(1, int(getattr(self.params, "accumulate_grad_steps", 1)))
        if acc_steps > self.K:
            raise ValueError(
                f"accumulate_grad_steps ({acc_steps}) must be ≤ num_support_sets K ({self.K}) "
                "to guarantee unique indices within each accumulation window."
            )
        VIS_IMAGE_N = min(16, self.K, acc_steps)

        # helper to draw a unique k-sequence for a window of length win_len (≤ K)
        def draw_unique_k_sequence(K, win_len, device):
            # random permutation of [0..K-1], then take the first win_len
            perm = torch.randperm(K, device=device)
            return perm[:win_len]

        # state for the current accumulation window
        current_z = None
        k_seq = None
        k_ptr = 0
        win_len = None  # number of micro-steps in the current window (handles last partial window)

        # --- hyperparams for wd (with sensible defaults) ---
        support_set_wd  = float(getattr(self.params, "support_set_wd", 0.05))
        reconstructor_wd = float(getattr(self.params, "reconstructor_wd", 0.001))
        betas = tuple(getattr(self.params, "adam_betas", (0.9, 0.999)))
        eps = float(getattr(self.params, "adam_eps", 1e-8))

        # --- restart flags (all default False) ---
        reset_lr          = bool(getattr(self.params, "reset_lr", True))
        reset_weight_decay = bool(getattr(self.params, "reset_weight_decay", False))
        reset_schedulers  = bool(getattr(self.params, "reset_schedulers", False))
        reset_start_iter  = bool(getattr(self.params, "reset_start_iter", False))

        # --- create optimizer(s) with split weight decay ---
        support_sets_optim = build_adamw(
            support_sets.PSI_SET,
            lr=self.params.support_set_lr,
            weight_decay=support_set_wd,
            extra_no_decay_names=(),                      
            extra_no_decay_params=(getattr(support_sets, "c", None),),  
            betas=betas,
            eps=eps,
        )
        support_sets_optim.add_param_group({"params": support_sets.F_POT_SET.parameters(), "weight_decay": 0.1, "lr": self.params.support_set_lr})
        reconstructor_optim = build_adamw(
            reconstructor,
            lr=self.params.reconstructor_lr,
            weight_decay=reconstructor_wd,
            extra_no_decay_names=(),
            betas=betas,
            eps=eps,
        )

        # --- create schedulers (we may re-init them below if needed) ---
        total_opt_steps = math.ceil(self.params.max_iter / acc_steps)
        warmup_steps = math.ceil(self.params.warmup_fraction * total_opt_steps)
        if reset_schedulers:
            sched_support = CosineScheduleWithWarmup(
                support_sets_optim, num_warmup_steps=warmup_steps,
                num_training_steps=total_opt_steps, last_epoch=-1
            )
            sched_recon = CosineScheduleWithWarmup(
                reconstructor_optim, num_warmup_steps=warmup_steps,
                num_training_steps=total_opt_steps, last_epoch=-1
            )

        # --- load models/opts/schedulers if checkpoint exists ---
        starting_opt_step = self.get_starting_iteration(
            support_sets, reconstructor,
            support_opt=support_sets_optim,
            recon_opt=reconstructor_optim,
            support_sched=sched_support,
            recon_sched=sched_recon,
        )

        # ===== apply restart logic =====
        # If only LR is reset: keep optimizer state but overwrite group LRs and
        # update schedulers' base_lrs so cosine scale uses the new starting LR.
        if reset_lr:
            for g in support_sets_optim.param_groups:
                g["lr"] = float(self.params.support_set_lr)
            for g in reconstructor_optim.param_groups:
                g["lr"] = float(self.params.reconstructor_lr)

        # If WD is reset: rebuild optimizers (this naturally drops moment state,
        # which is typically what you want when "restarting" WD).
        if reset_weight_decay:
            support_sets_optim = build_adamw(
                support_sets.PSI_SET, lr=self.params.support_set_lr, weight_decay=support_set_wd,
                extra_no_decay_names=("c",), betas=betas, eps=eps
            )
            support_sets_optim.add_param_group({"params": support_sets.F_POT_SET.parameters(), "weight_decay": 0.1, "lr": self.params.support_set_lr})
            reconstructor_optim = build_adamw(
                reconstructor, lr=self.params.reconstructor_lr, weight_decay=reconstructor_wd,
                extra_no_decay_names=(), betas=betas, eps=eps
            )



        half_range = self.params.num_support_timesteps // 2

        kl_mode         = getattr(self.params, "kl_mode", "gaussian")           # "gaussian" | "kde"
        kl_symmetric    = bool(getattr(self.params, "kl_symmetric", True))      # KL(P||Q) + KL(Q||P)
        kl_bandwidth    = getattr(self.params, "kl_bandwidth", None)            # for kde: None/"median"/"scott"/float
        kl_detach_ref   = bool(getattr(self.params, "kl_detach_reference", True))  # don't backprop through initial set
        self.kl_loss_fn = KLPath(
            mode=kl_mode,
            symmetric=kl_symmetric,
            bandwidth=kl_bandwidth,
            detach_reference=kl_detach_ref,
        )

        # If schedulers are reset: re-init them from step 0 (warmup restarts).
        if reset_schedulers or reset_weight_decay:
            sched_support = CosineScheduleWithWarmup(
                support_sets_optim, num_warmup_steps=warmup_steps,
                num_training_steps=total_opt_steps, last_epoch=-1
            )
            sched_recon = CosineScheduleWithWarmup(
                reconstructor_optim, num_warmup_steps=warmup_steps,
                num_training_steps=total_opt_steps, last_epoch=-1
            )
            # also reset the stored opt-step index so logs align
            if not reset_start_iter:
                # If you restarted schedulers but didn't explicitly request iteration reset,
                # we'll keep the iteration unless you *also* restarted WD (fresh start).
                if reset_weight_decay:
                    starting_opt_step = 0

        # start position (convert optimizer-step index -> micro-step index)
        if reset_start_iter:
            starting_micro = 1
            opt_step_idx = 0
        else:
            # Resume *after* the last finished optimizer step
            starting_micro = starting_opt_step * acc_steps + 1
            opt_step_idx = starting_opt_step

        # zero grads ONCE before the loop
        support_sets_optim.zero_grad(set_to_none=True)
        reconstructor_optim.zero_grad(set_to_none=True)

        t0 = time.time()
        opt_step_idx = 0

        # Parallelize if needed
        if self.multi_gpu:
            print("#. Parallelize G, R over {} GPUs...".format(torch.cuda.device_count()))
            generator = DataParallelPassthrough(generator)
            reconstructor = DataParallelPassthrough(reconstructor)
            cudnn.benchmark = True

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

        print("#. Start training from micro-step {}".format(starting_micro))
        print(f"#. Training loop: {starting_micro} to {self.params.max_iter}")

        # zero grads ONCE before the loop
        support_sets_optim.zero_grad(set_to_none=True)
        reconstructor_optim.zero_grad(set_to_none=True)

        # === main loop ===
        for micro_idx, iteration in enumerate(range(starting_micro, self.params.max_iter + 1), start=1):
            iter_t0 = time.time()


            # Sample index k and timestep t
            # ---- start of accumulation window? ----
            # micro_idx counts 1..N over micro-steps
            window_start = ((micro_idx - 1) % support_sets.num_support_sets) == 0

            if window_start:
                imgs_orig = [[] for _ in range(VIS_IMAGE_N)]
                imgs_step1 = [[] for _ in range(VIS_IMAGE_N)]
                imgs_step2 = [[] for _ in range(VIS_IMAGE_N)]
                # how many micro-steps remain including this one?
                micros_left = self.params.max_iter - iteration + 1
                win_len = min(support_sets.num_support_sets, micros_left)

                # 1) sample z ONCE per window
                current_z = sample_z(batch_size=self.params.batch_size,
                                    dim_z=generator.dim_z,
                                    truncation=self.params.z_truncation)
                if self.use_cuda:
                    current_z = current_z.cuda(non_blocking=True)
                elif self.use_mps:
                    current_z = current_z.to(self.device)

                if getattr(generator, "shift_in_w_space", False) is True:
                    current_z = generator.get_w(current_z)

                img1 = generator(current_z[:1])
                for i in range(VIS_IMAGE_N):
                    imgs_orig[i] = img1

                # 2) draw k indices WITHOUT replacement for this window
                k_seq = draw_unique_k_sequence(self.K, win_len, self.device)
                k_ptr = 0


            # use the shared z for this micro-step
            z = current_z

            # pick the next unique k for this micro-step
            k = int(k_seq[k_ptr].item())
            k_ptr += 1

            index = torch.tensor([k], device=self.device)
            t_idx = torch.randint(0, max(1, half_range - 1), (1,), device=self.device)
            time_step = t_idx.to(z).repeat(self.params.batch_size, 1)


            # KL loss and outputs
            (loss, classification_loss, loss_wave, kl_loss,
             logits, target, latent1, latent2, img_step1, img_step2) = self.loss(
                support_sets, generator, reconstructor, index, z, time_step, acc_steps
            )

            # For visual logging (keep here)
            if k < VIS_IMAGE_N:
                imgs_step1[k] = img_step1[:1]
                imgs_step2[k] = img_step2[:1]



            # ---- Enhanced analytics (safe device handling) ----            
            with torch.no_grad():
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                entropy = -(probs * (probs.clamp_min(1e-8).log())).sum(dim=1).mean()

                # Latent step norms
                delta1 = (latent1 - z).detach()
                delta2 = (latent2 - latent1).detach()
                step1_norm = float(delta1.norm(dim=1).mean())
                step2_norm = float(delta2.norm(dim=1).mean())

                # Per-MLP selected grad norm (only selected MLP has grads)
                grad_norm_selected = None
                if analytics:
                    mlp_params = list(support_sets.PSI_SET[int(index.item())].parameters())
                    if len(mlp_params) > 0:
                        g2 = 0.0
                        for p in mlp_params:
                            if p.grad is not None:
                                g2 += float(p.grad.detach().to('cpu').pow(2).sum())
                        grad_norm_selected = math.sqrt(max(g2, 1e-12))

                wave_dict = support_sets.get_losses()
                wave_dict['potential_std'] = wave_dict['potential_preds'].std()
                wave_dict['xf_now'] = wave_dict['xf_now'].norm(dim=-1).mean()
                self.stat_tracker.add_micro(
                    acc=float((preds == index.item()).float().mean().item()),
                    classification_loss=float(classification_loss.item()),
                    wave_loss=float(loss_wave.item()),
                    kl_loss=float(kl_loss.item()),
                    total_loss=float((loss).item()),
                    entropy=float(entropy.item()),
                    step1_norm=step1_norm,
                    step2_norm=step2_norm,
                    **wave_dict,
                )
                
                # Per-MLP analytics in tracker (CPU numpy)
                self.stat_tracker.update_per_k_after_micro(
                    true_k=int(index.item()),
                    preds=preds.detach().to('cpu').numpy(),
                    batch_size=int(self.params.batch_size),
                    grad_norm_selected_mlp=grad_norm_selected if analytics else None,
                )
            
            if (micro_idx % acc_steps == 0) or (iteration == self.params.max_iter):
                if analytics and self.tensorboard:
                    # Global grad norms (CPU, correct L2 norm)
                    def module_grad_norm(mod):
                        total_sq = 0.0
                        for p in mod.parameters():
                            if p.grad is not None:
                                total_sq += float(p.grad.detach().to('cpu').pow(2).sum().item())
                        return math.sqrt(total_sq)
                    gn_support = module_grad_norm(support_sets.PSI_SET)
                    gn_support_f = module_grad_norm(support_sets.F_POT_SET)
                    gn_recon = module_grad_norm(reconstructor)
                    self.tb_writer.add_scalar("train/grad_norm/psi_sets", gn_support, opt_step_idx)
                    self.tb_writer.add_scalar("train/grad_norm/f_potential_sets", gn_support_f, opt_step_idx)
                    self.tb_writer.add_scalar("train/grad_norm/reconstructor", gn_recon, opt_step_idx)
                # Gradient clipping before optimizer step
                torch.nn.utils.clip_grad_norm_(support_sets.parameters(), max_norm=3.0)
                torch.nn.utils.clip_grad_norm_(reconstructor.parameters(), max_norm=3.0)
                
                support_sets_optim.step()
                reconstructor_optim.step()
                support_sets_optim.zero_grad(set_to_none=True)
                reconstructor_optim.zero_grad(set_to_none=True)

                sched_support.step()
                sched_recon.step()
                opt_step_idx += 1

                # Snapshot LRs into tracker
                self.stat_tracker.set_lrs(
                    sched_support.get_last_lr()[0],
                    sched_recon.get_last_lr()[0],
                )

                # Window means (aggregated over the last acc_steps micro-steps)
                win_means = self.stat_tracker.close_window()

                # ============================ TensorBoard logging ============================
                if self.tensorboard:
                    # Scalars written at optimizer-step cadence
                    for key, value in win_means.items():
                        self.tb_writer.add_scalar(f"train/{key}", float(value), opt_step_idx)
                    self.tb_writer.add_scalar("train/support_sets_lr", self.stat_tracker.last_support_lr, opt_step_idx)
                    self.tb_writer.add_scalar("train/reconstructor_lr", self.stat_tracker.last_recon_lr, opt_step_idx)

                    if analytics:
                        # Wave speed c stats (support_sets.c: [K,1])
                        c_vals = support_sets.c.detach().view(-1).cpu().numpy()
                        # self.tb_writer.add_scalar("train/c_stats/mean", float(c_vals.mean()), opt_step_idx)
                        # self.tb_writer.add_scalar("train/c_stats/std", float(c_vals.std()), opt_step_idx)
                        # self.tb_writer.add_scalar("train/c_stats/min", float(c_vals.min()), opt_step_idx)
                        # self.tb_writer.add_scalar("train/c_stats/max", float(c_vals.max()), opt_step_idx)

                        # Periodic snapshots for heatmap/confusion
                        if (micro_idx//acc_steps % self.params.log_freq) == 0:
                            self.stat_tracker.snapshot_per_k_history(opt_step_idx)

                        # Optional histograms
                        if histograms:
                            self.tb_writer.add_histogram("train/logits", logits.detach().cpu().numpy(), opt_step_idx)
                            self.tb_writer.add_histogram("train/probs", probs.detach().cpu().numpy(), opt_step_idx)
                            self.tb_writer.add_histogram("train/latent_step_norm/step1",
                                                         (delta1.norm(dim=1)).detach().cpu().numpy(), opt_step_idx)
                            self.tb_writer.add_histogram("train/latent_step_norm/step2",
                                                         (delta2.norm(dim=1)).detach().cpu().numpy(), opt_step_idx)
                            self.tb_writer.add_histogram("train/potential_preds", wave_dict['potential_preds'].detach().cpu().numpy().reshape(-1), opt_step_idx)
                            self.tb_writer.add_histogram("per_mlp/ema_accuracy", self.stat_tracker.per_k_ema_acc, opt_step_idx)
                            self.tb_writer.add_histogram("per_mlp/ema_grad_norm", self.stat_tracker.per_k_ema_grad, opt_step_idx)
                            self.tb_writer.add_histogram("per_mlp/selection_counts", self.stat_tracker.per_k_select_counts, opt_step_idx)
                            self.tb_writer.add_histogram("meta/timestep_idx", time_step[:, 0].detach().cpu().numpy(), opt_step_idx)
                            self.tb_writer.add_histogram("meta/predicted_k", preds.detach().cpu().numpy(), opt_step_idx)
                            self.tb_writer.add_histogram("meta/true_k", target.detach().cpu().numpy(), opt_step_idx)

                    # Images & figures (on optimizer-step cadence)
                    if save_images and ((micro_idx)//acc_steps % self.params.log_freq) == 0:
                        print([isinstance(img, torch.Tensor) for img in imgs_step1])
                        print([isinstance(img, torch.Tensor) for img in imgs_step2])
                        self._log_image_triplet(self.tb_writer, "images", torch.cat(imgs_orig), torch.cat(imgs_step1), torch.cat(imgs_step2),
                                                opt_step_idx, n_vis=min(VIS_IMAGE_N, self.params.batch_size))

                        # Heatmap of per-MLP EMA accuracy over time
                        if len(self.stat_tracker.ema_history) >= 2:
                            hist_mat = np.stack(self.stat_tracker.ema_history, axis=1)  # [K, T_hist]
                            fig = self._plot_heatmap(hist_mat, title="Per-MLP EMA Accuracy over Time",
                                                        xlabel="optimizer step snapshot",
                                                        ylabel="MLP index k")
                            self.tb_writer.add_figure("per_mlp/accuracy_heatmap", fig, global_step=opt_step_idx)
                            plt.close(fig)

                        # Confusion matrix (normalized rows)
                        fig_c = self._plot_confusion(self.stat_tracker.confusion)
                        self.tb_writer.add_figure("classifier/confusion_matrix", fig_c, global_step=opt_step_idx)
                        plt.close(fig_c)
                # ============================ /TensorBoard logging ===========================

                # Timing & ETA (optimizer-step based)
                step_dt = time.time() - iter_t0
                self.stat_tracker.push_step_time(step_dt)
                elapsed_time = time.time() - t0
                mean_step_time = self.stat_tracker.mean_step_time()
                total_opt_steps = math.ceil(self.params.max_iter / acc_steps)
                eta = (total_opt_steps - opt_step_idx) * mean_step_time

                # Persist step stats to json + stdout progress
                self.stat_tracker.finalize_step(
                    step_idx=opt_step_idx,
                    window_means=win_means,
                    elapsed_from_start=elapsed_time,
                    mean_step_time=mean_step_time,
                    eta_seconds=eta,
                )
                self._write_stats_json()

                if micro_idx % self.params.log_freq == 0:
                    self.log_progress(opt_step_idx, mean_step_time, elapsed_time, eta)

                # Save checkpoint based on optimizer-step cadence (optional)
                if save_checkpoints and (opt_step_idx % self.params.ckp_freq == 0):
                    checkpoint_dict = {
                        'iter': opt_step_idx,
                        'support_sets': support_sets.state_dict(),
                        'reconstructor': reconstructor.state_dict(),
                        'support_opt': support_sets_optim.state_dict(),
                        'recon_opt': reconstructor_optim.state_dict(),
                        'support_sched': sched_support.state_dict(),
                        'recon_sched': sched_recon.state_dict(),
                    }
                    torch.save(checkpoint_dict, self.checkpoint)

        elapsed_time = time.time() - t0

        # Save final models
        support_sets_model_filename = osp.join(self.models_dir, 'support_sets.pt')
        torch.save(support_sets.state_dict(), support_sets_model_filename)

        reconstructor_model_filename = osp.join(self.models_dir, 'reconstructor.pt')
        torch.save(reconstructor.module.state_dict() if self.multi_gpu else reconstructor.state_dict(),
                   reconstructor_model_filename)

        for _ in range(10):
            print()
        print("#.Training completed -- Total elapsed time: {}.".format(sec2dhms(elapsed_time)))

        print("#. Copy {} to {}...".format(self.wip_dir, self.complete_dir))
        try:
            shutil.copytree(src=self.wip_dir, dst=self.complete_dir, ignore=shutil.ignore_patterns('checkpoint.pt'))
            print("  \\__Done!")
        except IOError as e:
            print("  \\__Already exists -- {}".format(e))