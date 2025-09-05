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

from torch.optim.lr_scheduler import _LRScheduler

class CosineScheduleWithWarmup(_LRScheduler):
    """
    A custom learning rate scheduler that implements a linear warmup followed
    by a cosine decay. This avoids the need for the `transformers` library.
    """
    def __init__(self, optimizer, num_warmup_steps: int, num_training_steps: int, last_epoch: int = -1):
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.num_warmup_steps:
            progress = float(self.last_epoch) / float(max(1, self.num_warmup_steps))
            return [base_lr * progress for base_lr in self.base_lrs]
        
        progress = float(self.last_epoch - self.num_warmup_steps) / float(max(1, self.num_training_steps - self.num_warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        
        return [base_lr * cosine_decay for base_lr in self.base_lrs]

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    return CosineScheduleWithWarmup(optimizer, num_warmup_steps, num_training_steps, last_epoch)
    

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

        # Array of iteration times
        self.iter_times = np.array([])

        # Statistics tracker (keeps rolling means printed to stdout)
        self.stat_tracker = TrainingStatTracker()

        # ========= Enhanced logging state (set later in train() once K is known) =========
        self.K = None
        self.ema_decay = getattr(self.params, "ema_decay", 0.9)  # moving average decay for per-MLP accuracy/grad
        self.per_k_ema_acc = None     # [K] (CPU)
        self.per_k_ema_grad = None    # [K] (CPU)
        self.per_k_select_counts = None  # [K] (CPU, long)
        self.confusion = None         # [K, K] (CPU, long)
        self.ema_history = []         # list of np.array(K,)
        self.iter_history = []        # list of iteration indices
        self.max_history = getattr(self.params, "ema_max_history", 200)  # cap heatmap history length

    # ------------------------ helpers for TB visuals ------------------------

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
        Logs three grids: original, step1, step2 and their absolute differences.
        """
        b = min(n_vis, x0.size(0))
        x0n = self._to_uint8_images(x0[:b])
        x1n = self._to_uint8_images(x1[:b])
        x2n = self._to_uint8_images(x2[:b])

        grid0 = make_grid(x0n, nrow=b)
        grid1 = make_grid(x1n, nrow=b)
        grid2 = make_grid(x2n, nrow=b)

        # abs diffs vs original
        diff1 = (x1n - x0n).abs()
        diff2 = (x2n - x0n).abs()
        grid_d1 = make_grid(diff1, nrow=b)
        grid_d2 = make_grid(diff2, nrow=b)

        writer.add_image(f"{tag_prefix}/orig", grid0, iteration)
        writer.add_image(f"{tag_prefix}/step1", grid1, iteration)
        writer.add_image(f"{tag_prefix}/step2", grid2, iteration)
        writer.add_image(f"{tag_prefix}/diff_step1_abs", grid_d1, iteration)
        writer.add_image(f"{tag_prefix}/diff_step2_abs", grid_d2, iteration)

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

    def _plot_confusion(self, conf_mat):
        """
        conf_mat: torch.Tensor [K, K] on CPU, rows = true k, cols = predicted k
        """
        cm = conf_mat.float()
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

    def get_starting_iteration(self, support_sets, reconstructor):
        starting_iter = 1
        if osp.isfile(self.checkpoint):
            checkpoint_dict = torch.load(self.checkpoint, map_location=self.device)
            starting_iter = checkpoint_dict['iter']
            support_sets.load_state_dict(checkpoint_dict['support_sets'])
            reconstructor.load_state_dict(checkpoint_dict['reconstructor'])
        return starting_iter

    def log_progress(self, iteration, mean_iter_time, elapsed_time, eta):
        stats = self.stat_tracker.get_means()

        # Update training statistics json file
        with open(self.stats_json) as f:
            stats_dict = json.load(f)
        stats_dict.update({iteration: stats})
        with open(self.stats_json, 'w') as out:
            json.dump(stats_dict, out)

        # Flush training statistics tracker
        self.stat_tracker.flush()

        update_progress("  \\__.Training [bs: {}] [iter: {:06d}/{:06d}] ".format(
            self.params.batch_size, iteration, self.params.max_iter), self.params.max_iter, iteration + 1)
        if iteration < self.params.max_iter - 1:
            print()
        print("      \\__Batch accuracy Index      : {:.03f}".format(stats['accuracy_index']))
        print("      \\__Classification loss       : {:.08f}".format(stats['classification_loss']))
        print("      \\__Wave loss (PDE-JVP combo) : {:.08f}".format(stats['wave_loss']))
        print("      \\__Total loss                : {:.08f}".format(stats['total_loss']))
        print("         ===================================================================")
        print("      \\__Mean iter time            : {:.3f} sec".format(mean_iter_time))
        print("      \\__Elapsed time              : {}".format(sec2dhms(elapsed_time)))
        print("      \\__ETA                       : {}".format(sec2dhms(eta)))
        print("         ===================================================================")
        update_stdout(10)

    # ---------- Helper: estimate W stats (diag Gaussian) ----------
    def _estimate_w_diag_stats(self, generator, n_samples: int, batch_size: int):
        """
        Estimate mean and diagonal variance of the W-space by sampling z -> w.
        Returns (mu: [D], inv_var: [D]) on self.device.
        """
        mu = None
        m2 = None
        seen = 0
        while seen < n_samples:
            this_bs = min(batch_size, n_samples - seen)
            z = sample_z(batch_size=this_bs, dim_z=generator.dim_z,
                        truncation=self.params.z_truncation).to(self.device)
            with torch.no_grad():
                w = generator.get_w(z)  # [B, D] (or [B, D'] flattened)
            if mu is None:
                D = w.shape[1]
                mu = torch.zeros(D, device=self.device, dtype=w.dtype)
                m2 = torch.zeros(D, device=self.device, dtype=w.dtype)
            seen += this_bs
            delta = w.mean(dim=0) - mu
            mu = mu + delta * (this_bs / seen)
            # second moment accum (Welford)
            m2 = m2 + ((w - mu).pow(2).sum(dim=0))
        var = (m2 / max(1, (seen - 1))).clamp_min(1e-8)
        inv_var = 1.0 / var
        return mu, inv_var

    # ---------- Helper: energy gradient (Gaussian / diag-Gaussian) ----------
    def _latent_energy_grad(self, latents, use_w, mu=None, inv_var=None):
        """
        Returns ∇E(latent) where E is Gaussian energy.
        If use_w is False (Z-space): ∇E = z (since E=0.5||z||^2).
        If use_w is True (W-space):  diag-Gaussian approx => ∇E = (w - mu) * inv_var.
        """
        if not use_w:
            return latents  # z
        # W-space diag Gaussian approx
        return (latents - mu[None, :]) * inv_var[None, :]

    # ---------- Main method: contrastive pretraining ----------
    def contrastive_pretrain_potentials(self, generator, support_sets):
        """
        Contrastive latent-only pretraining for potentials:
        - Keeps gradient steps within the latent distribution
        - Encourages mutual orthogonality and local consistency
        Uses only generator.get_w if shift_in_w_space=True. No image synthesis.
        """
        print("#. Contrastive pretraining of potentials (latent-only)")
        device = self.device
        support_sets = support_sets.to(device).train()
        generator = generator.to(device).eval()

        # ----------------- Hyperparameters (with sane defaults) -----------------
        steps          = int(getattr(self.params, "pretrain_steps", 2000))
        bs             = int(getattr(self.params, "pretrain_batch_size", self.params.batch_size))
        lr             = float(getattr(self.params, "pretrain_lr", 1e-4))
        t_rand         = bool(getattr(self.params, "pretrain_time_random", True))
        cons_sigma     = float(getattr(self.params, "pretrain_consistency_sigma", 0.05))  # local noise std
        target_norm    = float(getattr(self.params, "pretrain_target_grad_norm", 1.0))
        lambda_orth    = float(getattr(self.params, "pretrain_lambda_orth", 1.0))
        lambda_in      = float(getattr(self.params, "pretrain_lambda_in", 1.0))
        lambda_cons    = float(getattr(self.params, "pretrain_lambda_consistency", 0.2))
        lambda_norm    = float(getattr(self.params, "pretrain_lambda_norm", 0.0))
        log_freq       = int(getattr(self.params, "pretrain_log_freq", 100))
        use_amp        = bool(getattr(self.params, "pretrain_amp", False))

        K = self.params.num_support_sets
        T = self.params.num_support_timesteps
        eps = 1e-8

        # -------------- Decide latent space and estimate distribution -----------
        use_w = bool(getattr(generator, "shift_in_w_space", False))
        mu = None
        inv_var = None
        if use_w:
            n_stats = int(getattr(self.params, "pretrain_w_stats_samples", 50000))
            stats_bs = int(getattr(self.params, "pretrain_w_stats_batch", 1024))
            print(f"   - Estimating W diag-Gaussian stats with {n_stats} samples...")
            mu, inv_var = self._estimate_w_diag_stats(generator, n_stats, stats_bs)
            # cache on module for later use if desired
            support_sets.register_buffer("w_mu", mu)
            support_sets.register_buffer("w_inv_var", inv_var)
        else:
            print("   - Using Z prior N(0,I); no stats needed.")

        # -------------------------- Optimizer -----------------------------------
        opt = torch.optim.Adam(support_sets.parameters(), lr=lr)

        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        # ------------------------ Training loop ---------------------------------
        t0 = time.time()
        
        for it in range(1, steps + 1):
            z = sample_z(batch_size=bs, dim_z=generator.dim_z,
                        truncation=self.params.z_truncation).to(device)

            with torch.no_grad():
                lat = generator.get_w(z, truncation_psi=self.params.z_truncation) if use_w else z  # [B, D]

            # Random or fixed time input
            if t_rand:
                t_in = torch.randint(0, max(1, T), (bs, 1), device=device, dtype=lat.dtype)
            else:
                t_in = torch.zeros(bs, 1, device=device, dtype=lat.dtype)

            # Energy gradient (out-of-distribution local direction)
            gradE = self._latent_energy_grad(lat, use_w, mu, inv_var)  # [B, D]
            gradE_norm = gradE.norm(dim=1, keepdim=True).clamp_min(eps)

            # Compute gradients g_k = ∇_lat u^k(lat, t)
            # We build them in a list to keep each graph separate (memory-friendly)
            g_list = []
            g_pert_list = []
            if cons_sigma > 0:
                lat_pert = (lat + cons_sigma * torch.randn_like(lat))

            opt.zero_grad(set_to_none=True)
            
            # AMP only helps inside forwards; we still need create_graph for backprop through g_k
            autocast = torch.cuda.amp.autocast if use_amp else torch.autocast
            with autocast(device_type='cuda', enabled=use_amp):
                for k in range(K):
                    mlp = support_sets.MLP_SET[k]
                    # base grads
                    lat_req = lat.detach().requires_grad_(True)
                    t_req   = t_in.detach()  # we don't need grads wrt time in pretrain
                    u = mlp(lat_req, t_req)           # (B,1)
                    g = torch.autograd.grad(u.sum(), lat_req, create_graph=True)[0]  # (B,D)
                    g_list.append(g)

                    if cons_sigma > 0:
                        latp_req = lat_pert.detach().requires_grad_(True)
                        up = mlp(latp_req, t_req)
                        gp = torch.autograd.grad(up.sum(), latp_req, create_graph=True)[0]
                        g_pert_list.append(gp)

            # Stack: [B, K, D]
            G = torch.stack(g_list, dim=1)
            G_norm = G.norm(dim=-1, keepdim=True).clamp_min(eps)
            G_unit = G / G_norm

            # ---------- Orthogonality: ||G G^T - I||_F^2 (per batch, mean) ----------
            # Gram per sample: [B, K, K]
            Gram = torch.matmul(G_unit, G_unit.transpose(1, 2))
            I = torch.eye(K, device=device, dtype=Gram.dtype)[None, :, :]
            off = Gram - I
            L_orth = (off.pow(2).sum(dim=(1, 2)) / (K * (K - 1) + eps)).mean()

            # ---------- In-distribution: minimize cos^2(g_k, gradE) ----------
            # Broadcast gradE to [B, K, D]
            GE = gradE[:, None, :]
            GE_unit = GE / gradE_norm[:, None, :]
            cos_g_GE = (G_unit * GE_unit).sum(dim=-1)  # [B, K]
            L_in = (cos_g_GE.pow(2)).mean()

            # ---------- Consistency: local directional stability ----------
            if cons_sigma > 0:
                GP = torch.stack(g_pert_list, dim=1)        # [B, K, D]
                GP_unit = GP / GP.norm(dim=-1, keepdim=True).clamp_min(eps)
                cos_cons = (G_unit * GP_unit).sum(dim=-1)   # [B, K]
                L_cons = (1.0 - cos_cons).mean()
            else:
                L_cons = torch.zeros((), device=device, dtype=lat.dtype)

            # ---------- Norm regularizer (optional) ----------
            if lambda_norm > 0:
                L_norm = ((G_norm.squeeze(-1) - target_norm).pow(2)).mean()
            else:
                L_norm = torch.zeros((), device=device, dtype=lat.dtype)

            # Total loss
            loss = (lambda_orth * L_orth
                    + lambda_in * L_in
                    + lambda_cons * L_cons
                    + lambda_norm * L_norm)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            # Logging
            if (it % log_freq) == 0:
                dt = time.time() - t0
                print(f"[pretrain {it:06d}/{steps:06d}] "
                    f"loss={float(loss):.5f}  L_orth={float(L_orth):.5f} "
                    f"L_in={float(L_in):.5f}  L_cons={float(L_cons):.5f}  "
                    f"L_norm={float(L_norm):.5f}  ({dt:.1f}s)")
                t0 = time.time()

        print("#. Contrastive pretraining complete.")
        
    def train(self, generator, support_sets, reconstructor):

        histograms = False
        save_images = False
        save_checkpoints = False
        analytics = False
        # Save initial `support_sets` model as `support_sets_init.pt`
        torch.save(support_sets.state_dict(), osp.join(self.models_dir, 'support_sets_init.pt'))

        # Set modes/devices
        generator = generator.to(self.device).eval()
        support_sets = support_sets.to(self.device).train()
        reconstructor = reconstructor.to(self.device).train()

        # Initialize enhanced logging arrays (on CPU) now that K is known
        self.K = self.params.num_support_sets
        cpu = torch.device('cpu')
        self.per_k_ema_acc = torch.zeros(self.K, dtype=torch.float32, device=cpu)
        self.per_k_ema_grad = torch.zeros(self.K, dtype=torch.float32, device=cpu)
        self.per_k_select_counts = torch.zeros(self.K, dtype=torch.long, device=cpu)
        self.confusion = torch.zeros((self.K, self.K), dtype=torch.long, device=cpu)

        # Optimizers
        # Starting iter (maybe resume)
        starting_iter = self.get_starting_iteration(support_sets, reconstructor)
        acc_steps = max(1, int(getattr(self.params, "accumulate_grad_steps", 1)))
        # === before the loop, after K is known ===
        acc_steps = max(1, int(getattr(self.params, "accumulate_grad_steps", 1)))
        if acc_steps > self.K:
            raise ValueError(f"accumulate_grad_steps ({acc_steps}) must be ≤ num_support_sets K ({self.K}) "
                            "to guarantee unique indices within each accumulation window.")

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

        # --- create optimizer(s) first ---
        support_sets_optim = torch.optim.Adam(support_sets.parameters(), lr=self.params.support_set_lr)
        reconstructor_optim = torch.optim.Adam(reconstructor.parameters(), lr=self.params.reconstructor_lr)

        # --- scheduler should count OPTIMIZER steps, not micro-steps ---
        total_opt_steps = math.ceil(self.params.max_iter / acc_steps)
        warmup_steps = math.ceil(self.params.warmup_fraction * total_opt_steps)

        # before the loop
        completed_micro = starting_iter - 1
        opt_step_idx = completed_micro // acc_steps   # number of optimizer steps already done

        # init schedulers with the right position (if not loading state)
        sched_support = CosineScheduleWithWarmup(
            support_sets_optim, num_warmup_steps=warmup_steps, num_training_steps=total_opt_steps,
            last_epoch=opt_step_idx - 1  # so next .step() advances to opt_step_idx
        )
        sched_recon = CosineScheduleWithWarmup(
            reconstructor_optim, num_warmup_steps=warmup_steps, num_training_steps=total_opt_steps,
            last_epoch=opt_step_idx - 1
        )

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

        # Early exit if complete
        if starting_iter == self.params.max_iter:
            print("#. This experiment has already been completed and can be found @ {}".format(self.wip_dir))
            print("#. Copy {} to {}...".format(self.wip_dir, self.complete_dir))
            try:
                shutil.copytree(src=self.wip_dir, dst=self.complete_dir, ignore=shutil.ignore_patterns('checkpoint.pt'))
                print("  \\__Done!")
            except IOError as e:
                print("  \\__Already exists -- {}".format(e))
            sys.exit()
        print("#. Start training from iteration {}".format(starting_iter))

        t0 = time.time()
        

        # Training loop
        print(f"#. Training loop: {starting_iter} to {self.params.max_iter}")
        for micro_idx, iteration in enumerate(range(starting_iter, self.params.max_iter + 1), start=1):
            iter_t0 = time.time()


            # Sample index k and timestep t
            # ---- start of accumulation window? ----
            # micro_idx counts 1..N over micro-steps
            window_start = ((micro_idx - 1) % acc_steps) == 0

            if window_start:
                # how many micro-steps remain including this one?
                micros_left = self.params.max_iter - iteration + 1
                win_len = min(acc_steps, micros_left)

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

                # 2) draw k indices WITHOUT replacement for this window
                k_seq = draw_unique_k_sequence(self.K, win_len, self.device)
                k_ptr = 0

                # Original images
                if iteration % self.params.log_freq == 0 and save_images:
                    with torch.no_grad():
                        img_orig = generator(current_z[:8])

            # use the shared z for this micro-step
            z = current_z

            # pick the next unique k for this micro-step
            k = int(k_seq[k_ptr].item())
            k_ptr += 1

            index = torch.tensor([k], device=self.device)
            half_range = self.params.num_support_timesteps // 2
            t_idx = torch.randint(0, max(1, half_range - 1), (1,), device=self.device)
            time_stamp = t_idx.to(z).repeat(self.params.batch_size, 1)

            # Wave traversal step
            energy, latent1, latent2, loss_wave = support_sets(index.item(), z, time_stamp, generator)

            # Images after step 1 and 2
            img_step1 = generator(latent1)
            img_step2 = generator(latent2)

            # Classifier
            predicted_support_sets_indices, _ = reconstructor(img_step1, img_step2)

            # Targets
            target = index.repeat(self.params.batch_size)
            classification_loss = self.cross_entropy(predicted_support_sets_indices, target)


            loss = self.params.lambda_cls * classification_loss + self.params.lambda_pde * loss_wave
            loss = loss / acc_steps
            loss.backward()

            # ---- Enhanced analytics (safe device handling) ----
            with torch.no_grad():
                logits = predicted_support_sets_indices
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                entropy = -(probs * (probs.clamp_min(1e-8).log())).sum(dim=1).mean().detach()
                k = int(index.item())
                acc_k = (preds == k).float().mean().detach().to('cpu')
                self.per_k_ema_acc[k] = self.per_k_ema_acc[k] * self.ema_decay + acc_k * (1.0 - self.ema_decay)
                self.per_k_select_counts[k] += int(self.params.batch_size)

                # Snapshot per-MLP EMA accuracy for heatmap
                if self.tensorboard and iteration % self.params.log_freq == 0 and analytics:
                    # Predictions, probs, entropy
                    # Update per-MLP EMA accuracy and confusion on CPU

                    preds_cpu = preds.detach().to('cpu')
                    counts = torch.bincount(preds_cpu, minlength=self.K).to(self.confusion.dtype)
                    self.confusion[k, :].add_(counts)

                    # Per-MLP grad norm EMA (only selected MLP has grads)
                    mlp_params = list(support_sets.MLP_SET[k].parameters())
                    if len(mlp_params) > 0:
                        g2 = 0.0
                        for p in mlp_params:
                            if p.grad is not None:
                                g2 += float(p.grad.detach().to('cpu').pow(2).sum())
                        grad_norm = math.sqrt(max(g2, 1e-12))
                        self.per_k_ema_grad[k] = self.per_k_ema_grad[k] * self.ema_decay + grad_norm * (1.0 - self.ema_decay)



                    # Global grad norms (CPU)
                    def module_grad_norm(mod):
                        tot = 0.0
                        for p in mod.parameters():
                            if p.grad is not None:
                                tot += float(p.grad.detach().to('cpu').pow(2).sum())
                        return math.sqrt(max(tot, 1e-12))

                    gn_support = module_grad_norm(support_sets)
                    gn_recon = module_grad_norm(reconstructor)

                    # Latent step norms
                    delta1 = (latent1 - z).detach()
                    delta2 = (latent2 - latent1).detach()
                    step1_norm = delta1.norm(dim=1).mean()
                    step2_norm = delta2.norm(dim=1).mean()
                    self.ema_history.append(self.per_k_ema_acc.detach().cpu().numpy().copy())
                    self.iter_history.append(iteration)
                    if len(self.ema_history) > self.max_history:
                        self.ema_history = self.ema_history[-self.max_history:]
                        self.iter_history = self.iter_history[-self.max_history:]

            if (micro_idx % acc_steps == 0) or (iteration == self.params.max_iter):
                support_sets_optim.step()
                reconstructor_optim.step()
                support_sets_optim.zero_grad(set_to_none=True)
                reconstructor_optim.zero_grad(set_to_none=True)

                sched_support.step()
                sched_recon.step()
                opt_step_idx += 1
                # Update statistics tracker
                self.stat_tracker.update(accuracy_index=(preds == target).to(torch.float32).mean().detach(),
                                        classification_loss=classification_loss.item(),
                                        wave_loss=loss_wave.item(),
                                        total_loss=loss.item(),
                                        support_sets_lr=sched_support.get_last_lr()[0],
                                        reconstructor_lr=sched_recon.get_last_lr()[0])
                # ============================ TensorBoard logging ============================
                if self.tensorboard:
                    with torch.no_grad():
                        # Scalars
                        means = self.stat_tracker.get_means()
                        for key, value in means.items():
                            self.tb_writer.add_scalar(f"train/{key}", value, iteration)
                        self.tb_writer.add_scalar("train/entropy", float(entropy), iteration)
                        if analytics:
                            self.tb_writer.add_scalar("train/grad_norm/support_sets", gn_support, iteration)
                            self.tb_writer.add_scalar("train/grad_norm/reconstructor", gn_recon, iteration)
                            self.tb_writer.add_scalar("train/latent_step_norm/step1_mean", float(step1_norm), iteration)
                            self.tb_writer.add_scalar("train/latent_step_norm/step2_mean", float(step2_norm), iteration)

                        # wave speed c stats (support_sets.c: [K,1])
                        c_vals = support_sets.c.detach().view(-1).cpu().numpy()
                        self.tb_writer.add_scalar("train/c_stats/mean", float(c_vals.mean()), iteration)
                        self.tb_writer.add_scalar("train/c_stats/std", float(c_vals.std()), iteration)
                        self.tb_writer.add_scalar("train/c_stats/min", float(c_vals.min()), iteration)
                        self.tb_writer.add_scalar("train/c_stats/max", float(c_vals.max()), iteration)

                        # Histograms
                        if histograms:
                            self.tb_writer.add_histogram("train/logits", logits.detach().cpu().numpy(), iteration)
                            self.tb_writer.add_histogram("train/probs", probs.detach().cpu().numpy(), iteration)
                            self.tb_writer.add_histogram("train/latent_step_norm/step1", delta1.norm(dim=1).detach().cpu().numpy(), iteration)
                            self.tb_writer.add_histogram("train/latent_step_norm/step2", delta2.norm(dim=1).detach().cpu().numpy(), iteration)
                            self.tb_writer.add_histogram("train/c_values", c_vals, iteration)

                            # Per-MLP histograms (CPU arrays)
                            self.tb_writer.add_histogram("per_mlp/ema_accuracy", self.per_k_ema_acc.detach().cpu().numpy(), iteration)
                            self.tb_writer.add_histogram("per_mlp/ema_grad_norm", self.per_k_ema_grad.detach().cpu().numpy(), iteration)
                            self.tb_writer.add_histogram("per_mlp/selection_counts", self.per_k_select_counts.detach().cpu().numpy(), iteration)

                            # Meta histograms
                            self.tb_writer.add_histogram("meta/timestep_idx", time_stamp[:, 0].detach().cpu().numpy(), iteration)
                            self.tb_writer.add_histogram("meta/predicted_k", preds.detach().cpu().numpy(), iteration)
                            self.tb_writer.add_histogram("meta/true_k", target.detach().cpu().numpy(), iteration)

                        # Images & figures
                        if opt_step_idx % self.params.log_freq == 0 and save_images:
                            self._log_image_triplet(self.tb_writer, "images", img_orig, img_step1, img_step2, iteration,
                                                    n_vis=min(8, self.params.batch_size))

                            # Heatmap of per-MLP EMA accuracy over time
                            if len(self.ema_history) >= 2:
                                hist_mat = np.stack(self.ema_history, axis=1)  # [K, T_hist]
                                fig = self._plot_heatmap(hist_mat, title="Per-MLP EMA Accuracy over Time",
                                                        xlabel="log step (every log_freq iters)",
                                                        ylabel="MLP index k")
                                self.tb_writer.add_figure("per_mlp/accuracy_heatmap", fig, global_step=iteration)
                                plt.close(fig)

                            # Confusion matrix
                            fig_c = self._plot_confusion(self.confusion)  # CPU tensor
                            self.tb_writer.add_figure("classifier/confusion_matrix", fig_c, global_step=iteration)
                            plt.close(fig_c)

                # ============================ /TensorBoard logging ===========================

                # Timing
                iter_t = time.time()
                self.iter_times = np.append(self.iter_times, iter_t - iter_t0)
                elapsed_time = iter_t - t0
                mean_iter_time = self.iter_times.mean()
                eta = elapsed_time * ((self.params.max_iter - iteration) / (iteration - starting_iter + 1))

                # Log progress in stdout
                if iteration % self.params.log_freq == 0:
                    self.log_progress(iteration, mean_iter_time, elapsed_time, eta)

                # Save checkpoint
                if iteration % self.params.ckp_freq == 0 and save_checkpoints:
                    checkpoint_dict = {
                        'iter': iteration,
                        'support_sets': support_sets.state_dict(),
                        'reconstructor': reconstructor.module.state_dict() if self.multi_gpu else reconstructor.state_dict()
                    }
                    torch.save(checkpoint_dict, self.checkpoint)
        # === End of training loop ===

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