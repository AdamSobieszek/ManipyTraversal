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

VIS_IMAGE_N = 15

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

    # ---------- Helper: estimate W stats (full Gaussian) ----------
    @torch.no_grad()
    def _estimate_w_full_stats(self, generator, n_samples: int, batch_size: int, shrink: float = 0.01, jitter: float = 1e-6):
        """
        Estimate mean and *full* covariance of W-space by sampling z -> w.
        Returns (mu: [D], Sigma_inv: [D,D], chol_precision: [D,D]) on self.device.

        - 'shrink' applies Σ <- (1-shrink)Σ + shrink*tr(Σ)/D * I  (Ledoit-Wolf style ridge).
        - 'jitter' adds tiny εI for numerical stability.
        """
        device = self.device
        mu = None
        # Second-moment accumulator for online covariance (Welford in matrix form)
        # We keep running mean and running covariance via batch-wise updates.
        # See "parallel/online covariance" formula.
        cov = None
        seen = 0

        while seen < n_samples:
            this_bs = min(batch_size, n_samples - seen)
            z = sample_z(batch_size=this_bs, dim_z=generator.dim_z, truncation=self.params.z_truncation).to(device)
            w = generator.get_w(z)  # [B, D]
            B, D = w.shape

            if mu is None:
                mu = torch.zeros(D, device=device, dtype=w.dtype)
                cov = torch.zeros(D, D, device=device, dtype=w.dtype)

            # batch stats
            mu_b = w.mean(dim=0)                                   # [D]
            Xc = (w - mu_b)                                        # [B, D]
            cov_b = (Xc.T @ Xc) / max(1, B - 1)                    # [D, D]

            # combine running + batch (online)
            new_seen = seen + B
            delta = (mu_b - mu)
            mu_new = mu + delta * (B / new_seen)

            # covariance merge (Chan–Golub–LeVeque)
            cov = ( (seen - 1) / max(1, new_seen - 1) ) * cov \
                + ( (B - 1)   / max(1, new_seen - 1) ) * cov_b \
                + ( seen * B / max(1, new_seen * (new_seen - 1)) ) * torch.ger(delta, delta)

            mu = mu_new
            seen = new_seen

        # Final covariance regularization + shrinkage
        # Ridge toward spherical: α tr(Σ)/D I
        trace = torch.trace(cov)
        D = cov.shape[0]
        cov = (1.0 - shrink) * cov + shrink * (trace / max(1, D)) * torch.eye(D, device=cov.device, dtype=cov.dtype)
        cov = cov + jitter * torch.eye(D, device=cov.device, dtype=cov.dtype)

        # Invert robustly via Cholesky
        L = torch.linalg.cholesky(cov)               # Σ = L L^T
        Sigma_inv = torch.cholesky_inverse(L)        # Σ^{-1}
        # (optional) store precision Cholesky R s.t. Σ^{-1} = R R^T for fast Mahalanobis
        # Compute R via chol of precision: numerically stable using solve_triangular
        # Here we just reuse Sigma_inv's chol:
        R = torch.linalg.cholesky(Sigma_inv)         # precision factor

        return mu, Sigma_inv, R
    # ---------- Helper: energy gradient (Gaussian / full-Gaussian) ----------
    def _latent_energy_grad(self, latents, use_w, mu=None, Sigma_inv=None):
        """
        Returns ∇E(latent) for Gaussian energy E = 0.5 (x - μ)^T Σ^{-1} (x - μ).
        - Z-space (prior N(0,I)): ∇E = z.
        - W-space full Gaussian:  ∇E = (w - μ) Σ^{-1}.
        """
        if not use_w:
            return latents  # z
        # full precision
        diff = latents - mu[None, :]                      # [B, D]
        return diff @ Sigma_inv                           # [B, D]
    # ---------- Helper: Mahalanobis distance and support penalty ----------
    def _mahalanobis_sq(self, x, mu, R_precision):  # x: [B,D]
        diff = x - mu[None, :]
        y = diff @ R_precision.T                     # y = R (x - mu)
        return (y * y).sum(dim=-1)                   # [B]

    def _support_penalty(self, x, use_w, mu=None, R_precision=None, r2_thresh=None, p=2):
        """
        Penalize leaving support:  L = E[ ReLU(D^2 - r^2)^p ].
        - W-space: D^2 = (x-μ)^T Σ^{-1} (x-μ) = ||R(x-μ)||^2
        - Z-space: D^2 = ||x||^2
        - p ∈ {1,2}
        """
        if use_w:
            d2 = self._mahalanobis_sq(x, mu, R_precision)   # [B]
        else:
            d2 = (x * x).sum(dim=-1)

        if r2_thresh is None:
            # Failsafe: if unset, do not penalize
            return torch.zeros((), device=x.device, dtype=x.dtype)

        # Ensure thresholds match device/dtype
        r2 = r2_thresh.to(device=x.device, dtype=x.dtype)
        exced = (d2 - r2).clamp_min(0.0)
        if p == 1:
            return exced.sum()
        return (exced).pow(p).sum()

    def contrastive_pretrain_potentials(self, generator, support_sets):
        """
        Latent-only contrastive pretraining with *PDE traversal*:
        - Works in W-space if generator.shift_in_w_space == True (uses only get_w).
        - For each selected potential k, rolls z from i=0..t using WavePDE._per_step
        (same discretization & truncation as training), accumulates PDE loss,
        and applies contrastive terms at the selected step i=t.
        """
        import time, math
        device = self.device
        support_sets = support_sets.to(device).train()
        generator = generator.to(device).eval()

        # ----------------- Hyperparameters -----------------
        steps       = int(getattr(self.params, "pretrain_steps", 200))
        B          = int(getattr(self.params, "pretrain_batch_size", 32))
        lr          = float(getattr(self.params, "pretrain_lr", 1e-3))
        t_rand      = bool(getattr(self.params, "pretrain_time_random", True))
        cons_sigma  = float(getattr(self.params, "pretrain_consistency_sigma", 0.01))
        target_norm = float(getattr(self.params, "pretrain_target_grad_norm", 0.0))  # (unused by default)
        # contrastive weights
        lambda_orth = float(getattr(self.params, "pretrain_lambda_orth", 1.0))
        lambda_in   = float(getattr(self.params, "pretrain_lambda_in", 1.0))
        lambda_cons = float(getattr(self.params, "pretrain_lambda_consistency", 1.0))
        lambda_norm = float(getattr(self.params, "pretrain_lambda_norm", .0))
        lambda_sup = float(getattr(self.params, "pretrain_lambda_support", .0))
        p_power = int(getattr(self.params, "pretrain_support_power", 2))
        # PDE/IC weights (default to WavePDE's current settings)
        lambda_pde  = float(getattr(self.params, "pretrain_lambda_pde",
                                    getattr(support_sets, "lambda_pde", 1.)))
        lambda_ic   = float(getattr(self.params, "pretrain_lambda_ic",
                                    getattr(support_sets, "lambda_ic", 0.0)))

        # how many potentials to hit per iteration (<= K)
        K           = int(getattr(self.params, "num_support_sets", support_sets.num_support_sets))
        K_per_step  = int(getattr(self.params, "pretrain_k_per_step", K))  # set <K for speed

        T_all       = int(getattr(self.params, "num_support_timesteps", support_sets.num_support_timesteps))
        half_range  = max(1, T_all // 2)
        log_freq    = int(getattr(self.params, "pretrain_log_freq", 100))
        use_amp     = bool(getattr(self.params, "pretrain_amp", False))
        eps         = 1e-8
        # ----------------- Choose latent space & stats -----------------
        use_w = bool(getattr(generator, "shift_in_w_space", False))
        mu = Sigma_inv = R_prec = None
        r2_thresh = None
        q = float(getattr(self.params, "pretrain_support_quantile", 0.9))

        if use_w:
            n_stats  = int(getattr(self.params, "pretrain_w_stats_samples", 50000))
            stats_bs = int(getattr(self.params, "pretrain_w_stats_batch", 1024))
            shrink   = float(getattr(self.params, "pretrain_w_stats_shrink", 0.01))
            print(f"   - Estimating W full-Gaussian stats with {n_stats} samples...")
            mu, Sigma_inv, R_prec = self._estimate_w_full_stats(generator, n_stats, stats_bs,
                                                                shrink=shrink, jitter=1e-6)

            # Optional: empirical r^2 quantile in W
            n_q = min(20000, n_stats)
            d2_vals, seen = [], 0
            with torch.no_grad():
                while seen < n_q:
                    this_bs = min(stats_bs, n_q - seen)
                    zq = sample_z(batch_size=this_bs, dim_z=generator.dim_z,
                                truncation=self.params.z_truncation).to(self.device)
                    wq = generator.get_w(zq)
                    d2_vals.append(self._mahalanobis_sq(wq, mu, R_prec))   # ||R (w - mu)||^2
                    seen += this_bs
            d2_all = torch.cat(d2_vals, dim=0)
            r2_thresh = torch.quantile(d2_all, q).detach()

            # Register buffers (optional)
            support_sets.register_buffer("w_mu", mu)
            support_sets.register_buffer("w_Sigma_inv", Sigma_inv)
            support_sets.register_buffer("w_R_prec", R_prec)
            support_sets.register_buffer("w_r2_thresh", r2_thresh)
        else:
            print("   - Using Z prior N(0,I); no W stats needed.")
            # For Z ~ N(0,I_d), D^2 = ||z||^2 ~ Chi^2(df=d). Use chi-square quantile.
            df = int(generator.dim_z)

            t_q = torch.tensor(q, device=device, dtype=torch.float32)
            df_t = torch.tensor(float(df), device=device, dtype=torch.float32)
            # Fallback: Wilson–Hilferty approximation
            normal0 = torch.distributions.Normal(
                torch.tensor(0.0, device=device, dtype=torch.float32),
                torch.tensor(1.0, device=device, dtype=torch.float32)
            )
            z = normal0.icdf(t_q)  # this is implemented
            w = 1.0 - 2.0 / (9.0 * df_t) + z * torch.sqrt(2.0 / (9.0 * df_t))
            # clamp to avoid tiny negative due to numerical noise before cubing
            w = torch.clamp(w, min=1e-6)
            r2_thresh = df_t * (w ** 3)/2
            # Optionally cache as buffer
            support_sets.register_buffer("z_r2_thresh", r2_thresh)

        # ----------------- Optimizer -----------------
        opt = torch.optim.AdamW(support_sets.parameters(), lr=lr, weight_decay=1e-5)
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        autocast = (torch.cuda.amp.autocast if device.type == 'cuda' else torch.autocast)

        # Temporarily turn off JVP inside WavePDE (latent-only here)
        prev_lambda_jvp = float(getattr(support_sets, "lambda_jvp", 0.0))
        if hasattr(support_sets, "lambda_jvp"):
            support_sets.lambda_jvp = 0.0

        t_print = time.time()
        for it in range(1, steps + 1):
            # ---- sample batch in chosen latent space ----
            z = sample_z(batch_size=B, dim_z=generator.dim_z,
                        truncation=self.params.z_truncation).to(device)
            with torch.no_grad():
                lat0 = generator.get_w(z, truncation_psi=self.params.z_truncation) if use_w else z  # [B, D]



            # energy gradient at *selected* step locations will be computed later per-k
            # here we keep lat0 for rollouts
            # select K' potentials (without replacement)
            if K_per_step < K:
                k_indices = torch.randperm(K, device=device)[:K_per_step].tolist()
            else:
                k_indices = list(range(K))

            # accumulators across selected potentials
            Gk_list   = []  # [B, D] grads at selected step (normalized by OEMS in _per_step)
            Zk_list   = []  # [B, D] lat locations at selected step (for in-distribution)
            PDE_list  = []  # scalars per k
            IC_list   = []  # scalars per k (optional)
            GP_list   = []  # [B, D] perturbed grads for consistency
            all_G_list = []  # [B*timesteps, D] all grads
            Zk_all_list = []  # [B, D] locations at selected step
            opt.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, enabled=use_amp):
                for k in k_indices:
                    mlp_k = support_sets.MLP_SET[k]
                    c_k   = support_sets.c[k:k+1]  # [1,1]

                    z_curr = lat0
                    pde_acc = 0.0
                    g_sel = None
                    z_sel = None
                    direction = np.random.choice([-1, 1])
                    denom=0
                    for i_ind, i in enumerate((range(half_range) if direction == +1 else range(0, -half_range, -1))):
                        denom += 1
                        t_i = torch.full((B, 1), float(i), device=device, dtype=lat0.dtype, requires_grad=True)

                        u_i, u_z_i, pde_res_i, z_curr = support_sets._per_step(mlp_k, z_curr, t_i, c_k, direction)
                        # accumulate PDE residual like forward()
                        pde_acc = pde_acc + (pde_res_i.pow(2).mean())
                        # capture at target i
                        if i_ind == 0:
                            g_sel = u_z_i
                            z_sel = z_curr

                        if lambda_in > 0:
                            Zk_all_list.append(z_curr)   # [B, D]
                            
                        if lambda_norm > 0 or lambda_in > 0:
                            all_G_list.append(u_z_i)

                    # match forward(): average over number of steps processed (≈ i)
                    denom = max(1, denom)  
                    PDE_list.append(pde_acc / denom)

                    Gk_list.append(g_sel)   # [B, D]
                    Zk_list.append(z_sel)   # [B, D]
                    # local consistency at selected step
                    if lambda_cons > 0:
                        z_pert = (lat0+cons_sigma*torch.randn_like(lat0)).detach().requires_grad_(True)
                        t_i = torch.full((B, 1), 0 if direction == +1 else 0, device=device, dtype=lat0.dtype, requires_grad=True)
                        u_i, u_z_i, pde_res_i, z_curr = support_sets._per_step(mlp_k, z_pert, t_i, c_k, -1*direction)

                        
                        GP_list.append(u_z_i)

            # Stack across selected k: [B, K', D]
            G = torch.stack(Gk_list, dim=1)                     # grads at selected step
            Zs = torch.stack(Zk_list, dim=1)                    # locations at selected step
            Z_all = torch.stack(Zk_all_list, dim=0)   
            all_G = torch.stack(all_G_list, dim=0)                 # locations at selected step
            G_norm = G.norm(dim=-1, keepdim=True).clamp_min(eps)
            G_unit = G / G_norm

            # Orthogonality across potentials
            Kp = G.shape[1]
            Gram = torch.matmul(G_unit, G_unit.transpose(1, 2))  # [B, K', K']
            I = torch.eye(Kp, device=device, dtype=Gram.dtype)[None]
            L_orth = ((Gram - I).pow(2).sum(dim=(1, 2)) / max(1, Kp * (Kp - 1))).mean()

            # In-distribution alignment at each k's selected location
            if use_w:
                GE = (Z_all - mu[None, None, :]) @ Sigma_inv   # [B,K',D]
            else:
                GE = Z_all
            all_G_unit = all_G / all_G.norm(dim=-1, keepdim=True).clamp_min(eps)
            cos_g_GE = (GE * all_G_unit).sum(dim=-1)       # [B,K']
            L_in = (cos_g_GE.pow(2)).mean()

            # Consistency
            if lambda_cons > 0:
                GP = torch.stack(GP_list, dim=1)                # [B, K', D]
                # GP_unit = GP / GP.norm(dim=-1, keepdim=True).clamp_min(eps)
                cos_cons = (GP + G).pow(2).sum(dim=-1)       # [B, K']
                L_cons = (cos_cons).mean()
            else:
                L_cons = torch.zeros((), device=device, dtype=G.dtype)


            # Optional norm target for |g_k|
            
            L_norm = torch.zeros((), device=device, dtype=G.dtype)

            # PDE & IC aggregates
            L_pde = torch.stack(PDE_list).mean() if len(PDE_list) > 0 else torch.zeros((), device=device)
            L_ic  = torch.stack(IC_list).mean() if (lambda_ic > 0 and len(IC_list) > 0) else torch.zeros((), device=device)

            # Total loss
            loss = (lambda_orth * L_orth
                + lambda_in   * L_in
                + lambda_cons * L_cons
                + lambda_norm * L_norm
                + lambda_pde  * L_pde
                + lambda_ic   * L_ic
                ) 

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            # Logging
            if it % log_freq == 0:
                dt = time.time() - t_print
                print(f"[pretrain {it:06d}/{steps:06d}] "
                    f"loss={float(loss):.5f} | "
                    f"orth={float(L_orth):.5f} in={float(L_in):.5f} cons={float(L_cons):.5f} "
                    f"norm={float(L_norm):.5f} pde={float(L_pde):.5f} "
                    f"cons={float(L_cons):.5f} "
                    f"(dt={dt:.1f}s)"
                    )
                t_print = time.time()

        # restore original lambda_jvp
        if hasattr(support_sets, "lambda_jvp"):
            support_sets.lambda_jvp = prev_lambda_jvp

        print("#. Contrastive pretraining complete.")


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
        start_iter = 1
        if osp.isfile(self.checkpoint):
            ckpt = torch.load(self.checkpoint, map_location=self.device)
            start_iter = int(ckpt.get('iter', 1))

            # Model weights (allow non-strict to be robust to minor changes)
            support_sets.load_state_dict(ckpt['support_sets'], strict=False)
            reconstructor.load_state_dict(ckpt['reconstructor'], strict=False)

            # Optimizers (if both objects and states exist)
            if support_opt is not None and 'support_opt' in ckpt:
                support_opt.load_state_dict(ckpt['support_opt'])
            if recon_opt is not None and 'recon_opt' in ckpt:
                recon_opt.load_state_dict(ckpt['recon_opt'])

            # Schedulers (if both objects and states exist)
            if support_sched is not None and 'support_sched' in ckpt:
                support_sched.load_state_dict(ckpt['support_sched'])
            if recon_sched is not None and 'recon_sched' in ckpt:
                recon_sched.load_state_dict(ckpt['recon_sched'])

        return start_iter
    # ------------------------ helpers for TB visuals ------------------------
    def _write_stats_json(self):
        # Update training statistics json file (optimizer-step keyed)
        with open(self.stats_json, 'w') as out:
            json.dump(self.stat_tracker.stats_by_step, out)

    def log_progress(self, step_idx, mean_step_time, elapsed_time, eta):
        # Stdout progress (now on optimizer-step cadence)
        stats = self.stat_tracker.stats_by_step.get(int(step_idx), {})
        total_opt_steps = math.ceil(self.params.max_iter / max(1, int(getattr(self.params, "accumulate_grad_steps", 1))))
        update_progress(
            "  \\__.Training [bs: {}] [opt-step: {:06d}/{:06d}] ".format(
                self.params.batch_size, step_idx, total_opt_steps
            ),
            total_opt_steps,
            step_idx + 1,
        )
        if step_idx < total_opt_steps - 1:
            print()
        print("      \\__Batch accuracy Index      : {:.03f}".format(stats.get('accuracy_index', 0.0)))
        print("      \\__Classification loss       : {:.08f}".format(stats.get('classification_loss', 0.0)))
        print("      \\__Wave loss (PDE-JVP combo) : {:.08f}".format(stats.get('wave_loss', 0.0)))
        print("      \\__Total loss                : {:.08f}".format(stats.get('total_loss', 0.0)))
        print("         ===================================================================")
        print("      \\__Mean opt-step time        : {:.3f} sec".format(mean_step_time))
        print("      \\__Elapsed time              : {}".format(sec2dhms(elapsed_time)))
        print("      \\__ETA                       : {}".format(sec2dhms(eta)))
        print("         ===================================================================")
        update_stdout(10)

    def train(self, generator, support_sets, reconstructor):
        histograms = False
        save_images = True
        save_checkpoints = True
        analytics = True
        if not osp.isfile(self.checkpoint):
            self.contrastive_pretrain_potentials(generator, support_sets)
            # Save initial `support_sets` model as `support_sets_init.pt`
            torch.save(support_sets.state_dict(), osp.join(self.models_dir, 'support_sets_init.pt'))
        else:
            print("#. checkpoint found, skipping contrastive pretraining.")

        # Set modes/devices
        generator = generator.to(self.device).eval()
        support_sets = support_sets.to(self.device).train()
        reconstructor = reconstructor.to(self.device).train()

        # Initialize enhanced logging arrays (now owned by stat_tracker) once K is known
        self.K = int(self.params.num_support_sets)
        self.stat_tracker.init_per_k(self.K)

        # Optimizers
        # Starting iter (maybe resume)
        
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
        support_sets_optim = torch.optim.AdamW(support_sets.parameters(), lr=self.params.support_set_lr, weight_decay=0.001)
        reconstructor_optim = torch.optim.Adam(reconstructor.parameters(), lr=self.params.reconstructor_lr)

        # --- create schedulers ---
        total_opt_steps = math.ceil(self.params.max_iter / acc_steps)
        warmup_steps = math.ceil(self.params.warmup_fraction * total_opt_steps)
        sched_support = CosineScheduleWithWarmup(support_sets_optim, num_warmup_steps=warmup_steps,
                                                num_training_steps=total_opt_steps, last_epoch=-1)
        sched_recon   = CosineScheduleWithWarmup(reconstructor_optim, num_warmup_steps=warmup_steps,
                                                num_training_steps=total_opt_steps, last_epoch=-1)

        # --- NOW load everything (models + opts + schedulers) if checkpoint exists ---
        starting_iter = self.get_starting_iteration(
            support_sets, reconstructor,
            # support_opt=support_sets_optim,
            # recon_opt=reconstructor_optim,
            # support_sched=sched_support,
            # recon_sched=sched_recon,
        )
        starting_iter = starting_iter*acc_steps

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
            half_range = self.params.num_support_timesteps // 2
            t_idx = torch.randint(0, max(1, half_range - 1), (1,), device=self.device)
            time_stamp = t_idx.to(z).repeat(self.params.batch_size, 1)

            # Wave traversal step
            energy, latent1, latent2, loss_wave = support_sets(index.item(), z, time_stamp, generator)

            # Images after step 1 and 2
            img_step1 = generator(latent1)
            img_step2 = generator(latent2)

            # Store images for visualization
            if k < VIS_IMAGE_N:
                imgs_step1[k] = img_step1[:1]
                imgs_step2[k] = img_step2[:1]

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
                entropy = -(probs * (probs.clamp_min(1e-8).log())).sum(dim=1).mean()

                # Latent step norms
                delta1 = (latent1 - z).detach()
                delta2 = (latent2 - latent1).detach()
                step1_norm = float(delta1.norm(dim=1).mean())
                step2_norm = float(delta2.norm(dim=1).mean())

                # Per-MLP selected grad norm (only selected MLP has grads)
                grad_norm_selected = None
                if analytics:
                    mlp_params = list(support_sets.MLP_SET[int(index.item())].parameters())
                    if len(mlp_params) > 0:
                        g2 = 0.0
                        for p in mlp_params:
                            if p.grad is not None:
                                g2 += float(p.grad.detach().to('cpu').pow(2).sum())
                        grad_norm_selected = math.sqrt(max(g2, 1e-12))

                # Accumulate micro-step means into the tracker (unscaled totals for readability)
                self.stat_tracker.add_micro(
                    acc=float((preds == index.item()).float().mean().item()),
                    classification_loss=float(classification_loss.item()),
                    wave_loss=float(loss_wave.item()),
                    total_loss=float((self.params.lambda_cls * classification_loss + self.params.lambda_pde * loss_wave).item()),
                    entropy=float(entropy.item()),
                    step1_norm=step1_norm,
                    step2_norm=step2_norm,
                )

                # Per-MLP analytics in tracker (CPU numpy)
                self.stat_tracker.update_per_k_after_micro(
                    true_k=int(index.item()),
                    preds=preds.detach().to('cpu').numpy(),
                    batch_size=int(self.params.batch_size),
                    grad_norm_selected_mlp=grad_norm_selected if analytics else None,
                )
                
            if (micro_idx % acc_steps == 0) or (iteration == self.params.max_iter):
                # Optimizer step (after acc_steps micro-steps, or at the very end)
                support_sets_optim.step()
                reconstructor_optim.step()
                support_sets_optim.zero_grad(set_to_none=True)
                reconstructor_optim.zero_grad(set_to_none=True)

                # LR schedulers step once per optimizer step
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
                        self.tb_writer.add_scalar("train/c_stats/mean", float(c_vals.mean()), opt_step_idx)
                        self.tb_writer.add_scalar("train/c_stats/std", float(c_vals.std()), opt_step_idx)
                        self.tb_writer.add_scalar("train/c_stats/min", float(c_vals.min()), opt_step_idx)
                        self.tb_writer.add_scalar("train/c_stats/max", float(c_vals.max()), opt_step_idx)

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
                            self.tb_writer.add_histogram("per_mlp/ema_accuracy", self.stat_tracker.per_k_ema_acc, opt_step_idx)
                            self.tb_writer.add_histogram("per_mlp/ema_grad_norm", self.stat_tracker.per_k_ema_grad, opt_step_idx)
                            self.tb_writer.add_histogram("per_mlp/selection_counts", self.stat_tracker.per_k_select_counts, opt_step_idx)
                            self.tb_writer.add_histogram("meta/timestep_idx", time_stamp[:, 0].detach().cpu().numpy(), opt_step_idx)
                            self.tb_writer.add_histogram("meta/predicted_k", preds.detach().cpu().numpy(), opt_step_idx)
                            self.tb_writer.add_histogram("meta/true_k", target.detach().cpu().numpy(), opt_step_idx)

                    # Images & figures (on optimizer-step cadence)
                    if save_images and ((micro_idx)//acc_steps % self.params.log_freq) == 0:
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