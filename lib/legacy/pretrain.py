
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
    cons_sigma  = float(getattr(self.params, "pretrain_consistency_sigma", 0.01))
    # contrastive weights
    lambda_orth = float(getattr(self.params, "pretrain_lambda_orth", 1.0))
    lambda_in   = float(getattr(self.params, "pretrain_lambda_in", 1.0))
    lambda_cons = float(getattr(self.params, "pretrain_lambda_consistency", 1.0))
    lambda_sup = float(getattr(self.params, "pretrain_lambda_support", .0))
    p_power = int(getattr(self.params, "pretrain_support_power", 2))
    # PDE/IC weights (default to WavePDE's current settings)
    lambda_pde  = float(getattr(self.params, "pretrain_lambda_pde",
                                getattr(support_sets, "lambda_pde", 1.)))
    lambda_ic   = float(getattr(self.params, "pretrain_lambda_ic",
                                getattr(support_sets, "lambda_ic", 0.0)))

    # how many potentials to hit per iteration (<= K)
    K           = support_sets.num_support_sets
    K_per_step  = int(getattr(self.params, "pretrain_k_per_step", K))  # set <K for speed
    half_range  = max(1, support_sets.num_support_timesteps // 2)
    log_freq    = int(getattr(self.params, "pretrain_log_freq", 100))
    use_amp     = bool(getattr(self.params, "pretrain_amp", False))
    eps         = 1e-8
    # ----------------- Choose latent space & stats -----------------
    use_w = bool(getattr(generator, "shift_in_w_space", False))
    r2_thresh = mu = Sigma_inv = R_prec = None
    q = float(getattr(self.params, "pretrain_support_quantile", 0.9))

    if use_w:
        n_stats  = int(getattr(self.params, "pretrain_w_stats_samples", 50000))
        stats_bs = int(getattr(self.params, "pretrain_w_stats_batch", 1024))
        shrink   = float(getattr(self.params, "pretrain_w_stats_shrink", 0.01))
        print(f"   - Estimating W full-Gaussian stats with {n_stats} samples...")
        mu, Sigma_inv, R_prec = self._estimate_w_full_stats(generator, n_stats, stats_bs,
                                                            shrink=shrink, jitter=1e-6)

    opt = build_adamw(
        support_sets,
            lr=float(getattr(self.params, "pretrain_lr", 1e-3)),
        weight_decay=1e-2,
        extra_no_decay_names=('c',),
    )
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
                mlp_k = support_sets.PSI_SET[k]
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
                        Zk_all_list.append(z_curr) 
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

        # PDE & IC aggregates
        L_pde = torch.stack(PDE_list).mean() if len(PDE_list) > 0 else torch.zeros((), device=device)
        L_ic  = torch.stack(IC_list).mean() if (lambda_ic > 0 and len(IC_list) > 0) else torch.zeros((), device=device)

        # Total loss
        loss = (lambda_orth * L_orth
            + lambda_in   * L_in
            + lambda_cons * L_cons
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
                f"pde={float(L_pde):.5f} "
                f"cons={float(L_cons):.5f} "
                f"(dt={dt:.1f}s)"
                )
            t_print = time.time()

    # restore original lambda_jvp
    if hasattr(support_sets, "lambda_jvp"):
        support_sets.lambda_jvp = prev_lambda_jvp

    print("#. Contrastive pretraining complete.")

