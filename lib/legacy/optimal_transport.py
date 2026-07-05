import numpy as np
import torch
import ot  # POT

def _pairwise_traversal_cost(X0, F0, X1, F1, *,
                             eps_df=1e-6,
                             ratio_power=2,
                             forward_only=True,
                             large_cost_val=1e10):
    """
    Cost C_{ij} = ||x0_i - x1_j||^p / (max(f1_j - f0_i, eps_df))^ratio_power
    - p=1 if you pass dist (L2), p=2 if you square the L2.
    Here we use p=2 (squared Euclidean) for stability; set ratio_power=1 to match ||dx||/|df|.
    """
    # distances
    # torch.cdist can be memory hungry; fallback: (X0**2).sum + (X1**2).sum - 2 X0 X1^T
    dist2 = torch.cdist(X0, X1, p=2.0) ** 2  # (m,n)
    df = F1.unsqueeze(0) - F0.unsqueeze(1)   # (m,n)

    if forward_only:
        # forbid non-increasing f: assign huge cost
        denom = torch.clamp(df, min=eps_df)
        C = dist2 / (denom ** ratio_power)
        C = torch.where(df > 0.0, C, torch.full_like(C, large_cost_val))
    else:
        denom = torch.clamp(df.abs(), min=eps_df)
        C = dist2 / (denom ** ratio_power)
    return C


@torch.no_grad()
def weighted_ot_coupling_with_traversal_cost(
    x_all: torch.Tensor,
    f_func,
    sample_weights: torch.Tensor = None,    # (N,) nonnegative
    # ------ splitting ------
    split_quantile: float = 0.5,            # e.g., 0.5: lower vs upper half
    # ------ traversal cost ------
    eps_df: float = 1e-6,
    ratio_power: int = 2,                   # 1 -> ||dx||/|df|, 2 -> ||dx||^2/|df|^2
    forward_only: bool = True,
    large_cost_val: float = 1e10,
    # ------ OT solver ------
    solver: str = "sinkhorn",               # "sinkhorn" | "emd" | "unbalanced"
    reg: float = 5e-2,                      # entropic regularization for sinkhorn
    unb_tau_a: float = 1.0,                 # unbalanced KL strength (source)
    unb_tau_b: float = 1.0,                 # unbalanced KL strength (target)
    ot_numItermax: int = 10_000,
    # ------ device/dtype ------
    device=None,
    dtype=None,
):
    """
    Returns
    -------
    (X0, F0, a), (X1, F1, b), Pi
      X0: (m,D), F0: (m,), a: (m,)   -- normalized marginals
      X1: (n,D), F1: (n,), b: (n,)
      Pi: (m,n) torch.float tensor, nonnegative, sum(Pi)=1
    """
    if device is None: device = x_all.device
    if dtype  is None: dtype  = x_all.dtype

    N, D = x_all.shape
    f_all = f_func(x_all).reshape(-1)  # (N,)
    thr = torch.quantile(f_all.float(), split_quantile).to(device=device, dtype=dtype)

    low_mask  = f_all <= thr
    high_mask = f_all  >  thr

    X0, F0 = x_all[low_mask],  f_all[low_mask]
    X1, F1 = x_all[high_mask], f_all[high_mask]
    if X0.numel() == 0 or X1.numel() == 0:
        raise ValueError("Split produced an empty side; adjust split_quantile.")

    # per-sample weights -> per-side marginals a, b (normalize to 1)
    if sample_weights is None:
        sample_weights = torch.ones(N, device=device, dtype=dtype)
    w0 = sample_weights[low_mask].clamp_min(0)
    w1 = sample_weights[high_mask].clamp_min(0)

    if w0.sum() <= 0 or w1.sum() <= 0:
        raise ValueError("All weights zero on one side.")

    a = (w0 / w0.sum()).to(device=device, dtype=torch.float64)  # POT prefers float64
    b = (w1 / w1.sum()).to(device=device, dtype=torch.float64)

    # cost
    C = _pairwise_traversal_cost(
        X0.to(device), F0.to(device),
        X1.to(device), F1.to(device),
        eps_df=eps_df, ratio_power=ratio_power,
        forward_only=forward_only, large_cost_val=large_cost_val
    )
    C_np = C.detach().cpu().numpy().astype(np.float64)
    # numerics
    C_np[~np.isfinite(C_np)] = large_cost_val

    a_np = a.detach().cpu().numpy()
    b_np = b.detach().cpu().numpy()

    # solve
    if solver == "emd":
        Pi_np = ot.emd(a_np, b_np, C_np, numItermax=ot_numItermax)
    elif solver == "sinkhorn":
        Pi_np = ot.sinkhorn(a_np, b_np, C_np, reg=reg, numItermax=ot_numItermax)
    elif solver == "unbalanced":
        # KL-unbalanced Sinkhorn; tau are relaxation strengths (bigger = closer to balanced)
        Pi_np = ot.unbalanced.sinkhorn_knopp_unbalanced(a_np, b_np, C_np,
                                                        reg, unb_tau_a, unb_tau_b,
                                                        numItermax=ot_numItermax)
    else:
        raise ValueError("solver must be one of {'emd','sinkhorn','unbalanced'}.")

    Pi = torch.from_numpy(Pi_np).to(device=device, dtype=torch.float32)
    # normalize for safety (some solvers already produce sum=1)
    total = Pi.sum()
    if total > 0:
        Pi = Pi / total

    # Return side data (float32 for the model), but keep a,b in float32 too
    return (X0.to(dtype), F0.to(dtype), a.to(torch.float32)), \
           (X1.to(dtype), F1.to(dtype), b.to(torch.float32)), \
           Pi


@torch.no_grad()
def sample_pairs_from_coupling(
    X0: torch.Tensor, F0: torch.Tensor, a: torch.Tensor,
    X1: torch.Tensor, F1: torch.Tensor, b: torch.Tensor,
    Pi: torch.Tensor,
    J: int = 1,                         # samples per source
    eps_df: float = 1e-6,
    fallback_to_min_cost: bool = True,
    C: torch.Tensor = None,             # optional precomputed cost (m,n) to speed fallback
    ):
    """
    Draw J partner indices per source row i from categorical(Pi[i,:]),
    building (x0, x1, u_hat) pairs. Returns:
      x0_pairs, x1_pairs, u_hat_pairs, pair_weights
    where pair_weights sum to 1 and approximate the transport mass across pairs.
    """
    device = X0.device
    dtype  = X0.dtype
    m, n = X0.shape[0], X1.shape[0]
    if Pi.shape != (m, n):
        raise ValueError("Pi shape must match (m,n).")

    # Row-normalized probabilities for categorical draws
    row_mass = Pi.sum(dim=1, keepdim=True)  # (m,1)
    probs = torch.where(row_mass > 0, Pi / row_mass, torch.zeros_like(Pi))

    # Prepare outputs
    x0_list, x1_list, uh_list, w_list = [], [], [], []

    # Optional fallback: precompute argmin costs for invalid rows
    if fallback_to_min_cost and (C is None):
        # cheap approximate cost only if needed later (avoid O(mn) if not)
        pass

    for i in range(m):
        pi_row_sum = row_mass[i].item()
        if pi_row_sum <= 0:
            if not fallback_to_min_cost:
                continue
            # Fallback: pick nearest valid j (min cost with df>0)
            if C is None:
                # compute cost row on the fly
                dx = X1 - X0[i:i+1]           # (n,D)
                dist2 = (dx**2).sum(dim=1)    # (n,)
                df = F1 - F0[i]
                denom = torch.clamp(df, min=eps_df)
                cost_i = dist2 / (denom**2)
                cost_i = torch.where(df > 0, cost_i, torch.full_like(cost_i, float('inf')))
            else:
                cost_i = C[i]
            j = torch.argmin(cost_i).item()
            js = [j]*J
            row_prob = torch.zeros(n, device=device, dtype=dtype)
            row_prob[j] = 1.0
        else:
            # sample J partners with replacement
            js = torch.multinomial(probs[i].clamp_min(0), num_samples=J, replacement=True).tolist()
            row_prob = probs[i]

        x0i = X0[i].expand(J, -1)            # (J,D)
        x1j = X1[js]                          # (J,D)
        df  = (F1[js] - F0[i]).unsqueeze(1)   # (J,1)
        uhat = (x1j - x0i) / (df.abs() + eps_df)

        # weights for the J pairs: split the row mass equally among its J samples
        # this produces an unbiased MC estimate of row expectations
        if pi_row_sum > 0:
            pair_w = (pi_row_sum / J) * torch.ones(J, device=device, dtype=dtype)
        else:
            pair_w = (1.0 / (m*J)) * torch.ones(J, device=device, dtype=dtype)  # tiny mass

        x0_list.append(x0i)
        x1_list.append(x1j)
        uh_list.append(uhat)
        w_list.append(pair_w)

    if len(x0_list)==0:
        # empty
        D = X0.shape[1]
        z = torch.empty(0, D, device=device, dtype=dtype)
        return z, z, z, torch.empty(0, device=device, dtype=dtype)

    x0_pairs = torch.cat(x0_list, dim=0)
    x1_pairs = torch.cat(x1_list, dim=0)
    u_hat_pairs = torch.cat(uh_list, dim=0)
    pair_weights = torch.cat(w_list, dim=0)
    # normalize weights to sum to 1
    s = pair_weights.sum()
    if s > 0:
        pair_weights = pair_weights / s

    return x0_pairs, x1_pairs, u_hat_pairs, pair_weights
