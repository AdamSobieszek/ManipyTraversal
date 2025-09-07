# KLPath.py
import torch
from torch import nn
from torch.linalg import slogdet

class KLPath(nn.Module):
    """
    Empirical KL penalty between two sets of samples.

    Args:
        mode: "gaussian" | "kde"
        symmetric: if True, loss = KL(P||Q) + KL(Q||P)
        bandwidth: (kde only) float, or "median", or "scott". If None -> "median".
        detach_reference: if True, the first argument to forward() is treated as constant
                          (no grads through initial samples).
        eps: numerical jitter.
    """
    def __init__(
        self,
        mode: str = "gaussian",
        symmetric: bool = False,
        bandwidth: float | str | None = None,
        detach_reference: bool = False,
        eps: float = 1e-6,
    ):
        super().__init__()
        assert mode in {"gaussian", "kde"}
        self.mode = mode
        self.symmetric = symmetric
        self.bandwidth = bandwidth
        self.detach_reference = detach_reference
        self.eps = eps

    # ---------- utils ----------
    def _flatten(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.size(0), -1)

    def _pairwise_sq_dists(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        # X: [N,D], Y: [M,D] -> [N,M]
        # ||x-y||^2 = ||x||^2 + ||y||^2 - 2 x·y
        X2 = (X * X).sum(-1, keepdim=True)
        Y2 = (Y * Y).sum(-1, keepdim=True).T
        return (X2 + Y2 - 2.0 * X @ Y.T).clamp_min(0.0)

    # ---------- Gaussian (fit mean/cov, closed-form KL) ----------
    def _fit_mvn(self, X: torch.Tensor):
        N, D = X.shape
        mu = X.mean(0, keepdim=True)  # [1,D]
        Xc = X - mu
        # Biased covariance (1/N) for stability; add eps*I
        cov = (Xc.T @ Xc) / max(N, 1)
        cov = cov + torch.eye(D, device=X.device, dtype=X.dtype) * self.eps
        return mu.squeeze(0), cov

    def _kl_mvn(self, mu0, S0, mu1, S1) -> torch.Tensor:
        # KL(N0 || N1) for full-cov Gaussians
        D = mu0.numel()
        # log det via slogdet for stability
        sign0, logdet0 = slogdet(S0)
        sign1, logdet1 = slogdet(S1)
        # Inverses
        S1_inv = torch.linalg.inv(S1)
        diff = (mu1 - mu0).unsqueeze(1)  # [D,1]
        tr_term = torch.trace(S1_inv @ S0)
        quad = (diff.transpose(0,1) @ S1_inv @ diff).squeeze()
        kl = 0.5 * (logdet1 - logdet0 - D + tr_term + quad)
        return kl

    # ---------- KDE-based empirical KL ----------
    def _select_bandwidth(self, X: torch.Tensor, Y: torch.Tensor) -> float:
        if isinstance(self.bandwidth, (float, int)):
            return float(self.bandwidth)
        rule = self.bandwidth or "median"
        D = X.size(1)
        if rule == "median":
            # Median heuristic on combined data
            Z = torch.cat([X, Y], dim=0)
            with torch.no_grad():
                # sample for speed if huge
                if Z.size(0) > 4096:
                    idx = torch.randperm(Z.size(0), device=Z.device)[:4096]
                    Zs = Z[idx]
                else:
                    Zs = Z
                d2 = self._pairwise_sq_dists(Zs, Zs)
                # exclude zeros on diagonal
                d2 = d2 + torch.eye(d2.size(0), device=d2.device, dtype=d2.dtype) * float("inf")
                med = d2.min(dim=1)[0].median().sqrt().item()
            h = max(med, self.eps)
            return h
        elif rule == "scott":
            # Scott’s rule: h ~ N^{-1/(D+4)} scaled by std per dim -> use global scalar
            std = X.std(dim=0).mean().clamp_min(self.eps).item()
            h = std * (X.size(0) ** (-1.0 / (D + 4)))
            return max(h, self.eps)
        else:
            raise ValueError(f"Unknown bandwidth rule: {self.bandwidth}")

    def _log_gaussian_kde(self, X: torch.Tensor, centers: torch.Tensor, h: float, leave_one_out: bool) -> torch.Tensor:
        """
        log p_h(X) where p_h(x) = (1/M) sum_j N(x | c_j, h^2 I)
        X: [N,D], centers: [M,D]
        Returns: [N]
        """
        N, D = X.shape
        M = centers.size(0)
        d2 = self._pairwise_sq_dists(X, centers)  # [N,M]
        # log (1/M sum_j exp(-||x-cj||^2 / (2 h^2))) - D/2 log(2π h^2)
        log_weights = -d2 / (2.0 * (h ** 2))
        # leave-one-out if X and centers are the same tensor (by identity)
        if leave_one_out and X.data_ptr() == centers.data_ptr() and N == M:
            # subtract -inf on diagonal to remove self term
            inf_mask = torch.eye(N, device=X.device, dtype=torch.bool)
            log_weights = log_weights.masked_fill(inf_mask, float("-inf"))
            Z = torch.logsumexp(log_weights, dim=1) - torch.log(torch.tensor(M - 1.0, device=X.device, dtype=X.dtype))
        else:
            Z = torch.logsumexp(log_weights, dim=1) - torch.log(torch.tensor(M * 1.0, device=X.device, dtype=X.dtype))
        norm_const = -0.5 * D * (math.log(2.0 * math.pi) + 2.0 * math.log(h))
        return Z + norm_const

    def _kl_kde(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Empirical KL(P||Q) with KDEs:
            1/N sum_i [ log p(x_i) - log q(x_i) ]
        where p, q are Gaussian-KDE with bandwidth h (same h for both, from combined data).
        """
        h = self._select_bandwidth(X, Y)
        log_p = self._log_gaussian_kde(X, centers=X, h=h, leave_one_out=True)   # LOO for p
        log_q = self._log_gaussian_kde(X, centers=Y, h=h, leave_one_out=False)  # plain for q
        return (log_p - log_q).mean()

    # ---------- public ----------
    def forward(self, initial: torch.Tensor, manipulated: torch.Tensor) -> torch.Tensor:
        """
        Args:
            initial: samples from P, shape [N, ...]
            manipulated: samples from Q, shape [M, ...]
        Returns:
            scalar loss (>= 0 for Gaussian KL; KDE estimate can be slightly <0 due to estimation noise)
        """
        X = self._flatten(initial)
        Y = self._flatten(manipulated)

        if self.detach_reference:
            X = X.detach()

        if self.mode == "gaussian":
            muX, SX = self._fit_mvn(X)
            muY, SY = self._fit_mvn(Y)
            kl_xy = self._kl_mvn(muX, SX, muY, SY)
            if self.symmetric:
                kl_yx = self._kl_mvn(muY, SY, muX, SX)
                return kl_xy + kl_yx
            else:
                return kl_xy

        # KDE mode
        kl_xy = self._kl_kde(X, Y)
        if self.symmetric:
            kl_yx = self._kl_kde(Y, X)
            return (kl_xy + kl_yx)
        else:
            return kl_xy