import math
from typing import Dict, Tuple, Optional, List

import torch
from torch import nn
from torch.autograd import grad
from torch.func import jvp as jvp_fwd


# ================================================================
# Core stacked layers (vectorized over support-set axis K)
# ================================================================
class StackedLinear(nn.Module):
    """
    Per-k linear layers evaluated in parallel.
    Weight: [K, out, in], Bias: [K, out]
    Forward expects x: [B, K, in] -> y: [B, K, out]
    """
    def __init__(self, K: int, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.K = int(K)
        self.in_features = int(in_features)
        self.out_features = int(out_features)

        # [K, out, in]
        self.weight = nn.Parameter(
            torch.empty(self.K, self.out_features, self.in_features)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(self.K, self.out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        # K independent Kaiming-uniform initializations
        for k in range(self.K):
            nn.init.kaiming_uniform_(self.weight.data[k], a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias.data, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, K, in]
        # y[b,k,o] = sum_i x[b,k,i] * W[k,o,i] + b[k,o]
        y = torch.einsum("bki,koi->bko", x, self.weight)
        if self.bias is not None:
            y = y + self.bias.unsqueeze(0)  # [1,K,out]
        return y


class OutputBatchNormPerK(nn.Module):
    """
    BatchNorm applied per (k, channel) using batch statistics over B only.
    x: [B, K, C] -> y: [B, K, C]

    Keeps running_mean/var per (K,C). Supports affine per (K,C).
    This preserves BatchNorm semantics for each potential independently.
    """
    def __init__(
        self,
        K: int,
        C: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ):
        super().__init__()
        self.K = int(K)
        self.C = int(C)
        self.eps = float(eps)
        self.momentum = float(momentum)
        self.affine = bool(affine)
        self.track_running_stats = bool(track_running_stats)

        if self.affine:
            self.weight = nn.Parameter(torch.ones(self.K, self.C))  # gamma
            self.bias = nn.Parameter(torch.zeros(self.K, self.C))   # beta
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        if self.track_running_stats:
            self.register_buffer("running_mean", torch.zeros(self.K, self.C))
            self.register_buffer("running_var", torch.ones(self.K, self.C))
            self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)
            self.register_buffer("num_batches_tracked", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 3 and x.size(1) == self.K and x.size(2) == self.C, (
            f"Expected [B,K,{self.C}] got {tuple(x.shape)}")
        B = x.size(0)

        if self.training:
            mean = x.mean(dim=0)  # [K,C]
            var = x.var(dim=0, unbiased=False)  # [K,C]

            if self.track_running_stats:
                with torch.no_grad():
                    self.num_batches_tracked += 1
                    m = self.momentum
                    self.running_mean.mul_(1 - m).add_(m * mean)
                    self.running_var.mul_(1 - m).add_(m * var)
        else:
            if self.track_running_stats:
                mean = self.running_mean
                var = self.running_var
            else:
                # fall back to batch stats if not tracking
                mean = x.mean(dim=0)
                var = x.var(dim=0, unbiased=False)

        y = (x - mean.unsqueeze(0)) / torch.sqrt(var.unsqueeze(0) + self.eps)
        if self.affine:
            y = y * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)
        return y


class StackedSinusoidalPositionEmbeddings(nn.Module):
    """
    Sinusoidal embedding broadcast over K.
    Input: t [B, K, 1] -> emb [B, K, E]; E must be even.
    """
    def __init__(self, emb_dim: int):
        super().__init__()
        assert emb_dim % 2 == 0, "time embedding dim should be even"
        self.emb_dim = int(emb_dim)
        half_dim = emb_dim // 2
        emb_scale = math.log(10000.0) / max(half_dim - 1, 1)
        freqs = torch.exp(torch.arange(half_dim) * -emb_scale)
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: [B,K,1]
        angles = t * self.freqs.view(1, 1, -1).to(t.dtype)  # [B,K,half]
        return torch.cat([angles.sin(), angles.cos()], dim=-1)  # [B,K,E]


# ================================================================
# Stacked potentials: f (SemanticPotential) and ψ (SliceEnergy)
#   - Hidden layers have NO BatchNorm per your requirement
#   - Only the FINAL output of f (and optionally ψ) is BatchNormed per-k
# ================================================================
class StackedSemanticPotential(nn.Module):
    def __init__(self, K: int, n_in: int, n_out: int = 1, final_activation: nn.Module = nn.Identity()):
        super().__init__()
        self.K = int(K)
        self.n_in = int(n_in)
        self.n_out = int(n_out)

        hidden = self.n_in
        self.fc1 = StackedLinear(self.K, self.n_in, hidden)
        self.act1 = nn.Tanh()

        self.fc2 = StackedLinear(self.K, hidden, hidden)
        self.act2 = nn.Tanh()

        self.fc3 = StackedLinear(self.K, hidden, hidden)
        self.act3 = nn.Tanh()

        self.fc4 = StackedLinear(self.K, hidden, self.n_out)

        # Additional linear component from input to output, initialized to random unit directions (per k)
        self.dir_linear = StackedLinear(self.K, self.n_in, self.n_out, bias=False)
        with torch.no_grad():
            W = self.dir_linear.weight.data  # [K, out(=1), in]
            # normalize each row vector (per k, per out)
            norms = W.norm(dim=-1, keepdim=True).clamp_min_(1e-8)  # [K,1,1]
            W.normal_()
            W.div_(norms)

        # Output-only BatchNorm per k
        self.out_bn = OutputBatchNormPerK(self.K, self.n_out)
        self.final_activation = final_activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,K,D]
        h = self.act1(self.fc1(x))
        h = self.act2(self.fc2(h))
        h = self.act3(self.fc3(h))
        out = self.fc4(h) + self.dir_linear(x)  # [B,K,1]
        out = self.out_bn(out)
        return self.final_activation(out)


class StackedSliceEnergy(nn.Module):
    def __init__(self, K: int, n_in: int, n_out: int = 1, final_activation: nn.Module = nn.Identity(),
                 apply_output_bn: bool = False):
        super().__init__()
        self.K = int(K)
        self.n_in = int(n_in)
        self.n_out = int(n_out)

        # x pathway
        self.layer_x = StackedLinear(self.K, self.n_in, self.n_in)
        self.activation1 = nn.Tanh()

        # time pathway (uses sinusoidal embeddings of size n_in)
        self.layer_pos = StackedSinusoidalPositionEmbeddings(self.n_in)
        self.layer_time = StackedLinear(self.K, self.n_in, self.n_in)
        self.activation2 = nn.GELU()
        self.layer_time2 = StackedLinear(self.K, self.n_in, self.n_in)

        # fusion + output
        self.layer_fusion = StackedLinear(self.K, self.n_in, self.n_in)
        self.activation3 = nn.Tanh()
        self.layer_out = StackedLinear(self.K, self.n_in, self.n_out)
        self.activation4 = nn.Tanh()

        self.apply_output_bn = bool(apply_output_bn)
        self.out_bn = OutputBatchNormPerK(self.K, self.n_out) if self.apply_output_bn else None

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        # x: [B,K,D], time: [B,K,1]
        xh = self.activation1(self.layer_x(x))
        t_feat = self.layer_pos(time)
        t_feat = self.layer_time(t_feat)
        t_feat = self.activation2(t_feat)
        t_feat = self.layer_time2(t_feat)

        h = self.activation3(self.layer_fusion(xh + t_feat))
        out = self.activation4(self.layer_out(h))  # [B,K,1]
        if self.apply_output_bn:
            out = self.out_bn(out)
        return out


# ================================================================
# WavePDE: fully parallel over K
# ================================================================
class WavePDE(nn.Module):
    """
    Parallel (stacked) implementation over K support sets.

    Inputs are standard batch latents z: [B, D]. Internally we expand to [B, K, D]
    and compute f (SemanticPotential) and ψ (SliceEnergy) for all K in parallel.

    Public API:
      - forward(z, t_index, direction=+1): returns (potential_preds_bk, latent1_bk, latent2_bk, L_total_mean)
          * potential_preds_bk: [B, K, 1] (detached)
          * latent1_bk / latent2_bk: [B, K, D]
          * L_total_mean: scalar (mean over B and K)
      - get_losses(): dict of last per-term means (scalars) + tensors for diagnostics
      - inference(z, direction=+1): rollout trajectory per step, returns list of [B, K, D]
    """
    def __init__(
        self,
        num_support_sets: int,
        num_support_timesteps: int,
        support_vectors_dim: int,
        n_laplace_probes: int = 1,
        lambdas: Optional[Dict] = None,
        apply_bn_on_psi_output: bool = False,
    ):
        super().__init__()
        self.num_support_sets = int(num_support_sets)
        self.num_support_timesteps = int(num_support_timesteps)
        self.support_vectors_dim = int(support_vectors_dim)
        self.n_laplace_probes = int(n_laplace_probes)

        # Optional learnable step-size per k (kept for future use)
        self.c = nn.Parameter(torch.full((self.num_support_sets, 1), 1.0))

        # Stacked potentials
        self.PSI = StackedSliceEnergy(
            K=self.num_support_sets,
            n_in=self.support_vectors_dim,
            n_out=1,
            final_activation=nn.Identity(),
            apply_output_bn=apply_bn_on_psi_output,
        )
        self.F = StackedSemanticPotential(
            K=self.num_support_sets,
            n_in=self.support_vectors_dim,
            n_out=1,
            final_activation=nn.Identity(),
        )

        # Loss weights (defaults)
        default_lambdas = dict(
            ot=.0,
            kin=0.0,
            sliceHJ=0.0,
            foot=0.0,
            unitspeed=1.0,
            div=0.0,
            BB=0.,
            tan=0.0,
            epsilon=0.0,
        )
        self.lambdas = default_lambdas if lambdas is None else {**default_lambdas, **lambdas}

        # For logging/telemetry
        self._acc: Dict[str, torch.Tensor] = {}

    # ---------------- Hutchinson Laplacian (stacked) ----------------
    def _laplacian_hutchinson_single(self, u_z: torch.Tensor, z_leaf: torch.Tensor) -> torch.Tensor:
        """Single-probe Laplacian using existing (u_z, z) graph.
        Shapes: u_z, z_leaf: [B, K, D] -> returns [B, K, 1]
        """
        v = torch.empty_like(z_leaf).bernoulli_(0.5).mul_(2).sub_(1)
        Hv = grad((u_z * v).sum(), z_leaf, create_graph=True, retain_graph=True)[0]
        lap = (Hv * v).sum(dim=-1, keepdim=True)  # [B,K,1]
        return lap

    def _laplacian_hutchinson_multi(self, mlp: nn.Module, z: torch.Tensor, t: torch.Tensor, m: int) -> torch.Tensor:
        """
        Vectorized m-probe Laplacian: recompute u_z on (m×B,K) batch so it depends on z_rep.
        z: [B,K,D], t: [B,K,1] -> returns [B,K,1]
        """
        B, K, D = z.shape
        z_rep = z.detach().unsqueeze(0).repeat(m, 1, 1, 1).reshape(m * B, K, D).requires_grad_(True)
        t_rep = t.detach().unsqueeze(0).repeat(m, 1, 1, 1).reshape(m * B, K, 1).requires_grad_(True)

        u_rep = mlp(z_rep, t_rep)  # [mB, K, 1]
        u_z_rep = grad(u_rep.sum(), z_rep, create_graph=True, retain_graph=True)[0]  # [mB, K, D]

        v_rep = torch.empty_like(z_rep).bernoulli_(0.5).mul_(2).sub_(1)
        Hv_rep = grad((u_z_rep * v_rep).sum(), z_rep, create_graph=True, retain_graph=True)[0]
        lap = (Hv_rep * v_rep).sum(dim=-1).reshape(m, B, K).mean(dim=0, keepdim=False).unsqueeze(-1)  # [B,K,1]
        return lap

    def _laplacian_hutchinson(self, psi_mlp: nn.Module, z_leaf: torch.Tensor, t: torch.Tensor, u_z: torch.Tensor) -> torch.Tensor:
        m = max(1, self.n_laplace_probes)
        if m == 1:
            return self._laplacian_hutchinson_single(u_z, z_leaf)
        else:
            return self._laplacian_hutchinson_multi(psi_mlp, z_leaf, t, m)

    # ---------------- Divergence (Hutchinson, stacked) ----------------
    def _divergence_hutchinson(self, v_tensor: torch.Tensor, z_leaf: torch.Tensor) -> torch.Tensor:
        eps = torch.empty_like(z_leaf).bernoulli_(0.5).mul_(2).sub_(1)
        Jv_eps = grad((v_tensor * eps).sum(), z_leaf, create_graph=True, retain_graph=True)[0]
        div_est = (Jv_eps * eps).sum(dim=-1, keepdim=True)  # [B,K,1]
        return div_est

    # ---------------- Forward-mode time derivs (stacked) ----------------
    def _u_t_and_u_tt_forward_mode(self, psi_mlp: nn.Module, z: torch.Tensor, t: torch.Tensor):
        """
        z: [B,K,D] treated as constant; t: [B,K,1]
        Returns (u_t, u_tt): both [B,K,1]
        """
        z_det = z.detach()

        def f_t(t_in):
            return psi_mlp(z_det, t_in)

        ones = torch.ones_like(t)
        _, u_t = jvp_fwd(f_t, (t,), (ones,))

        def ut_func(t_in):
            return jvp_fwd(f_t, (t_in,), (torch.ones_like(t_in),))[1]

        _, u_tt = jvp_fwd(ut_func, (t,), (ones,))
        return u_t, u_tt

    # ---------------- One PDE step (stacked) ----------------

    def _per_step(self, z_bk: torch.Tensor, direction: int = +1):
        """
        One PDE-informed step for ALL K in parallel.
        Inputs:
        z_bk: [B,K,D]  (current state for this step)
        Returns:
        losses: dict of [B,K,1] tensors
        extras: dict with tensors (e.g., f_now, xf_now) at this step
        z_leaf: [B,K,D] leaf used for grads
        z_next: [B,K,D] next state
        """
        device, dtype = z_bk.device, z_bk.dtype
        B, K, D = z_bk.shape
        lam = self.lambdas

        # Make a leaf that we can differentiate through
        z_leaf = z_bk.detach().clone().requires_grad_(True)

        # Δt
        dt = torch.full((B, K, 1), float(direction), device=device, dtype=dtype)

        # f and its gradient at current z
        t_now = self.F(z_leaf)  # [B,K,1]  (BatchNorm here is per-K)
        gradf_now = grad(t_now.sum(), z_leaf, create_graph=True, retain_graph=True)[0]  # [B,K,D]
        norm2_f_now = (gradf_now.pow(2).sum(dim=-1, keepdim=True) + 1e-8)              # [B,K,1]
        Xf_now = gradf_now / norm2_f_now                                               # [B,K,D]
        t_stop = t_now.detach()

        # ψ at t and t+Δt
        psi_now  = self.PSI(z_leaf, t_stop)       # [B,K,1]
        psi_next = self.PSI(z_leaf, t_stop + dt)  # [B,K,1]

        # ∇ψ
        gpsi_now = grad(psi_now.sum(),  z_leaf, create_graph=True, retain_graph=True)[0]   # [B,K,D]
        gpsi_next = grad(psi_next.sum(), z_leaf, create_graph=True, retain_graph=True)[0]  # [B,K,D]
        psi_t_next, psi_tt_next = self._u_t_and_u_tt_forward_mode(self.PSI, z_leaf, t_stop + dt)  # [B,K,1]

        # footpoint
        x_hat = z_leaf + gpsi_next                                                       # [B,K,D]
        t_next = self.F(x_hat)                                                           # [B,K,1]
        gradf_next = grad(t_next.sum(), x_hat, create_graph=True, retain_graph=True)[0]  # [B,K,D]
        norm2_f_next = (gradf_next.pow(2).sum(dim=-1, keepdim=True) + 1e-8)              # [B,K,1]
        Xf_next = gradf_next / norm2_f_next                                              # [B,K,D]

        # velocity & step
        v = Xf_now + gpsi_now                                                            # [B,K,D]
        z_next = z_leaf + dt * v                                                         # [B,K,D]

        # losses (per step)
        zero = torch.zeros((), device=device, dtype=dtype)

        L_ot = (gpsi_next - (Xf_now + Xf_next).detach() / 2.0).pow(2).sum(dim=-1, keepdim=True)  # [B,K,1]

        if lam.get("kin", 0.0) > 0.0:
            kin_res = psi_tt_next - gpsi_next.norm(dim=-1, keepdim=True) / (norm2_f_next.sqrt())
            L_kin = kin_res.pow(2)
        else:
            L_kin = zero

        if lam.get("sliceHJ", 0.0) > 0.0:
            eps2 = float(lam.get("epsilon", 0.0)) ** 2
            if eps2 > 0.0:
                lap_now = self._laplacian_hutchinson(self.PSI, z_leaf, t_stop, gpsi_next)  # [B,K,1]
            else:
                lap_now = torch.zeros_like(psi_now)
            sliceHJ_res = psi_now + 0.5 * (gpsi_next.pow(2).sum(dim=-1, keepdim=True)) - 0.5 * eps2 * lap_now
            L_sliceHJ = sliceHJ_res.pow(2)
        else:
            L_sliceHJ = zero

        if lam.get("foot", 0.0) > 0.0:
            L_foot = (t_next - (t_now + dt)).pow(2)
        else:
            L_foot = zero

        if lam.get("unitspeed", 0.0) > 0.0:
            unitspeed_res = (gradf_now * v).sum(dim=-1, keepdim=True) - 1.0
            L_unitspeed = unitspeed_res.pow(2)
        else:
            L_unitspeed = zero

        if lam.get("div", 0.0) > 0.0:
            div_v = self._divergence_hutchinson(v, z_leaf)  # [B,K,1]
            s_prior = -z_leaf
            divprior_res = div_v + (s_prior * v).sum(dim=-1, keepdim=True)
            L_div_prior = divprior_res.pow(2)
        else:
            L_div_prior = zero

        if lam.get("BB", 0.0) > 0.0:
            L_BB = v.pow(2).sum(dim=-1, keepdim=True)
        else:
            L_BB = zero

        losses = dict(
            ot=L_ot, kin=L_kin, sliceHJ=L_sliceHJ, foot=L_foot,
            unitspeed=L_unitspeed, div=L_div_prior, BB=L_BB, tan=zero,
        )
        extras = dict(
            z_next=z_next, v=v, x_hat=x_hat, f_now=t_now, psi_now=psi_now,
            psi_next=psi_next, t_now=t_now, xf_now=Xf_now,
        )
        return losses, extras, z_leaf, z_next

    def forward(self, z: torch.Tensor, t_index: torch.Tensor, direction: int = +1):
        """
        z: [B,D], t_index: [B,1] integers in [0, T-1]
        Returns:
        potential_preds: [B,K,1] (detached)
        latent1_bk, latent2_bk: [B,K,D]
        L_total_mean: scalar
        """
        device = z.device
        B, D = z.shape
        K = self.num_support_sets
        T = max(1, int(self.num_support_timesteps))

        if t_index.ndim == 1:
            t_index = t_index.unsqueeze(-1)
        i_target = torch.clamp(t_index, 0, T - 1).long().squeeze(-1)  # [B]

        # current state
        z_curr = z.unsqueeze(1).expand(B, K, D).contiguous()  # [B,K,D]

        # accumulators over steps (per B,K)
        sum_terms = {k: 0.0 for k in ["ot", "kin", "sliceHJ", "foot", "unitspeed", "div", "BB", "tan"]}

        latent1_bk = None
        latent2_bk = None
        last_extras = None

        step_iter = range(T) if direction == +1 else reversed(range(T))
        for i in step_iter:
            losses_i, ex_i, z_leaf, z_next = self._per_step(z_curr, direction=direction)

            with torch.no_grad():
                step_delta_norms = (z_next - z_leaf).norm(dim=-1,keepdim=True)
                latent_noise = torch.randn_like(z_next)
                latent_noise = latent_noise / latent_noise.norm(dim=-1,keepdim=True) * step_delta_norms / 5
            z_next = z_next + latent_noise

            for name, val in losses_i.items():
                sum_terms[name] = sum_terms[name] + val  # [B,K,1]

            # capture pair at i_target (broadcast across K)
            mask_b = (i_target == i).view(B, 1, 1)  # [B,1,1]
            if latent1_bk is None:
                latent1_bk = torch.where(mask_b, z_leaf, torch.zeros_like(z_leaf))
                latent2_bk = torch.where(mask_b, z_next, torch.zeros_like(z_next))
            else:
                latent1_bk = torch.where(mask_b, z_leaf, latent1_bk)
                latent2_bk = torch.where(mask_b, z_next, latent2_bk)

            # advance chain
            z_curr = z_next
            last_extras = ex_i

        # averages over steps
        for name in sum_terms:
            sum_terms[name] = sum_terms[name] / float(T)  # [B,K,1]

        L_total_per_bk = (
            self.lambdas.get("ot", 0.0) * sum_terms["ot"]
            + self.lambdas.get("kin", 0.0) * sum_terms["kin"]
            + self.lambdas.get("sliceHJ", 0.0) * sum_terms["sliceHJ"]
            + self.lambdas.get("foot", 0.0) * sum_terms["foot"]
            + self.lambdas.get("unitspeed", 0.0) * sum_terms["unitspeed"]
            + self.lambdas.get("div", 0.0) * sum_terms["div"]
            + self.lambdas.get("BB", 0.0) * sum_terms["BB"]
            + self.lambdas.get("tan", 0.0) * sum_terms["tan"]
        )
        L_total_mean = L_total_per_bk.mean()

        potential_preds = last_extras["f_now"].detach()  # [B,K,1]
        self._acc = {k: (v.mean() if torch.is_tensor(v) else torch.as_tensor(v)) for k, v in sum_terms.items()}
        self._acc["potential_preds"] = potential_preds
        self._acc["xf_now"] = last_extras["xf_now"]
        return potential_preds, latent1_bk, latent2_bk+latent_noise, L_total_mean


    def get_losses(self) -> Dict[str, torch.Tensor]:
        return self._acc

    @torch.enable_grad()
    def inference(self, z: torch.Tensor, direction: int = +1) -> List[torch.Tensor]:
        """
        Rollout latents without computing losses. Returns trajectory list of [B,K,D].
        """
        B, D = z.shape
        K = self.num_support_sets
        T = max(1, int(self.num_support_timesteps))
        traj: List[torch.Tensor] = []

        # Initialize z_leaf from z once
        z_leaf = (
            z.unsqueeze(1)
            .expand(B, K, D)
            .contiguous()
            .clone()
            .requires_grad_(True)
        )
        traj.append(z_leaf)

        for _ in (range(T) if direction == +1 else reversed(range(T))):
            # Reuse part of _per_step but skip loss building
            dt = torch.full((B, K, 1), float(direction), device=z.device, dtype=z.dtype)
            t_now = self.F(z_leaf)
            gradf_now = grad(t_now.sum(), z_leaf, create_graph=True, retain_graph=True)[0]
            norm2_f_now = (gradf_now.pow(2).sum(dim=-1, keepdim=True) + 1e-8)
            Xf_now = gradf_now / norm2_f_now
            t_stop = t_now.detach()
            psi_now = self.PSI(z_leaf, t_stop)
            gpsi_now = grad(psi_now.sum(), z_leaf, create_graph=True, retain_graph=True)[0]
            v = Xf_now + gpsi_now
            z_next = z_leaf + dt * v
            traj.append(z_next)
            z_leaf = z_next
        return traj
