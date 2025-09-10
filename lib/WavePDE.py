import math
from typing import Dict, Optional, List

import torch
from torch import nn


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
    def __init__(self, K: int, n_in: int, n_out: int = 1, n_hidden: int = 128,  activation: nn.Module = nn.Tanh(), final_activation: nn.Module = nn.Identity()):
        super().__init__()
        self.K = int(K)
        self.n_in = int(n_in)
        self.n_out = int(n_out)
        self.n_hidden = int(n_hidden)

        self.fc1 = StackedLinear(self.K, self.n_in, self.n_in)
        self.act1 = activation

        self.fc2 = StackedLinear(self.K, self.n_in, self.n_hidden)
        self.act2 = activation

        self.fc3 = StackedLinear(self.K, self.n_hidden, self.n_hidden)
        self.act3 = activation

        self.fc4 = StackedLinear(self.K, self.n_hidden, self.n_out)

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


    def update_y_distributions(self, y: torch.Tensor):
        """
        Update the density estimates for each k. This should stay implementation-agnostic with a plug-in estimator class call.
        """
        pass


class StackedSliceEnergy(nn.Module):
    def __init__(self, K: int, n_in: int, n_out: int = 1, n_hidden: int = 64, final_activation: nn.Module = nn.Identity(),
                 apply_output_bn: bool = False):
        super().__init__()
        self.K = int(K)
        self.n_in = int(n_in)
        self.n_out = int(n_out)
        self.n_hidden = int(n_hidden)


        # x pathway
        self.layer_x = StackedLinear(self.K, self.n_in, self.n_in)
        self.activation1 = nn.Tanh()

        # time pathway (uses sinusoidal embeddings of size n_in)
        self.layer_pos = StackedSinusoidalPositionEmbeddings(self.n_hidden)
        self.layer_time = StackedLinear(self.K, self.n_hidden, self.n_hidden)
        self.activation2 = nn.GELU()
        self.layer_time2 = StackedLinear(self.K, self.n_hidden, self.n_in)

        # fusion + output
        self.layer_fusion = StackedLinear(self.K, self.n_in, self.n_hidden)
        self.activation3 = nn.Tanh()
        self.layer_out = StackedLinear(self.K, self.n_hidden, self.n_out)
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
from lib.pde_ops import PDEState
from lib.pde_losses import build_losses 

class WavePDE(nn.Module):
    """
    K-parallel WavePDE powered by PDEState and modular PDE losses.
    """
    def __init__(
        self,
        num_support_sets: int,
        num_support_timesteps: int,
        support_vectors_dim: int,
        lambdas: Dict[str, float] = {},                   # ONLY what you want active (may include 0.0)
        n_laplace_probes: int = 1,
        apply_bn_on_psi_output: bool = False,
        # PDEState config
        time_ad: str = "reverse",
        detach_between_steps: bool = True,
        eps_norm2: float = 1e-8,
        divergence_probes: int = 1,
        rng: str = "rademacher",
        seed: Optional[int] = None,
        # optional: prior score function for DivPrior (defaults to Gaussian score -x)
        prior_score: Optional[callable] = None,
    ):
        super().__init__()
        self.num_support_sets = int(num_support_sets)
        self.num_support_timesteps = int(num_support_timesteps)
        self.support_vectors_dim = int(support_vectors_dim)

        # Learnable per-k scale (kept for parity; not wired to dt by default)
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

        # PDEState config for each step
        self._pde_cfg = dict(
            time_ad=time_ad,
            detach_between_steps=detach_between_steps,
            eps_norm2=eps_norm2,
            laplace_probes=int(n_laplace_probes),
            divergence_probes=int(divergence_probes),
            rng=rng,
            seed=seed,
        )

        # Build modular loss list from registry (only specified keys are included).
        # Pass modules/params needed by certain losses through ctx.
        epsilon = float(lambdas.get("epsilon", 0.0))  # a scalar param (not a loss)
        self.losses, self._needs_next = build_losses(
            lambdas,
            F=self.F,
            epsilon=epsilon,
            prior_score=prior_score,
        )

        # for telemetry
        self._acc: Dict[str, torch.Tensor] = {}

    # ---- one step ----
    def _per_step(self, z_bkd: torch.Tensor, direction: int = +1):
        st = PDEState(
            f=self.F,
            psi=self.PSI,
            z=z_bkd,
            direction=direction,
            need_next=self._needs_next,
            dt_value=1.0,  # use ±1 step; wire self.c here if desired
            **self._pde_cfg,
        )

        # compute & sum selected losses [B,K,1]
        per_bk = [L(st) for L in self.losses]
        L_sum = sum(per_bk) if per_bk else st.zeros()  # if no losses, zero tensor

        # next latent (semi-implicit Euler @ now)
        x_next = st.x_next()

        # (optional) same small step noise as before
        with torch.no_grad():
            step_delta_norms = (x_next - st.x()).norm(dim=-1, keepdim=True)
            latent_noise = torch.randn_like(x_next)
            latent_noise = latent_noise / latent_noise.norm(dim=-1, keepdim=True).clamp_min_(1e-12)
            latent_noise = latent_noise * (step_delta_norms / 5.0)
            x_next_noisy = x_next + latent_noise

        return st, x_next_noisy, L_sum

    # ---- unrolled training ----
    def forward(self, z: torch.Tensor, t_index: torch.Tensor, direction: int = +1):
        """
        Returns:
          potential_preds: [B,K,1] (detached)
          latent1_bk, latent2_bk: [B,K,D] (pair captured at t_index)
          L_total_mean: scalar
        """
        B, D = z.shape
        K = self.num_support_sets
        T = max(1, int(self.num_support_timesteps))

        i_target = torch.clamp(t_index.view(-1), 0, T - 1).long()

        # expand once to K stacks
        z_curr = z.unsqueeze(1).expand(B, K, D).contiguous()

        latent1_bk = torch.zeros_like(z_curr)
        latent2_bk = torch.zeros_like(z_curr)

        L_accum = None  # accumulate per-[B,K,1]

        step_iter = range(T) if direction == +1 else reversed(range(T))
        for i in step_iter:
            st, x_next, L_step = self._per_step(z_curr, direction=direction)

            # accumulate loss per step
            L_accum = (L_step if L_accum is None else L_accum + L_step)

            # capture (latent1, latent2) at the requested index
            mask_b = (i_target == i).view(B, 1, 1)
            latent1_bk = torch.where(mask_b, z_curr, latent1_bk)
            latent2_bk = torch.where(mask_b, x_next, latent2_bk)

            # advance
            z_curr = x_next

        # average over steps
        L_total_per_bk = L_accum / float(T) if L_accum is not None else st.zeros()
        L_total_mean = L_total_per_bk.mean()

        # telemetry
        potential_preds = st.f().detach()
        
        self._acc = {
            "xf_now": st.Xf(),
            "L_mean": L_total_mean.detach(),
            **st.state["losses"],
        }
        return potential_preds, latent1_bk, latent2_bk, L_total_mean

    def get_losses(self) -> Dict[str, torch.Tensor]:
        return self._acc

    @torch.enable_grad()
    def inference(self, z: torch.Tensor, direction: int = +1) -> List[torch.Tensor]:
        B, D = z.shape
        K = self.num_support_sets
        T = max(1, int(self.num_support_timesteps))
        traj: List[torch.Tensor] = []

        z_curr = z.unsqueeze(1).expand(B, K, D).contiguous()
        traj.append(z_curr)

        for _ in (range(T) if direction == +1 else reversed(range(T))):
            st = PDEState(
                f=self.F,
                psi=self.PSI,
                z=z_curr,
                direction=direction,
                need_next=False,
                dt_value=1.0,
                **self._pde_cfg,
            )
            z_next = st.x() + st.dt() * st.v("now")
            traj.append(z_next)
            z_curr = z_next
        return traj
