# WavePDE.py 
import torch
from torch import nn
import math
from torch.autograd import grad
from torch.autograd.functional import jvp as jvp_rev
from torch.func import jvp as jvp_fwd


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0, "time embedding dim should be even"
        self.dim = dim
        half_dim = dim // 2
        emb_scale = math.log(10000.0) / max(half_dim - 1, 1)
        freqs = torch.exp(torch.arange(half_dim) * -emb_scale)  # fp32; cast on use
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        time = time.squeeze(-1)
        freqs = self.freqs.to(time.dtype)
        angles = time.unsqueeze(-1) * freqs
        return torch.cat([angles.sin(), angles.cos()], dim=-1)

class SemanticPotential(nn.Module):
     def __init__(self,  n_in: int, n_out: int, final_activation: nn.Module = nn.Identity()):
         super().__init__()
         # Network structure from the notebook example
         self.network = nn.Sequential(
            nn.Linear(n_in, n_in*2),
            nn.BatchNorm1d(n_in*2),
            nn.Tanh(),

             nn.Linear(n_in*2, n_in*2),
             nn.BatchNorm1d(n_in*2),
             nn.Tanh(),
             nn.Dropout(0.1), 

             nn.Linear(n_in*2, n_in*2),
             nn.BatchNorm1d(n_in*2),
             nn.Tanh(),

             nn.Linear(n_in*2, n_out)
         )
         self.final_activation = final_activation

     def forward(self, x):
         return self.final_activation(self.network(x))




class SliceEnergy(nn.Module):
    def __init__(self, n_in: int, n_out: int, final_activation: nn.Module = nn.Identity()):
        super().__init__()
        self.layer_x = nn.Linear(n_in, n_in)
        self.activation1 = nn.Tanh()

        self.layer_pos = SinusoidalPositionEmbeddings(n_in)
        self.layer_time = nn.Linear(n_in, n_in)
        self.activation2 = nn.GELU()
        self.layer_time2 = nn.Linear(n_in, n_in)

        self.layer_fusion = nn.Linear(n_in, n_in)
        self.activation3 = nn.Tanh()
        self.layer_out = nn.Linear(n_in, n_out)
        self.activation4 = nn.Tanh()

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        x = self.activation1(self.layer_x(x))
        t_feat = self.layer_pos(time)
        t_feat = self.layer_time(t_feat)
        t_feat = self.activation2(t_feat)
        t_feat = self.layer_time2(t_feat)
        h = self.activation3(self.layer_fusion(x + t_feat))
        return self.activation4(self.layer_out(h))


class WavePDE(nn.Module):
    def __init__(
        self,
        num_support_sets: int,
        num_support_timesteps: int,
        support_vectors_dim: int,
        n_laplace_probes: int = 1,
        lambdas: dict = None,
    ):
        super().__init__()
        self.num_support_sets = num_support_sets
        self.num_support_timesteps = num_support_timesteps
        self.support_vectors_dim = support_vectors_dim
        self.n_laplace_probes = int(n_laplace_probes)

        self.c = nn.Parameter(torch.full((num_support_sets, 1), 1.))
        self.MLP_SET = nn.ModuleList(
            [SliceEnergy(n_in=support_vectors_dim, n_out=1, final_activation=nn.Identity()) for _ in range(num_support_sets)]
        )
        self.SEM_SET = nn.ModuleList(
            [SemanticPotential(n_in=support_vectors_dim, n_out=1, final_activation=nn.Identity()) for _ in range(num_support_sets)]
        )

        # default weights and epsilon (can be overridden by user-supplied dict)
        default_lambdas = dict(
            kin=1.0,
            sliceHJ=1.0,
            foot=1.0,
            unitspeed=1.0,
            div=1.0,
            BB=1e-2,
            tan=0.0,        # optional; set >0 to enable tangential-norm diagnostic penalty
            epsilon=0.0     # viscous slice HJ (0.0 => inviscid)
        )
        self.lambdas = default_lambdas if lambdas is None else {**default_lambdas, **lambdas}

    # ---------- Hutchinson Laplacian ----------
    def _laplacian_hutchinson_single(self, u_z: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Single-probe Laplacian using existing (u_z, z) graph."""
        v = torch.empty_like(z).bernoulli_(0.5).mul_(2).sub_(1)
        Hv = grad((u_z * v).sum(), z, create_graph=True, retain_graph=True)[0]
        lap = (Hv * v).sum(dim=1, keepdim=True)  # [B,1]
        return lap

    def _laplacian_hutchinson_multi(self, mlp: nn.Module, z: torch.Tensor, t: torch.Tensor, m: int) -> torch.Tensor:
        """
        Vectorized m-probe Laplacian: recompute u_z on (m×B) batch so that it depends on z_rep.
        """
        B, D = z.shape
        # Repeat inputs along a probe axis; make them leafs for grad
        z_rep = z.detach().unsqueeze(0).repeat(m, 1, 1).reshape(m * B, D).requires_grad_(True)
        t_rep = t.detach().unsqueeze(0).repeat(m, 1, 1).reshape(m * B, 1).requires_grad_(True)

        # Compute u_z on replicated batch
        u_rep = mlp(z_rep, t_rep)
        u_z_rep = grad(u_rep.sum(), z_rep, create_graph=True, retain_graph=True)[0]  # [mB, D]

        # One reverse pass to get all H v
        v_rep = torch.empty_like(z_rep).bernoulli_(0.5).mul_(2).sub_(1)
        Hv_rep = grad((u_z_rep * v_rep).sum(), z_rep, create_graph=True, retain_graph=True)[0]  # [mB, D]

        lap = (Hv_rep * v_rep).sum(dim=1).reshape(m, B).mean(dim=0, keepdim=True).T  # [B,1]
        return lap

    def _laplacian_hutchinson(self, mlp: nn.Module, z: torch.Tensor, t: torch.Tensor, u_z: torch.Tensor) -> torch.Tensor:
        m = max(1, self.n_laplace_probes)
        if m == 1:
            return self._laplacian_hutchinson_single(u_z, z)
        else:
            return self._laplacian_hutchinson_multi(mlp, z, t, m)

    # ---------- Divergence (Hutchinson) ----------
    def _divergence_hutchinson(self, v_tensor: torch.Tensor, z_leaf: torch.Tensor) -> torch.Tensor:
        """
        Estimate div v at z_leaf using Hutchinson's trick.
        v_tensor must be computed from z_leaf (i.e., shares graph).
        """
        eps = torch.empty_like(z_leaf).bernoulli_(0.5).mul_(2).sub_(1)
        Jv_eps = grad((v_tensor * eps).sum(), z_leaf, create_graph=True, retain_graph=True)[0]
        div_est = (Jv_eps * eps).sum(dim=1, keepdim=True)
        return div_est  # [B,1]

    # ---------- Forward-mode u_t and u_tt ----------
    def _u_t_and_u_tt_forward_mode(self, mlp: nn.Module, z: torch.Tensor, t: torch.Tensor):
        """
        Pure forward-mode for time derivs; z treated as constant (detached).
        Returns (u_t, u_tt), both [B,1].
        """
        z_det = z.detach()

        def f_t(t_in):
            return mlp(z_det, t_in)

        ones = torch.ones_like(t)
        _, u_t = jvp_fwd(f_t, (t,), (ones,))

        def ut_func(t_in):
            # re-evaluate JVP to keep graph correct for jvp over it
            return jvp_fwd(f_t, (t_in,), (torch.ones_like(t_in),))[1]

        _, u_tt = jvp_fwd(ut_func, (t,), (ones,))
        return u_t, u_tt

    # ---------- One PDE step ----------
    def _per_step(self, psi_mlp: nn.Module, f_mlp: nn.Module, z: torch.Tensor, direction: int = +1):
        """
        One PDE-informed step from z -> z_next and all loss components for this step.

        Gradients:
          - Time inputs to ψ use detached t_stop = f(z).detach() and (t_stop + Δt) to avoid pushing gradients to f via time.
          - f receives gradients via: f(x_hat) in footpoint loss and ∇f in unit-speed + velocity projector.
          - ψ receives gradients via its spatial gradients and values; NOT via time (detached).
        """
        B, D = z.shape
        device, dtype = z.device, z.dtype
        # leaf variables for this step (we do not need grads wrt input z across steps)
        z_req = z.detach().requires_grad_(True)

        # step size Δt (constant 1 * direction)
        dt = torch.full((B, 1), float(direction), device=device, dtype=dtype)

        # evaluate time labels
        t_fwd = f_mlp(z_req)            # gradients flow to f
        t_stop = t_fwd.detach()         # detached time label for ψ inputs

        # ψ at current and next time (with detached times)
        psi_now  = psi_mlp(z_req, t_stop)              # [B,1]
        psi_next = psi_mlp(z_req, t_stop + dt)         # [B,1]

        # ∇_x ψ at t and t+Δt
        gpsi_now  = grad(psi_now.sum(),  z_req, create_graph=True, retain_graph=True)[0]   # [B,D]
        gpsi_next = grad(psi_next.sum(), z_req, create_graph=True, retain_graph=True)[0]   # [B,D]

        # footpoint (OT/slice projection)
        x_hat = z_req + gpsi_next  # [B,D]

        # ∇ f at footpoint, projector Pf, transverse gradient Xf
        f_hat     = f_mlp(x_hat)  # [B,1] (used in foot loss; keeps grads to f and ψ)
        gradf_hat = grad(f_hat.sum(), x_hat, create_graph=True, retain_graph=True)[0]  # [B,D]
        norm2_f   = (gradf_hat.pow(2).sum(dim=1, keepdim=True) + 1e-8)                 # [B,1]
        Xf_hat    = gradf_hat / norm2_f                                                # [B,D]

        I = torch.eye(D, device=device, dtype=dtype).unsqueeze(0).expand(B, D, D)
        outer = gradf_hat.unsqueeze(2) @ gradf_hat.unsqueeze(1)                        # [B,D,D]
        Pf_hat = I - outer / norm2_f.unsqueeze(2)                                      # [B,D,D]

        # Velocity (minimal two-potential): v = Xf(x_hat) + Pf(x_hat) ∇ψ(x, t)
        tang = (Pf_hat @ gpsi_now.unsqueeze(2)).squeeze(2)                              # [B,D]
        v    = Xf_hat + tang                                                            # [B,D]

        # Advance one step (explicit Euler for rollout)
        z_next = z + dt * v  # this uses z (not z_req) to keep the outer graph clean

        # ---------- Losses (per step) ----------
        # Kinematic coupling: ∂_t ψ - ||∇ψ|| / ||∇f(x_hat)||
        kin_res = (psi_next - psi_now) / dt.abs() - gpsi_next.norm(dim=1, keepdim=True) / (norm2_f.sqrt())
        L_kin = (kin_res.pow(2)).mean()

        # Slice HJ (viscous): ψ + 0.5||∇ψ||^2 - 0.5 ε^2 Δψ
        lap_now = self._laplacian_hutchinson(psi_mlp, z_req, t_stop, gpsi_now)          # [B,1]
        eps2 = float(self.lambdas.get("epsilon", 0.0)) ** 2
        sliceHJ_res = psi_now + 0.5 * (gpsi_now.pow(2).sum(dim=1, keepdim=True)) - 0.5 * eps2 * lap_now
        L_sliceHJ = (sliceHJ_res.pow(2)).mean()

        # Footpoint consistency: f(x_hat) ≈ t_fwd + Δt
        L_foot = ((f_hat - (t_fwd + dt)).pow(2)).mean()

        # Unit f-speed: ∇f(x) · v(x,t) = 1
        f_z     = f_mlp(z_req)
        gradf_z = grad(f_z.sum(), z_req, create_graph=True, retain_graph=True)[0]
        unitspeed_res = (gradf_z * v).sum(dim=1, keepdim=True) - 1.0
        L_unitspeed = (unitspeed_res.pow(2)).mean()

        # Incompressibility wrt prior N(0,I): div v + s_prior·v = 0, with s_prior(x) = -x
        div_v = self._divergence_hutchinson(v, z_req)                # [B,1]
        s_prior = -z_req                                             # [B,D]
        divprior_res = div_v + (s_prior * v).sum(dim=1, keepdim=True)
        L_div_prior = (divprior_res.pow(2)).mean()

        # BB action regularizer
        L_BB = (v.pow(2).sum(dim=1, keepdim=True)).mean()

        # Optional: tangential-norm diagnostic regularizer (detached to avoid training ψ or f)
        if self.lambdas.get("tan", 0.0) > 0.0:
            gpsi_now_stop = gpsi_now.detach()
            Pf_hat_stop   = Pf_hat.detach()
            L_tan = (( (Pf_hat_stop @ gpsi_now_stop.unsqueeze(2)).squeeze(2) ).pow(2).sum(dim=1, keepdim=True)).mean()
        else:
            L_tan = torch.zeros((), device=device, dtype=dtype)

        # package
        losses = dict(
            kin=L_kin,
            sliceHJ=L_sliceHJ,
            foot=L_foot,
            unitspeed=L_unitspeed,
            div=L_div_prior,
            BB=L_BB,
            tan=L_tan
        )
        extras = dict(
            z_next=z_next,
            v=v,
            x_hat=x_hat,
            psi_now=psi_now,
            psi_next=psi_next,
            t_fwd=t_fwd
        )
        return losses, extras

    # ---------- Public API ----------
    def forward(self, index: int, z: torch.Tensor, t: torch.Tensor, direction: int = +1):
        """
        Unroll num_support_timesteps PDE-informed steps and accumulate losses.
        Returns:
          energy  : average BB action over steps (scalar tensor)
          latent1 : a reference latent at step i_target (with grads)
          latent2 : its next-step latent (detached, no grads)
          loss    : weighted total loss
        """
        psi_k = self.MLP_SET[index]
        f_k   = self.SEM_SET[index]
        # c_k is available if you want to make Δt learnable; we keep Δt=1 here
        # c_k = self.c[index:index+1]

        B, D = z.shape
        device, dtype = z.device, z.dtype

        T = max(1, int(self.num_support_timesteps))
        # choose a pair index from provided t (clamped into [0, T-1])
        i_target = int(torch.clamp(t[0], 0, T - 1).item()) if t.numel() > 0 else 0

        # accumulators
        acc = {k: 0.0 for k in ["kin", "sliceHJ", "foot", "unitspeed", "div", "BB", "tan"]}
        z_curr = z
        latent1, latent2 = None, None

        for i in (range(T) if direction == +1 else reversed(range(T))):
            losses_i, ex_i = self._per_step(psi_k, f_k, z_curr, direction=direction)
            z_next = ex_i["z_next"]

            # accumulate (keep graphs!)
            for name, val in losses_i.items():
                acc[name] = acc[name] + val

            # pick the classifier pair at i_target
            if latent1 is None and i == i_target:
                latent1 = z_curr
                latent2 = z_next.detach()

            # move on
            z_curr = z_next

        # averages
        for name in acc:
            acc[name] = acc[name] / float(T)

        # total weighted loss
        L_total = (
            self.lambdas["kin"]      * acc["kin"]
          + self.lambdas["sliceHJ"]  * acc["sliceHJ"]
          + self.lambdas["foot"]     * acc["foot"]
          + self.lambdas["unitspeed"]* acc["unitspeed"]
          + self.lambdas["div"]      * acc["div"]
          + self.lambdas["BB"]       * acc["BB"]
          + self.lambdas.get("tan",0.0) * acc["tan"]
        )

        # default latent pair if not set
        if latent1 is None:
            latent1 = z
            latent2 = z_curr.detach()

        energy = acc["BB"]  # BB action proxy (lower = shorter path)
        return energy, latent1, latent2, L_total

    @torch.enable_grad()
    def inference(self, index: int, z: torch.Tensor, t: torch.Tensor, direction: int = +1):
        """
        Rollout latents without computing losses. Returns the full latent trajectory list.
        """
        psi_k = self.MLP_SET[index]
        f_k   = self.SEM_SET[index]

        T = max(1, int(self.num_support_timesteps))
        traj = [z]
        z_curr = z
        for _ in (range(T) if direction == +1 else reversed(range(T))):
            # single step (reuse _per_step but ignore losses and keep grads for consistency)
            losses_i, ex_i = self._per_step(psi_k, f_k, z_curr, direction=direction)
            z_curr = ex_i["z_next"]
            traj.append(z_curr)
        return traj