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
        self.PSI_SET = nn.ModuleList(
            [SliceEnergy(n_in=support_vectors_dim, n_out=1, final_activation=nn.Identity()) for _ in range(num_support_sets)]
        )
        self.F_POT_SET = nn.ModuleList(
            [SemanticPotential(n_in=support_vectors_dim, n_out=1, final_activation=nn.Identity()) for _ in range(num_support_sets)]
        )

        self.lambdas = lambdas

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
    def _per_step(self, mlp: nn.Module, z: torch.Tensor, t: torch.Tensor, c_k: torch.Tensor, direction: int = +1):
        z_req = z.detach().requires_grad_(True)
        t_req = t.detach().requires_grad_(True)
        # finish this function

    # ---------- Public API ----------
    def forward(self, index: int, z: torch.Tensor, t: torch.Tensor, direction: int = +1):
        mlp_k = self.PSI_SET[index]
        sem_k = self.F_POT_SET[index]
        c_k = self.c[index:index+1]  # [1,1]
        B, D = z.shape
        device, dtype = z.device, z.dtype
        half_range = self.num_support_timesteps // 2
        target_i = int(t[0].item())


        z_curr = z
        for i in (range(half_range) if direction == +1 else range(0, -half_range, -1)):
            t_i = torch.full((B, 1), float(i), device=device, dtype=dtype, requires_grad=True)
            # steps
            if i == target_i:
                latent1 = z_curr
                latent2 = (z_next).detach()
                # finish this function

                break

            z_curr = z_next

        # finish this function

        loss = #
        return energy, latent1, latent2, loss

    @torch.enable_grad()
    def inference(self, index: int, z: torch.Tensor, t: torch.Tensor, direction: int = +1):
        # finish this function