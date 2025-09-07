# WavePDE.py (fix Hutchinson grad graph: reuse for m=1, vectorize with recompute for m>1)
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


class MLP(nn.Module):
    def __init__(self, n_in: int, n_out: int, final_activation: nn.Module = nn.Tanh()):
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
        generator_projection: nn.Module | None = None,
        lambda_pde: float = 1.0,
        lambda_jvp: float = 0.0,
        lambda_ic: float = 0.0,
        final_activation: nn.Module = nn.Identity(),
    ):
        super().__init__()
        self.num_support_sets = num_support_sets
        self.num_support_timesteps = num_support_timesteps
        self.support_vectors_dim = support_vectors_dim
        self.n_laplace_probes = int(n_laplace_probes)

        self.c = nn.Parameter(torch.full((num_support_sets, 1), 1.))
        self.MLP_SET = nn.ModuleList(
            [MLP(n_in=support_vectors_dim, n_out=1, final_activation=final_activation) for _ in range(num_support_sets)]
        )
        self.proj = generator_projection if generator_projection is not None else nn.Identity()

        self.lambda_pde = lambda_pde
        self.lambda_jvp = lambda_jvp
        self.lambda_ic = lambda_ic

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

        u = mlp(z_req, t_req)                                 # [B,1]
        u_z = grad(u.sum(), z_req, create_graph=True)[0]      # [B,D]
        u_z = self.oems_parametrization(u_z)


        u_t, u_tt = self._u_t_and_u_tt_forward_mode(mlp, z_req, t_req)  # [B,1] each
        lap = self._laplacian_hutchinson(mlp, z_req, t_req, u_z)        # [B,1]

        pde_res = u_tt - (c_k ** 2) * lap                     # [B,1]
        z_next = (z_req + direction * u_z).detach()
        return u, direction*u_z, pde_res, z_next

    # ---------- Projected JVP ----------
    def _projected_jvp(self, generator, z: torch.Tensor, v: torch.Tensor):
        def Gproj(z_in):
            return self.proj(generator(z_in))
        _, jvp_val = jvp_rev(Gproj, (z,), (v,), create_graph=True)
        return jvp_val

    def oems_parametrization(self, grad):
        # Normalize grad to unit norm
        norm = grad.norm(dim=-1, keepdim=True) + 1e-8

        u = grad / norm
        # Softly encourage the norm to stay in [min_norm, max_norm] via a smooth function
        min_norm, max_norm = 1e-4, 10
        # Use a smooth sigmoid-based scaling to softly push norm into [min_norm, max_norm]
        # The scale is close to 1 in the interval, smoothly shrinks/grows outside
        scale = 1/norm
        # scale = (
        #     torch.sigmoid(10 * (norm - min_norm)) * torch.sigmoid(10 * (max_norm - norm))
        #     + (norm < min_norm).float() * (norm / min_norm)
        #     + (norm > max_norm).float() * (max_norm / norm)
        # )
        u = u * scale
        return u
    
    # ---------- Public API ----------
    def forward(self, index: int, z: torch.Tensor, t: torch.Tensor, generator, direction: int = +1):
        mlp_k = self.MLP_SET[index]
        c_k = self.c[index:index+1]  # [1,1]
        B, D = z.shape
        device, dtype = z.device, z.dtype
        half_range = self.num_support_timesteps // 2
        target_i = int(t[0].item())

        loss_pde_acc = 0.0
        mse_ic = None
        energy = None
        latent1 = None
        latent2 = None
        mse_jvp = None

        z_curr = z
        for i in (range(half_range) if direction == +1 else range(0, -half_range, -1)):
            t_i = torch.full((B, 1), float(i), device=device, dtype=dtype, requires_grad=True)

            u_i, u_z_i, pde_res_i, z_next = self._per_step(mlp_k, z_curr, t_i, c_k, direction)
            loss_pde_acc = loss_pde_acc + (pde_res_i.pow(2).mean())
            # IC at i=0
            if i == 0 and self.lambda_ic > 0:
                mse_ic = -(u_z_i.pow(2).sum(dim=1)).mean()
            
            # u_z_i = self.oems_parametrization(u_z_i)

            
            if i == target_i:
                latent1 = z_curr

                z1 = latent1.detach().requires_grad_(True)
                energy = u_i
                latent2 = (z_next).detach()
                if self.lambda_jvp > 0:
                    jvp_val = self._projected_jvp(generator, z1, u_z_i)
                    mse_jvp = (jvp_val.pow(2).mean())

                break

            z_curr = z_next

        if mse_ic is None:
            mse_ic = torch.tensor(0.0, device=device, dtype=dtype, requires_grad=True)
        if mse_jvp is None:
            mse_jvp = torch.tensor(0.0, device=device, dtype=dtype, requires_grad=True)
        if energy is None:
            raise ValueError("energy is None")

            
        loss = self.lambda_ic * mse_ic + self.lambda_pde * (loss_pde_acc / max(1, i)) - self.lambda_jvp * mse_jvp
        return energy, latent1, latent2, loss

    @torch.enable_grad()
    def inference(self, index: int, z: torch.Tensor, t: torch.Tensor, generator=None, direction: int = +1):
        mlp_k = self.MLP_SET[index]
        B = z.size(0)
        t = t if t.dim() == 2 else t.view(B, 1)
        z_req = z.detach().requires_grad_(True)
        t_req = t.detach().requires_grad_(True)
        u = mlp_k(z_req, t_req)
        u_z = grad(u.sum(), z_req, create_graph=False)[0]
        u_z = self.oems_parametrization(u_z)
        return u, direction * u_z