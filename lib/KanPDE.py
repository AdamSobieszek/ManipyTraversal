import math
from typing import Dict, Tuple, Optional, List

import torch
from torch import nn
from torch.autograd import grad
from torch.func import jvp as jvp_fwd

from lib.pde_ops import PDEState
from lib.pde_losses import build_losses 

import torch
import torch.nn as nn
import torch.nn.functional as F

# Re-using your robust StackedLinear from the context
class StackedLinear(nn.Module):
    """
    Per-k linear layers evaluated in parallel.
    Weight: [K, out, in], Bias: [K, out]
    """
    def __init__(self, K: int, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.K = int(K)
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.weight = nn.Parameter(torch.empty(self.K, self.out_features, self.in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.K, self.out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        for k in range(self.K):
            nn.init.kaiming_uniform_(self.weight.data[k], a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias.data, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, K, in] -> y: [B, K, out]
        y = torch.einsum("bki,koi->bko", x, self.weight)
        if self.bias is not None:
            y = y + self.bias.unsqueeze(0)
        return y

# ================================================================
# New Component: Neural Vielbein Generator
# ================================================================
class NeuralVielbeinFrame(nn.Module):
    """
    The 'Hive Mind' Backbone.
    Maps position x -> Orthonormal Frame Matrix E(x) [B, D, K].
    
    This separates the Geometry (directions) from the Topology (potential values).
    """
    def __init__(self, in_dim: int, K: int, hidden_dim: int = 128):
        super().__init__()
        self.in_dim = in_dim
        self.K = K
        
        # Standard MLP backbone
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            # Output raw vectors which we will orthonormalize
            nn.Linear(hidden_dim, in_dim * K) 
        )
    def gram_schmidt(self, vectors: torch.Tensor) -> torch.Tensor:
            """
            vectors: [B, D, K]
            Returns orthonormalized vectors [B, D, K] using Classical GS.
            Safe for Autograd (no in-place operations).
            """
            B, D, K = vectors.shape
            
            # We will store the orthonormalized columns in a list
            q_vectors = []
            
            for k in range(K):
                # Take k-th raw vector
                v = vectors[:, :, k]  # [B, D]
                
                # Subtract projection onto previous vectors
                if k > 0:
                    # To avoid indexing a tensor that we are building (which causes issues),
                    # we stack the *already computed* vectors from the list.
                    # current_basis: [B, D, k]
                    previous_basis = torch.stack(q_vectors, dim=2)
                    
                    # Coefficients: (v . q_j)
                    # [B, 1, D] @ [B, D, k] -> [B, 1, k]
                    coeffs = torch.bmm(v.unsqueeze(1), previous_basis)
                    
                    # Reconstruct projections: 
                    # [B, 1, k] * [B, D, k] (broadcasting needs permutation or einsum)
                    # einsum: b 1 k, b d k -> b d
                    projections = torch.einsum('bik,bdk->bd', coeffs, previous_basis)
                    
                    v = v - projections
                
                # Normalize
                norm = v.norm(p=2, dim=1, keepdim=True).clamp_min(1e-8)
                v_normalized = v / norm
                
                # Append to list
                q_vectors.append(v_normalized)
                
            # Stack along the last dimension to reform [B, D, K]
            Q = torch.stack(q_vectors, dim=2)
                
            return Q

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, D] (or [B, K, D], we will handle flattening)
        if x.dim() == 3:
            # If input is [B, K, D], we assume the frame depends on the position 
            # defined by the specific support set index, or we flatten.
            # Usually x is the same for all K if it's the state, 
            # but in Flow-on-Flows x might be [B,K,D].
            # We process them independently.
            B, K_in, D = x.shape
            x_flat = x.view(-1, D)
            raw = self.net(x_flat) # [B*K, D*K_out]
            raw = raw.view(B*K_in, D, self.K)
            ortho = self.gram_schmidt(raw) # [B*K, D, K]
            return ortho.view(B, K_in, D, self.K)
        else:
            # x: [B, D]
            B, D = x.shape
            raw = self.net(x) # [B, D*K]
            raw = raw.view(B, D, self.K)
            return self.gram_schmidt(raw) # [B, D, K]

# ================================================================
# Replacement Class: Neural Vielbein Potential
# ================================================================
class StackedVielbeinPotential(nn.Module):
    """
    REPLACES: StackedSemanticPotential
    
    Instead of N independent MLPs, this learns:
    1. A shared Orthonormal Frame E(x) (The Vielbein)
    2. N independent 1D Activation Profiles (The KAN edges)
    
    F^k(x) = Profile_k( <x, E_k(x)> )
    """
    def __init__(
        self,
        K: int,
        n_in: int,
        n_out: int = 1, # kept for API compatibility, usually 1
        n_hidden: int = 128,
        activation: nn.Module = None, # Deprecated, internal KAN uses SiLU
        final_activation: nn.Module = nn.Identity(),
    ):
        super().__init__()
        self.K = K
        self.n_in = n_in
        self.n_out = n_out
        
        # 1. The Geometry Learner (Shared)
        self.frame_net = NeuralVielbeinFrame(n_in, K, n_hidden)
        
        # 2. The Topology Learner (Per-K)
        self.profile = LearnableProfile(K, hidden_dim=32)
        
        # Keep API compatibility components
        self.final_activation = final_activation
        self.register_buffer("running_mean", torch.zeros(self.K, self.n_out))
        self.update_batchnorm = True
        
        # Linear residual for stability
        self.c = nn.Parameter(torch.full((self.K, 1), 1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, K, D] (Support set positions)
        Returns: [B, K, 1] (Potentials)
        """
        B, K_in, D = x.shape
        
        # 1. Compute the Local Frame E(x)
        # E shape: [B, K_in, D, K_basis]
        E = self.frame_net(x) 
        
        # 2. Extract the k-th basis vector for the k-th input
        # We want to match the K_in index with the K_basis index.
        # torch.diagonal extracts the diagonal along the specified dims 
        # and places it at the END.
        # Input: [B, K, D, K] -> Diagonal(dim1=1, dim2=3) -> [B, D, K]
        E_diagonal = torch.diagonal(E, dim1=1, dim2=3)
        
        # Permute back to [B, K, D] to match x
        E_k = E_diagonal.permute(0, 2, 1)
        
        # 3. Compute Coordinate along the manifold
        # coord = <x, e_k>
        # [B, K, D] * [B, K, D] -> [B, K, 1]
        coords = (x * E_k).sum(dim=-1, keepdim=True)
        
        # 4. Apply 1D Profile (The KAN step)
        out_energy = self.profile(coords)
        
        # 5. API Compatibility (EMA, Scaling)
        out = out_energy * self.c
        
        if self.training and self.update_batchnorm:
            with torch.no_grad():
                batch_mean = out.mean(dim=0)
                self.running_mean.lerp_(batch_mean, 0.1)

        out_centered = out - self.running_mean
        
        return self.final_activation(out_centered)
        # 1. Compute the Local Frame E(x)
        # E shape: [B, K_in, D, K_basis]
        # Note: We compute a frame *at* each support point. 
        
        # 2. Project x onto the frame to get "Invariant Coordinates"
        # We need the coordinate corresponding to k-th potential.
        # Ideally, potential k uses the k-th basis vector.
        # E[..., k] is the vector for potential k.
        
        # Extract the specific basis vector for each k in the batch
        # We want E_{b,k} corresponding to index k.
        # Since x is [B, K, D], and we want F^k(x_{b,k}), 
        # we need the k-th column of the frame computed at x_{b,k}.
        
        # E is [B, K, D, K]. Diagonal extraction:
        # We want basis vector k for input k.
        # `basis_vectors = torch.diagonal(E, dim1=1, dim2=3)` # This might be tricky dimensionally
        # Let's be explicit:
        # We have K parallel inputs. We want K parallel outputs.
        # For input k, we use basis vector k.
        
        # E: [B, K_input, D, K_basis]
        # We want to match K_input index with K_basis index.
        # Gather is complex here. Let's use masking or einsum.
        
        # Slice: E[:, k, :, k] -> [B, D] vector for the k-th slot
        # E_k: [B, K, D] (The k-th basis vector for the k-th input point)
        
        # 3. Compute Coordinate along the manifold
        # coord = <x, e_k>
        # [B, K, D] * [B, K, D] -> [B, K, 1]
        
        # 4. Apply 1D Profile (The KAN step)
        # [B, K, 1] -> [B, K, 1]
        
        # 5. API Compatibility (EMA, Scaling)

import math
import torch
from torch import nn


# ================================================================
# New Component: Learnable KAN-like Activation Profile
# ================================================================
class LearnableProfile(nn.Module):
    """
    Represents the scalar function psi_k(u).
    Input: [B, K, 1], Output: [B, K, 1]
    """
    def __init__(self, K: int, hidden_dim: int = 32):
        super().__init__()
        # Using 1x1 convolutions (groups=K) or StackedLinear is equivalent.
        # StackedLinear is explicit and easy to read.
        self.l1 = StackedLinear(K, 1, hidden_dim)
        self.l2 = StackedLinear(K, hidden_dim, hidden_dim)
        self.l2 = StackedLinear(K, hidden_dim, hidden_dim)
        self.l3 = StackedLinear(K, hidden_dim, 1)
        self.act = nn.SiLU() 

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        h = self.act(self.l1(u))
        h = self.act(self.l2(h))
        return self.l3(h)

class StackedLinear(nn.Module):
    """ Standard K-independent Linear Layers """
    def __init__(self, K: int, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.K = int(K)
        self.weight = nn.Parameter(torch.empty(self.K, out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.K, out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        # Kaiming init
        for k in range(self.K):
            nn.init.kaiming_uniform_(self.weight.data[k], a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.weight.shape[2]
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias.data, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, K, in]
        # y: [B, K, out]
        return torch.einsum("bki,koi->bko", x, self.weight) + (self.bias.unsqueeze(0) if self.bias is not None else 0)

class StackedKanPotential(nn.Module):
    """
    High-Performance Amortized Potential.
    
    Structure:
      1. Project x onto K learnable directions w_k (The "Basis").
      2. Apply learnable non-linear profile psi_k (The "KAN Edge").
      
    F^k(x) = psi_k( <x, w_k> )
    """
    def __init__(
        self,
        K: int,
        n_in: int,
        n_out: int = 1,
        n_hidden: int = 32, # Hidden dim for the profile MLP
        final_activation: nn.Module = nn.Identity(),
        orthogonal_init: bool = True,
        **kwargs # Ignore unused args from previous versions
    ):
        super().__init__()
        self.K = K
        self.n_in = n_in
        self.n_out = n_out # Usually 1

        # 1. The Basis (The Directions)
        # We use a StackedLinear with out_features=1 to represent dot product <x, w_k>
        # Input: [B, K, D], Output: [B, K, 1]
        self.projector = StackedLinear(K, n_in, 1, bias=False)
        
        # 2. The Profile (The Curvature)
        # Input: [B, K, 1], Output: [B, K, 1]
        self.profile = LearnableProfile(K, hidden_dim=n_hidden)

        # 3. API Compatibility
        self.final_activation = final_activation
        self.register_buffer("running_mean", torch.zeros(self.K, self.n_out))
        self.update_batchnorm = True
        self.c = nn.Parameter(torch.full((self.K, 1), 1.0))
        
        if orthogonal_init:
            self._init_orthogonal_basis()

    def _init_orthogonal_basis(self):
        """
        Initialize the K vectors to be mutually orthogonal to start with.
        This provides a 'good start' without enforcing it during training.
        """
        if self.n_in >= self.K:
            # If dim >= K, we can find K orthogonal vectors
            w = torch.empty(self.n_in, self.K)
            if torch.cuda.is_available():
                nn.init.orthogonal_(w)
            else:
                device = w.device
                w = w.to(torch.device('cpu'))
                nn.init.orthogonal_(w)
                w = w.to(device)

            # Assign to projector weights: [K, 1, D]
            with torch.no_grad():
                self.projector.weight.data.copy_(w.t().unsqueeze(1))
        else:
            # If dim < K, standard init is fine
            pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, K, D]
        returns: [B, K, 1]
        """
        # 1. Projection (Linear Basis)
        # u = <x, w_k>
        u = self.projector(x) # [B, K, 1]
        
        # 2. Non-linear Profile (KAN)
        # energy = psi_k(u)
        out_energy = self.profile(u)
        
        # 3. Scaling & Centering
        out = out_energy * self.c
        
        if self.training and self.update_batchnorm:
            with torch.no_grad():
                batch_mean = out.mean(dim=0)
                self.running_mean.lerp_(batch_mean, 0.1)

        out_centered = out - self.running_mean
        
        return self.final_activation(out_centered)
    
    def ortho_regularization(self):
        """
        Optional: Call this in your loss function if you see mode collapse.
        Returns || W^T W - I ||^2
        """
        # W: [K, D] (squeeze the output dim)
        W = self.projector.weight.squeeze(1) 
        
        # Normalize rows to check angular orthogonality only
        W_norm = W / W.norm(dim=1, keepdim=True).clamp_min(1e-8)
        
        # Gram matrix
        gram = W_norm @ W_norm.T # [K, K]
        
        # Compare to Identity
        I = torch.eye(self.K, device=W.device)
        return (gram - I).pow(2).mean()

class SkipSliceEnergy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        return (torch.zeros_like(x)*x).sum(dim=-1, keepdim=True)

class KanPDE(nn.Module):
    """
    K-parallel KanPDE powered by PDEState and modular PDE losses.
    Now equipped with Neural Vielbein Frames for intrinsic orthogonality.
    """
    def __init__(
        self,
        num_support_sets: int,
        num_support_timesteps: int,
        support_vectors_dim: int,
        lambdas: Dict[str, float] = {}, 
        n_laplace_probes: int = 1,
        apply_bn_on_psi_output: bool = False,
        # Model complexity
        n_hidden: int = 128,  # New: Controls FrameNet and Profile complexity
        # PDEState config
        time_ad: str = "reverse",
        detach_between_steps: bool = False,
        eps_norm2: float = 1e-5,
        divergence_probes: int = 1,
        rng: str = "rademacher",
        seed: Optional[int] = None,
        prior_score: Optional[callable] = None,
        only_potential: bool = True, # Defaults to true as requested
    ):
        super().__init__()
        self.num_support_sets = int(num_support_sets)
        self.num_support_timesteps = int(num_support_timesteps)
        self.support_vectors_dim = int(support_vectors_dim)

        # Learnable per-k scale 
        self.c = nn.Parameter(torch.full((self.num_support_sets, 1), 1.0))

        # Time-independent assumption
        self.PSI = SkipSliceEnergy()

        # ============================================================
        # REPLACEMENT: Neural Vielbein Potential
        # ============================================================
        # We replace StackedSemanticPotential with the Vielbein version.
        # Note: We drop 'activation' as the KAN profile handles it internally.
        self.F = StackedKanPotential(
            K=self.num_support_sets,
            n_in=self.support_vectors_dim,
            n_hidden=n_hidden, 
            final_activation=nn.Identity(),
        )
        
        # Optional: Geometric Initialization
        # Ensures the initial frame is close to Identity (or uniform random)
        # rather than collapsed, helping the Gram-Schmidt process early on.
        self._init_geometric_weights()

        # ============================================================

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

        # Build modular loss list
        epsilon = float(lambdas.get("epsilon", 0.0))
        self.losses, self._needs_next = build_losses(
            lambdas,
            F=self.F,
            epsilon=epsilon,
            prior_score=prior_score,
        )

        # for telemetry
        self._acc: Dict[str, torch.Tensor] = {}

    def _init_geometric_weights(self):
        """
        Helper to initialize the FrameNet specifically.
        We want the 'raw' vectors entering Gram-Schmidt to be linearly independent.
        """
        # Access the backbone output layer
        # (Assuming implementation: self.F.frame_net.net[-1])
        if hasattr(self.F, 'frame_net'):
            final_layer = self.F.frame_net.net[-1]
            # Orthogonal initialization helps Gram-Schmidt not explode gradients early
            if torch.cuda.is_available():
                nn.init.orthogonal_(final_layer.weight)
            else:
                print("WARNING: Orthogonal initialization not available on CPU. Using random initialization.")
            if final_layer.bias is not None:
                nn.init.zeros_(final_layer.bias)

    # ---- one step ----
    def _per_step(self, z_bkd: torch.Tensor, dt: torch.Tensor=1.0, direction: int = +1):
        st = PDEState(
            f=self.F,
            psi=self.PSI,
            z=z_bkd,
            need_next=self._needs_next,
            dt_value=dt,  # use ±1 step; wire self.c here if desired
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
            latent_noise = latent_noise * (step_delta_norms.clamp_min(step_delta_norms.mean().item()/3) / 5.0)
        x_next_noisy = x_next + latent_noise

        return st, x_next_noisy, L_sum, st.dt()

    # ---- unrolled training ----
    def forward(self, z: torch.Tensor, t_index: torch.Tensor, dt: torch.Tensor, direction: int = +1, w_avg: torch.Tensor = None):
        """
        Returns:
          potential_preds: [B,K,1] (detached)
          latent1_bk, latent2_bk: [B,K,D] (pair captured at t_index)
          L_total_mean: scalar
        """
        B, D = z.shape
        K = self.num_support_sets
        T = max(1, int(self.num_support_timesteps)-1)

        if t_index.ndim == 1:
            t_index = t_index.unsqueeze(-1)
        i_target = torch.clamp(t_index, 0, T - 1).long().squeeze(-1)
        
        # expand once to K stacks
        if w_avg is not None:
            z = z - w_avg.reshape(1,D)
        z_curr = z.unsqueeze(1).expand(B, K, D).contiguous()
        potential_preds = []
        latent1_bk = None
        latent2_bk = None
        last_st: Optional[PDEState] = None

        L_accum = None  # accumulate per-[B,K,1]

        step_iter = range(T)# if  else reversed(range(T))
        self.F.update_batchnorm = True
        for i in step_iter:
            st, x_next, L_step, dt = self._per_step(z_curr, dt=dt, direction=direction)


            potential_preds.append(st.f().detach())
            # accumulate loss per step
            L_accum = (L_step if L_accum is None else L_accum + L_step)

            # capture (latent1, latent2) at the requested index
            mask_b = (i_target == i).view(B, 1, 1)
            if latent1_bk is None:
                latent1_bk = torch.where(mask_b, z_curr, torch.zeros_like(z_curr))
                latent2_bk = torch.where(mask_b, x_next, torch.zeros_like(x_next))
            else:
                latent1_bk = torch.where(mask_b, z_curr, latent1_bk)
                latent2_bk = torch.where(mask_b, x_next, latent2_bk)

            # advance
            z_curr = x_next
            last_st = st
            # dont move the batchnorm at subsequent steps
            self.F.update_batchnorm = False


        # average over steps
        L_total_per_bk = L_accum / float(T) if L_accum is not None else last_st.zeros()
        L_total_mean = L_total_per_bk.mean()

        # for predicting the attribute
        potential_preds.append(last_st.f("next").detach())
        potential_preds = torch.cat(potential_preds, dim=-1)

        self._acc = {
            "xf_now": last_st.Xf() if last_st is not None else torch.zeros(B, K, D, device=z.device, dtype=z.dtype),
            "L_mean": L_total_mean.detach(),
            **last_st.state["losses"],
        }

        if w_avg is not None:
            latent1_bk = latent1_bk + w_avg.reshape(1,1,D)
            latent2_bk = latent2_bk + w_avg.reshape(1,1,D)
        return potential_preds, latent1_bk, latent2_bk, L_total_mean, last_st.delta_y().detach()

    def get_losses(self) -> Dict[str, torch.Tensor]:
        return self._acc

    @torch.enable_grad()
    def inference(self, z: torch.Tensor, t_index: torch.Tensor = None, dt: torch.Tensor = 1.0, direction: int = +1) -> List[torch.Tensor]:
        if len(z.shape) == 3:
            B, K, D = z.shape
            z_curr = z
        else:
            B, D = z.shape
            K = self.num_support_sets
            z_curr = z.unsqueeze(1).expand(B, K, D).contiguous()


        st, x_next, L_step, dt = self._per_step(z_curr, dt=dt, direction=direction)
        return z_curr, x_next-z_curr
