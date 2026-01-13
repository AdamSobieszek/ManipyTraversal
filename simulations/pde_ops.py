# pde_ops_nopsi.py
# ---------------------------------------------------------------------
# Minimal, self-contained PDE ops for *single-potential* PINNs (no psi).
# Single class PDEState with lazy, cached primitives:
#   x, dt, f, f_grad, f_laplace, Xf, v, v_div, x_next, delta_y
#
# Timepoint handled via when ∈ {"now","next"} (default "now"}.
# ---------------------------------------------------------------------

from __future__ import annotations
from typing import Optional, Tuple, Dict, Any, Literal

import torch
from torch import nn
from torch.autograd import grad

When = Literal["now", "next"]

# --- RNG helpers (generator-safe across PyTorch versions) ---

def _make_gen(seed: Optional[int], device: torch.device) -> Optional[torch.Generator]:
    if seed is None:
        return None
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    return g

def _randn_like(x: torch.Tensor, gen: Optional[torch.Generator]) -> torch.Tensor:
    if gen is None:
        return torch.randn_like(x)
    return torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=gen)

def _rand_rademacher_like(x: torch.Tensor, gen: Optional[torch.Generator]) -> torch.Tensor:
    if gen is None:
        r = torch.randint(0, 2, x.shape, device=x.device)
    else:
        r = torch.randint(0, 2, x.shape, device=x.device, generator=gen)
    return r.to(dtype=x.dtype).mul_(2).sub_(1)

def _rand_vec_like(x: torch.Tensor, mode: str, gen: Optional[torch.Generator]) -> torch.Tensor:
    if mode == "gaussian":
        return _randn_like(x, gen)
    if mode == "rademacher":
        return _rand_rademacher_like(x, gen)
    raise ValueError(f"rng must be 'rademacher' or 'gaussian', got {mode!r}")

# ----------------------- helpers -----------------------

def _ensure_bkd(z: torch.Tensor) -> Tuple[torch.Tensor, int]:
    """Return z as [B,K,D] and inferred K."""
    if z.dim() == 2:
        B, D = z.shape
        return z.unsqueeze(1), 1
    if z.dim() == 3:
        return z, z.shape[1]
    raise ValueError(f"z must be [B,D] or [B,K,D], got {tuple(z.shape)}")

def _grad_or_zeros(y: torch.Tensor, x: torch.Tensor, create_graph: bool) -> torch.Tensor:
    """
    Safe grad(y.sum(), x). If y has no grad path to x, return zeros_like(x).
    """
    if not x.requires_grad:
        return torch.zeros_like(x)
    g = grad(y.sum(), x, create_graph=create_graph, allow_unused=True)[0]
    if g is None:
        g = torch.zeros_like(x)
    return g

# --- robust HVP / Laplacian that handle affine cases cleanly ---

def _hvp_from_grad(g: torch.Tensor,
                   x: torch.Tensor,
                   v: torch.Tensor,
                   create_graph: bool) -> torch.Tensor:
    """
    Hessian-vector product H_x(u) @ v using g = ∇_x u.
    If (g·v).sum() has no grad path to x (e.g., g is x-constant), return zeros.
    """
    s = (g * v).sum()
    if not s.requires_grad or not x.requires_grad:
        return torch.zeros_like(x)
    Hv = grad(s, x, create_graph=create_graph, allow_unused=True)[0]
    if Hv is None:
        Hv = torch.zeros_like(x)
    return Hv

def _laplacian_hutch(g: torch.Tensor,
                     x: torch.Tensor,
                     probes: int,
                     mode: str,
                     gen: Optional[torch.Generator],
                     create_graph: bool) -> torch.Tensor:
    if probes <= 0:
        raise ValueError("probes must be >= 1 for Laplacian.")
    acc = 0.0
    for _ in range(int(probes)):
        v = _rand_vec_like(x, mode, gen)
        Hv = _hvp_from_grad(g, x, v, create_graph=create_graph)
        acc = acc + (Hv * v).sum(dim=-1, keepdim=True)
    return acc / float(probes)

# --- robust divergence Hutchinson (handles x-independent fields) ---

def _divergence_hutch(v_field: torch.Tensor,
                      x: torch.Tensor,
                      probes: int,
                      mode: str,
                      gen: Optional[torch.Generator],
                      create_graph: bool) -> torch.Tensor:
    """
    Hutchinson divergence: trace(J_v).
    If (v·eps).sum() has no grad path to x (v independent of x), return zeros.
    """
    if probes <= 0:
        raise ValueError("probes must be >= 1 for divergence.")
    acc = 0.0
    for _ in range(int(probes)):
        eps = _rand_vec_like(x, mode, gen)
        s = (v_field * eps).sum()
        if not s.requires_grad or not x.requires_grad:
            Jv_eps = torch.zeros_like(x)
        else:
            Jv_eps = grad(s, x, create_graph=create_graph, allow_unused=True)[0]
            if Jv_eps is None:
                Jv_eps = torch.zeros_like(x)
        acc = acc + (Jv_eps * eps).sum(dim=-1, keepdim=True)
    return acc / float(probes)

# ----------------------- PDEState (no psi) -----------------------

class PDEState:
    """
    Lazy, cached PDE primitives for a single step (no psi).

    Required:
      - f: nn.Module mapping x -> [B,K,1]
      - z: current latents [B,D] or [B,K,D]
      - direction: +1 or -1

    Config kwargs (all optional):
      detach_between_steps: bool = False
      dt_value: float = 1.0
      track_param_through_xgrads: bool = True
      eps_norm2: float = 1e-8
      laplace_probes: int = 1
      divergence_probes: int = 1
      rng: str = "rademacher"   # or "gaussian"
      seed: Optional[int] = None
    """

    def __init__(self,
                 f: nn.Module,
                 z: torch.Tensor,
                 direction: int = +1,
                 **config):
        self.f_m = f

        z_bk, _ = _ensure_bkd(z)
        self.B, self.K, self.D = z_bk.shape
        self.device, self.dtype = z_bk.device, z_bk.dtype
        self.direction = direction

        # defaults
        self.cfg: Dict[str, Any] = {
            "detach_between_steps": False,
            "dt_value": 1.0,
            "track_param_through_xgrads": True,
            "eps_norm2": 1e-8,
            "laplace_probes": 1,
            "divergence_probes": 1,
            "rng": "rademacher",
            "seed": None,
        }
        self.cfg.update(config)

        # cache
        self.state: Dict[Any, Any] = {}

        # core tensors
        if self.cfg["detach_between_steps"]:
            x_leaf = z_bk.detach().clone().requires_grad_(True)
        else:
            x_leaf = z_bk.requires_grad_(True)

        dt_value = self.cfg["dt_value"]
        if not isinstance(self.direction, torch.Tensor) and not isinstance(dt_value, torch.Tensor):
            dt = torch.full((self.B, self.K, 1),
                            float(dt_value) * float(self.direction),
                            device=self.device, dtype=self.dtype)
        else:
            dt = (self.direction * dt_value).reshape(self.B, -1, 1)
            if self.K > dt.shape[1]:
                dt = dt.repeat(1, self.K // dt.shape[1], 1)

        self.state["x"] = x_leaf           # [B,K,D]
        self.state["dt"] = dt              # [B,K,1]
        self.state["losses"] = {}

        # optional RNG for Hutchinson
        self._gen = _make_gen(self.cfg["seed"], self.device)

    # ------------- basics -------------

    def x(self) -> torch.Tensor:
        return self.state["x"]

    def dt(self) -> torch.Tensor:
        return self.state["dt"]

    def zeros(self) -> torch.Tensor:
        """A [B,K,1] zero tensor on the state's device/dtype."""
        return torch.zeros((self.B, self.K, 1), device=self.device, dtype=self.dtype)

    # ------------- f and geometry -------------

    def f(self, when: When = "now") -> torch.Tensor:
        key = ("f", when)
        if key not in self.state:
            if when == "now":
                self.state[key] = self.f_m(self.x())
            elif when == "next":
                self.state[key] = self.f_m(self.x_next())
            else:
                raise ValueError("when must be 'now' or 'next'")
        return self.state[key]

    def f_grad(self, when: When = "now") -> torch.Tensor:
        key = ("f_grad", when)
        if key not in self.state:
            create_graph = bool(self.cfg["track_param_through_xgrads"])
            xw = self.x() if when == "now" else self.x_next()
            self.state[key] = _grad_or_zeros(self.f(when), xw, create_graph=create_graph)
        return self.state[key]

    def f_laplace(self, when: When = "now", probes: Optional[int] = None) -> torch.Tensor:
        key = ("f_laplace", when, probes)
        if key not in self.state:
            p = int(self.cfg["laplace_probes"] if probes is None else probes)
            xw = self.x() if when == "now" else self.x_next()
            self.state[key] = _laplacian_hutch(
                g=self.f_grad(when),
                x=xw,
                probes=p,
                mode=self.cfg["rng"],
                gen=self._gen,
                create_graph=True,
            )
        return self.state[key]

    def Xf(self, when: When = "now") -> torch.Tensor:
        key = ("Xf", when)
        if key not in self.state:
            g = self.f_grad(when)
            norm2 = g.pow(2).sum(dim=-1, keepdim=True).add_(float(self.cfg["eps_norm2"]))
            self.state[key] = g / norm2
        return self.state[key]

    # ------------- velocity and divergence -------------

    def v(self, when: When = "now") -> torch.Tensor:
        """
        Velocity field without psi: v = Xf.
        (If you had additional terms previously from psi_grad, add them externally.)
        """
        key = ("v", when)
        if key not in self.state:
            self.state[key] = self.Xf(when)
        return self.state[key]

    def v_div(self, when: When = "now", probes: Optional[int] = None) -> torch.Tensor:
        key = ("v_div", when, probes)
        if key not in self.state:
            p = int(self.cfg["divergence_probes"] if probes is None else probes)
            xw = self.x() if when == "now" else self.x_next()
            self.state[key] = _divergence_hutch(
                v_field=self.v(when),
                x=xw,
                probes=p,
                mode=self.cfg["rng"],
                gen=self._gen,
                create_graph=True,
            )
        return self.state[key]

    # ------------- stepping / convenience -------------

    def x_next(self) -> torch.Tensor:
        if "x_next" not in self.state:
            self.state["x_next"] = self.x() + self.dt() * self.v("now")
        return self.state["x_next"]

    def delta_y(self) -> torch.Tensor:
        """
        f(next) - stopgrad(f(now)) to use as a one-step target / residual term.
        """
        if "delta_y" not in self.state:
            self.state["delta_y"] = self.f("next") - self.f("now").detach()
        return self.state["delta_y"]