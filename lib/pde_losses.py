# pde_losses.py
from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Callable, Type
import inspect, sys

import torch
from torch import nn
from torch.autograd import grad

from lib.pde_ops import PDEState

# ---------------- Base ----------------

class PDELoss(nn.Module):
    name: str = "loss"
    needs_next: bool = False  # subclasses toggle when they require "next"

    def __init__(self, lam: float, **kwargs):
        super().__init__()
        self.lam = float(lam)
        self.ctx = kwargs

    def _loss(self, st: PDEState) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, st: PDEState) -> torch.Tensor:
        loss = self._loss(st) if self.lam != 0.0 else st.zeros()
        st.state["losses"][f"L_{self.name}"] = loss.detach().mean()
        return self.lam * loss


# ---------------- Losses ----------------

class OT(PDELoss):
    """Placeholder; define your optimal transport term here."""
    name = "ot"
    def _loss(self, st: PDEState) -> torch.Tensor:
        return st.zeros()

class BB(PDELoss):
    """Benamou–Brenier kinetic energy: ||v||^2."""
    name = "bb"
    def _loss(self, st: PDEState) -> torch.Tensor:
        return st.v("now").pow(2).sum(dim=-1, keepdim=True)

class UnitSpeed(PDELoss):
    """(<∇f, v> - 1)^2."""
    name = "unitspeed"
    def _loss(self, st: PDEState) -> torch.Tensor:
        res = (st.f_grad() * st.v("now")).sum(dim=-1, keepdim=True) - 1.0
        return res.pow(2)
     
class SliceHJ(PDELoss):
    r"""
    HJ residual at "now": (ψ + 0.5||∇ψ||^2 - 0.5 ε^2 Δψ)^2.
    Pass epsilon via `epsilon=...` (defaults to 0.0).
    """
    name = "slicehj"
    def _loss(self, st: PDEState) -> torch.Tensor:
        eps = float(self.ctx.get("epsilon", 0.0))
        H = 0.5 * st.psi_grad("now").pow(2).sum(dim=-1, keepdim=True)
        if eps > 0.0:
            lap = st.psi_laplace("now")
        else:
            lap = st.zeros()
        res = st.psi("now") + H - 0.5 * (eps ** 2) * lap
        return res.pow(2)
             
class HJ(PDELoss):
    r"""
    HJ residual at "now": (ψ + 0.5||∇ψ||^2 - 0.5 ε^2 Δψ)^2.
    Pass epsilon via `epsilon=...` (defaults to 0.0).
    """
    name = "hj"
    def _loss(self, st: PDEState) -> torch.Tensor:
        eps = float(self.ctx.get("epsilon", 0.0))
        H = 0.5 * st.psi_grad("now").pow(2).sum(dim=-1, keepdim=True)
        if eps > 0.0:
            lap = st.psi_laplace("now")
        else:
            lap = st.zeros()
        res = st.psi("now") + H - 0.5 * (eps ** 2) * lap
        return res.pow(2)

class Kinetic(PDELoss):
    r"""
    Kinetic term (wave-like): ψ_tt(next) - ||∇ψ(next)|| / ||∇f(x_hat)||, with x_hat := x + ∇ψ(next).
    Requires the semantic potential module F: pass as `F=...` when constructing.
    """
    name = "kin"
    def _loss(self, st: PDEState) -> torch.Tensor:
        F: nn.Module = self.ctx["F"]  # required
        eps_norm2 = float(st.cfg["eps_norm2"])

        psi_tt_next = st.psi_tt("next")                         # [B,K,1]
        gpsi_next   = st.psi_grad("next")                       # [B,K,D]
        gnorm       = gpsi_next.norm(dim=-1, keepdim=True)      # [B,K,1]

        x_hat = st.x() + gpsi_next                              # [B,K,D]
        t_hat = F(x_hat)                                        # [B,K,1]
        gradf_hat = grad(t_hat.sum(), x_hat, create_graph=True)[0]
        denom = gradf_hat.pow(2).sum(dim=-1, keepdim=True).add(eps_norm2).sqrt()

        res = psi_tt_next - (gnorm / denom)
        return res.pow(2)

class Footpoint(PDELoss):
    r"""
    Footpoint consistency: f(x + ∇ψ(now)) - ( f(x) + dt ) squared.
    Requires F: pass as `F=...`.
    """
    name = "foot"
    def _loss(self, st: PDEState) -> torch.Tensor:
        F: nn.Module = self.ctx["F"]  # required
        x_hat = st.x() + st.psi_grad("now")
        t_next_from_hat = F(x_hat)
        return (t_next_from_hat - (st.f() + st.dt())).pow(2)

class DivPrior(PDELoss):
    r"""
    Divergence prior: ( ∇·v(now) + <s_prior(x), v(now)> )^2.
    Default s_prior(x) = -x (Gaussian prior). You can pass a custom callable via `prior_score=...`.
    """
    name = "div"
    def _loss(self, st: PDEState) -> torch.Tensor:
        div_v = st.v_div("now")                                  # [B,K,1]
        prior_score: Optional[Callable[[torch.Tensor], torch.Tensor]] = self.ctx.get("prior_score", None)
        s_prior = prior_score(st.x()) if prior_score is not None else -st.x()
        res = div_v + (s_prior * st.v("now")).sum(dim=-1, keepdim=True)
        return res.pow(2)

class Tangency(PDELoss):
    """Placeholder for tangency constraints if you add one later."""
    name = "tan"
    def _loss(self, st: PDEState) -> torch.Tensor:
        res = (st.f_grad() * st.psi_grad()).sum(dim=-1, keepdim=True)
        return res.pow(2)

class FConvex(PDELoss):
    r"""
    Trace-based convexity penalty for f.

    Uses Hutchinson's estimator of the Laplacian:
        q ≈ E_v [ v^T (∇^2 f) v ] = tr(∇^2 f)
    and penalizes violations of q >= margin via a hinge.

    Ctx:
      - probes: int (default 8)     # number of Hutchinson probes
      - margin: float (default 0.0) # require average curvature >= margin
    """
    name = "fconvex"
    def _loss(self, st: PDEState) -> torch.Tensor:
        probes = int(self.ctx.get("probes", 1))
        margin = float(self.ctx.get("margin", 0.0))
        q = st.f_laplace(probes=probes)              # [B,K,1], ≈ average v^T Hf v
        viol = (margin - q).clamp_min(0.0)           # hinge on negative average curvature
        return viol.pow(2)

class FGradNormEMA(PDELoss):
    r"""
    Per-head EMA target for ||∇f|| with deviation penalty.

    Keeps an [K,1] EMA buffer of the batch-mean gradient norm for each head
    and penalizes per-sample deviations from that running target.

    Ctx (optional):
      - ema_beta: float in [0,1) (default 0.99)
      - relative: bool (default False)  # if True, penalize relative error
      - eps: float (default 1e-8)       # stability for relative mode
    """
    name = "fgnorm"
    @torch.no_grad()
    def _init_or_update_ema(self, gnorm: torch.Tensor):
        gnorm = gnorm.mean(dim=0, keepdim=True)
        if "ema_fgnorm" not in self.ctx:
            self.ctx["ema_fgnorm"] = torch.ones_like(gnorm)
        else:
            self.ctx["ema_fgnorm"].lerp_(gnorm, 1.0 - self.ctx.get("ema_beta", 0.99))

    def _loss(self, st: PDEState) -> torch.Tensor:
        gnorm = st.f_grad().norm(dim=-1, keepdim=True)
        self._init_or_update_ema(gnorm.clone().detach())
        if self.ctx.get("relative", False):
            diff = (gnorm - self.ctx["ema_fgnorm"]) / (self.ctx["ema_fgnorm"].abs() + self.ctx.get("eps", 1e-8))
        else:
            diff = gnorm - self.ctx["ema_fgnorm"]
        return diff.pow(2)

# ---------------- Public registry API ----------------

def _normalize(key: str) -> str:
    # tolerate case and punctuation: "sliceHJ" -> "slicehj", "BB" -> "bb"
    return "".join(ch for ch in key.lower() if ch.isalnum())

def loss_registry() -> Dict[str, Type[PDELoss]]:
    """
    Discover all PDELoss subclasses in this module and index by their `name`.
    Returns {normalized_name: LossClass}.
    """
    reg: Dict[str, Type[PDELoss]] = {}
    for _, obj in inspect.getmembers(sys.modules[__name__], inspect.isclass):
        if issubclass(obj, PDELoss) and obj is not PDELoss:
            name = getattr(obj, "name", None)
            if isinstance(name, str) and name:
                reg[_normalize(name)] = obj
    return reg

def build_losses(lambda_dict: Dict[str, float], **ctx) -> Tuple[List[PDELoss], bool]:
    """
    Factory: given {"bb": 1.0, "kin": 0.5, ...}, instantiate ONLY those losses.
    Keys are matched case-insensitively (and punctuation-insensitively).
    Returns (loss_list, needs_next_any).
    """
    reg = loss_registry()
    losses: List[PDELoss] = []  
    needs_next_any = False
    for raw_name, lam in lambda_dict.items():
        key = _normalize(raw_name)
        cls = reg.get(key, None)
        if cls is None:
            # Unknown key: ignore (useful for parameters like "epsilon")
            continue
        inst = cls(lam, **ctx)
        losses.append(inst)
        if getattr(cls, "needs_next", False):
            needs_next_any = True
    return losses, needs_next_any
