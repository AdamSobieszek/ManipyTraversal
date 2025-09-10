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
    name = "ot"
    def _loss(self, st: PDEState) -> torch.Tensor:
        return st.zeros()

class BB(PDELoss):
    name = "bb"
    def _loss(self, st: PDEState) -> torch.Tensor:
        return st.v("now").pow(2).sum(dim=-1, keepdim=True)

class UnitSpeed(PDELoss):
    name = "unitspeed"
    def _loss(self, st: PDEState) -> torch.Tensor:
        res = (st.f_grad() * st.v("now")).sum(dim=-1, keepdim=True) - 1.0
        return res.pow(2)

class SliceHJ(PDELoss):
    name = "slicehj"
    def _loss(self, st: PDEState) -> torch.Tensor:
        eps = float(self.ctx.get("epsilon", 0.0))
        H = 0.5 * st.psi_grad("now").pow(2).sum(dim=-1, keepdim=True)
        lap = st.psi_laplace("now") if eps > 0.0 else st.zeros()
        res = st.psi("now") + H - 0.5 * (eps**2) * lap
        return res.pow(2)

class Kinetic(PDELoss):
    name = "kin"
    needs_next = True
    def _loss(self, st: PDEState) -> torch.Tensor:
        F: nn.Module = self.ctx["F"]
        eps_norm2 = float(st.cfg["eps_norm2"])
        psi_tt_next = st.psi_tt("next")
        gpsi_next = st.psi_grad("next")
        gnorm = gpsi_next.norm(dim=-1, keepdim=True)
        x_hat = st.x() + gpsi_next
        t_hat = F(x_hat)
        gradf_hat = grad(t_hat.sum(), x_hat, create_graph=True)[0]
        denom = gradf_hat.pow(2).sum(dim=-1, keepdim=True).add(eps_norm2).sqrt()
        res = psi_tt_next - (gnorm / denom)
        return res.pow(2)

class Footpoint(PDELoss):
    name = "foot"
    def _loss(self, st: PDEState) -> torch.Tensor:
        F: nn.Module = self.ctx["F"]
        x_hat = st.x() + st.psi_grad("now")
        t_next_from_hat = F(x_hat)
        return (t_next_from_hat - (st.f() + st.dt())).pow(2)

class DivPrior(PDELoss):
    name = "div"
    def _loss(self, st: PDEState) -> torch.Tensor:
        prior_score: Optional[Callable[[torch.Tensor], torch.Tensor]] = self.ctx.get("prior_score", None)
        div_v = st.v_div("now")
        s_prior = prior_score(st.x()) if prior_score is not None else -st.x()
        res = div_v + (s_prior * st.v("now")).sum(dim=-1, keepdim=True)
        return res.pow(2)

class Tangential(PDELoss):
    name = "tan"
    def _loss(self, st: PDEState) -> torch.Tensor:
        return st.zeros()




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
