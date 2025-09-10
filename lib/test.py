# tests/print_pde_ops_check.py
import torch
import math

from pde_ops import PDEState  # import your package/module

torch.set_printoptions(precision=6, sci_mode=True)

# ---------- Simple, analytically tractable potentials ----------

class FLinear(torch.nn.Module):
    """
    f(x) = a · x + b
    ∇f = a
    Δf = 0
    """
    def __init__(self, a: torch.Tensor, b: float):
        super().__init__()
        self.register_buffer("a", a.view(1, 1, -1))   # [1,1,D] for BK broadcast
        self.register_buffer("b", torch.tensor(b, dtype=a.dtype))

    def forward(self, x_bkd: torch.Tensor) -> torch.Tensor:
        return (x_bkd * self.a).sum(dim=-1, keepdim=True) + self.b


class PSIQ(torch.nn.Module):
    """
    ψ(x,t) = 0.5 * λ * ||x||^2 + α * t + β
    ∇_x ψ = λ x
    Δψ = λ D
    ψ_t = α
    ψ_tt = 0
    """
    def __init__(self, lam: float, alpha: float, beta: float, dtype=torch.float32):
        super().__init__()
        self.register_buffer("lam", torch.tensor(lam, dtype=dtype))
        self.register_buffer("alpha", torch.tensor(alpha, dtype=dtype))
        self.register_buffer("beta", torch.tensor(beta, dtype=dtype))

    def forward(self, x_bkd: torch.Tensor, t_bk1: torch.Tensor) -> torch.Tensor:
        quad = 0.5 * self.lam * (x_bkd.pow(2)).sum(dim=-1, keepdim=True)
        return quad + self.alpha * t_bk1 + self.beta


# ---------- Helpers ----------

def check_close(name, actual, expected, atol=1e-12, rtol=0.0):
    ok = torch.allclose(actual, expected, atol=atol, rtol=rtol)
    max_abs_diff = (actual - expected).abs().max().item()
    print(f"{name:>18}: {'PASS' if ok else 'FAIL'} | max|Δ|={max_abs_diff:.3e}")
    if not ok:
        print("  actual  :", actual)
        print("  expected:", expected)
    return ok


def main():
    dtype = torch.float64  # high precision for crisp comparisons

    # Dimensions
    B, K, D = 2, 2, 3

    # Known constants
    a_vec = torch.tensor([1.0, -2.0, 0.5], dtype=dtype)[:D]
    b = 0.3
    lam = 2.5
    alpha = 1.7
    beta = -0.2
    dt_val = 0.1

    # Modules
    f_mod = FLinear(a_vec.to(dtype), b).to(dtype)
    psi_mod = PSIQ(lam=lam, alpha=alpha, beta=beta, dtype=dtype)

    # Input x ∈ R^{B×K×D}
    xs = torch.arange(B * K * D, dtype=dtype).reshape(B, K, D) / 10.0
    x_bkd = xs + 0.3

    # Build PDE state (single step)
    # NOTE: we use time_ad="forward" so psi_tt is computed via JVP (ψ_tt=0 here).
    st = PDEState(
        f=f_mod,
        psi=psi_mod,
        z=x_bkd,
        direction=+1,
        need_next=True,
        dt_value=dt_val,
        time_ad="forward",
        detach_between_steps=True,
        laplace_probes=1,
        divergence_probes=1,
        rng="rademacher",
        seed=42,
        track_param_through_xgrads=True,
        eps_norm2=1e-12,
    )

    # Expected analytics
    f_expected = (x_bkd * a_vec.view(1, 1, -1)).sum(-1, keepdim=True) + b
    f_grad_expected = a_vec.view(1, 1, -1).expand(B, K, D)
    f_laplace_expected = torch.zeros(B, K, 1, dtype=dtype)

    a_norm2 = (a_vec ** 2).sum()
    Xf_expected = (a_vec / a_norm2).view(1, 1, -1).expand(B, K, D)

    t_now = f_expected.detach()
    t_next = t_now + dt_val

    psi_now_expected = 0.5 * lam * (x_bkd ** 2).sum(-1, keepdim=True) + alpha * t_now + beta
    psi_next_expected = psi_now_expected + alpha * dt_val
    psi_grad_expected = lam * x_bkd
    psi_laplace_expected = torch.full((B, K, 1), lam * D, dtype=dtype)
    psi_t_expected = torch.full((B, K, 1), alpha, dtype=dtype)
    psi_tt_expected = torch.zeros(B, K, 1, dtype=dtype)

    v_expected = Xf_expected + psi_grad_expected
    v_div_expected = torch.full((B, K, 1), lam * D, dtype=dtype)  # div(const + λx) = λD
    x_next_expected = x_bkd + dt_val * v_expected

    # Checks (NOW)
    n_pass = 0; n_total = 0
    def run(name, actual, expected):
        nonlocal n_pass, n_total
        n_total += 1
        if check_close(name, actual, expected): n_pass += 1

    print("=== NOW ===")
    run("x", st.x(), x_bkd)
    run("dt", st.dt(), torch.full((B, K, 1), dt_val, dtype=dtype))
    run("f", st.f(), f_expected)
    run("f_grad", st.f_grad(), f_grad_expected)
    run("f_laplace", st.f_laplace(), f_laplace_expected)
    run("Xf", st.Xf(), Xf_expected)

    run("psi", st.psi("now"), psi_now_expected)
    run("psi_grad", st.psi_grad("now"), psi_grad_expected)
    run("psi_laplace", st.psi_laplace("now"), psi_laplace_expected)
    run("psi_t", st.psi_t("now"), psi_t_expected)
    run("psi_tt", st.psi_tt("now"), psi_tt_expected)

    run("v", st.v("now"), v_expected)
    run("v_div", st.v_div("now"), v_div_expected)
    run("x_next", st.x_next(), x_next_expected)

    # Checks (NEXT)
    print("\n=== NEXT ===")
    run("psi[next]", st.psi("next"), psi_next_expected)
    run("psi_grad[next]", st.psi_grad("next"), psi_grad_expected)   # same as now (x-only)
    run("psi_laplace[next]", st.psi_laplace("next"), psi_laplace_expected)
    run("psi_t[next]", st.psi_t("next"), psi_t_expected)
    run("psi_tt[next]", st.psi_tt("next"), psi_tt_expected)
    run("v_div[next]", st.v_div("next"), v_div_expected)

    # Composite loss sanity & backward
    print("\n=== Composite loss (sanity) ===")
    c = 1.0
    L = (st.psi_tt() - (c ** 2) * st.psi_laplace()).mean()
    print("loss value:", float(L.detach()))
    L.backward()
    print("x.grad finite:", torch.isfinite(st.x().grad).all().item())

    print(f"\nRESULT: {n_pass}/{n_total} checks passed.")

if __name__ == "__main__":
    main()

# tests/print_pde_ops_quadratic_check.py
import torch
from pde_ops import PDEState

torch.set_printoptions(precision=6, sci_mode=True)

# ---------- Quadratic f and psi with closed-form derivatives ----------

class FQuad(torch.nn.Module):
    """
    f(x) = 0.5 * x^T A x + b^T x + c
    ∇f = A x + b
    Δf = tr(A)
    """
    def __init__(self, A: torch.Tensor, b: torch.Tensor, c: float = 0.0):
        super().__init__()
        assert A.shape[0] == A.shape[1] == b.numel()
        A = 0.5 * (A + A.T)
        self.register_buffer("A", A)                               # [D,D]
        self.register_buffer("b", b.view(1,1,-1))                  # [1,1,D]
        self.register_buffer("c", torch.tensor(float(c), dtype=A.dtype))

    def forward(self, x_bkd: torch.Tensor) -> torch.Tensor:
        Ax = torch.einsum("bkd,dd->bkd", x_bkd, self.A)
        quad = 0.5 * (x_bkd * Ax).sum(dim=-1, keepdim=True)
        lin  = (x_bkd * self.b).sum(dim=-1, keepdim=True)
        return quad + lin + self.c


class PSIQuadT(torch.nn.Module):
    """
    ψ(x,t) = 0.5 * x^T Q x + r^T x + α t + 0.5 β t^2 + γ
    ∇_x ψ = Q x + r
    Δψ = tr(Q)
    ψ_t = α + β t
    ψ_tt = β
    """
    def __init__(self, Q: torch.Tensor, r: torch.Tensor, alpha: float, beta: float, gamma: float = 0.0):
        super().__init__()
        assert Q.shape[0] == Q.shape[1] == r.numel()
        Q = 0.5 * (Q + Q.T)
        self.register_buffer("Q", Q)                               # [D,D]
        self.register_buffer("r", r.view(1,1,-1))                  # [1,1,D]
        self.register_buffer("alpha", torch.tensor(float(alpha), dtype=Q.dtype))
        self.register_buffer("beta", torch.tensor(float(beta), dtype=Q.dtype))
        self.register_buffer("gamma", torch.tensor(float(gamma), dtype=Q.dtype))

    def forward(self, x_bkd: torch.Tensor, t_bk1: torch.Tensor) -> torch.Tensor:
        Qx = torch.einsum("bkd,dd->bkd", x_bkd, self.Q)
        quad = 0.5 * (x_bkd * Qx).sum(dim=-1, keepdim=True)
        lin  = (x_bkd * self.r).sum(dim=-1, keepdim=True)
        time = self.alpha * t_bk1 + 0.5 * self.beta * (t_bk1 ** 2)
        return quad + lin + time + self.gamma


# ---------- Utilities ----------

def check_close(name, actual, expected, atol=1e-8, rtol=1e-8):
    ok = torch.allclose(actual, expected, atol=atol, rtol=rtol)
    diff = (actual - expected).abs().max().item()
    print(f"{name:>22}: {'PASS' if ok else 'FAIL'} | max|Δ|={diff:.3e}")
    if not ok:
        print("  actual  :", actual)
        print("  expected:", expected)
    return ok

def exact_divergence_from_basis(v_field: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Compute exact divergence via coordinate basis:
      div(v) = sum_i e_i^T J_v e_i
    where J_v is Jacobian wrt x. This is exact and deterministic.
    """
    B,K,D = x.shape
    total = torch.zeros(B, K, 1, dtype=x.dtype, device=x.device)
    for i in range(D):
        eps = torch.zeros_like(x)
        eps[..., i] = 1.0
        Jv_eps = torch.autograd.grad((v_field * eps).sum(), x, create_graph=True)[0]
        diag_i = (Jv_eps * eps).sum(-1, keepdim=True)  # ∂v_i/∂x_i
        total = total + diag_i
    return total

def main():
    dtype = torch.float64
    B, K, D = 2, 2, 3

    # Symmetric A, Q and nontrivial b, r
    A = torch.diag(torch.tensor([2.0, -1.0, 0.5], dtype=dtype))
    b = torch.tensor([0.3, -0.2, 0.1], dtype=dtype)
    Q = torch.diag(torch.tensor([1.0, 2.0, 3.0], dtype=dtype))
    r = torch.tensor([0.1, -0.1, 0.2], dtype=dtype)
    alpha, beta, c, gamma = 1.2, -0.7, 0.05, -0.2
    dt_val = 0.1

    f_mod = FQuad(A, b, c).to(dtype)
    psi_mod = PSIQuadT(Q, r, alpha, beta, gamma).to(dtype)

    # Inputs
    xs = torch.arange(B * K * D, dtype=dtype).reshape(B, K, D) / 10.0
    x_bkd = xs + 0.25

    # State (use many probes for Hutchinson; still stochastic, but tight)
    st = PDEState(
        f=f_mod,
        psi=psi_mod,
        z=x_bkd,
        direction=+1,
        need_next=True,
        dt_value=dt_val,
        time_ad="forward",            # ψ_tt = β exactly
        detach_between_steps=True,
        track_param_through_xgrads=True,
        eps_norm2=1e-12,
        laplace_probes=2048,
        divergence_probes=8192,       # high probe count for low-variance div estimate
        rng="rademacher",
        seed=12345,
    )

    # ---- Closed-form targets ----
    Ax = torch.einsum("bkd,dd->bkd", x_bkd, A)
    g  = Ax + b.view(1,1,-1)                        # ∇f
    s  = (g * g).sum(-1, keepdim=True)              # g·g
    f_expected      = 0.5 * (x_bkd * Ax).sum(-1, keepdim=True) + (x_bkd*b.view(1,1,-1)).sum(-1, keepdim=True) + c
    f_grad_expected = g
    f_lap_expected  = torch.full((B,K,1), torch.trace(A), dtype=dtype)

    Xf_expected = g / s

    t_now  = f_expected.detach()
    t_next = t_now + dt_val

    Qx = torch.einsum("bkd,dd->bkd", x_bkd, Q)
    psi_now_expected  = 0.5*(x_bkd*Qx).sum(-1,keepdim=True) + (x_bkd*r.view(1,1,-1)).sum(-1,keepdim=True) \
                        + alpha*t_now + 0.5*beta*(t_now**2) + gamma
    psi_next_expected = 0.5*(x_bkd*Qx).sum(-1,keepdim=True) + (x_bkd*r.view(1,1,-1)).sum(-1,keepdim=True) \
                        + alpha*t_next + 0.5*beta*(t_next**2) + gamma
    psi_grad_expected = Qx + r.view(1,1,-1)
    psi_lap_expected  = torch.full((B,K,1), torch.trace(Q), dtype=dtype)
    psi_t_now_exp     = alpha + beta * t_now
    psi_t_next_exp    = alpha + beta * t_next
    psi_tt_exp        = torch.full((B,K,1), beta, dtype=dtype)

    v_now_expected = Xf_expected + psi_grad_expected

    # Exact divergence: tr(Q) + div(Xf) with div(Xf) = (tr(A)*s - 2 g^T A g)/s^2
    Ag = torch.einsum("bkd,dd->bkd", g, A)
    gTAg = (g * Ag).sum(-1, keepdim=True)
    div_Xf_exact = (torch.trace(A) * s - 2.0 * gTAg) / (s * s)
    v_div_expected = psi_lap_expected + div_Xf_exact

    x_next_expected = x_bkd + dt_val * v_now_expected

    # ---- Checks ----
    print("=== NOW (theory vs. autograd) ===")
    check_close("x", st.x(), x_bkd)
    check_close("dt", st.dt(), torch.full((B,K,1), dt_val, dtype=dtype))

    check_close("f", st.f(), f_expected)
    check_close("f_grad", st.f_grad(), f_grad_expected)
    check_close("f_laplace", st.f_laplace(), f_lap_expected, atol=5e-4, rtol=1e-6)

    check_close("Xf", st.Xf(), Xf_expected)

    check_close("psi(now)", st.psi("now"), psi_now_expected)
    check_close("psi_grad(now)", st.psi_grad("now"), psi_grad_expected)
    check_close("psi_laplace(now)", st.psi_laplace("now"), psi_lap_expected, atol=5e-4, rtol=1e-6)

    check_close("psi_t(now)", st.psi_t("now"), psi_t_now_exp)
    check_close("psi_tt(now)", st.psi_tt("now"), psi_tt_exp)

    check_close("v(now)", st.v("now"), v_now_expected)

    # Hutchinson vs theory (looser tol), PLUS exact divergence from basis for reference
    v_div_now_est = st.v_div("now")                  # stochastic estimator
    v_div_now_exact_autograd = exact_divergence_from_basis(st.v("now"), st.x())
    check_close("v_div(now) (Hutch vs theory)", v_div_now_est, v_div_expected, atol=4e-2, rtol=1e-3)
    check_close("v_div(now) (exact vs theory)", v_div_now_exact_autograd, v_div_expected, atol=1e-8, rtol=1e-8)

    check_close("x_next", st.x_next(), x_next_expected)

    print("\n=== NEXT (theory vs. autograd) ===")
    check_close("psi(next)", st.psi("next"), psi_next_expected)
    check_close("psi_grad(next)", st.psi_grad("next"), psi_grad_expected)
    check_close("psi_laplace(next)", st.psi_laplace("next"), psi_lap_expected, atol=5e-4, rtol=1e-6)
    check_close("psi_t(next)", st.psi_t("next"), psi_t_next_exp)
    check_close("psi_tt(next)", st.psi_tt("next"), psi_tt_exp)

    v_div_next_est = st.v_div("next")
    check_close("v_div(next) (Hutch vs theory)", v_div_next_est, v_div_expected, atol=4e-2, rtol=1e-3)

    # Composite loss that depends on x (so backward is meaningful)
    print("\n=== Composite loss & backward ===")
    c = 1.0
    # Use an x-dependent objective to ensure requires_grad=True
    L = (
        0.3 * st.psi().mean() +
        0.2 * st.f().mean() +
        0.1 * st.v().pow(2).mean() +
        0.4 * (st.psi_t().pow(2).mean())    # depends on t(f(x)) as well
        - 0.05 * (c ** 2) * st.psi_laplace().mean()
    )
    print("loss value:", float(L.detach()))
    L.backward()
    print("x.grad finite:", torch.isfinite(st.x().grad).all().item())

if __name__ == "__main__":
    main()

# tests/print_pde_ops_edge_cases.py
import torch
from pde_ops import PDEState

torch.set_printoptions(precision=6, sci_mode=True)

def check_close(name, actual, expected, atol=1e-6, rtol=1e-6):
    ok = torch.allclose(actual, expected, atol=atol, rtol=rtol)
    diff = (actual - expected).abs().max().item()
    print(f"{name:>28}: {'PASS' if ok else 'FAIL'} | max|Δ|={diff:.3e}")
    if not ok:
        print("  actual  :", actual)
        print("  expected:", expected)
    return ok

def exact_divergence_from_basis(v_field, x):
    B,K,D = x.shape
    total = torch.zeros(B, K, 1, dtype=x.dtype, device=x.device)
    for i in range(D):
        eps = torch.zeros_like(x)
        eps[..., i] = 1.0
        Jv_eps = torch.autograd.grad((v_field * eps).sum(), x, create_graph=True)[0]
        total = total + Jv_eps[..., i:i+1]
    return total

# --- Modules for tests ---

class FQuad(torch.nn.Module):
    def __init__(self, A, b, c=0.0):
        super().__init__()
        A = 0.5*(A + A.T)
        self.register_buffer("A", A)
        self.register_buffer("b", b.view(1,1,-1))
        self.register_buffer("c", torch.tensor(float(c), dtype=A.dtype))
    def forward(self, x):
        Ax = torch.einsum("bkd,dd->bkd", x, self.A)
        return 0.5*(x*Ax).sum(-1, keepdim=True) + (x*self.b).sum(-1, keepdim=True) + self.c

class PSI_Param(torch.nn.Module):
    """
    ψ(x,t) = 0.5 x^T Q x + t u^T x + α t + 0.5 β t^2 + γ
    (Q fixed buffer; u, α, β are learnable Parameters for param-grad tests.)
    """
    def __init__(self, Q, u, alpha=0.7, beta=-0.4, gamma=0.0, learn_time_params=True):
        super().__init__()
        Q = 0.5*(Q + Q.T)
        self.register_buffer("Q", Q)
        self.u = torch.nn.Parameter(u.view(1,1,-1))
        if learn_time_params:
            self.alpha = torch.nn.Parameter(torch.tensor(float(alpha), dtype=Q.dtype))
            self.beta  = torch.nn.Parameter(torch.tensor(float(beta), dtype=Q.dtype))
        else:
            self.register_buffer("alpha", torch.tensor(float(alpha), dtype=Q.dtype))
            self.register_buffer("beta",  torch.tensor(float(beta),  dtype=Q.dtype))
        self.register_buffer("gamma", torch.tensor(float(gamma), dtype=Q.dtype))
    def forward(self, x, t):
        Qx  = torch.einsum("bkd,dd->bkd", x, self.Q)
        quad = 0.5*(x*Qx).sum(-1, keepdim=True)
        linx = (t*self.u * x).sum(-1, keepdim=True)  # t u^T x (broadcast)
        time = self.alpha * t + 0.5 * self.beta * (t**2)
        return quad + linx + time + self.gamma

def main():
    dtype = torch.float64
    B,K,D = 2,2,3
    xs = torch.arange(B*K*D, dtype=dtype).reshape(B,K,D)/10.0 + 0.2
    # Matrices/vectors
    A = torch.diag(torch.tensor([2.0, -1.0, 0.5], dtype=dtype))
    b = torch.tensor([0.0, 0.0, 0.0], dtype=dtype)     # keep g nontrivial via x*A
    Q = torch.diag(torch.tensor([1.5, 0.5, 2.0], dtype=dtype))
    u = torch.tensor([0.3, -0.1, 0.2], dtype=dtype)
    alpha, beta, gamma = 1.1, -0.6, 0.0
    dt = 0.1

    # Modules
    f_mod = FQuad(A, b, c=0.0).to(dtype)
    psi_mod = PSI_Param(Q, u, alpha=alpha, beta=beta, gamma=gamma).to(dtype)

    # ---------- 1) Reverse vs Forward time AD parity + param grads ----------
    print("=== 1) Time AD parity & param grads ===")
    # Reverse-mode
    st_rev = PDEState(f_mod, psi_mod, xs, direction=+1,
                      need_next=True, dt_value=dt, time_ad="reverse",
                      detach_between_steps=True, laplace_probes=256, divergence_probes=256,
                      track_param_through_xgrads=True, eps_norm2=1e-12, seed=123)
    # Forward-mode
    st_fwd = PDEState(f_mod, psi_mod, xs, direction=+1,
                      need_next=True, dt_value=dt, time_ad="forward",
                      detach_between_steps=True, laplace_probes=256, divergence_probes=256,
                      track_param_through_xgrads=True, eps_norm2=1e-12, seed=123)

    # Targets
    t_now  = st_rev.f().detach()
    t_next = t_now + dt
    
    ux = (st_rev.x() * psi_mod.u).sum(-1, keepdim=True)  # u^T x
    psi_t_now_exp  = ux + psi_mod.alpha + psi_mod.beta * t_now
    psi_t_next_exp = ux + psi_mod.alpha + psi_mod.beta * t_next
    psi_tt_exp     = torch.full((B,K,1), float(psi_mod.beta.detach()), dtype=dtype)

    check_close("psi_t(now) reverse",  st_rev.psi_t("now"),  psi_t_now_exp)
    check_close("psi_t(now) forward",  st_fwd.psi_t("now"),  psi_t_now_exp)
    check_close("psi_tt(now) reverse", st_rev.psi_tt("now"), psi_tt_exp)
    check_close("psi_tt(now) forward", st_fwd.psi_tt("now"), psi_tt_exp)

    # Reverse-mode should allow param grads:
    loss_rev = (st_rev.psi_t("now")**2 + st_rev.psi_tt("now")**2).mean()
    for p in psi_mod.parameters():
        if p.grad is not None: p.grad.zero_()
    loss_rev.backward(retain_graph=True)
    print("  param grads (reverse) non-None:",
          all((p.grad is not None) for p in psi_mod.parameters()))

    # Forward-mode usually won't propagate param grads through u_t/u_tt:
    for p in psi_mod.parameters():
        if p.grad is not None: p.grad.zero_()
    loss_fwd = (st_fwd.psi_t("now")**2 + st_fwd.psi_tt("now")**2).mean()
    loss_fwd.backward()
    print("  param grads (forward) non-None:",
          all((p.grad is not None and p.grad.abs().sum()>0) for p in psi_mod.parameters()))

    # ---------- 2) t-dependent spatial gradient (psi_grad now vs next) ----------
    print("\n=== 2) t-dependent psi_grad ===")
    dpsi = st_rev.psi_grad("next") - st_rev.psi_grad("now")
    expected = dt * psi_mod.u.expand_as(dpsi)
    check_close("psi_grad(next)-psi_grad(now)", dpsi, expected)

    # ---------- 3) Zero ∥∇f∥ (robust Xf) ----------
    print("\n=== 3) Zero gradient in f (Xf robust) ===")
    f_zero = FQuad(torch.zeros_like(A), torch.zeros_like(b)).to(dtype)
    st_zero = PDEState(f_zero, psi_mod, xs, direction=+1, dt_value=dt, need_next=False,
                       time_ad="reverse", detach_between_steps=True, eps_norm2=1e-9)
    Xf_zero = st_zero.Xf()
    print("  Xf finite (no NaNs/Infs):", torch.isfinite(Xf_zero).all().item())
    check_close("v equals psi_grad when ∇f=0", st_zero.v(), st_zero.psi_grad())

    # ---------- 4) Divergence: Hutchinson vs exact & analytic ----------
    print("\n=== 4) Divergence checks ===")
    st_div = PDEState(f_mod, psi_mod, xs, direction=+1, dt_value=dt,
                      need_next=False, time_ad="forward", laplace_probes=1024,
                      divergence_probes=8192, rng="rademacher", seed=999,
                      detach_between_steps=True)
    v_now = st_div.v()
    # Exact autograd divergence
    div_exact = exact_divergence_from_basis(v_now, st_div.x())
    # Analytic: tr(Q) + (tr(A)s - 2 g^T A g)/s^2
    Ax = torch.einsum("bkd,dd->bkd", xs, A)
    g  = Ax + b.view(1,1,-1)
    s  = (g*g).sum(-1, keepdim=True)
    Ag = torch.einsum("bkd,dd->bkd", g, A)
    gTAg = (g*Ag).sum(-1, keepdim=True)
    trA = torch.trace(A)
    trQ = torch.trace(Q)
    eps = float(st_div.cfg["eps_norm2"])  # or st_rev.cfg[...] in your earlier block
    s   = (g*g).sum(-1, keepdim=True)
    s_eps = s + eps
    div_Xf = (torch.trace(A) * s_eps - 2.0 * gTAg) / (s_eps * s_eps)
    div_analytic = torch.trace(Q) + div_Xf
    check_close("div v (exact autograd vs analytic)", div_exact, div_analytic, atol=1e-8, rtol=1e-8)
    check_close("div v (Hutch vs analytic)", st_div.v_div(), div_analytic, atol=4e-2, rtol=1e-3)

    # ---------- 5) direction = -1 ----------
    print("\n=== 5) direction = -1 (reverse step) ===")
    st_back = PDEState(f_mod, psi_mod, xs, direction=-1, dt_value=dt, need_next=True, time_ad="forward")
    check_close("dt negative", st_back.dt(), -torch.full_like(st_back.dt(), dt))
    check_close("t_next = t_now + dt", st_back._t("next"), st_back._t("now") + st_back.dt())
    check_close("x_next = x + dt*v", st_back.x_next(), st_back.x() + st_back.dt()*st_back.v())

    # ---------- 6) need_next guard ----------
    print("\n=== 6) need_next guard ===")
    st_guard = PDEState(f_mod, psi_mod, xs, direction=+1, dt_value=dt, need_next=False)
    try:
        st_guard.psi("next")
        print("  ERROR: expected ValueError when calling next without need_next")
    except ValueError:
        print("  PASS: raised ValueError as expected for next without need_next")

    # ---------- 7) RNG reproducibility ----------
    print("\n=== 7) RNG reproducibility ===")
    st_a = PDEState(f_mod, psi_mod, xs, direction=+1, dt_value=dt, divergence_probes=2048, seed=111)
    st_b = PDEState(f_mod, psi_mod, xs, direction=+1, dt_value=dt, divergence_probes=2048, seed=111)
    st_c = PDEState(f_mod, psi_mod, xs, direction=+1, dt_value=dt, divergence_probes=2048, seed=112)
    eq_ab = torch.allclose(st_a.v_div(), st_b.v_div(), atol=0, rtol=0)
    diff_ac = not torch.allclose(st_a.v_div(), st_c.v_div(), atol=0, rtol=0)
    print("  same seed -> identical:", eq_ab, " | different seed -> different:", diff_ac)

    # ---------- 8) probes=0 raises ----------
    print("\n=== 8) probes=0 raises ===")
    try:
        st_a.psi_laplace(probes=0)
        print("  ERROR: expected ValueError for probes=0 in Laplacian")
    except ValueError:
        print("  PASS: Laplacian probes=0 raised ValueError")
    try:
        st_a.v_div(probes=0)
        print("  ERROR: expected ValueError for probes=0 in divergence")
    except ValueError:
        print("  PASS: Divergence probes=0 raised ValueError")

    # ---------- 9) [B,D] broadcast to [B,1,D] ----------
    print("\n=== 9) [B,D] broadcast ===")
    x_bd = xs[:,0,:]  # [B,D]
    st_bd = PDEState(f_mod, psi_mod, x_bd, direction=+1, dt_value=dt)
    check_close("x shapes match after expand", st_bd.x(), x_bd.unsqueeze(1))
    check_close("v match K=1", st_bd.v(), PDEState(f_mod, psi_mod, x_bd.unsqueeze(1), direction=+1, dt_value=dt).v())

    # ---------- 10) dtype toggle ----------
    print("\n=== 10) dtype float32 ===")
    xs32 = xs.to(torch.float32)
    f_mod32 = FQuad(A.to(torch.float32), b.to(torch.float32)).to(torch.float32)
    psi_mod32 = PSI_Param(Q.to(torch.float32), u.to(torch.float32), alpha=float(alpha), beta=float(beta)).to(torch.float32)
    st32 = PDEState(f_mod32, psi_mod32, xs32, dt_value=float(dt), need_next=True, time_ad="forward")
    print("  float32 forward OK. psi_tt:", st32.psi_tt().mean().item())

if __name__ == "__main__":
    main()
