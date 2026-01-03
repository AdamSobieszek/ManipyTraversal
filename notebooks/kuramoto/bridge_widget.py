import numpy as np
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ipywidgets as widgets
from IPython.display import display

# ----------------------------
# Helpers: sphere wireframe + line segments for arrows
# ----------------------------
def _sphere_wireframe_traces(n_lat: int = 9, n_lon: int = 18):
    traces = []
    lat_vals = np.linspace(-0.8 * np.pi / 2, 0.8 * np.pi / 2, n_lat)
    lon = np.linspace(0, 2*np.pi, 200)
    for phi in lat_vals:
        x = np.cos(phi) * np.cos(lon)
        y = np.cos(phi) * np.sin(lon)
        z = np.sin(phi) * np.ones_like(lon)
        traces.append(go.Scatter3d(
            x=x, y=y, z=z, mode="lines",
            line=dict(color="lightgrey", width=1),
            hoverinfo="skip", showlegend=False
        ))
    lon_vals = np.linspace(0, 2*np.pi, n_lon, endpoint=False)
    lat = np.linspace(-np.pi/2, np.pi/2, 200)
    for lam in lon_vals:
        x = np.cos(lat) * np.cos(lam)
        y = np.cos(lat) * np.sin(lam)
        z = np.sin(lat)
        traces.append(go.Scatter3d(
            x=x, y=y, z=z, mode="lines",
            line=dict(color="lightgrey", width=1),
            hoverinfo="skip", showlegend=False
        ))
    return traces


def _segments3d(a: np.ndarray, b: np.ndarray):
    """
    Build a single polyline-with-gaps from pairs of points.
    a, b: [M,3]
    returns x,y,z arrays with None separators for Plotly.
    """
    xs, ys, zs = [], [], []
    for i in range(a.shape[0]):
        xs += [float(a[i,0]), float(b[i,0]), None]
        ys += [float(a[i,1]), float(b[i,1]), None]
        zs += [float(a[i,2]), float(b[i,2]), None]
    return xs, ys, zs


# ----------------------------
# Reference kernel + dynamic Sinkhorn on a chain
# ----------------------------
def gaussian_kernel_matrix(x: torch.Tensor, y: torch.Tensor, eps: float):
    """
    K(i,j) = exp(-||x_i - y_j||^2 / eps)
    x: [M, d], y: [N, d]
    """
    # squared distances via (x-y)^2 = x^2 + y^2 - 2 x.y
    x2 = (x**2).sum(dim=1, keepdim=True)          # [M,1]
    y2 = (y**2).sum(dim=1, keepdim=True).T        # [1,N]
    xy = x @ y.T                                   # [M,N]
    dist2 = (x2 + y2 - 2*xy).clamp(min=0.0)
    K = torch.exp(-dist2 / eps)
    return K


@torch.no_grad()
def dynamic_sinkhorn_chain(
    nodes_list,     # list of [M_t, d]
    mu_list,        # list of [M_t] (targets, must sum to 1 each)
    eps_kernel=0.2, # kernel temperature for K_t
    n_iters=200,
    update_times=None,  # list of times to enforce; default all
    store_every=1,
    tiny=1e-12
):
    """
    Chain graph IPFP / Sinkhorn over time slices:
      candidate path density ~ prod_t u_t(x_t) K_t(x_t, x_{t+1})

    Uses message passing to compute implied marginals:
      alpha_{t+1} = (alpha_t * u_t) @ K_t
      beta_t      = K_t @ (u_{t+1} * beta_{t+1})
      m_t         ∝ u_t * alpha_t * beta_t

    Then updates:
      u_t <- u_t * (mu_t / m_t)

    Returns history dict for visualization.
    """
    T = len(nodes_list) - 1
    assert len(mu_list) == T + 1

    device = nodes_list[0].device
    d = nodes_list[0].shape[1]

    # Make sure mu's are normalized
    mu_list = [m / (m.sum() + tiny) for m in mu_list]

    # Precompute reference kernels K_t
    K_list = []
    for t in range(T):
        K_list.append(gaussian_kernel_matrix(nodes_list[t], nodes_list[t+1], eps_kernel).clamp(min=tiny))

    # Which times to enforce
    if update_times is None:
        update_times = list(range(T+1))

    # Initialize u_t
    u_list = [torch.ones_like(mu_list[t]) for t in range(T+1)]

    # History (stored on CPU)
    hist = {
        "u": [],            # list over stored iters of list[u_t np]
        "marg": [],         # list over stored iters of list[m_t np]
        "resid_l1": [],     # [stored_iters, T+1]
        "ybar": [],         # list over stored iters of list[ybar_t np] for t=0..T-1, each [M_t, d]
    }

    def forward_messages(u_list):
        alpha = [None]*(T+1)
        alpha[0] = torch.ones_like(mu_list[0])  # base measure for visualization; constraints handled via updates
        for t in range(T):
            # alpha_{t+1}(j) = sum_i alpha_t(i) * u_t(i) * K_t(i,j)
            alpha[t+1] = (alpha[t] * u_list[t]) @ K_list[t]
        return alpha

    def backward_messages(u_list):
        beta = [None]*(T+1)
        beta[T] = torch.ones_like(mu_list[T])
        for t in reversed(range(T)):
            # beta_t(i) = sum_j K_t(i,j) * u_{t+1}(j) * beta_{t+1}(j)
            beta[t] = K_list[t] @ (u_list[t+1] * beta[t+1])
        return beta

    def implied_marginals(u_list, alpha, beta):
        marg = []
        for t in range(T+1):
            m = (u_list[t] * alpha[t] * beta[t]).clamp(min=tiny)
            m = m / m.sum()
            marg.append(m)
        return marg

    def barycentric_push(u_list, beta):
        """
        For each transition t, define the current Doob-tilted kernel:
          Q_t(i,j) = K_t(i,j) * u_{t+1}(j) * beta_{t+1}(j) / beta_t(i)

        Then ybar_t(i) = sum_j Q_t(i,j) * node_{t+1}[j]
        """
        ybar_list = []
        for t in range(T):
            # weights on columns (j)
            col_w = (u_list[t+1] * beta[t+1]).clamp(min=tiny)          # [M_{t+1}]
            numer = K_list[t] * col_w.unsqueeze(0)                     # [M_t, M_{t+1}]
            denom = beta[t].clamp(min=tiny).unsqueeze(1)               # [M_t,1]
            Q = numer / denom                                          # row-stochastic (approximately)
            ybar = Q @ nodes_list[t+1]                                 # [M_t, d]
            ybar_list.append(ybar)
        return ybar_list

    # Main loop
    for it in range(n_iters+1):
        alpha = forward_messages(u_list)
        beta  = backward_messages(u_list)
        marg  = implied_marginals(u_list, alpha, beta)

        # Residuals
        resid = torch.stack([(marg[t] - mu_list[t]).abs().sum() for t in range(T+1)], dim=0)

        # Store
        if it % store_every == 0:
            ybar_list = barycentric_push(u_list, beta)
            hist["u"].append([u.detach().cpu().numpy() for u in u_list])
            hist["marg"].append([m.detach().cpu().numpy() for m in marg])
            hist["resid_l1"].append(resid.detach().cpu().numpy())
            hist["ybar"].append([yb.detach().cpu().numpy() for yb in ybar_list])

        # Update u_t to enforce marginals (skip if not constrained)
        for t in update_times:
            m = marg[t].clamp(min=tiny)
            u_list[t] = (u_list[t] * (mu_list[t] / m)).clamp(min=tiny)

        # Optional: mild renorm to prevent explosion (keeps scale moderate)
        for t in range(T+1):
            u_list[t] = u_list[t] / (u_list[t].mean() + tiny)

    hist["resid_l1"] = np.stack(hist["resid_l1"], axis=0)  # [I, T+1]
    return hist


# ----------------------------
# Widget: visualize iterations + chosen transition step
# ----------------------------
class SinkhornChain3DWidget:
    """
    Visualize dynamic Sinkhorn iterations on a time-chain.

    - nodes_list: list of [M_t,3] supports for each time slice t=0..T
    - mu_list: list of [M_t] target marginals on those supports
    - hist: output of dynamic_sinkhorn_chain()
    """

    def __init__(
        self,
        nodes_list,
        mu_list,
        hist,
        title="Dynamic Sinkhorn on a Chain (3D)",
        point_size=4,
        show_targets=True,
        max_arrows=80,
        arrow_width=3,
        colors_time=None,
    ):
        self.nodes_list = [x.detach().cpu().numpy() for x in nodes_list]
        self.mu_list = [m.detach().cpu().numpy() for m in mu_list]
        self.hist = hist
        self.I = len(hist["u"])  # stored iterations
        self.T = len(nodes_list) - 1

        self.point_size = point_size
        self.show_targets = show_targets
        self.max_arrows = max_arrows
        self.arrow_width = arrow_width

        if colors_time is None:
            # time slice colors
            palette = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b"]
            self.colors_time = (palette * ((self.T+2)//len(palette)))[:self.T+1]
        else:
            self.colors_time = colors_time

        # Precompute metrics figure arrays
        self.resid = self.hist["resid_l1"]  # [I, T+1]
        self.it_axis = np.arange(self.I)

        # Init figures + widgets
        self._init_sphere_fig(title)
        self._init_metrics_fig()
        self._init_widgets()

        self.layout = widgets.HBox([
            self.sphere_fig,
            widgets.VBox([self.controls_box, self.metrics_fig])
        ])
        display(self.layout)

    def _init_sphere_fig(self, title):
        self.sphere_fig = go.FigureWidget()

        # Wireframe
        for tr in _sphere_wireframe_traces():
            self.sphere_fig.add_trace(tr)
        self._wire_count = len(self.sphere_fig.data)

        # Initial indices
        it0 = 0
        t0 = 0
        self._current_t = t0

        x_t = self.nodes_list[t0]
        x_tp1 = self.nodes_list[t0+1]

        # subsample arrows
        idx = np.arange(x_t.shape[0])
        if idx.shape[0] > self.max_arrows:
            idx = np.random.choice(idx, size=self.max_arrows, replace=False)
        idx = np.sort(idx)

        ybar = self.hist["ybar"][it0][t0]          # [M_t,3]
        a = x_t[idx]
        b = ybar[idx]

        segx, segy, segz = _segments3d(a, b)

        # Time-t support points
        self.sphere_fig.add_trace(go.Scatter3d(
            x=x_t[:,0], y=x_t[:,1], z=x_t[:,2],
            mode="markers",
            marker=dict(size=self.point_size, color=self.colors_time[t0]),
            name=f"nodes t={t0}"
        ))

        # Time-(t+1) support points
        self.sphere_fig.add_trace(go.Scatter3d(
            x=x_tp1[:,0], y=x_tp1[:,1], z=x_tp1[:,2],
            mode="markers",
            marker=dict(size=self.point_size, color=self.colors_time[t0+1]),
            name=f"nodes t={t0+1}"
        ))

        # Arrows / flow segments
        self.sphere_fig.add_trace(go.Scatter3d(
            x=segx, y=segy, z=segz,
            mode="lines",
            line=dict(color="black", width=self.arrow_width),
            hoverinfo="skip",
            name="barycentric push"
        ))

        self._idx_nodes_t = self._wire_count
        self._idx_nodes_tp1 = self._wire_count + 1
        self._idx_arrows = self._wire_count + 2
        self._arrow_idx_cache = idx

        self.sphere_fig.update_layout(
            title=title,
            width=700, height=700,
            scene=dict(
                aspectmode="data",
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
            ),
            margin=dict(l=10, r=10, t=50, b=10),
            showlegend=False
        )

    def _init_metrics_fig(self):
        self.metrics_fig = go.FigureWidget(make_subplots(
            rows=2, cols=1,
            subplot_titles=("L1 residual by time-slice", "max residual across slices"),
            vertical_spacing=0.18
        ))

        # Residual curves per time slice
        for t in range(self.T+1):
            self.metrics_fig.add_trace(go.Scatter(
                x=self.it_axis, y=self.resid[:, t],
                mode="lines",
                name=f"t={t}"
            ), row=1, col=1)

        # Moving dot (one per time slice) — we’ll draw ONE dot for max, and one for selected t
        self.metrics_fig.add_trace(go.Scatter(
            x=[0], y=[float(self.resid[0].max())],
            mode="markers",
            marker=dict(size=10, color="red"),
            showlegend=False
        ), row=2, col=1)

        # Max curve
        self.metrics_fig.add_trace(go.Scatter(
            x=self.it_axis, y=self.resid.max(axis=1),
            mode="lines",
            name="max_t"
        ), row=2, col=1)

        self.metrics_fig.update_layout(
            width=650, height=520,
            margin=dict(l=20, r=20, t=50, b=20),
            showlegend=True
        )
        self.metrics_fig.update_xaxes(title_text="Stored iteration index", row=1, col=1)
        self.metrics_fig.update_xaxes(title_text="Stored iteration index", row=2, col=1)

    def _init_widgets(self):
        # Iteration play/slider
        self.play = widgets.Play(
            value=0, min=0, max=self.I-1, step=1,
            interval=80, description="Play", show_repeat=False
        )
        self.iter_slider = widgets.IntSlider(
            value=0, min=0, max=self.I-1,
            description="Iter:", continuous_update=False,
            layout=widgets.Layout(width="320px")
        )
        widgets.jslink((self.play, "value"), (self.iter_slider, "value"))

        # Transition step selector
        self.t_slider = widgets.IntSlider(
            value=0, min=0, max=self.T-1,
            description="Step t→t+1:",
            continuous_update=False,
            layout=widgets.Layout(width="320px")
        )

        self.iter_slider.observe(self._update_frame, names="value")
        self.t_slider.observe(self._update_frame, names="value")

        self.controls_box = widgets.VBox([
            widgets.HBox([self.play, self.iter_slider]),
            self.t_slider
        ])

    def _update_frame(self, change):
        it = int(self.iter_slider.value)
        t = int(self.t_slider.value)

        x_t = self.nodes_list[t]
        x_tp1 = self.nodes_list[t+1]

        # keep a fixed subset of arrows for stability when switching iterations
        idx = self._arrow_idx_cache
        if idx.max() >= x_t.shape[0]:
            idx = np.arange(min(x_t.shape[0], self.max_arrows))

        ybar = self.hist["ybar"][it][t]  # [M_t,3]
        a = x_t[idx]
        b = ybar[idx]
        segx, segy, segz = _segments3d(a, b)

        # Update plots
        with self.sphere_fig.batch_update():
            self.sphere_fig.data[self._idx_nodes_t].x = x_t[:,0]
            self.sphere_fig.data[self._idx_nodes_t].y = x_t[:,1]
            self.sphere_fig.data[self._idx_nodes_t].z = x_t[:,2]

            self.sphere_fig.data[self._idx_nodes_tp1].x = x_tp1[:,0]
            self.sphere_fig.data[self._idx_nodes_tp1].y = x_tp1[:,1]
            self.sphere_fig.data[self._idx_nodes_tp1].z = x_tp1[:,2]

            self.sphere_fig.data[self._idx_arrows].x = segx
            self.sphere_fig.data[self._idx_arrows].y = segy
            self.sphere_fig.data[self._idx_arrows].z = segz

        # Update metrics max-dot
        with self.metrics_fig.batch_update():
            maxr = float(self.resid[it].max())
            # data index: residual lines (T+1 traces) then max-dot then max-line
            idx_maxdot = (self.T + 1)
            self.metrics_fig.data[idx_maxdot].x = [it]
            self.metrics_fig.data[idx_maxdot].y = [maxr]


# ----------------------------
# Example usage (replace with your own supports / marginals)
# ----------------------------
def random_points_on_sphere(N: int, d: int = 3):
    x = torch.randn(N, d)
    x = x / (x.norm(dim=-1, keepdim=True) + 1e-9)
    return x

# Build a 3-slice chain (t=0,1,2). You can increase slices.
device = torch.device("cpu")
M0, M1, M2 = 120, 120, 120
nodes0 = random_points_on_sphere(M0, 3).to(device)
nodes1 = random_points_on_sphere(M1, 3).to(device)
nodes2 = random_points_on_sphere(M2, 3).to(device)
nodes_list = [nodes0, nodes1, nodes2]

# Make target marginals with mild concentration (toy)
def soft_cap(mu, power=1.0):
    mu = mu.clamp(min=1e-12)
    mu = mu**power
    return mu / mu.sum()

# Example: prefer +z at t=0, prefer -z at t=2, uniform at t=1
mu0 = soft_cap((nodes0[:,2] + 1.1), power=3.0)
mu1 = torch.ones(M1, device=device) / M1
mu2 = soft_cap((-nodes2[:,2] + 1.1), power=3.0)
mu_list = [mu0, mu1, mu2]

hist = dynamic_sinkhorn_chain(
    nodes_list, mu_list,
    eps_kernel=0.35,
    n_iters=250,
    store_every=1,
    update_times=[0,1,2]  # enforce all slices; set [0,2] for endpoints-only
)

widget = SinkhornChain3DWidget(
    nodes_list, mu_list, hist,
    title="Dynamic Sinkhorn (chain) — barycentric push of Q_t^{(n)}",
    point_size=3,
    max_arrows=60,
    arrow_width=3
)
