import numpy as np
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ipywidgets as widgets
from IPython.display import display

# --- 1. Helper Functions ---

def mobius_sphere(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """
    Applies the Möbius transformation defined by w to points x.
    Map: f_w(x) = (1 - |w|^2) / |x - w|^2 * (x - w) - w
    """
    w = w.view(1, -1)
    diff = x - w
    w_sq = (w**2).sum(dim=1, keepdim=True)
    diff_sq = (diff**2).sum(dim=1, keepdim=True)
    scale = (1 - w_sq) / (diff_sq + 1e-8)
    return scale * diff - w

# --- 2. The Interactive Widget ---

class KuramotoWidget:
    """
    Interactive Dashboard for 2D Kuramoto/Möbius Dynamics.
    
    Layout:
    [ Left Col: Disk Animation ]  [ Right Col: Time Slider & Metric Plots ]
    """
    def __init__(
        self,
        traj_w: torch.Tensor,
        base_points: torch.Tensor,
        title: str = "2D Kuramoto Dynamics",
        point_size: int = 8,
        w_color: str = "red",
        osc_color: str = "blue",
    ):
        # --- Data Prep ---
        self.traj_w = traj_w.detach().cpu()
        self.base_points = base_points.detach().cpu()
        self.T = self.traj_w.shape[0]
        self.point_size = point_size
        self.w_color = w_color
        self.osc_color = osc_color
        
        # 1. Precompute Simulation State (Positions & Metrics)
        self._precompute_dynamics()
        
        # 2. Build Figures
        self._init_disk_figure(title)
        self._init_metrics_figure()
        
        # 3. Build Widgets & Layout
        self._init_widgets()
        
        # 4. Display Layout
        # HBox( [DiskFigure, VBox([Controls, MetricsFigure])] )
        self.layout = widgets.HBox([
            self.disk_fig,
            widgets.VBox([
                self.controls_box, 
                self.metrics_fig
            ])
        ])
        display(self.layout)

    def _precompute_dynamics(self):
        """Pre-calculates all x_i(t) and metric values for the slider."""
        x_all = []
        metric_w_mag = []
        metric_order = []
        
        for t in range(self.T):
            w_t = self.traj_w[t]
            
            # Dynamics: x(t) = M_{w(t)}(base)
            x_t = mobius_sphere(self.base_points, w_t)
            x_all.append(x_t)
            
            # Metric 1: Magnitude of w
            w_mag = torch.norm(w_t).item()
            metric_w_mag.append(w_mag)
            
            # Metric 2: Order Parameter R = | (1/N) * sum(x_i) |
            centroid = torch.mean(x_t, dim=0)
            r_mag = torch.norm(centroid).item()
            metric_order.append(r_mag)
            
        self.x_all = torch.stack(x_all).numpy() # [T, N, 2]
        self.traj_w_np = self.traj_w.numpy()    # [T, 2]
        
        self.metrics_data = {
            "w_mag": np.array(metric_w_mag),
            "order_p": np.array(metric_order)
        }
        self.time_steps = np.arange(self.T)

    def _init_disk_figure(self, title):
        """Initializes the Left Panel (Poincaré Disk)."""
        self.disk_fig = go.FigureWidget()
        
        # Static: Unit Circle
        theta = np.linspace(0, 2*np.pi, 200)
        self.disk_fig.add_trace(go.Scatter(
            x=np.cos(theta), y=np.sin(theta), mode='lines', 
            line=dict(color='black', width=2), hoverinfo='skip', name="Boundary"
        ))
        
        # Dynamic: Oscillators (Trace 1)
        x0 = self.x_all[0]
        self.disk_fig.add_trace(go.Scatter(
            x=x0[:,0], y=x0[:,1], mode='markers',
            marker=dict(size=self.point_size, color=self.osc_color),
            name="Oscillators"
        ))
        
        # Dynamic: w(t) (Trace 2)
        w0 = self.traj_w_np[0]
        self.disk_fig.add_trace(go.Scatter(
            x=[w0[0]], y=[w0[1]], mode='markers',
            marker=dict(size=self.point_size+4, color=self.w_color, symbol='x'),
            name="w(t)"
        ))
        
        # Dynamic: w path (Trace 3)
        self.disk_fig.add_trace(go.Scatter(
            x=[w0[0]], y=[w0[1]], mode='lines',
            line=dict(color=self.w_color, width=1, dash='dot'),
            name="w path"
        ))

        self.disk_fig.update_layout(
            title=title, width=500, height=500,
            xaxis=dict(range=[-1.1, 1.1], visible=False, scaleanchor="y", scaleratio=1),
            yaxis=dict(range=[-1.1, 1.1], visible=False),
            margin=dict(l=10, r=10, t=40, b=10),
            showlegend=False,
            template="plotly_white"
        )

    def _init_metrics_figure(self):
        """Initializes the Right Panel (Time Series Metrics)."""
        self.metrics_fig = go.FigureWidget(make_subplots(
            rows=2, cols=1, 
            subplot_titles=("Order Parameter R(t)", "Control Magnitude |w(t)|"),
            vertical_spacing=0.15
        ))
        
        # --- Plot 1: Order Parameter ---
        self.metrics_fig.add_trace(go.Scatter(
            x=self.time_steps, y=self.metrics_data["order_p"],
            mode='lines', line=dict(color='purple'), name="R(t)"
        ), row=1, col=1)
        
        self.metrics_fig.add_trace(go.Scatter(
            x=[0], y=[self.metrics_data["order_p"][0]],
            mode='markers', marker=dict(size=10, color='red'), showlegend=False
        ), row=1, col=1)
        
        # --- Plot 2: |w| Magnitude ---
        self.metrics_fig.add_trace(go.Scatter(
            x=self.time_steps, y=self.metrics_data["w_mag"],
            mode='lines', line=dict(color='green'), name="|w(t)|"
        ), row=2, col=1)
        
        self.metrics_fig.add_trace(go.Scatter(
            x=[0], y=[self.metrics_data["w_mag"][0]],
            mode='markers', marker=dict(size=10, color='red'), showlegend=False
        ), row=2, col=1)

        # Layout and Axis Locking
        self.metrics_fig.update_layout(
            width=500, height=500,
            margin=dict(l=20, r=20, t=40, b=20),
            showlegend=False,
            template="plotly_white"
        )
        
        # --- FIX: Explicitly lock X-Axis ranges so they don't move ---
        self.metrics_fig.update_xaxes(title_text="Time", range=[0, self.T], row=1, col=1)
        self.metrics_fig.update_xaxes(title_text="Time", range=[0, self.T], row=2, col=1)
        
        # Optional: Lock Y-axes to reasonable defaults (0 to 1) if desired
        self.metrics_fig.update_yaxes(range=[-0.05, 1.05], row=1, col=1) # R is [0,1]
        self.metrics_fig.update_yaxes(range=[-0.05, 1.05], row=2, col=1) # |w| is < 1

    def _init_widgets(self):
        """Sets up the Play, Pause, and Slider controls."""
        self.play = widgets.Play(
            value=0, min=0, max=self.T-1,
            step=1, interval=50, 
            description="Press play",
            show_repeat=False
        )
        
        self.slider = widgets.IntSlider(
            value=0, min=0, max=self.T-1,
            description="Time:",
            layout=widgets.Layout(width='300px')
        )
        
        widgets.jslink((self.play, 'value'), (self.slider, 'value'))
        self.slider.observe(self._update_frame, names='value')
        
        self.controls_box = widgets.HBox([self.play, self.slider])

    def _update_frame(self, change):
        """Callback: Updates both figures when slider moves."""
        t = change['new']
        
        # 1. Update Disk
        x_t = self.x_all[t]
        w_t = self.traj_w_np[t]
        path_slice = self.traj_w_np[:t+1]
        
        with self.disk_fig.batch_update():
            # Oscillators
            self.disk_fig.data[1].x = x_t[:, 0]
            self.disk_fig.data[1].y = x_t[:, 1]
            # w(t) marker
            self.disk_fig.data[2].x = [w_t[0]]
            self.disk_fig.data[2].y = [w_t[1]]
            # w path
            self.disk_fig.data[3].x = path_slice[:, 0]
            self.disk_fig.data[3].y = path_slice[:, 1]

        # 2. Update Metrics (Only move the red dot)
        with self.metrics_fig.batch_update():
            # R(t) dot (Trace 1)
            self.metrics_fig.data[1].x = [t]
            self.metrics_fig.data[1].y = [self.metrics_data["order_p"][t]]
            
            # |w| dot (Trace 3)
            self.metrics_fig.data[3].x = [t]
            self.metrics_fig.data[3].y = [self.metrics_data["w_mag"][t]]