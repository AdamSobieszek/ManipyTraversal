import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
import os
import torch
import torch.nn as nn
import torch.optim as optim

# --- Configuration ---
OUTPUT_DIR = 'animations'
GRID_RES = 25
RANGE = 3.0
FPS = 20

# Animation Timing
FRAMES_PER_EPOCH_VIEW = 20  # How long to linger on a specific training stage (breathing grid)
TRANSITION_FRAMES = 10      # How fast to morph weights between training stages
TRAINING_EPOCHS = 30        # Total "checkpoints" to show
TRAINING_STEPS_PER_EPOCH = 200 # Gradient descent steps between checkpoints
LEARNING_RATE = 0.001
WD = 0.

# Global Device Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Math & Neural Network Helpers (Numpy for Visualization) ---

def tanh_np(x):
    return np.tanh(x)

def sech2_np(x):
    return 1.0 - np.tanh(x)**2

def get_TF_and_F(X, Y, model_state_dict, clip_threshold=15.0):
    """
    Calculates Conformal Transform T_F(x) and Scalar Field F(x).
    Uses Numpy for visualization compatibility.
    The state dict is from a PyTorch module (Linear -> Tanh -> Linear -> Tanh -> Linear).
    """
    # Unpack
    # Layer 1: weight, bias, Layer 2: weight, bias, Layer 3: weight, bias
    W1 = model_state_dict['layers.0.weight']
    B1 = model_state_dict['layers.0.bias']
    W2 = model_state_dict['layers.2.weight']
    B2 = model_state_dict['layers.2.bias']
    W3 = model_state_dict['layers.4.weight']
    B3 = model_state_dict['layers.4.bias']

    def forward_np(pts):
        # Forward through the MLP (NumPy implementation)
        Z1 = W1 @ pts + B1[:, None]
        H1 = np.tanh(Z1)
        Z2 = W2 @ H1 + B2[:, None]
        H2 = np.tanh(Z2)
        Z3 = W3 @ H2 + B3[:, None]
        # Output: shape (1, N)
        return Z3[0]

    # Prepare shape
    shape = X.shape
    pts = np.vstack([X.ravel(), Y.ravel()])

    # Forward
    F_val = forward_np(pts)

    # Gradients via finite differences
    eps = 1e-4
    Tx = np.zeros_like(F_val)
    Ty = np.zeros_like(F_val)
    # Centered differences
    for i, delta, arr in [(0, [eps,0], Tx), (1, [0,eps], Ty)]:
        pts_perturb = pts.copy()
        pts_perturb[i] += delta[i]
        F_p = forward_np(pts_perturb)
        pts_perturb = pts.copy()
        pts_perturb[i] -= delta[i]
        F_m = forward_np(pts_perturb)
        arr[:] = (F_p - F_m) / (2*eps)

    # Conformal: T_F = grad F / |grad F|^2
    Gx = Tx
    Gy = Ty
    NormSq = Gx**2 + Gy**2
    epsilon = 1e-6
    NormSq = np.maximum(NormSq, epsilon)
    Tx_cf = Gx / NormSq
    Ty_cf = Gy / NormSq

    # Soft Clip for visualization
    mag = np.sqrt(Tx_cf**2 + Ty_cf**2)
    scale_factor = np.tanh(mag / clip_threshold) * clip_threshold / (mag + epsilon)
    Tx_cf *= scale_factor
    Ty_cf *= scale_factor

    return Tx_cf.reshape(shape), Ty_cf.reshape(shape), F_val.reshape(shape)

# --- PyTorch MLP Model ---

class BigMLP(nn.Module):
    def __init__(self, in_dim=2, hidden=128, hidden2=64):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden2),
            nn.Tanh(),
            nn.Linear(hidden2, 1)
        )

    def forward(self, x):
        """
        x: (N, 2) or (2, N)
        """
        if x.shape[0] == 2 and x.ndim == 2:
            x = x.T
        return self.layers(x).squeeze(-1)

def compute_hessian_penalty(model, X_train, Y_train):
    """
    (Optional, not used here.)
    """
    # Not implemented for big model in this demo.
    return torch.tensor(0.0, device=DEVICE)

def loss_function_torch(model, X_train, Y_train, Targets):
    pts = torch.stack([X_train.flatten(), Y_train.flatten()], dim=1).to(DEVICE)
    F_pred = model(pts)
    # MSE Loss
    mse_loss = torch.mean((F_pred - Targets) ** 2)
    convexity_penalty = torch.tensor(0.0, device=DEVICE)
    beta = 0.1
    return mse_loss + beta * convexity_penalty, mse_loss, convexity_penalty

def model_state_to_numpy(model):
    return {k: v.detach().cpu().numpy().copy() for k,v in model.state_dict().items()}

def generate_training_stages():
    print("Initializing Training Simulation with BigMLP (PyTorch)...")
    model = BigMLP().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WD)

    # Training Data: Non-isotropic Gaussian
    np.random.seed(42)
    N_SAMPLES = 200

    # Generate X: stretch along one axis
    X_raw = np.random.randn(N_SAMPLES, 2)
    X_raw[:, 0] *= 1.5 # Stretch x-axis
    X_raw[:, 1] *= .5 # Compress y-axis

    # Rotate slightly
    theta = np.radians(30)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    X_raw = X_raw @ R.T

    Xt_np = X_raw[:, 0]
    Yt_np = X_raw[:, 1]

    # Generate Targets Y: scaled sigmoid of linear combination + noise
    w_true = np.array([0.4, 0.8])
    linear_combination = X_raw @ w_true
    Targets_np = 1.0 / (1.0 + np.exp(-linear_combination*2)) * 5.0 + 0.5 * np.random.randn(N_SAMPLES)

    Xt = torch.tensor(Xt_np, dtype=torch.float32, device=DEVICE)
    Yt = torch.tensor(Yt_np, dtype=torch.float32, device=DEVICE)
    Targets = torch.tensor(Targets_np, dtype=torch.float32, device=DEVICE)

    stages = []
    # Capture initial state
    with torch.no_grad():
        loss_val, mse, conv = loss_function_torch(model, Xt, Yt, Targets)
        loss_item = loss_val.item()

    stages.append({
        "state_dict": model_state_to_numpy(model),
        "epoch": 0,
        "loss": loss_item,
    })

    for epoch in range(1, TRAINING_EPOCHS + 1):
        # Perform gradient descent steps
        for _ in range(TRAINING_STEPS_PER_EPOCH):
            optimizer.zero_grad()
            loss, mse, conv = loss_function_torch(model, Xt, Yt, Targets)
            loss.backward() 
            optimizer.step()

        with torch.no_grad():
            loss_val, mse, conv = loss_function_torch(model, Xt, Yt, Targets)
            loss_item = loss_val.item()
        print(f"  Epoch {epoch}/{TRAINING_EPOCHS} - Total Loss: {loss_item:.4f} (MSE: {mse.item():.4f}, Conv: {conv.item():.4f})")
        stages.append({
            "state_dict": model_state_to_numpy(model),
            "epoch": epoch,
            "loss": loss_item,
        })
    return stages, Xt_np, Yt_np, Targets_np

# --- Generate the Data ---
STAGES, X_TRAIN, Y_TRAIN, Y_RATINGS = generate_training_stages()

print(f"Dataset Coverage Verification:")
print(f"  Grid Range: [{-RANGE}, {RANGE}]")
print(f"  X Data Range: [{X_TRAIN.min():.2f}, {X_TRAIN.max():.2f}]")
print(f"  Y Data Range: [{Y_TRAIN.min():.2f}, {Y_TRAIN.max():.2f}]")
if X_TRAIN.min() < -RANGE or X_TRAIN.max() > RANGE or Y_TRAIN.min() < -RANGE or Y_TRAIN.max() > RANGE:
    print("  -> Data extends BEYOND grid limits (clipping may occur in viz)")
else:
    print("  -> Data is contained within grid limits")

# --- Visualization Setup ---
plt.style.use('dark_background')

def generate_grid_lines(xmin, xmax, ymin, ymax, steps):
    lines = []
    # Vertical
    x = np.linspace(xmin, xmax, steps)
    y = np.linspace(ymin, ymax, steps*5)
    for xi in x: lines.append(np.column_stack([np.full_like(y, xi), y]))
    # Horizontal
    y = np.linspace(ymin, ymax, steps)
    x = np.linspace(xmin, xmax, steps*5)
    for yi in y: lines.append(np.column_stack([x, np.full_like(x, yi)]))
    return lines

grid_lines_base = generate_grid_lines(-RANGE, RANGE, -RANGE, RANGE, GRID_RES)

fig, ax = plt.subplots(figsize=(10, 10))
ax.set_xlim(-RANGE, RANGE)
ax.set_ylim(-RANGE, RANGE)
ax.set_aspect('equal')
ax.axis('off')

# Coloring points by their targets
scatter_obj = ax.scatter(X_TRAIN, Y_TRAIN, c=Y_RATINGS, cmap='plasma', s=19, alpha=0.9, label='Training Data', zorder=10)
cbar = plt.colorbar(scatter_obj, ax=ax, pad=0.01, shrink=0.7)
cbar.set_label("Target Ratings", color='white')
cbar.ax.yaxis.set_tick_params(color='white')
plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

lc = LineCollection([], colors='cyan', linewidths=0.8, alpha=0.7)
ax.add_collection(lc)
# Coloring scheme: Cyan vertical, Magenta horizontal
lc_colors = ['cyan'] * (len(grid_lines_base)//2) + ['magenta'] * (len(grid_lines_base)//2)
lc.set_color(lc_colors)

title_obj = ax.set_title("", fontsize=16, pad=20, color='white')

# --- Animation Logic ---
def get_animation_state(frame_idx):
    """
    Calculates the exact MLP state_dict and text for the current frame.
    """
    total_stages = len(STAGES)
    frames_per_block = FRAMES_PER_EPOCH_VIEW + TRANSITION_FRAMES

    stage_idx = frame_idx // frames_per_block
    local_frame = frame_idx % frames_per_block

    if stage_idx >= total_stages:
        stage_idx = total_stages - 1
        local_frame = FRAMES_PER_EPOCH_VIEW

    current_stage = STAGES[stage_idx]
    next_stage = STAGES[min(stage_idx + 1, total_stages - 1)]

    # Morph between two state_dicts
    if local_frame < FRAMES_PER_EPOCH_VIEW:
        sd = current_stage["state_dict"]
        epoch_label = current_stage["epoch"]
        loss_label = current_stage["loss"]
        morph_status = "Observing Geometry"
    else:
        t_raw = (local_frame - FRAMES_PER_EPOCH_VIEW) / TRANSITION_FRAMES
        t = t_raw * t_raw * (3 - 2 * t_raw) # Smoothstep
        # Interpolate all weights
        sd = {}
        for k in current_stage["state_dict"]:
            v0 = current_stage["state_dict"][k]
            v1 = next_stage["state_dict"][k]
            sd[k] = (1-t)*v0 + t*v1
        epoch_label = f"{current_stage['epoch']} \u2192 {next_stage['epoch']}"
        loss_label = (1-t)*current_stage["loss"] + t*next_stage["loss"]
        morph_status = "Backpropagating Convexity Loss..."

    return sd, epoch_label, loss_label, morph_status

def update(frame):
    state_dict, epoch_lbl, loss_lbl, status = get_animation_state(frame)

    pulse = 0.15 * np.sin(frame * 0.1)

    new_segments = []
    for segment in grid_lines_base:
        lx, ly = segment[:, 0], segment[:, 1]
        Tx, Ty, _ = get_TF_and_F(lx, ly, state_dict)
        lx_new = lx + pulse * Tx
        ly_new = ly + pulse * Ty
        new_segments.append(np.column_stack([lx_new, ly_new]))

    lc.set_segments(new_segments)
    # Title
    title_main = r"$\bf{Training\ f_\theta(x)}$"
    status_str = status
    title_str = (
        title_main + "\n"
        + f"Epoch: {epoch_lbl} | Loss: {loss_lbl:.4f}\n"
        + f"{status_str}"
    )
    title_obj.set_text(title_str)
    return lc, title_obj

total_frames = len(STAGES) * (FRAMES_PER_EPOCH_VIEW + TRANSITION_FRAMES)

print(f"Rendering Training Animation ({total_frames} frames)...")
ani = animation.FuncAnimation(fig, update, frames=total_frames, interval=50, blit=False)

save_path = os.path.join(OUTPUT_DIR, 'conformal_training_flow2.mp4')
try:
    ani.save(save_path, writer='ffmpeg', fps=FPS, dpi=100)
    print(f"Saved {save_path}")
except Exception as e:
    print(f"Video save failed: {e}")
    # plt.show() # Disable show in script mode
