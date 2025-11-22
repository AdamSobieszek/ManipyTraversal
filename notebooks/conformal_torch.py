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
WD = 0.1

# Global Device Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Math & Neural Network Helpers (Numpy for Visualization) ---

def tanh_np(x):
    return np.tanh(x)

def sech2_np(x):
    return 1.0 - np.tanh(x)**2

def get_TF_and_F(X, Y, W, B, U, clip_threshold=15.0):
    """
    Calculates Conformal Transform T_F(x) and Scalar Field F(x).
    Uses Numpy for visualization compatibility.
    """
    shape = X.shape
    pts = np.vstack([X.ravel(), Y.ravel()])
    
    # Forward
    Z = W @ pts + B[:, None]
    H = tanh_np(Z)
    F_val = U @ H
    
    # Gradient \nabla F
    D = sech2_np(Z)
    # grad_F = W^T * (D * u)
    weighted_sensitivities = D * U[:, None]
    Grad = W.T @ weighted_sensitivities
    
    # Conformal Map T_F = Grad / |Grad|^2
    Gx, Gy = Grad[0], Grad[1]
    NormSq = Gx**2 + Gy**2
    epsilon = 1e-6
    NormSq = np.maximum(NormSq, epsilon)
    
    Tx = Gx / NormSq
    Ty = Gy / NormSq
    
    # Soft Clip for visualization
    mag = np.sqrt(Tx**2 + Ty**2)
    scale_factor = np.tanh(mag / clip_threshold) * clip_threshold / (mag + epsilon)
    
    return (Tx * scale_factor).reshape(shape), (Ty * scale_factor).reshape(shape), F_val.reshape(shape)

# --- PyTorch Training Logic ---

# --- PyTorch Training Logic ---

def forward_torch(pts, W, B, U):
    """Simple forward pass for training using PyTorch."""
    # pts: (2, N)
    # W: (Hidden, 2)
    # B: (Hidden,)
    # U: (Hidden,)
    
    # Z = W @ pts + B
    Z = torch.matmul(W, pts) + B.unsqueeze(1)
    H = torch.tanh(Z)
    # output = U @ H
    return torch.matmul(U.unsqueeze(0), H).squeeze(0)

def compute_hessian_penalty(W, B, U, X_train, Y_train):
    """
    Computes the convexity penalty based on the minimum eigenvalue of the Hessian.
    Penalty = mean(ReLU(-lambda_min)^2)
    """
    # We need to compute Hessian w.r.t inputs for each point
    # To avoid autograd issues with slicing, we create fresh leaf tensors for each point.
    
    total_penalty = 0.0
    N = X_train.numel()
    
    # Detach data to use as values for new leaf inputs
    X_flat = X_train.flatten().detach()
    Y_flat = Y_train.flatten().detach()
    
    # We must enable grad to compute derivatives w.r.t p, even if called in no_grad context
    with torch.enable_grad():
        for i in range(N):
            # Create a leaf tensor for input (2, 1)
            # We must use the same device as weights
            p = torch.tensor([[X_flat[i]], [Y_flat[i]]], dtype=torch.float32, device=W.device)
            p.requires_grad_(True)
            
            out = forward_torch(p, W, B, U) # scalar
            
            # First derivative
            # create_graph=True is essential for higher order derivatives
            grad = torch.autograd.grad(out, p, create_graph=True)[0] # (2, 1)
            
            # Second derivatives (Hessian columns)
            # d(grad[0])/dx
            grad_x = torch.autograd.grad(grad[0], p, create_graph=True, retain_graph=True)[0]
            # d(grad[1])/dx
            grad_y = torch.autograd.grad(grad[1], p, create_graph=True, retain_graph=True)[0]
            
            H = torch.cat([grad_x, grad_y], dim=1) # (2, 2)
            
            # Eigenvalues of 2x2 symmetric matrix
            # tr = Hxx + Hyy
            # det = Hxx*Hyy - Hxy*Hyx
            tr = H.trace()
            det = H[0,0]*H[1,1] - H[0,1]*H[1,0]
            
            # min_eig = (tr - sqrt(tr^2 - 4*det)) / 2
            gap = tr**2 - 4*det
            gap = torch.relu(gap) # Ensure non-negative for sqrt
            
            min_eig = (tr - torch.sqrt(gap)) / 2.0
            
            # Penalty: if min_eig < 0, add min_eig^2
            total_penalty += torch.relu(-min_eig)**2
        
    return total_penalty / N

def loss_function_torch(W, B, U, X_train, Y_train, Targets):
    pts = torch.stack([X_train.flatten(), Y_train.flatten()], dim=0).to(DEVICE)
    F_pred = forward_torch(pts, W, B, U)
    
    # MSE Loss
    mse_loss = torch.mean((F_pred - Targets)**2)
    
    # Convexity Penalty
    convexity_penalty = torch.tensor(0.0, device=DEVICE)  # compute_hessian_penalty(W, B, U, X_train, Y_train)
    
    # Total Loss
    # Weight the convexity penalty to enforce it
    beta = 0.1 
    return mse_loss + beta * convexity_penalty, mse_loss, convexity_penalty

def generate_training_stages():
    print("Initializing Training Simulation (PyTorch)...")
    
    # 1. Initialize Weights (The "Saddle" Configuration)
    W_init = np.array([[1.0, 1.0], 
                  [-1.0, 1.0],
                  [-0.5, -1.5]])
    B_init = np.array([0.0, 0.0, 1.0])
    U_init = np.array([1.0, -1.0, 0.5])
    
    # Convert to PyTorch Parameters
    W = torch.nn.Parameter(torch.tensor(W_init, dtype=torch.float32, device=DEVICE))
    B = torch.nn.Parameter(torch.tensor(B_init, dtype=torch.float32, device=DEVICE))
    U = torch.nn.Parameter(torch.tensor(U_init, dtype=torch.float32, device=DEVICE))
    
    # Optimizer
    # Using AdamW with high weight decay as requested
    optimizer = optim.AdamW([W, B, U], lr=LEARNING_RATE, weight_decay=WD)
    
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
    # y = sigmoid(w_true * x) * scale + noise

    w_true = np.array([0.1, 0.8])
    linear_combination = X_raw @ w_true
    Targets_np = 1.0 / (1.0 + np.exp(-linear_combination)) * 5.0 + 0.1 * np.random.randn(N_SAMPLES)
    
    Xt = torch.tensor(Xt_np, dtype=torch.float32, device=DEVICE)
    Yt = torch.tensor(Yt_np, dtype=torch.float32, device=DEVICE)
    Targets = torch.tensor(Targets_np, dtype=torch.float32, device=DEVICE)
    
    stages = []
    
    # Capture initial state
    with torch.no_grad():
        loss_val, mse, conv = loss_function_torch(W, B, U, Xt, Yt, Targets)
        loss_item = loss_val.item()
    
    stages.append({
        "W": W.detach().cpu().numpy().copy(), 
        "B": B.detach().cpu().numpy().copy(), 
        "U": U.detach().cpu().numpy().copy(),
        "epoch": 0, 
        "loss": loss_item
    })
    
    # Training Loop
    for epoch in range(1, TRAINING_EPOCHS + 1):
        # Perform gradient descent steps
        for _ in range(TRAINING_STEPS_PER_EPOCH):
            optimizer.zero_grad()
            loss, mse, conv = loss_function_torch(W, B, U, Xt, Yt, Targets)
            loss.backward()
            optimizer.step()
            
        # Checkpoint
        with torch.no_grad():
            loss_val, mse, conv = loss_function_torch(W, B, U, Xt, Yt, Targets)
            loss_item = loss_val.item()
        
        print(f"  Epoch {epoch}/{TRAINING_EPOCHS} - Total Loss: {loss_item:.4f} (MSE: {mse.item():.4f}, Conv: {conv.item():.4f})")
        
        stages.append({
            "W": W.detach().cpu().numpy().copy(), 
            "B": B.detach().cpu().numpy().copy(), 
            "U": U.detach().cpu().numpy().copy(),
            "epoch": epoch, 
            "loss": loss_item
        })
        
    return stages, Xt_np, Yt_np

# --- Generate the Data ---
STAGES, X_TRAIN, Y_TRAIN = generate_training_stages()

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

# Plot Training Data
ax.scatter(X_TRAIN, Y_TRAIN, c='yellow', s=10, alpha=0.6, label='Training Data', zorder=10)
# ax.legend(loc='upper right')

lc = LineCollection([], colors='cyan', linewidths=0.8, alpha=0.7)
ax.add_collection(lc)
# Coloring scheme: Cyan vertical, Magenta horizontal
lc_colors = ['cyan'] * (len(grid_lines_base)//2) + ['magenta'] * (len(grid_lines_base)//2)
lc.set_color(lc_colors)

title_obj = ax.set_title("", fontsize=16, pad=20, color='white')

# --- Animation Logic ---
# We sequence the animation: 
# For each stage:
#   Show sinusoidal breathing for FRAMES_PER_EPOCH_VIEW
#   Morph to next stage for TRANSITION_FRAMES

def get_animation_state(frame_idx):
    """
    Calculates the exact W, B, U and text for the current frame.
    """
    total_stages = len(STAGES)
    frames_per_block = FRAMES_PER_EPOCH_VIEW + TRANSITION_FRAMES
    
    stage_idx = frame_idx // frames_per_block
    local_frame = frame_idx % frames_per_block
    
    # Clamp to last stage
    if stage_idx >= total_stages:
        stage_idx = total_stages - 1
        local_frame = FRAMES_PER_EPOCH_VIEW # Stay static
    
    current_stage = STAGES[stage_idx]
    next_stage = STAGES[min(stage_idx + 1, total_stages - 1)]
    
    # Are we looking at the static view or transitioning?
    if local_frame < FRAMES_PER_EPOCH_VIEW:
        # Static View (Breathing)
        W, B, U = current_stage["W"], current_stage["B"], current_stage["U"]
        epoch_label = current_stage["epoch"]
        loss_label = current_stage["loss"]
        morph_status = "Observing Geometry"
    else:
        # Transitioning (Morphing weights)
        t_raw = (local_frame - FRAMES_PER_EPOCH_VIEW) / TRANSITION_FRAMES
        t = t_raw * t_raw * (3 - 2 * t_raw) # Smoothstep
        
        W = (1-t)*current_stage["W"] + t*next_stage["W"]
        B = (1-t)*current_stage["B"] + t*next_stage["B"]
        U = (1-t)*current_stage["U"] + t*next_stage["U"]
        
        epoch_label = f"{current_stage['epoch']} \u2192 {next_stage['epoch']}"
        loss_label = (1-t)*current_stage["loss"] + t*next_stage["loss"]
        morph_status = "Backpropagating Convexity Loss..."
        
    return W, B, U, epoch_label, loss_label, morph_status

def update(frame):
    W, B, U, epoch_lbl, loss_lbl, status = get_animation_state(frame)
    
    # Sinusoidal breathing for the grid
    # Pulse represents time 't' in the flow x -> x + t*T_F(x)
    pulse = 0.15 * np.sin(frame * 0.1)
    
    new_segments = []
    for segment in grid_lines_base:
        lx, ly = segment[:, 0], segment[:, 1]
        Tx, Ty, _ = get_TF_and_F(lx, ly, W, B, U)
        
        lx_new = lx + pulse * Tx
        ly_new = ly + pulse * Ty
        
        new_segments.append(np.column_stack([lx_new, ly_new]))
    
    lc.set_segments(new_segments)
    
    # Updated: safer title text (avoid mathtext commands not supported)
    # Only use math mode for bolded main title, the rest as plain text

    title_main = r"$\bf{Training\ Convexity\ Topology}$"
    # Prevent latex parsing of status
    if status == "Observing Geometry":
        status_str = status
    else:
        status_str = status

    # Only bold title, format rest safely as plain non-mathtext
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
