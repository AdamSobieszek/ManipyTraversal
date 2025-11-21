import argparse
import base64
import math
import os
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.io as pio
except ImportError:  # pragma: no cover - optional dependency
    go = None
    make_subplots = None
    pio = None

try:
    from PIL import Image
except ImportError:  # pragma: no cover - optional dependency
    Image = None
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# 1. Analytic teacher: disentangled square renderer + bendy latent warp
# ---------------------------------------------------------------------------

def square_from_latent(latent, img_size=32):
    """
    Analytic rendering of a black square on white background.
    latent[:, 0] controls square size, latent[:, 1] controls horizontal center.
    """
    device = latent.device
    B = latent.shape[0]

    size_frac = 0.6 * torch.sigmoid(latent[:, 0:1]) + 0.3  # ~[0.3, 0.9]
    half_size = size_frac * img_size / 2.0

    center_x = img_size * 0.5 + (img_size * 0.25) * torch.tanh(latent[:, 1:2])
    center_y = torch.full_like(center_x, img_size * 0.5)

    ys = torch.arange(img_size, device=device).view(1, img_size, 1).float()
    xs = torch.arange(img_size, device=device).view(1, 1, img_size).float()
    xs = xs.expand(B, -1, -1)
    ys = ys.expand(B, -1, -1)

    cx = center_x.view(B, 1, 1)
    cy = center_y.view(B, 1, 1)
    hs = half_size.view(B, 1, 1)

    inside_x = (xs >= cx - hs) & (xs <= cx + hs)
    inside_y = (ys >= cy - hs) & (ys <= cy + hs)
    mask = inside_x & inside_y

    img = torch.ones(B, img_size, img_size, device=device)
    img[mask] = 0.0
    return img.unsqueeze(1)


def bendy_invertible_transform(z, swirl_strength=1.35, radial_gain=1.35, bend=0.3):
    """
    Smooth, invertible transform that bends the latent grid by:
      1) squashing into a unit disk
      2) applying a monotonic radial warp
      3) swirling angles proportionally to radius^2
      4) adding a gentle sine-based shear.
    The mapping remains bijective because every step is monotonic or a diffeomorphism.
    """
    w = torch.tanh(z)  # keep latent inside (-1, 1)
    x = w[:, 0:1]
    y = w[:, 1:2]

    radius = torch.sqrt(torch.clamp(x * x + y * y, min=1e-8))
    angle = torch.atan2(y, x)

    gain = torch.tensor(radial_gain, device=z.device, dtype=z.dtype)
    max_r = torch.tanh(gain)
    radius_warp = torch.tanh(gain * radius) / max_r  # monotonic radial change

    angle_warp = angle + swirl_strength * radius_warp.pow(2)

    xp = radius_warp * torch.cos(angle_warp)
    yp = radius_warp * torch.sin(angle_warp)

    # Small shear keeps mapping invertible while bending isolines
    x_bent = xp + bend * torch.sin(math.pi * yp)
    y_bent = yp + bend * torch.sin(math.pi * x_bent)

    return torch.cat([x_bent, y_bent], dim=1)


def invert_bendy_transform(
    target,
    num_steps=60,
    step_size=0.5,
    tolerance=1e-5,
):
    """
    Approximate inverse of bendy_invertible_transform via gradient descent.
    target: (B,2) tensor in feature/disentangled space.
    Returns z such that bendy_invertible_transform(z) ~= target.
    """
    device = target.device
    z = torch.atanh(torch.clamp(target, -0.95, 0.95)).detach()
    for _ in range(num_steps):
        z = z.detach().requires_grad_(True)
        warped = bendy_invertible_transform(z)
        diff = warped - target
        loss = (diff * diff).sum(dim=1)
        grad = torch.autograd.grad(loss.sum(), z)[0]
        with torch.no_grad():
            z_next = z - step_size * grad
            if torch.max(torch.abs((z_next - z))) < tolerance:
                z = z_next
                break
            z = z_next
    return z.detach()


def teacher_image(z, img_size=32):
    latent = bendy_invertible_transform(z)
    return square_from_latent(latent, img_size=img_size)


# ---------------------------------------------------------------------------
# 2. Modernized convolutional generator
# ---------------------------------------------------------------------------

class LatentMapper(nn.Module):
    """Small MLP that inflates a 2D latent into a structured feature map seed."""

    def __init__(self, latent_dim, hidden_dim, out_dim):
        super().__init__()
        dims = [latent_dim, hidden_dim, hidden_dim, hidden_dim, out_dim]
        layers = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            if out_dim != dims[-1]:
                layers.append(nn.LayerNorm(out_dim))
                layers.append(nn.SiLU())
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


class ResUpsampleBlock(nn.Module):
    """Bilinear upsample + residual convolutions to keep training stable."""

    def __init__(self, in_ch, out_ch, upsample=True):
        super().__init__()
        self.upsample = upsample
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.skip = (
            nn.Conv2d(in_ch, out_ch, kernel_size=1)
            if in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x):
        identity = x
        if self.upsample:
            x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
            identity = F.interpolate(
                identity, scale_factor=2.0, mode="bilinear", align_corners=False
            )

        x = F.silu(self.conv1(x))
        x = F.silu(self.conv2(x))
        identity = self.skip(identity)
        return (x + identity) * (1.0 / math.sqrt(2.0))


class ConvGenerator(nn.Module):
    """
    Progressive decoder:
      latent -> mapper -> 4x4 seed -> residual upsampling stack -> 1-channel image.
    """

    def __init__(self, latent_dim=2, img_size=32, base_channels=128):
        super().__init__()
        self.img_size = img_size
        self.base_channels = base_channels

        mapped_dim = base_channels * 4 * 4
        self.mapper = LatentMapper(latent_dim, hidden_dim=128, out_dim=mapped_dim)

        self.blocks = nn.ModuleList(
            [
                ResUpsampleBlock(base_channels, base_channels // 2),  # 4 -> 8
                ResUpsampleBlock(base_channels // 2, base_channels // 4),  # 8 -> 16
                ResUpsampleBlock(base_channels // 4, base_channels // 8),  # 16 -> 32
            ]
        )

        self.to_image = nn.Sequential(
            nn.Conv2d(base_channels // 8, base_channels // 8, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_channels // 8, 1, kernel_size=1),
        )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(module.weight, a=0.2, mode="fan_in", nonlinearity="leaky_relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, z):
        mapped = self.mapper(z)
        x = mapped.view(-1, self.base_channels, 4, 4)
        for block in self.blocks:
            x = block(x)
        x = self.to_image(x)
        return torch.sigmoid(x)


# ---------------------------------------------------------------------------
# 3. Training utilities
# ---------------------------------------------------------------------------

def build_dataloader(num_samples, latent_dim, img_size, batch_size):
    z = torch.randn(num_samples, latent_dim)
    with torch.no_grad():
        imgs = teacher_image(z, img_size=img_size)
    dataset = TensorDataset(z, imgs)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def train_model(model, loader, optimizer, scheduler, epochs, device):
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for z_batch, img_batch in loader:
            z_batch = z_batch.to(device)
            img_batch = img_batch.to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                pred = model(z_batch)
                loss = F.mse_loss(pred, img_batch)

            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * z_batch.size(0)

        if scheduler is not None:
            scheduler.step()

        epoch_loss = running_loss / len(loader.dataset)
        print(f"Epoch {epoch + 1:03d}/{epochs} - MSE: {epoch_loss:.6f}")


@torch.no_grad()
def generate_latent_grid(model, device, steps=7, span=2.5):
    grid_lin = torch.linspace(-span, span, steps=steps)
    mesh = torch.stack(torch.meshgrid(grid_lin, grid_lin, indexing="ij"), dim=-1)
    latents = mesh.view(-1, 2).to(device)
    imgs = model(latents).cpu()
    return imgs, steps


def save_grid(imgs, steps, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(steps, steps, figsize=(8, 8))
    for idx, ax in enumerate(axes.flat):
        img = imgs[idx].squeeze().numpy()
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
        img_path = os.path.join(save_dir, f"img_{idx:02d}.jpg")
        plt.imsave(img_path, img, cmap="gray", vmin=0, vmax=1)
    plt.tight_layout()
    plt.show()
    print(f"Saved generated samples to '{save_dir}'.")


# ---------------------------------------------------------------------------
# 4. Plotly visualization utilities
# ---------------------------------------------------------------------------

def sample_l_shape_latents(num_points, low=0.05, high=0.95):
    """Sample normalized (x,y) coordinates inside an L-shaped region."""
    n_vertical = int(num_points * 0.6)
    n_horizontal = num_points - n_vertical

    # Horizontal bar along bottom
    h_x = torch.empty(n_horizontal).uniform_(low, high)
    h_y = torch.empty(n_horizontal).uniform_(low, 0.55)

    # Vertical bar along right side
    v_x = torch.empty(n_vertical).uniform_(0.45, high)
    v_y = torch.empty(n_vertical).uniform_(0.45, high)

    coords = torch.cat(
        [torch.stack([h_x, h_y], dim=1), torch.stack([v_x, v_y], dim=1)],
        dim=0,
    )
    perm = torch.randperm(coords.size(0))
    coords = coords[perm][:num_points]
    return coords


def normalized_to_feature_latent(coords, scale=3.0):
    """
    Map normalized (x,y) in [0,1] to the feature latent expected by square_from_latent.
    y-axis controls square size (latent dim 0), x-axis controls horizontal position (latent dim 1).
    """
    features = torch.zeros_like(coords)
    # dim0 (size latent) uses y-coordinate
    features[:, 0] = (coords[:, 1] - 0.5) * scale
    # dim1 (horizontal latent) uses x-coordinate
    features[:, 1] = (coords[:, 0] - 0.5) * scale
    return features


def colorize_from_coords(coords):
    """Generate hex colors using bilinear interpolation of magenta, blue, yellow palette."""
    bottom_left = np.array([1.0, 0.0, 1.0])
    bottom_right = np.array([0.0, 0.6, 1.0])
    top_right = np.array([1.0, 1.0, 0.25])
    top_left = np.array([0.8, 0.0, 0.8])

    colors = []
    coords_np = coords.cpu().numpy()
    for x, y in coords_np:
        x = float(np.clip(x, 0.0, 1.0))
        y = float(np.clip(y, 0.0, 1.0))
        bottom = bottom_left * (1 - x) + bottom_right * x
        top = top_left * (1 - x) + top_right * x
        rgb = bottom * (1 - y) + top * y
        rgb = np.clip(rgb, 0.0, 1.0)
        rgb255 = (rgb * 255).astype(np.uint8)
        colors.append(f"#{rgb255[0]:02x}{rgb255[1]:02x}{rgb255[2]:02x}")
    return colors


def make_hover_images(true_imgs, recon_imgs):
    """Return base64-encoded side-by-side images for hover display."""
    if Image is None:
        raise ImportError("Pillow is required for hover images. Install via `pip install pillow`.")
    imgs = []
    for gt, recon in zip(true_imgs, recon_imgs):
        gt_np = gt.squeeze().cpu().numpy()
        recon_np = recon.squeeze().cpu().numpy()
        combined = np.concatenate([gt_np, recon_np], axis=1)
        combined = (combined * 255.0).clip(0, 255).astype(np.uint8)
        image = Image.fromarray(combined, mode="L")
        buf = BytesIO()
        image.save(buf, format="PNG")
        imgs.append(base64.b64encode(buf.getvalue()).decode("ascii"))
    return imgs


def add_feature_grid(fig, row, col, x_vals, y_vals):
    for xv in x_vals:
        fig.add_trace(
            go.Scatter(
                x=[xv, xv],
                y=[y_vals[0], y_vals[-1]],
                mode="lines",
                line=dict(color="rgba(50,50,50,0.4)", width=1),
                hoverinfo="skip",
                showlegend=False,
            ),
            row=row,
            col=col,
        )
    for yv in y_vals:
        fig.add_trace(
            go.Scatter(
                x=[x_vals[0], x_vals[-1]],
                y=[yv, yv],
                mode="lines",
                line=dict(color="rgba(50,50,50,0.4)", width=1),
                hoverinfo="skip",
                showlegend=False,
            ),
            row=row,
            col=col,
        )


def add_warped_grid(
    fig,
    row,
    col,
    grid_x_vals,
    grid_y_vals,
    device,
    inv_steps,
    inv_lr,
):
    grid_color = "rgba(50,50,50,0.4)"
    y_samples = torch.linspace(
        float(grid_y_vals[0]),
        float(grid_y_vals[-1]),
        steps=160,
        device=device,
    )
    x_samples = torch.linspace(
        float(grid_x_vals[0]),
        float(grid_x_vals[-1]),
        steps=160,
        device=device,
    )

    for xv in grid_x_vals:
        xv_tensor = torch.full_like(y_samples, float(xv), device=device)
        feats = torch.stack([y_samples, xv_tensor], dim=1)
        z_line = invert_bendy_transform(feats, num_steps=inv_steps, step_size=inv_lr)
        z_np = z_line.cpu().numpy()
        fig.add_trace(
            go.Scatter(
                x=z_np[:, 1],
                y=z_np[:, 0],
                mode="lines",
                line=dict(color=grid_color, width=1),
                hoverinfo="skip",
                showlegend=False,
            ),
            row=row,
            col=col,
        )

    for yv in grid_y_vals:
        yv_tensor = torch.full_like(x_samples, float(yv), device=device)
        feats = torch.stack([yv_tensor, x_samples], dim=1)
        z_line = invert_bendy_transform(feats, num_steps=inv_steps, step_size=inv_lr)
        z_np = z_line.cpu().numpy()
        fig.add_trace(
            go.Scatter(
                x=z_np[:, 1],
                y=z_np[:, 0],
                mode="lines",
                line=dict(color=grid_color, width=1),
                hoverinfo="skip",
                showlegend=False,
            ),
            row=row,
            col=col,
        )


def write_interactive_html(fig, output_path, div_id="bendy-widget"):
    """
    Persist Plotly figure to HTML and inject JS that shows hover images near cursor.
    """
    if pio is None:
        raise ImportError("Plotly is required for widget mode. Install via `pip install plotly`.")

    height = fig.layout.height or 500
    hover_width = int(0.4 * height)
    hover_height = int(0.2 * height)

    hover_js = f"""
const plotDiv = document.getElementById('{div_id}');
const hoverBox = document.createElement('div');
hoverBox.id = 'hover-image-box';
hoverBox.style.position = 'fixed';
hoverBox.style.pointerEvents = 'none';
hoverBox.style.background = 'rgba(0,0,0,0.75)';
hoverBox.style.borderRadius = '4px';
hoverBox.style.padding = '6px';
hoverBox.style.display = 'none';
hoverBox.style.zIndex = '9999';
const hoverImg = document.createElement('img');
hoverImg.style.width = '{hover_width}px';
hoverImg.style.height = '{hover_height}px';
hoverImg.style.display = 'block';
hoverImg.style.objectFit = 'contain';
hoverBox.appendChild(hoverImg);
document.body.appendChild(hoverBox);

plotDiv.on('plotly_hover', function(eventData) {{
    const pt = eventData.points && eventData.points[0];
    if (!pt || !pt.customdata) {{
        hoverBox.style.display = 'none';
        return;
    }}
    hoverImg.src = 'data:image/png;base64,' + pt.customdata;
    const mouseEvent = eventData.event;
    hoverBox.style.left = (mouseEvent.clientX + 20) + 'px';
    hoverBox.style.top = (mouseEvent.clientY - 20) + 'px';
    hoverBox.style.display = 'block';
}});

plotDiv.on('plotly_unhover', function() {{
    hoverBox.style.display = 'none';
}});
"""

    pio.write_html(
        fig,
        file=output_path,
        include_plotlyjs="cdn",
        auto_open=True,
        full_html=True,
        div_id=div_id,
        post_script=hover_js,
    )


def run_plotly_widget(args, device):
    if go is None or make_subplots is None:
        raise ImportError("Plotly is required for widget mode. Install via `pip install plotly`.")
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(
            f"Checkpoint '{args.checkpoint}' not found. Train first or provide a valid path."
        )
    model = ConvGenerator(latent_dim=args.latent_dim, img_size=args.img_size).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()

    coords = sample_l_shape_latents(args.num_widget_points)
    features = normalized_to_feature_latent(coords)
    colors = colorize_from_coords(coords)

    features_device = features.to(device)
    true_imgs = square_from_latent(features_device, img_size=args.img_size).cpu()
    z_latents = invert_bendy_transform(
        features_device,
        num_steps=args.inversion_steps,
        step_size=args.inversion_lr,
    )
    with torch.no_grad():
        recon_imgs = model(z_latents).cpu()

    hover_imgs = make_hover_images(true_imgs, recon_imgs)

    x_feat = features[:, 1].cpu().numpy()
    y_feat = features[:, 0].cpu().numpy()
    x_z = z_latents[:, 1].cpu().numpy()
    y_z = z_latents[:, 0].cpu().numpy()

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Disentangled Feature Space",
            "Generator Latent Space (Z)",
        ),
    )

    grid_x = np.linspace(-1.5, 1.5, 9)
    grid_y = np.linspace(-1.5, 1.5, 9)

    add_feature_grid(fig, 1, 1, grid_x, grid_y)
    add_warped_grid(
        fig,
        1,
        2,
        grid_x,
        grid_y,
        device,
        args.inversion_steps,
        args.inversion_lr,
    )

    hover_template = "x: %{x:.2f}<br>y: %{y:.2f}<extra></extra>"

    for subplot_col, x_vals, y_vals in [
        (1, x_feat, y_feat),
        (2, x_z, y_z),
    ]:
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="markers",
                marker=dict(size=6, color=colors),
                customdata=hover_imgs,
                hovertemplate=hover_template,
                showlegend=False,
            ),
            row=1,
            col=subplot_col,
        )

    fig.update_xaxes(title_text="Horizontal coordinate", row=1, col=1)
    fig.update_yaxes(title_text="Size coordinate", row=1, col=1)
    fig.update_xaxes(title_text="Z dim 2", row=1, col=2)
    fig.update_yaxes(title_text="Z dim 1", row=1, col=2)

    fig.update_layout(
        height=500,
        width=1000,
        title="Interactive Mapping Between Feature Space and Generator Latent Space",
    )
    output_path = os.path.abspath(args.widget_html)
    write_interactive_html(fig, output_path)
    print(f"Wrote interactive widget to '{output_path}' and opened it in your browser.")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a toy generator with bendy latent semantics.")
    parser.add_argument("--mode", choices=["train", "widget"], default="train", help="Run training loop or launch Plotly widget.")
    parser.add_argument("--num-samples", type=int, default=40000, help="Number of synthetic training pairs.")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=2e-3, help="Initial learning rate.")
    parser.add_argument("--latent-dim", type=int, default=2, help="Latent dimensionality.")
    parser.add_argument("--img-size", type=int, default=32, help="Generated image size.")
    parser.add_argument("--output-dir", type=str, default="generated_jpg_examples", help="Directory to store samples.")
    parser.add_argument("--checkpoint", type=str, default="conv_generator.pt", help="Path to save or load generator weights.")
    parser.add_argument("--widget-html", type=str, default="latent_widget.html", help="Output HTML file for widget mode.")
    parser.add_argument("--num-widget-points", type=int, default=600, help="Number of samples to visualize in widget mode.")
    parser.add_argument("--inversion-steps", type=int, default=60, help="Number of steps for latent inverse iterations.")
    parser.add_argument("--inversion-lr", type=float, default=0.5, help="Step size used for inversion iterations.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    if args.mode == "train":
        loader = build_dataloader(
            num_samples=args.num_samples,
            latent_dim=args.latent_dim,
            img_size=args.img_size,
            batch_size=args.batch_size,
        )

        model = ConvGenerator(latent_dim=args.latent_dim, img_size=args.img_size).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            betas=(0.9, 0.99),
            weight_decay=1e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        train_model(model, loader, optimizer, scheduler, args.epochs, device)

        torch.save(model.state_dict(), args.checkpoint)
        print(f"Saved checkpoint to '{args.checkpoint}'.")

        model.eval()
        imgs, steps = generate_latent_grid(model, device, steps=7, span=2.5)
        save_grid(imgs, steps, args.output_dir)
    else:
        run_plotly_widget(args, device)


if __name__ == "__main__":
    main()
