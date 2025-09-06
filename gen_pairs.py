"""
Generate paired images for VP metric using a trained experiment directory.

Example:
    python gen_pairs.py --exp /path/to/exp_dir --cuda
"""

import argparse
import json
import os
import os.path as osp
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib import *  # brings in GAN_WEIGHTS, GAN_RESOLUTIONS, WavePDE, etc.
from models.gan_load import (
    build_biggan, build_proggan, build_stylegan2, build_stylegan2mps, build_sngan
)

# ------------------
# Helpers
# ------------------

class ModelArgs:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def sample_z(batch_size, dim_z, device, truncation=None):
    """Sample latent z with optional truncation."""
    if truncation is None or truncation == 1.0:
        return torch.randn(batch_size, dim_z, device=device)
    else:
        from scipy.stats import truncnorm
        z_np = truncnorm.rvs(-truncation, truncation, size=(batch_size, dim_z))
        return torch.from_numpy(z_np).to(device=device, dtype=torch.float32)

def build_gan(gan_type, target_classes, stylegan2_resolution, shift_in_w_space, device, use_mps):
    # BigGAN
    if gan_type == 'BigGAN':
        G = build_biggan(
            pretrained_gan_weights=GAN_WEIGHTS[gan_type]['weights'][GAN_RESOLUTIONS[gan_type]],
            target_classes=target_classes
        )
    # ProgGAN
    elif gan_type == 'ProgGAN':
        G = build_proggan(
            pretrained_gan_weights=GAN_WEIGHTS[gan_type]['weights'][GAN_RESOLUTIONS[gan_type]]
        )
    # StyleGAN2
    elif gan_type == 'StyleGAN2':
        if use_mps:
            G = build_stylegan2mps(
                pretrained_gan_weights=GAN_WEIGHTS[gan_type]['weights'][stylegan2_resolution],
                resolution=stylegan2_resolution,
                shift_in_w_space=shift_in_w_space
            )
        else:
            G = build_stylegan2(
                pretrained_gan_weights=GAN_WEIGHTS[gan_type]['weights'][stylegan2_resolution],
                resolution=stylegan2_resolution,
                shift_in_w_space=shift_in_w_space
            )
    # SNGAN family
    else:
        G = build_sngan(
            pretrained_gan_weights=GAN_WEIGHTS[gan_type]['weights'][GAN_RESOLUTIONS[gan_type]],
            gan_type=gan_type
        )

    G = G.to(device).eval()
    return G

def load_support_sets(exp_models_dir, device):
    """
    Load WavePDE from checkpoint. We’ll:
      1) Read args.json to get K, T.
      2) Instantiate WavePDE(K, T, D) after we create G (so we know dim_z).
      3) Load weights from checkpoint dict (robust to a few key layouts).
    """
    # Read args.json
    args_json_file = osp.join(osp.dirname(exp_models_dir), 'args.json')
    if not osp.isfile(args_json_file):
        raise FileNotFoundError(f"File not found: {args_json_file}")
    a = ModelArgs(**json.load(open(args_json_file)))

    # Choose checkpoint
    ckpt_path = osp.join(exp_models_dir, 'checkpoint.pt')
    if not osp.isfile(ckpt_path):
        # fall back to last support_sets-*.pt
        cands = sorted([f for f in os.listdir(exp_models_dir) if f.startswith('support_sets-')])
        if not cands:
            raise FileNotFoundError(f"No checkpoint found in {exp_models_dir}")
        ckpt_path = osp.join(exp_models_dir, cands[-1])

    ckpt = torch.load(ckpt_path, map_location=device)

    return a, ckpt, ckpt_path

def robust_load_waves(S: nn.Module, ckpt):
    """
    Try a few common layouts to load WavePDE weights from checkpoint.
    """
    sd = None
    if isinstance(ckpt, dict):
        if 'support_sets' in ckpt and isinstance(ckpt['support_sets'], dict):
            sd = ckpt['support_sets']
        elif 'state_dict' in ckpt and isinstance(ckpt['state_dict'], dict):
            # if state_dict looks like WavePDE already
            if any(k.startswith('MLP_SET') or k == 'c' for k in ckpt['state_dict'].keys()):
                sd = ckpt['state_dict']
        elif all(isinstance(k, str) for k in ckpt.keys()):
            # checkpoint is the state_dict itself
            if any(k.startswith('MLP_SET') or k == 'c' for k in ckpt.keys()):
                sd = ckpt
    if sd is None:
        raise RuntimeError("Could not find WavePDE weights in checkpoint. Expected keys like 'support_sets' or 'MLP_SET.*'")
    S.load_state_dict(sd, strict=True)

# ------------------
# Main
# ------------------

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Generate paired images for VP metric')
    p.add_argument('--exp', type=str, required=True, help="experiment dir (created by train.py)")
    p.add_argument('--shift-steps', type=int, default=16, help="# shifts per direction (unused for PDE rollout)")
    p.add_argument('--eps', type=float, default=0.2, help="shift magnitude (unused for PDE rollout)")
    p.add_argument('--shift-leap', type=int, default=1, help="frame stride for saving (unused here)")
    p.add_argument('--batch-size', type=int, default=2, help="generator batch size")
    p.add_argument('--img-size', type=int, default=256, help="saved image size (resized)")
    p.add_argument('--img-quality', type=int, default=75, help="JPEG quality")
    p.add_argument('--gif', action='store_true', help="(unused)")
    p.add_argument('--gif-size', type=int, default=256)
    p.add_argument('--gif-fps', type=int, default=30)
    # Device flags to mirror train.py
    p.add_argument('--cuda', dest='cuda', action='store_true', help="use CUDA")
    p.add_argument('--no-cuda', dest='cuda', action='store_false', help="no CUDA")
    p.add_argument('--mps', dest='mps', action='store_true', help="use MPS")
    p.add_argument('--no-mps', dest='mps', action='store_false', help="no MPS")
    p.set_defaults(cuda=False, mps=True)

    args = p.parse_args()

    # Device selection
    cuda_avail = torch.cuda.is_available()
    mps_avail = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    use_cuda = args.cuda and cuda_avail
    use_mps = args.mps and mps_avail
    device = torch.device('cuda' if use_cuda else ('mps' if use_mps else 'cpu'))

    # Resolve paths
    if not osp.isdir(args.exp):
        raise NotADirectoryError(f"Invalid experiment directory: {args.exp}")
    models_dir = osp.join(args.exp, 'models')
    if not osp.isdir(models_dir):
        raise NotADirectoryError(f"Invalid models directory: {models_dir}")

    # Load args + checkpoint metadata
    a, ckpt, ckpt_path = load_support_sets(models_dir, device)

    # Build generator (aligned with train.py)
    G = build_gan(
        gan_type=a.__dict__['gan_type'],
        target_classes=a.__dict__.get('biggan_target_classes', None),
        stylegan2_resolution=a.__dict__.get('stylegan2_resolution', 1024),
        shift_in_w_space=a.__dict__.get('shift_in_w_space', False),
        device=device,
        use_mps=use_mps
    ).eval()

    # Instantiate WavePDE with D = G.dim_z (same as train.py), then load weights
    S = WavePDE(
        num_support_sets=a.__dict__['num_support_sets'],
        num_support_timesteps=a.__dict__['num_support_timesteps'],
        support_vectors_dim=G.dim_z
    ).to(device).eval()
    robust_load_waves(S, ckpt)

    # IMPORTANT: do NOT mutate activations differently from training.
    # (The old script forced Identity for StyleGAN2; that would mismatch trained weights.)

    # Output directory
    out_dir = osp.join(args.exp, 'vp_pairs')
    os.makedirs(out_dir, exist_ok=True)

    # Pair generation config
    n_samples = 40_000
    B = int(args.batch_size)
    assert n_samples % B == 0, "Choose batch-size that divides n_samples"
    n_batches = n_samples // B

    # PDE rollout length: match training (half_range = T // 2)
    half_range = S.num_support_timesteps // 2

    # Truncation (if present in args.json)
    z_trunc = a.__dict__.get('z_truncation', None)

    all_labels = []

    for i in range(n_batches):
        print(f'Generating image pairs {i+1}/{n_batches} ...')

        # Sample batch z on device, with truncation if specified
        z0 = sample_z(B, G.dim_z, device=device, truncation=z_trunc)

        # Choose ONE potential index per pair (keep same across the mini-batch to simplify labels)
        k = int(torch.randint(0, S.num_support_sets, (1,), device=device).item())

        # Optionally move to W space for StyleGAN2
        if a.__dict__.get('shift_in_w_space', False) and hasattr(G, 'get_w'):
            with torch.no_grad():
                z_cur = G.get_w(z0)
        else:
            z_cur = z0

        # Rollout by PDE: latent_{t+1} = latent_t + ∇_z u(latent_t, t)
        with torch.no_grad():
            for step in range(half_range):
                t_b = torch.full((B, 1), float(step), device=device, dtype=z_cur.dtype)
                _, dz = S.inference(k, z_cur, t_b)  # returns (u, ∇u)
                z_cur = z_cur + dz

        # One-hot labels for VP
        nz = S.num_support_sets
        label = np.zeros((B, nz), dtype=np.float32)
        label[:, k] = 1.0
        all_labels.append(label)

        # Generate images
        with torch.no_grad():
            img1 = G(z0)
            img2 = G(z_cur)

        # Resize to requested output size
        if args.img_size is not None:
            img1 = F.interpolate(img1, size=(args.img_size, args.img_size), mode="bilinear", align_corners=False)
            img2 = F.interpolate(img2, size=(args.img_size, args.img_size), mode="bilinear", align_corners=False)

        # Save pairs as JPEG
        # Convert from [-1,1] RGB to uint8 BGR for cv2
        img1 = img1.clamp(-1, 1)
        img2 = img2.clamp(-1, 1)
        for j in range(B):
            a1 = img1[j].detach().cpu().numpy().transpose(1, 2, 0)  # HWC, RGB
            a2 = img2[j].detach().cpu().numpy().transpose(1, 2, 0)
            pair = np.concatenate([a1, a2], axis=1)
            pair = ((pair + 1.0) * 127.5).round().astype(np.uint8)
            pair = pair[:, :, ::-1]  # RGB -> BGR
            cv2.imwrite(
                osp.join(out_dir, f'pair_{i * B + j:06d}.jpg'),
                pair,
                [int(cv2.IMWRITE_JPEG_QUALITY), int(args.img_quality)]
            )

    labels = np.concatenate(all_labels, axis=0)
    np.save(osp.join(out_dir, 'labels.npy'), labels)
    print(f"Done. Saved {n_samples} pairs and labels.npy to {out_dir}")