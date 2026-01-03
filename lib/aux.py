import os
import os.path as osp
import json
import argparse
import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import _LRScheduler
import sys
import math
import time
from scipy.stats import truncnorm
from PIL import Image, ImageDraw



def choose_device() -> torch.device:
        # Device selection
    cuda_available = torch.cuda.is_available()
    mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    device = torch.device('cuda' if cuda_available else ('mps' if mps_available else 'cpu'))

    # Set default tensor type for CUDA only (no MPS default tensor type exists)
    if cuda_available:
        torch.set_default_device(torch.device('cuda'))
    elif mps_available:
        torch.set_default_device(torch.device('mps'))
        torch.set_default_dtype(torch.float32)
    else:
        torch.set_default_device(torch.device('cpu'))

    return device

import math
from torch.optim.lr_scheduler import _LRScheduler

class PhasedCosineWithRestarts(_LRScheduler):
    """
    Cosine LR with warmup + SGDR-style restarts + phase offset (radians).

    For step s:
      if s < warmup_steps:
          lr = base_lr * (s+1)/warmup_steps
      else:
          let u = s - warmup_steps
          cycle_len = T0 * (Tmult ** cycle_idx)
          pos  = (u - sum_prev_cycles) / cycle_len   in [0,1)
          lr   = min_lr + 0.5*(base_lr-min_lr) * (1 + cos(2π*pos + phase))

    Notes:
      • base_lrs are captured from the optimizer's param_groups at construction.
      • min_lr can be scalar or per-group list (length == len(base_lrs)).
      • phase is a scalar in radians (e.g., 0 for support, π for reconstructor).
      • last_epoch is the *number of steps already taken* (PyTorch convention).
    """
    def __init__(self, optimizer,
                 warmup_steps: int,
                 T0: int,
                 Tmult: float = 1.0,
                 min_lr=1e-6,
                 phase: float = 0.0,
                 last_epoch: int = -1):
        self.warmup_steps = int(max(0, warmup_steps))
        self.T0 = int(max(1, T0))
        self.Tmult = float(max(1.0, Tmult))
        self.phase = float(phase)
        self._cycle_boundaries = None  # built lazily
        self.base_lrs = [g['lr'] for g in optimizer.param_groups]

        if isinstance(min_lr, (list, tuple)):
            assert len(min_lr) == len(self.base_lrs), "min_lr list must match number of param groups"
            self.min_lrs = list(map(float, min_lr))
        else:
            self.min_lrs = [float(min_lr)] * len(self.base_lrs)

        super().__init__(optimizer, last_epoch=last_epoch)

    def _locate_cycle(self, u: int):
        # u = steps since warmup (>=0). Return (cycle_idx, pos_in_cycle[0,1), cycle_len)
        if self._cycle_boundaries is None:
            self._cycle_boundaries = []
        # Expand boundaries until u is within range
        total = 0
        c = 0
        while True:
            L = int(round(self.T0 * (self.Tmult ** c)))
            if u < total + L:
                pos = (u - total) / max(1, L)
                return c, pos, L
            total += L
            c += 1

    def get_lr(self):
        s = self.last_epoch  # steps completed
        # Warmup
        if s < self.warmup_steps:
            scale = (s + 1) / max(1, self.warmup_steps)
            return [base * scale for base in self.base_lrs]

        # After warmup
        u = s - self.warmup_steps
        _, pos, _ = self._locate_cycle(u)
        # Cosine with phase
        cos_arg = 2.0 * math.pi * pos + self.phase
        cval = 0.5 * (1.0 + math.cos(cos_arg))
        # Per-group LR
        lrs = []
        for base, minlr in zip(self.base_lrs, self.min_lrs):
            lrs.append(minlr + (base - minlr) * cval)
        return lrs


@torch.no_grad()
def sample_z(batch_size, generator, params, device = torch.device('cuda')):
    """
    Instead of sampling batch_size independent random vectors,
    sample one random vector and generate the rest as an orthonormal basis
    (Gram-Schmidt) to it. If batch_size > generator.dim_z, will pad with zeros.
    """
    dim_z = generator.dim_z

    # Draw one random vector
    z0 = torch.randn(dim_z, device=device)
    z0_norm = z0.norm()
    z0 = z0 / (z0_norm + 1e-8)

    # Create orthonormal basis (including z0 as the first vector)
    basis = [z0]
    for _ in range(1, min(batch_size, dim_z)):
        v = torch.randn(dim_z, device=device)
        # Gram-Schmidt orthogonalization
        for b in basis:
            v = v - (v @ b) * b
        v_norm = v.norm()
        if v_norm < 1e-8:
            # If degenerate, resample
            v = torch.randn(dim_z, device=device)
            for b in basis:
                v = v - (v @ b) * b
            v_norm = v.norm()
            if v_norm < 1e-8:
                v = torch.zeros_like(v)
        else:
            v = v / v_norm
        basis.append(v)
    # Stack basis vectors
    z = torch.stack(basis, dim=0)*z0_norm
    # If batch_size > dim_z, pad with zeros
    if batch_size > dim_z:
        pad = torch.zeros(batch_size - dim_z, dim_z, device=device)
        z = torch.cat([z, pad], dim=0)
    # If batch_size < dim_z, truncate
    if z.shape[0] > batch_size:
        z = z[:batch_size]

    # Move to correct device if needed
    if z.device.type == 'cuda':
        z = z.cuda(non_blocking=True)

    # Optionally shift in w-space and apply truncation
    if getattr(generator, "shift_in_w_space", False):
        z = generator.get_w(z)
        if getattr(params, "z_truncation", None) is not None:
            z_mean = z.mean(dim=0, keepdim=True)
            z = (z - z_mean) * params.z_truncation + z_mean
    else:
        if getattr(params, "z_truncation", None) is not None:
            z = z * params.z_truncation

    return z
       
def create_exp_dir(args, new_experiment=False):
    """Create output directory for current experiment under experiments/wip/ and save given the arguments (json) and
    the given command (bash script).

    Experiment's directory name format:

        <gan_type>(-<stylegan2_resolution>)(-{Z,W})-<reconstructor_type>-K<num_support_sets>-
            D<num_support_dipoles>(-LearnAlphas)(-LearnGammas)-eps<min_shift_magnitude>_<max_shift_magnitude>
    E.g.:

        experiments/wip/ProgGAN-ResNet-K200-N32-LearnGammas-eps0.35_0.5

    Args:
        args (argparse.Namespace): the namespace object returned by `parse_args()` for the current run

    """
    if new_experiment:
        print("Creating new experiment\n"+"-"*30+"\n"*2)
    exp_dir = "{}".format(args.gan_type)
    if args.gan_type == 'StyleGAN2':
        exp_dir += '-{}'.format(args.stylegan2_resolution)
        if args.shift_in_w_space:
            exp_dir += '-W'
        else:
            exp_dir += '-Z'
    if args.gan_type == 'BigGAN':
        biggan_classes = '-'
        for c in args.biggan_target_classes:
            biggan_classes += '{}'.format(c)
        exp_dir += '{}'.format(biggan_classes)
    exp_dir += "-{}".format(args.reconstructor_type)
    exp_dir += "-K{}-D{}".format(args.num_support_sets, args.num_support_timesteps)
    if new_experiment:
        exp_dir += f"__{time.strftime('%Y%m%d_%H%M%S')}"
    else:
        # grep all folders that start with exp_dir
        exp_dirs = [d for d in os.listdir("experiments/wip") if d.startswith(exp_dir)]
        # exclude folders that do not contain checkpoint.pt as a file in their recursive folder structure
        exp_dirs = [d for d in exp_dirs if osp.isfile(osp.join("experiments/wip", d, "models", "checkpoint.pt"))]
        # sort by last modified time
        exp_dirs.sort(key=lambda x: os.path.getmtime(osp.join("experiments/wip", x)))
        #  set exp_dir to the newest folder
        exp_dir = exp_dirs[-1]
        print(f"Using existing experiment: {exp_dir}\n"+"-"*30+"\n"*2)
    # Create output directory (wip)
    wip_dir = osp.join("experiments", "wip", exp_dir)
    os.makedirs(wip_dir, exist_ok=True)
    # Save args namespace object in json format
    with open(osp.join(wip_dir, 'args.json'), 'w') as args_json_file:
        json.dump(args.__dict__, args_json_file)

    # Save the given command in a bash script file
    with open(osp.join(wip_dir, 'command.sh'), 'w') as command_file:
        command_file.write('#!/usr/bin/bash\n')
        command_file.write(' '.join(sys.argv) + '\n')

    return exp_dir
import torch
from torch import nn
from typing import Iterable, Sequence, Union, Optional

# -----------------------------
# Hyperparameter helper function
# --------------------------------

def _norm_classes():
    return (
        nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm,
        nn.LayerNorm, nn.GroupNorm,
        nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d,
        nn.LocalResponseNorm,
    )


def split_weight_decay_groups(
    model: nn.Module,
    weight_decay: float,
    extra_no_decay_names: Sequence[str] = (),
    extra_no_decay_params: Sequence[Optional[torch.nn.Parameter]] = (),
    include_1d_as_no_decay: bool = True,
):
    """
    Build AdamW param groups:
      - Decay: 'true' weights.
      - No-decay: biases, norm params, explicitly provided params, and (optionally) 1D tensors.
    Uses PARAM IDENTITY to ensure correct bucketing.

    Args:
        model: the module to scan for parameters.
        weight_decay: WD to apply to the decay group.
        extra_no_decay_names: param names or suffixes to exclude from WD (exact or ".suffix" match).
        extra_no_decay_params: explicit Parameter objects to exclude (by identity).
        include_1d_as_no_decay: if True, any 1D tensor (e.g., LayerNorm/Bias) goes to no-decay.

    Returns:
        A list of param-group dicts suitable for torch.optim.AdamW.
    """
    norm_types = _norm_classes()

    # 1) Collect no-decay by identity (explicit params)
    no_decay_ids = set(id(p) for p in extra_no_decay_params if p is not None)

    # 2) Add norm layer params by identity
    for m in model.modules():
        if isinstance(m, norm_types):
            for p in m.parameters(recurse=False):
                no_decay_ids.add(id(p))

    # 3) Name-based rules (bias and explicit suffix matches)
    name_rules = set(extra_no_decay_names or ())

    def name_is_extra(n: str) -> bool:
        # exact or endswith(".name")
        return (n in name_rules) or any(n.endswith(f".{x}") for x in name_rules)

    # 4) Final bucketing
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if (
            id(p) in no_decay_ids
            or n.endswith("bias")
            or name_is_extra(n)
            or (include_1d_as_no_decay and p.ndim == 1)
        ):
            no_decay.append(p)
        else:
            decay.append(p)

    groups = []
    if decay:
        groups.append({"params": decay, "weight_decay": float(weight_decay)})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    return groups


def _as_param_groups(obj, weight_decay: float):
    """Best-effort conversion of various inputs into AdamW param groups.

    Accepts:
      - nn.Module -> single WD-specified group (caller typically wants split_weight_decay_groups instead)
      - Iterable[Parameter] -> one group
      - Sequence[dict] (already param groups) -> returned as-is
    """
    if isinstance(obj, nn.Module):
        return [{"params": [p for p in obj.parameters() if p.requires_grad], "weight_decay": float(weight_decay)}]

    # Already param groups
    if isinstance(obj, (list, tuple)) and len(obj) > 0 and isinstance(obj[0], dict):
        return list(obj)

    # Iterable of parameters
    try:
        it = iter(obj)  # type: ignore
    except TypeError:
        raise TypeError("build_adamw: unsupported input type for 'model'/'params' argument.")
    params = [p for p in it if isinstance(p, torch.nn.Parameter)]
    if not params:
        raise ValueError("build_adamw: received an iterable with no Parameters.")
    return [{"params": params, "weight_decay": float(weight_decay)}]


def build_adamw(
    model: Union[nn.Module, Sequence[dict], Iterable[torch.nn.Parameter]],
    lr: float,
    weight_decay: float,
    extra_no_decay_names: Sequence[str] = (),
    extra_no_decay_params: Sequence[Optional[torch.nn.Parameter]] = (),
    betas=(0.9, 0.999),
    eps=1e-8,
    include_1d_as_no_decay: bool = True,
):
    """
    Flexible AdamW builder.

    If `model` is an nn.Module -> build groups using split_weight_decay_groups (norms/bias excluded).
    If `model` is an iterable of Parameters -> single group (uses provided weight_decay).
    If `model` is a sequence of param-group dicts -> passed through unchanged.

    Note: We always pass `weight_decay=0.0` to the optimizer itself and rely on group-level WD.
    """
    if isinstance(model, nn.Module):
        groups = split_weight_decay_groups(
            model,
            weight_decay,
            extra_no_decay_names=extra_no_decay_names,
            extra_no_decay_params=extra_no_decay_params,
            include_1d_as_no_decay=include_1d_as_no_decay,
        )
    else:
        groups = _as_param_groups(model, weight_decay)

    return torch.optim.AdamW(groups, lr=lr, betas=betas, eps=eps, weight_decay=0.0)




# aux.py

def module_grad_norm(mod):
    total_sq = 0.0
    for p in mod.parameters():
        if p.grad is not None:
            total_sq += float(p.grad.detach().to('cpu').pow(2).sum().item())
    return math.sqrt(total_sq)


# ------------------------ per-k grad norms (stacked) ------------------------
@torch.no_grad()
def _per_k_grad_norms(support_sets) -> np.ndarray:
    K = support_sets.num_support_sets
    # Keep accumulation on the same device as gradients (prevents device sync/copies)
    try:
        dev = next(support_sets.parameters()).device
    except StopIteration:
        dev = torch.device("cpu")
    g2 = torch.zeros(K, dtype=torch.float32, device=dev)

    for p in support_sets.parameters():
        g = p.grad
        if g is None:
            continue
        g = g.detach().float()
        if g.ndim == 0:
            continue
            
        # find which axis corresponds to K (don’t assume it’s dim 0)
        axes_with_K = [ax for ax, sz in enumerate(g.shape) if sz == K]
        if not axes_with_K:
            continue
        k_ax = axes_with_K[0]
        if k_ax != 0:
            g = g.movedim(k_ax, 0)  # put K in front
        g2 += g.reshape(K, -1).pow(2).sum(dim=1)
    return torch.sqrt(torch.clamp(g2, min=1e-12)).detach().cpu().numpy()

    
def _pack_BK(x_bk: torch.Tensor, *, return_b_idx: bool = True):
    """
    Flatten [B,K,...] → [B*K,...] in a *known* order and return the mapping + targets.
    Order: (b=0,k=0..K-1), (b=1,k=0..K-1), ...
    """
    assert x_bk.dim() >= 2, f"Expected [B,K,...], got {tuple(x_bk.shape)}"
    B, K = x_bk.shape[:2]

    # Make contiguous only if needed; use reshape to avoid copies when possible.
    x_bk_c = x_bk if x_bk.is_contiguous() else x_bk.contiguous()
    flat = x_bk_c.reshape(B * K, *x_bk_c.shape[2:])

    # Mapping & targets (same device as input)
    dev = x_bk.device
    k_idx = torch.arange(K, device=dev).repeat(B)                # [0..K-1, 0..K-1, ...]
    targets = k_idx                                              # class k for row (b,k)
    b_idx = torch.arange(B, device=dev).repeat_interleave(K) if return_b_idx else None  # [0,0,...,1,1,...]

    return flat, targets, (b_idx, k_idx), (B, K)








# ============================================================
# TrainingStatTracker (updated to support global_opt_step)
# ============================================================
class TrainingStatTracker(object):
    """
    Tracks metrics at two levels:
      - micro-step accumulation (within a grad-acc window)
      - optimizer-step aggregates (emitted once per window)

    Also tracks per-MLP analytics (EMA accuracy, EMA grad-norm, selection counts, confusion),
    optional histories for heatmaps, and learning rates.
    """

    def __init__(self, ema_decay: float = 0.9, ema_max_history: int = 200):
        # Window (micro-step) accumulators
        self._reset_window()

        # Global optimizer-step index (used by logging)
        # Convention: this marks the CURRENT step id used for logging;
        # it is incremented AFTER finalize_step() completes.
        self.global_opt_step: int = 0

        # LRs (latest seen per optimizer-step)
        self.last_support_lr = 0.0
        self.last_recon_lr = 0.0

        # Timing
        self.iter_times = np.array([])  # seconds per opt step

        # Per-MLP analytics (set after K is known)
        self.K = None
        self.ema_decay = float(ema_decay)
        self.ema_max_history = int(ema_max_history)
        self.per_k_ema_acc = None         # [K] float
        self.per_k_ema_grad = None        # [K] float
        self.per_k_select_counts = None   # [K] long
        self.confusion = None             # [K, K] long  (row=true, col=pred)
        self.ema_history = []             # list of np.array([K]) snapshots
        self.iter_history = []            # matching optimizer-step indices for heatmap

        # JSON-like store of per-step aggregates (string keys)
        self.stats_by_step = {}  # {step_idx: dict}

    # ---------- window (micro-steps) ----------
    def _reset_window(self):
        self.win_count = 0
        self.win_sum =  dict()

    def _acc(self, key: str, val: float | None):
        if val is None:  # allow optional arguments
            return
        self.win_sum[key] = self.win_sum.get(key, 0.0) + float(val)

    def add_micro(
        self,
        *,
        acc: float,
        classification_loss: float,
        total_loss: float,
        entropy: float = 0.0,
        step1_norm: float = 0.0,
        step2_norm: float = 0.0,
        potential_std: float = 0.0,
        xf_now: float = 0.0,
        # ---- allow arbitrary extras without breaking ----
        **extras,
    ):
        """Accumulate values from a micro-step; all inputs are Python floats."""
        self.win_count += 1
        self._acc('accuracy_index', acc)
        self._acc('L_classification', classification_loss)
        self._acc('total_loss', total_loss)
        self._acc('entropy', entropy)
        self._acc('step1_norm', step1_norm)
        self._acc('step2_norm', step2_norm)
        self._acc('potential_std', potential_std)
        self._acc('xf_now', xf_now)
        # PDE components

        # Any extra scalar metrics can be merged automatically
        for k, v in extras.items():
            try:
                self._acc(k, v)
            except Exception:
                # ignore non-scalar or malformed extras
                pass

    def close_window(self):
        """Return window means and reset micro accumulators."""
        denom = max(1, self.win_count)
        means = {k: (v / denom) for k, v in self.win_sum.items()}
        self._reset_window()
        return means

    # ---------- per-MLP analytics ----------
    def init_per_k(self, K: int):
        """Call once when K is known."""
        self.K = int(K)
        self.per_k_ema_acc = np.zeros(self.K, dtype=np.float32)
        self.per_k_ema_grad = np.zeros(self.K, dtype=np.float32)
        self.per_k_select_counts = np.zeros(self.K, dtype=np.int64)
        self.confusion = np.zeros((self.K, self.K), dtype=np.int64)
        self.ema_history.clear()
        self.iter_history.clear()

    def update_per_k_after_micro(
        self,
        *,
        true_k: int,
        preds: np.ndarray,         # shape [B] int64 on CPU
        batch_size: int,
        grad_norm_selected_mlp: float | None = None,
    ):
        """
        Update EMA accuracy, selection counts, and confusion for the selected k of THIS micro-step.
        - Only the selected MLP's grad-norm is meaningful to track.
        """
        if self.K is None:
            return
        true_k = int(true_k)
        # acc for this micro-batch against the selected k
        acc = float((preds == true_k).mean())
        self.per_k_ema_acc[true_k] = self.per_k_ema_acc[true_k] * self.ema_decay + acc * (1.0 - self.ema_decay)
        self.per_k_select_counts[true_k] += int(batch_size)

        # confusion row update (count predicted classes)
        binc = np.bincount(preds, minlength=self.K).astype(np.int64)
        self.confusion[true_k, :] += binc

        # grad norm EMA (only for the MLP that received grads)
        if grad_norm_selected_mlp is not None:
            g = float(grad_norm_selected_mlp)
            self.per_k_ema_grad[true_k] = self.per_k_ema_grad[true_k] * self.ema_decay + g * (1.0 - self.ema_decay)

    def snapshot_per_k_history(self, step_idx: int):
        """Keep a thin history (capped) for heatmaps."""
        if self.K is None:
            return
        self.ema_history.append(self.per_k_ema_acc.copy())
        self.iter_history.append(int(step_idx))
        if len(self.ema_history) > self.ema_max_history:
            self.ema_history = self.ema_history[-self.ema_max_history:]
            self.iter_history = self.iter_history[-self.ema_max_history:]

    # ---------- LRs ----------
    def set_lrs(self, support_lr: float, recon_lr: float):
        self.last_support_lr = float(support_lr)
        self.last_recon_lr = float(recon_lr)

    # Allow trainer to sync starting index (resume)
    def set_opt_step(self, step_idx: int):
        self.global_opt_step = int(step_idx)

    # ---------- per-step finalize ----------
    def finalize_step(
        self,
        *,
        step_idx: int,
        window_means: dict,
        elapsed_from_start: float,
        mean_step_time: float,
        eta_seconds: float,
    ):
        """
        Called once per optimizer step to store a compact dictionary of metrics.
        - Stores under the provided step_idx
        - Exposes both legacy and new metric keys for backward compatibility
        - Increments global_opt_step AFTER storing
        """
        rec = dict(window_means)

        # Backward-compat aliases expected by some logs
        if 'classification_loss' not in rec and 'L_classification' in rec:
            rec['classification_loss'] = rec['L_classification']
        if 'kl_loss' not in rec and 'L_kl' in rec:
            rec['kl_loss'] = rec['L_kl']

        rec.update({
            'support_sets_lr': self.last_support_lr,
            'reconstructor_lr': self.last_recon_lr,
            'mean_step_time_sec': float(mean_step_time),
            'elapsed_sec': float(elapsed_from_start),
            'eta_sec': float(eta_seconds),
        })
        self.stats_by_step[int(step_idx)] = rec

        # Advance the global step *after* storing
        self.global_opt_step = int(step_idx) + 1

    # ---------- time helpers ----------
    def push_step_time(self, dt_seconds: float):
        self.iter_times = np.append(self.iter_times, float(dt_seconds))

    def mean_step_time(self) -> float:
        return float(self.iter_times.mean()) if self.iter_times.size > 0 else 0.0


def update_progress(msg, total, progress):
    bar_length, status = 20, ""
    progress = float(progress) / float(total)
    if progress >= 1.:
        progress, status = 1, "\r\n"
    block = int(round(bar_length * progress))
    block_symbol = u"\u2588"
    empty_symbol = u"\u2591"
    text = "\r{}{} {:.0f}% {}".format(msg, block_symbol * block + empty_symbol * (bar_length - block),
                                      round(progress * 100, 0), status)
    sys.stdout.write(text)
    sys.stdout.flush()


def update_stdout(num_lines):
    """Move cursor up and clear lines in terminal-friendly way."""
    cursor_up = '\x1b[1A'
    erase_line = '\x1b[2K'
    for _ in range(num_lines):
        sys.stdout.write(cursor_up + erase_line + '\r')
    sys.stdout.flush()


def sec2dhms(t):
    """Convert seconds to 'DD days, HH hours, MM minutes, and SS seconds'."""
    t = int(t)
    day = t // (24 * 3600)
    t = t % (24 * 3600)
    hour = t // 3600
    t %= 3600
    minutes = t // 60
    t %= 60
    seconds = t
    return "%02d days, %02d hours, %02d minutes, and %02d seconds" % (day, hour, minutes, seconds)

def get_wh(img_paths):
    """Get width and height of images in given list of paths. Images are expected to have the same resolution.

    Args:
        img_paths (list): list of image paths

    Returns:
        width (int)  : the common images width
        height (int) : the common images height

    """
    img_widths = []
    img_heights = []
    for img in img_paths:
        img_ = Image.open(img)
        img_widths.append(img_.width)
        img_heights.append(img_.height)

    if len(set(img_widths)) == len(set(img_heights)) == 1:
        return img_widths[0], img_heights[1]
    else:
        raise ValueError("Inconsistent image resolutions in {}".format(img_paths))


def create_summarizing_gif(imgs_root, gif_filename, num_imgs=None, gif_size=None, gif_fps=30, gap=15, progress_bar_h=15,
                           progress_bar_color=(252, 186, 3)):
    """Create a summarizing GIF image given an images root directory (images generated across a certain latent path) and
    the number of images to appear as a static sequence. The resolution of the resulting GIF image will be
    ((num_imgs + 1) * gif_size, gif_size). That is, a static sequence of `num_imgs` images will be depicted in front of
    the animated GIF image (the latter will use all the available images in `imgs_root`).

    Args:
        imgs_root (str)            : directory of images (generated across a certain path)
        gif_filename (str)         : filename of the resulting GIF image
        num_imgs (int)             : number of images that will be used to build the static sequence before the
                                     animated part of the GIF
        gif_size (int)             : height of the GIF image (its width will be equal to (num_imgs + 1) * gif_size)
        gif_fps (int)              : GIF frames per second
        gap (int)                  : a gap between the static sequence and the animated path of the GIF
        progress_bar_h (int)       : height of the progress bar depicted to the bottom of the animated part of the GIF
                                     image. If a non-positive number is given, progress bar will be disabled.
        progress_bar_color (tuple) : color of the progress bar

    """
    # Check if given images root directory exists
    if not osp.isdir(imgs_root):
        raise NotADirectoryError("Invalid directory: {}".format(imgs_root))

    # Get all images under given root directory
    path_images = [osp.join(imgs_root, dI) for dI in os.listdir(imgs_root) if osp.isfile(osp.join(imgs_root, dI))]
    path_images.sort()

    # Set number of images to appear in the static sequence of the GIF
    num_images = len(path_images)
    if num_imgs is None:
        num_imgs = num_images
    elif num_imgs > num_images:
        num_imgs = num_images

    # Get paths of static images
    static_imgs = []
    for i in range(0, len(path_images), math.ceil(len(path_images) / num_imgs)):
        static_imgs.append(osp.join(imgs_root, '{:06}.jpg'.format(i)))
    num_imgs = len(static_imgs)

    # Get GIF image resolution
    if gif_size is not None:
        gif_w = gif_h = gif_size
    else:
        gif_w, gif_h = get_wh(static_imgs)

    # Create PIL static image
    static_img_pil = Image.new('RGB', size=(len(static_imgs) * gif_w, gif_h))
    for i in range(len(static_imgs)):
        static_img_pil.paste(Image.open(static_imgs[i]).resize((gif_w, gif_h)), (i * gif_w, 0))

    # Create PIL GIF frames
    gif_frames = []
    for i in range(len(path_images)):
        # Create new PIL frame
        gif_frame_pil = Image.new('RGB', size=((num_imgs + 1) * gif_w + gap, gif_h), color=(255, 255, 255))

        # Paste static image
        gif_frame_pil.paste(static_img_pil, (0, 0))

        # Paste current image
        gif_frame_pil.paste(Image.open(path_images[i]).resize((gif_w, gif_h)), (num_imgs * gif_w + gap, 0))

        # Draw progress bar
        if progress_bar_h > 0:
            gif_frame_pil_drawing = ImageDraw.Draw(gif_frame_pil)
            progress = (i / len(path_images)) * gif_w
            gif_frame_pil_drawing.rectangle(xy=[num_imgs * gif_w + gap, gif_h - progress_bar_h,
                                                num_imgs * gif_w + gap + progress, gif_h],
                                            fill=progress_bar_color)

        # Append to GIF frames list
        gif_frames.append(gif_frame_pil)

    # Save GIF file
    gif_frames[0].save(
        fp=gif_filename,
        append_images=gif_frames[1:],
        save_all=True,
        optimize=False,
        loop=0,
        duration=1000 // gif_fps)

from typing import Iterable, Sequence, Union, Optional
import math
from torch.optim.lr_scheduler import _LRScheduler
import math
from torch.optim.lr_scheduler import _LRScheduler

class CosineScheduleWithWarmup(_LRScheduler):
    """
    Linear warmup followed by a configurable decay (cosine by default).
    Pickle-safe: no lambdas or local callables are stored.
    """
    def __init__(self, optimizer, num_warmup_steps: int, num_training_steps: int, last_epoch: int = -1):
        self.num_warmup_steps = int(num_warmup_steps)
        self.num_training_steps = int(num_training_steps)
        # decay config (primitive types only -> pickle-safe)
        self.decay_kind = 'cosine'
        self.decay_power = 1.0
        super().__init__(optimizer, last_epoch)

    @staticmethod
    def _clamp01(x: float) -> float:
        return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x

    def _decay_factor(self, p: float) -> float:
        # p is in [0,1]
        if self.decay_kind == 'cosine':
            return 0.5 * (1.0 + math.cos(math.pi * p))   # 1 -> 0
        elif self.decay_kind == 'linear':
            return 1.0 - p                                # 1 -> 0
        elif self.decay_kind == 'constant':
            return 1.0                                    # hold LR
        elif self.decay_kind == 'poly':
            return (1.0 - p) ** float(self.decay_power)   # power decay
        else:
            raise ValueError(f"Unknown decay schedule '{self.decay_kind}'")

    def get_lr(self):
        # Warmup: 0 -> 1
        if self.last_epoch < self.num_warmup_steps:
            progress = self._clamp01(self.last_epoch / max(1, self.num_warmup_steps))
            return [base_lr * progress for base_lr in self.base_lrs]

        # Decay phase
        denom = max(1, self.num_training_steps - self.num_warmup_steps)
        p = self._clamp01((self.last_epoch - self.num_warmup_steps) / denom)
        factor = self._decay_factor(p)
        return [base_lr * factor for base_lr in self.base_lrs]

    def set_decay(self, schedule: str = 'cosine', **kwargs):
        """
        Change post-warmup decay on the fly (pickle-safe).
        schedule ∈ {'cosine', 'linear', 'constant', 'poly'}
        For 'poly', you can pass power=2.0, etc.
        """
        schedule = (schedule or 'cosine').lower()
        if schedule not in {'cosine', 'linear', 'constant', 'poly'}:
            raise ValueError("schedule must be one of {'cosine','linear','constant','poly'}")

        self.decay_kind = schedule
        if schedule == 'poly':
            self.decay_power = float(kwargs.get('power', 1.0))
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    return CosineScheduleWithWarmup(optimizer, num_warmup_steps, num_training_steps, last_epoch)



# image_logging.py
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


class ImageViz:
    """All image-creation helpers (moved out of Trainer)."""

    @staticmethod
    def to_uint01(x: torch.Tensor) -> torch.Tensor:
        x = x.detach().cpu()
        if torch.numel(x) == 0:
            return x
        if x.min() < 0.0:
            x = (x + 1.0) / 2.0
        return x.clamp(0.0, 1.0)

    @staticmethod
    def _pick_first_batch_and_K(t: torch.Tensor, k_vis: int) -> torch.Tensor:
        if t.ndim == 5:            # [B,K,C,H,W]
            return t[0, :k_vis]
        elif t.ndim == 4:          # [B,C,H,W]
            return t[:k_vis]
        elif t.ndim == 3:          # [C,H,W]
            return t.unsqueeze(0).repeat(k_vis, 1, 1, 1)
        else:
            raise ValueError(f"Unexpected tensor ndim={t.ndim} for visualization")

    @staticmethod
    def _infer_k_vis(step1_src: torch.Tensor, n_vis: int) -> int:
        if step1_src.ndim == 5:
            return min(int(n_vis), int(step1_src.shape[1]))
        elif step1_src.ndim == 4:
            return min(int(n_vis), int(step1_src.shape[0]))
        else:
            return int(n_vis)

    @staticmethod
    def _maybe_downscale(t: torch.Tensor, scale: Optional[float]) -> torch.Tensor:
        if scale is None or abs(scale - 1.0) < 1e-6:
            return t
        return F.interpolate(t.float(), scale_factor=scale, mode="area").type_as(t)

    @classmethod
    def make_triplet_grids(
        cls,
        x0: torch.Tensor,  # step1
        x1: torch.Tensor,  # step2
        x2: torch.Tensor,  # ref
        n_vis: int = 8,
        downscale: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        k_vis = cls._infer_k_vis(x0, n_vis)
        s1 = cls._pick_first_batch_and_K(x0, k_vis)
        s2 = cls._pick_first_batch_and_K(x1, k_vis)

        # reference shape handling
        if x2.ndim == 5:
            ref = x2[0, :k_vis]
        elif x2.ndim == 4:
            ref = x2.repeat(k_vis, 1, 1, 1) if x2.shape[0] == 1 else x2[:k_vis]
        elif x2.ndim == 3:
            ref = x2.unsqueeze(0).repeat(k_vis, 1, 1, 1)
        else:
            raise ValueError(f"Unexpected reference tensor ndim={x2.ndim}")

        # optional downscale
        ref = cls._maybe_downscale(ref, downscale)
        s1  = cls._maybe_downscale(s1,  downscale)
        s2  = cls._maybe_downscale(s2,  downscale)

        ref_n = cls.to_uint01(ref)
        s1_n  = cls.to_uint01(s1)
        s2_n  = cls.to_uint01(s2)

        grid_ref = make_grid(ref_n, nrow=k_vis)
        grid_s1  = make_grid(s1_n,  nrow=k_vis)
        grid_s2  = make_grid(s2_n,  nrow=k_vis)
        stacked  = torch.cat([grid_ref, grid_s1, grid_s2], dim=1)

        diff1 = (s1_n - ref_n)
        diff2 = (s2_n - s1_n)
        grid_d1 = make_grid(diff1, nrow=k_vis)
        grid_d2 = make_grid(diff2, nrow=k_vis)
        stacked_diff = torch.cat([grid_d1, grid_d2], dim=1)
        return stacked, stacked_diff

    @staticmethod
    def plot_heatmap(mat_t_by_k, K: int, title: str, xlabel: str, ylabel: str):
        arr = np.array(mat_t_by_k)
        if arr.ndim == 2 and arr.shape[0] != K:
            arr = arr.T
        fig, ax = plt.subplots(figsize=(max(6, arr.shape[1] * 0.15), max(4, K * 0.15)))
        im = ax.imshow(arr, aspect='auto', origin='lower', interpolation='nearest')
        ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        ax.set_yticks(np.arange(K)); ax.set_yticklabels([str(i) for i in range(K)])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        return fig

    @staticmethod
    def plot_confusion(conf_mat_nd: np.ndarray, K: int):
        cm = torch.tensor(conf_mat_nd, dtype=torch.float32)
        row_sums = cm.sum(dim=1, keepdim=True).clamp(min=1.0)
        cm_norm = (cm / row_sums).cpu().numpy()
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm_norm, interpolation='nearest', aspect='auto', origin='lower')
        ax.set_title("Classifier Confusion (row=true k, col=pred k)")
        ax.set_xlabel("predicted k"); ax.set_ylabel("true k")
        ax.set_xticks(np.arange(K)); ax.set_yticks(np.arange(K))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        return fig



class ImageLogger:
    """
    Writes images into the SAME TensorBoard run directory as your main SummaryWriter.
    Keeps only the last `keep_last_images` image events by:
      • opening a short-lived SummaryWriter with a `.images.<step>` suffix,
      • logging the images for that step,
      • closing it,
      • pruning older `events.*.images*` files in the same run directory.
    """
    def __init__(self, writer: SummaryWriter, keep_last_images: int = 50, downscale: Optional[float] = None):
        self.writer = writer
        self.keep_last_images = int(keep_last_images)
        self.downscale = downscale
        # Ensure we can glob files reliably regardless of SummaryWriter implementation.
        self._log_dir = Path(getattr(writer, "log_dir", ""))

    def _list_image_eventfiles(self):
        # PyTorch appends filename_suffix to event filename, so match *.images*
        return sorted(
            [p for p in self._log_dir.glob("events.out.tfevents.*.images*") if p.is_file()],
            key=lambda p: p.stat().st_mtime
        )

    def _prune_old_images(self):
        files = self._list_image_eventfiles()
        if self.keep_last_images <= 0:
            to_delete = files  # keep none
        else:
            to_delete = files[:-self.keep_last_images]
        for f in to_delete:
            try:
                f.unlink()
            except Exception:
                pass

    def log_triplet(self, tag_prefix: str, x0: torch.Tensor, x1: torch.Tensor, x2: torch.Tensor,
                    step: int, n_vis: int = 8):
        # short-lived writer to the SAME run dir; one event file per image step
        triplet, diffs = ImageViz.make_triplet_grids(x0, x1, x2, n_vis=n_vis, downscale=self.downscale)
        self.writer.add_image(f"{tag_prefix}/triplet", triplet, step)
        self.writer.add_image(f"{tag_prefix}/diff_triplet_abs", diffs, step)
        self.writer.flush()
        #  writer.close()
        # self._prune_old_images()
        
    def close(self): self.writer.close()

# =========================
# TensorBoard server + ngrok
# =========================
def tb_start(exp_dir: str):
    """
    Starts TensorBoard programmatically and optionally opens an ngrok tunnel.
    Uses env vars:
      TB_HOST, TB_PORT,
      NGROK_AUTHTOKEN, NGROK_DOMAIN, NGROK_BASIC_AUTH, NGROK_REGION
    Returns: (tb_writer, tb_url, tb_obj, run_logdir)
    """
    from tensorboard import program
    from torch.utils.tensorboard import SummaryWriter
    import os

    exp_root, run_name = exp_dir.split("__")
    tb_dir = os.path.join("experiments", "tensorboard", "wip", exp_root)
    run_dir = os.path.join(tb_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    tb_host = os.getenv("TB_HOST", "0.0.0.0")
    tb_port = int(os.getenv("TB_PORT", "6006"))

    tb = program.TensorBoard()
    tb.configure(argv=[
        None,
        "--logdir", tb_dir,
        "--host", tb_host,
        "--port", str(tb_port),
        "--reload_interval", "5",
    ])
    local_url = tb.launch()

    public_url = None
    ngrok_token = os.getenv("NGROK_AUTHTOKEN")
    if ngrok_token:
        try:
            from pyngrok import ngrok, conf
            cfg = conf.PyngrokConfig(auth_token=ngrok_token)
            conf.set_default(cfg)

            # Close any old tunnel on this port (useful for restarts)
            for t in ngrok.get_tunnels():
                if t.config.get("addr", "").endswith(f":{tb_port}"):
                    ngrok.disconnect(t.public_url)

            ngrok_hostname = os.getenv("NGROK_DOMAIN")   # reserved domain (premium)
            ngrok_auth = os.getenv("NGROK_BASIC_AUTH")   # "user:pass"
            ngrok_region = os.getenv("NGROK_REGION")     # e.g. "eu", "us"
            if ngrok_region:
                cfg.region = ngrok_region

            connect_kwargs = {"proto": "http", "addr": tb_port}
            if ngrok_hostname:
                connect_kwargs["hostname"] = ngrok_hostname
            if ngrok_auth:
                connect_kwargs["auth"] = ngrok_auth

            tunnel = ngrok.connect(**connect_kwargs)
            public_url = tunnel.public_url
        except Exception as e:
            print(f"[ngrok] Failed to create tunnel: {e}")

    tb_url = public_url or local_url
    print(f"#. TensorBoard local: {local_url}")
    if public_url:
        print(f"#. TensorBoard public: {public_url}", "\n" * 8)
    else:
        print("#. (No ngrok tunnel; set NGROK_AUTHTOKEN to expose publicly)")

    writer = SummaryWriter(log_dir=run_dir)
    return writer, tb_url, tb, run_dir


# =========================
# dt sampling (toggleable)
# =========================
@torch.no_grad()
def dt_temperature(step: int, total_opt_steps: int, start: float, end: float,
                   schedule: str = "cosine", anneal_fraction: float = 1.0) -> float:
    schedule = (schedule or "cosine").lower()
    anneal_steps = max(1, int(round(total_opt_steps * max(0.0, min(1.0, anneal_fraction)))))
    p = min(1.0, max(0.0, step / float(anneal_steps)))
    if schedule == "linear":
        return float(start + (end - start) * p)
    # cosine default
    cos_p = 0.5 * (1.0 + math.cos(math.pi * p))  # 1 -> 0
    return float(end + (start - end) * cos_p)


@torch.no_grad()
def sample_dt_legacy_uniform(B: int, device, half_range: int,
                             low: int = 4000, high: int = 5500) -> torch.Tensor:
    dt = torch.randint(low, high, (1, 1), device=device) / 5000.0
    dt_scale = 2.0 / max(1, (half_range - 1))
    dt = dt * dt_scale
    return dt.repeat(B, 1)


@torch.no_grad()
def sample_dt_chi_temp(B: int, device, *, step: int, total_opt_steps: int, half_range: int,
                       temp_start: float = 1.0, temp_end: float = 0.05,
                       schedule: str = "cosine", anneal_fraction: float = 1.0,
                       clip_max: float = 5.0, dtype=torch.float32) -> torch.Tensor:
    base_a = math.sqrt(math.pi / 8.0)
    temp = dt_temperature(step, total_opt_steps, temp_start, temp_end, schedule, anneal_fraction)
    a = float(base_a * temp)

    x = torch.randn((B, 3), device=device, dtype=dtype).mul_(a)
    dt_raw = x.norm(dim=1, keepdim=True)  # [B,1]
    if clip_max is not None and clip_max > 0:
        dt_raw = dt_raw.clamp(max=float(clip_max))

    dt_scale = 2.0 / max(1, (half_range - 1))
    return dt_raw.mul(dt_scale)


# =========================
# analytics helpers (short calls)
# =========================
@torch.no_grad()
def batch_acc_from_logits(logits: torch.Tensor, B: int, K: int, device) -> tuple[float, torch.Tensor]:
    preds = torch.argmax(logits, dim=1).view(B, K)
    true_2d = torch.arange(K, device=device).unsqueeze(0).expand(B, K)
    acc = float((preds == true_2d).float().mean().item())
    return acc, preds


@torch.no_grad()
def entropy_from_logits(logits: torch.Tensor) -> float:
    probs = torch.softmax(logits, dim=1)
    ent = -(probs * probs.clamp_min(1e-8).log()).sum(dim=1).mean()
    return float(ent.item())


@torch.no_grad()
def collect_wave_stats(support_sets, potential_preds: torch.Tensor) -> dict:
    wave_dict = support_sets.get_losses()
    wave_dict["potential_std"] = float(potential_preds.std().item())
    if "xf_now" in wave_dict and torch.is_tensor(wave_dict["xf_now"]):
        wave_dict["xf_now"] = float(wave_dict["xf_now"].norm(dim=-1).mean().item())
    # normalize to python floats
    out = {}
    for k, v in wave_dict.items():
        if torch.is_tensor(v):
            out[k] = float(v.item()) if v.numel() == 1 else v
        else:
            out[k] = float(v) if isinstance(v, (int, float)) else v
    return out


# =========================
# TB logging blocks (short names)
# =========================
def tb_scalars(writer, step: int, win_means: dict, stat_tracker):
    for k, v in win_means.items():
        writer.add_scalar(f"train/{k}", float(v), step)
    writer.add_scalar("train/support_sets_lr", float(stat_tracker.last_support_lr), step)
    writer.add_scalar("train/reconstructor_lr", float(stat_tracker.last_recon_lr), step)


def tb_grad_norms(writer, step: int, support_sets, reconstructor):
    gn_support = module_grad_norm(support_sets.PSI) + module_grad_norm(support_sets.F)
    gn_recon = module_grad_norm(reconstructor)
    writer.add_scalar("train/grad_norm/support_sets", gn_support, step)
    writer.add_scalar("train/grad_norm/reconstructor", gn_recon, step)


def tb_hists(writer, step: int, *, logits_det: torch.Tensor,
             potential_preds_det: torch.Tensor | None,
             K: int, log_potential: bool = True):
    writer.add_histogram("train/logits", logits_det, step)
    if log_potential and potential_preds_det is not None:
        for k in range(K):
            writer.add_histogram(
                f"potential_distribution/{k}",
                potential_preds_det[:, k].reshape(-1).detach(),
                step
            )


def tb_figs(writer, step: int, stat_tracker, K: int, log_freq: int):
    # snapshot history only when we're logging figures
    stat_tracker.snapshot_per_k_history(step)

    if len(stat_tracker.ema_history) >= 2:
        hist_mat = np.stack(stat_tracker.ema_history, axis=1)[:, -30:]
        fig = ImageViz.plot_heatmap(
            hist_mat, K=K,
            title="Per-MLP EMA Accuracy over Time",
            xlabel="optimizer step snapshot",
            ylabel="MLP index k",
        )
        writer.add_figure("per_mlp/accuracy_heatmap", fig, global_step=step)
        plt.close(fig)

        fig_c = ImageViz.plot_confusion(stat_tracker.confusion, K=K)
        writer.add_figure("classifier/confusion_matrix", fig_c, global_step=step)
        plt.close(fig_c)


@torch.no_grad()
def tb_images(img_logger, step: int, *, generator, z_first: torch.Tensor,
              img1_bk: torch.Tensor, img2_bk: torch.Tensor, n_vis: int):
    first_img = generator(z_first)  # [1,C,H,W]
    img_logger.log_triplet(
        tag_prefix="images",
        x0=img1_bk, x1=img2_bk, x2=first_img,
        step=step,
        n_vis=n_vis,
    )


def json_append_by_step(path: str, step: int, rec: dict):
    """Load a JSON dict (if exists), set rec at key=step (int), and write back."""
    os.makedirs(osp.dirname(path), exist_ok=True)
    if osp.isfile(path):
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except Exception:
            data = {}
    else:
        data = {}
    data[str(int(step))] = rec
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def torch_append_by_step(path: str, step: int, payload: dict):
    """
    Load a torch-saved dict (if exists), set payload at key=step (int), and save back.
    NOTE: payload should already be moved to CPU + detached.
    """
    os.makedirs(osp.dirname(path), exist_ok=True)
    if osp.isfile(path):
        try:
            data = torch.load(path, map_location="cpu")
            if not isinstance(data, dict):
                data = {}
        except Exception:
            data = {}
    else:
        data = {}
    data[int(step)] = payload
    torch.save(data, path)
