import sys
import os
import os.path as osp
import json
import argparse
import numpy as np
import torch
import math
import time
from scipy.stats import truncnorm
from PIL import Image, ImageDraw



def sample_z(batch_size, dim_z, truncation=None):
    """Sample a random latent code from multi-variate standard Gaussian distribution with/without truncation.

    Args:
        batch_size (int)   : batch size (number of latent codes)
        dim_z (int)        : latent space dimensionality
        truncation (float) : truncation parameter

    Returns:
        z (torch.Tensor)   : batch of latent codes
    """
    if truncation is None or truncation == 1.0:
        return torch.randn(batch_size, dim_z)
    else:
        return torch.from_numpy(truncnorm.rvs(-truncation, truncation, size=(batch_size, dim_z))).to(torch.float)


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

# aux.py
import sys
import time
import numpy as np

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
        self.win_sum = {
            'accuracy_index': 0.0,
            'classification_loss': 0.0,
            'wave_loss': 0.0,
            'total_loss': 0.0,
            'entropy': 0.0,
            'step1_norm': 0.0,
            'step2_norm': 0.0,
        }

    def add_micro(
        self,
        *,
        acc: float,
        classification_loss: float,
        wave_loss: float,
        total_loss: float,
        entropy: float = 0.0,
        step1_norm: float = 0.0,
        step2_norm: float = 0.0,
    ):
        """Accumulate values from a micro-step; all inputs are Python floats."""
        self.win_count += 1
        self.win_sum['accuracy_index'] += float(acc)
        self.win_sum['classification_loss'] += float(classification_loss)
        self.win_sum['wave_loss'] += float(wave_loss)
        self.win_sum['total_loss'] += float(total_loss)
        self.win_sum['entropy'] += float(entropy)
        self.win_sum['step1_norm'] += float(step1_norm)
        self.win_sum['step2_norm'] += float(step2_norm)

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
        """
        rec = dict(window_means)
        rec.update({
            'support_sets_lr': self.last_support_lr,
            'reconstructor_lr': self.last_recon_lr,
            'mean_step_time_sec': float(mean_step_time),
            'elapsed_sec': float(elapsed_from_start),
            'eta_sec': float(eta_seconds),
        })
        self.stats_by_step[int(step_idx)] = rec

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
