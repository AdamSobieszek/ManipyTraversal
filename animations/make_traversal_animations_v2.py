#!/usr/bin/env python3
"""
Traversal montage animator (v2)

Fixes a common failure mode in v1: if the montage width/height is not perfectly divisible
by the number of traversal columns, some extracted column-frames end up 1px (or more)
different in size. GIF encoders (and many video pipelines) require every frame to be the
same dimensions, which causes "one column GIF" generation to fail.

v2 guarantees consistent frame sizes by:
- Splitting columns using rounded proportional boundaries (more robust than w//n)
- Normalizing all extracted frames to the same size (pad-right/pad-bottom by default)

Outputs:
- out/gifs/<basename>.gif                 (per montage: columns animated forward+back)
- out/sets/set_000.mp4 + set_000.webm     (per N montages: each output column = one montage)
- out/sets/set_000.gif                    (optional)
- out/columns/<basename>/col_00.png ...   (optional: extracted columns)

Requirements:
  pip install pillow imageio
  ffmpeg on PATH
"""

from __future__ import annotations
import argparse
import math
import os
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import imageio.v2 as imageio

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def list_images(input_dir: Path) -> List[Path]:
    return [p for p in sorted(input_dir.iterdir()) if p.is_file() and p.suffix.lower() in IMG_EXTS]


def split_columns(im: Image.Image, n_cols: int) -> List[Image.Image]:
    """
    Split into n_cols vertical slices using rounded proportional boundaries.
    This is more stable than integer division when dimensions don't divide evenly.
    """
    w, h = im.size
    cols: List[Image.Image] = []
    for i in range(n_cols):
        left = round(i * w / n_cols)
        right = round((i + 1) * w / n_cols)
        cols.append(im.crop((left, 0, right, h)))
    return cols


def normalize_sizes(
    frames: List[Image.Image],
    target_size: Tuple[int, int] | None = None,
    mode: str = "pad",
    pad_color: Tuple[int, int, int] = (0, 0, 0),
) -> List[Image.Image]:
    """
    Ensure all frames are identical size.
    mode:
      - "pad": pad to max width/height (pad-right + pad-bottom) using pad_color
      - "crop": crop to min width/height (crop-right + crop-bottom)
      - "resize": resize all to the first frame's size (may introduce slight blur)
    """
    if not frames:
        return frames

    sizes = [f.size for f in frames]
    if target_size is None:
        if mode == "pad":
            target_w = max(w for w, _ in sizes)
            target_h = max(h for _, h in sizes)
        elif mode == "crop":
            target_w = min(w for w, _ in sizes)
            target_h = min(h for _, h in sizes)
        elif mode == "resize":
            target_w, target_h = sizes[0]
        else:
            raise ValueError(f"Unknown normalize mode: {mode}")
        target_size = (target_w, target_h)

    tw, th = target_size
    out: List[Image.Image] = []
    for f in frames:
        f = f.convert("RGB")
        w, h = f.size
        if (w, h) == (tw, th):
            out.append(f)
            continue

        if mode == "pad":
            canvas = Image.new("RGB", (tw, th), pad_color)
            # left/top aligned; pad to the right/bottom
            canvas.paste(f, (0, 0))
            out.append(canvas)
        elif mode == "crop":
            out.append(f.crop((0, 0, tw, th)))
        elif mode == "resize":
            out.append(f.resize((tw, th), Image.LANCZOS))
        else:
            raise ValueError(f"Unknown normalize mode: {mode}")

    return out


def pingpong(frames: List[Image.Image]) -> List[Image.Image]:
    if len(frames) <= 1:
        return frames
    # forward then backward, excluding endpoints to avoid duplicates
    return frames + frames[-2:0:-1]

def pingpong_indices(n: int) -> List[int]:
    if n <= 1:
        return list(range(n))
    idx = list(range(n))
    return idx + idx[-2:0:-1]


def build_mosaic_frames(
    transforms: List[List[Image.Image]],
    ncols: int,
    collate: str,
    repeat: int,
    pad_color: Tuple[int, int, int],
) -> Tuple[List[Image.Image], List[List[int]], List[List[int]]]:
    """
    Build a single "mosaic" animation similar to make_traversal_mosaic.py.

    - transforms: list length N; each item is a list of length ncols (column images)
    - collate:
        - "replace": batch-by-batch (like traversal_mosaic)
        - "rolling": after each full ping-pong cycle, shift transformations one column right;
                     run N cycles for N input transformations (so each transform appears as left column once)
    """
    if not transforms:
        return [], [], []

    stage = pingpong_indices(ncols)
    N = len(transforms)
    frames: List[Image.Image] = []
    ids_per_frame: List[List[int]] = []
    widths_per_frame: List[List[int]] = []

    if collate == "replace":
        batches = math.ceil(N / ncols)
        for _ in range(repeat):
            for b in range(batches):
                batch = transforms[b * ncols : (b + 1) * ncols]
                batch_ids = list(range(b * ncols, min((b + 1) * ncols, N)))
                # If the final batch is short, wrap within the batch.
                if len(batch) < ncols:
                    if len(batch) == 0:
                        continue
                    for k in range(ncols - len(batch)):
                        batch.append(batch[k % len(batch)])
                    for k in range(ncols - len(batch_ids)):
                        batch_ids.append(batch_ids[k % len(batch_ids)])
                for si in stage:
                    frame_cols = [batch[j][si] for j in range(ncols)]
                    frame_cols = normalize_sizes(frame_cols, mode="pad", pad_color=pad_color)
                    frames.append(hstack(frame_cols, pad_color=pad_color))
                    ids_per_frame.append(list(batch_ids))
                    widths_per_frame.append([im.size[0] for im in frame_cols])
        return frames, ids_per_frame, widths_per_frame

    if collate == "rolling":
        # Exactly like make_traversal_mosaic.py in spirit:
        # - A "cycle" is a full ping-pong (no spatial motion).
        # - Between cycles, update the rule:
        #     - each column takes the animation from its left neighbor
        #     - column 0 receives the next new animation
        #
        # With N input transforms, this yields N cycles. For early cycles where the
        # window isn't "full" yet, we repeat transforms[0] to avoid black/empty space.
        for _ in range(repeat):
            for c in range(N):
                window_ids = [max(c - j, 0) for j in range(ncols)]
                window = [transforms[i] for i in window_ids]
                for si in stage:
                    frame_cols = [window[j][si] for j in range(ncols)]
                    frame_cols = normalize_sizes(frame_cols, mode="pad", pad_color=pad_color)
                    frames.append(hstack(frame_cols, pad_color=pad_color))
                    ids_per_frame.append(list(window_ids))
                    widths_per_frame.append([im.size[0] for im in frame_cols])
        return frames, ids_per_frame, widths_per_frame

    raise ValueError(f"Unknown mosaic collate mode: {collate}")


def save_gif(frames: List[Image.Image], out_path: Path, fps: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arrs = [np.array(f.convert("RGB")) for f in frames]
    imageio.mimsave(str(out_path), arrs, duration=1 / fps, loop=0)


def hstack(images: List[Image.Image], pad_color: Tuple[int, int, int] = (0, 0, 0)) -> Image.Image:
    # Normalize heights (pad) so concatenation is stable.
    images = normalize_sizes(images, mode="pad", pad_color=pad_color)
    widths = [im.size[0] for im in images]
    h = images[0].size[1]
    out = Image.new("RGB", (sum(widths), h))
    x = 0
    for im in images:
        out.paste(im.convert("RGB"), (x, 0))
        x += im.size[0]
    return out


def _render_latex_label_cached(
    cache: Dict[Tuple[str, int], Image.Image],
    latex: str,
    fontsize: int,
) -> Image.Image:
    """
    Render a LaTeX mathtext label (white on transparent) and cache it.
    Uses matplotlib if available; falls back to plain PIL text.
    """
    key = (latex, fontsize)
    if key in cache:
        return cache[key]

    im: Optional[Image.Image] = None

    try:
        # Avoid matplotlib/fontconfig cache permission issues by pointing caches at /tmp.
        # (Especially helpful in sandboxed or restricted environments.)
        cache_dir = Path(tempfile.gettempdir()) / "traversal_mpl_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
        os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))

        import matplotlib

        matplotlib.use("Agg")  # headless
        import matplotlib.pyplot as plt
        from io import BytesIO

        fig = plt.figure(figsize=(0.01, 0.01), dpi=200)
        fig.patch.set_alpha(0.0)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")
        ax.set_facecolor((0, 0, 0, 0))
        ax.text(
            0,
            0,
            latex,
            fontsize=fontsize,
            color="white",
            ha="left",
            va="bottom",
        )

        buf = BytesIO()
        fig.savefig(buf, format="png", transparent=True, bbox_inches="tight", pad_inches=0.0)
        plt.close(fig)
        buf.seek(0)
        im = Image.open(buf).convert("RGBA")
    except Exception:
        im = None

    if im is None:
        from PIL import ImageDraw, ImageFont

        txt = latex.strip("$")
        font = ImageFont.load_default()
        dummy = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
        d = ImageDraw.Draw(dummy)
        bbox = d.textbbox((0, 0), txt, font=font)
        w = max(1, bbox[2] - bbox[0])
        h = max(1, bbox[3] - bbox[1])
        im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        d.text((0, 0), txt, fill=(255, 255, 255, 255), font=font)

    cache[key] = im
    return im


def add_bottom_label_strip(
    frames: List[Image.Image],
    col_widths_per_frame: List[List[int]],
    ids_per_frame: List[List[int]],
    *,
    strip_height: int,
    label_offset: int,
) -> List[Image.Image]:
    """
    Append a black strip at the bottom, with white LaTeX numbers centered under each column.
    """
    if not frames:
        return frames
    if len(frames) != len(col_widths_per_frame) or len(frames) != len(ids_per_frame):
        raise ValueError("Label metadata length mismatch")

    strip_h = max(8, int(strip_height))
    # Make labels about 2x smaller than before, relative to the strip height.
    fontsize = max(6, int(strip_h * 0.35))
    cache: Dict[Tuple[str, int], Image.Image] = {}
    bottom_pad = 2

    out: List[Image.Image] = []
    for fr, widths, ids in zip(frames, col_widths_per_frame, ids_per_frame):
        fr = fr.convert("RGB")
        w, h = fr.size

        canvas = Image.new("RGBA", (w, h + strip_h), (0, 0, 0, 255))
        canvas.paste(fr.convert("RGBA"), (0, 0))

        x0 = 0
        for cw, tid in zip(widths, ids):
            center_x = x0 + cw // 2
            label = rf"${tid + label_offset}$"
            lab_im = _render_latex_label_cached(cache, label, fontsize=fontsize)
            lx = int(center_x - lab_im.size[0] / 2)
            # Bottom-align the label to the bottom of the strip (with small padding).
            ly = int(h + strip_h - lab_im.size[1] - bottom_pad)
            canvas.alpha_composite(lab_im, dest=(lx, ly))
            x0 += cw

        out.append(canvas.convert("RGB"))
    return out


def save_video_ffmpeg(frames: List[Image.Image], out_path: Path, fps: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmpdir = Path(tempfile.mkdtemp(prefix="traversal_frames_"))
    try:
        # Ensure identical frame sizes for ffmpeg input
        frames = normalize_sizes(frames, mode="pad")

        for i, fr in enumerate(frames):
            fr.save(tmpdir / f"frame_{i:04d}.png")

        # ensure even dimensions for codecs like H.264
        vf = "pad=ceil(iw/2)*2:ceil(ih/2)*2"

        if out_path.suffix.lower() == ".mp4":
            cmd = [
                "ffmpeg", "-y",
                "-framerate", str(fps),
                "-i", str(tmpdir / "frame_%04d.png"),
                "-vf", vf,
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-crf", "18",
                "-preset", "medium",
                str(out_path),
            ]
        elif out_path.suffix.lower() == ".webm":
            cmd = [
                "ffmpeg", "-y",
                "-framerate", str(fps),
                "-i", str(tmpdir / "frame_%04d.png"),
                "-vf", vf,
                "-c:v", "libvpx-vp9",
                "-pix_fmt", "yuv420p",
                "-b:v", "0",
                "-crf", "32",
                str(out_path),
            ]
        else:
            raise ValueError(f"Unsupported video extension: {out_path.suffix}")

        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="/Users/adamsobieszek/PycharmProjects/runpod_traversals/nue/vp_traversals", type=Path, help="Folder with montage images")
    ap.add_argument("--output", default="/Users/adamsobieszek/PycharmProjects/runpod_traversals/nue/roll_along", type=Path, help="Output folder")

    ap.add_argument("--cols", type=int, default=6, help="Number of traversal columns per montage")
    ap.add_argument("--fps", type=int, default=24, help="Animation FPS")
    ap.add_argument("--repeat", type=int, default=1, help="Repeat the whole mosaic this many times")
    ap.add_argument("--verbose", action="store_true", help="Print parsed args")

    ap.add_argument(
        "--mosaic",
        action="store_true",
        help="Render a single mosaic animation (one GIF + one MP4 + one WebM). Skips per-montage GIFs and set outputs.",
    )
    ap.add_argument(
        "--mosaic-collate",
        choices=["replace", "rolling"],
        default="rolling",
        help=(
            "Mosaic collation mode. "
            "'replace' behaves like make_traversal_mosaic.py batches; "
            "'rolling' updates the per-column assignment between ping-pong cycles: each column takes the animation from its left, and column 0 gets the next new animation."
        ),
    )
    ap.add_argument("--mosaic-name", default="mosaic", help="Basename for mosaic outputs (no extension)")
    ap.add_argument(
        "--labels",
        dest="labels",
        action="store_true",
        help="Add bottom strip with LaTeX transform ids under each column (default when --mosaic).",
    )
    ap.add_argument(
        "--no-labels",
        dest="labels",
        action="store_false",
        help="Disable bottom label strip.",
    )
    ap.set_defaults(labels=None)
    ap.add_argument("--label-strip-height", type=int, default=28, help="Label strip height in pixels (default: 28)")
    ap.add_argument("--label-offset", type=int, default=0, help="Add this offset to displayed transform ids (default: 0)")

    ap.add_argument("--set-size", type=int, default=6, help="How many transformations per set-video")
    ap.add_argument(
        "--set-collate",
        choices=["replace", "rolling"],
        default="rolling",
        help=(
            "How to collate set videos. "
            "'replace' = keep fixed columns and update all each frame (default). "
            "'rolling' = shift each transformation one column to the right each frame (wider output; saved to mp4/webm)."
        ),
    )
    ap.add_argument("--include-incomplete", action="store_true",
                    help="Also render last set if it has fewer than --set-size montages")
    ap.add_argument("--no-set-gif", action="store_true",
                    help="Do not write set GIFs (only mp4/webm)")
    ap.add_argument("--dump-columns", action="store_true",
                    help="Also save extracted columns as PNGs under out/columns/<basename>/")
    ap.add_argument("--normalize", choices=["pad", "crop", "resize"], default="pad",
                    help="How to make extracted column frames the same size (default: pad)")
    ap.add_argument("--pad-color", default="0,0,0",
                    help="RGB pad color used when --normalize=pad, like 0,0,0 (default)")
    args = ap.parse_args()
    if args.verbose:
        print(args)
    pad_color = tuple(int(x) for x in args.pad_color.split(","))  # type: ignore
    if len(pad_color) != 3:
        raise SystemExit("--pad-color must be like R,G,B (e.g. 0,0,0)")

    montages = list_images(args.input)
    if not montages:
        raise SystemExit(f"No images found in {args.input}")

    # In mosaic mode we need the per-transform column images (not pingpong'd yet).
    per_montage_cols: List[List[Image.Image]] = []
    per_montage_frames: List[List[Image.Image]] = []

    # Load montages and split into columns.
    for p in montages:
        im = Image.open(p).convert("RGB")
        cols = split_columns(im, args.cols)

        # Optional: dump extracted column images
        if args.dump_columns:
            dump_dir = args.output / "columns" / p.stem
            dump_dir.mkdir(parents=True, exist_ok=True)
            # normalize so dumps are consistent too
            cols_norm = normalize_sizes(cols, mode=args.normalize, pad_color=pad_color)
            for i, c in enumerate(cols_norm):
                c.save(dump_dir / f"col_{i:02d}.png")

        cols = normalize_sizes(cols, mode=args.normalize, pad_color=pad_color)

        per_montage_cols.append(cols)

        frames = pingpong(cols)
        per_montage_frames.append(frames)

        # Per-montage "one column GIF" (skipped when --mosaic)
        if not args.mosaic:
            gif_out = args.output / "gifs" / f"{p.stem}.gif"
            save_gif(frames, gif_out, fps=args.fps)

    if args.mosaic:
        mosaic_frames, ids_per_frame, widths_per_frame = build_mosaic_frames(
            transforms=per_montage_cols,
            ncols=args.cols,
            collate=args.mosaic_collate,
            repeat=args.repeat,
            pad_color=pad_color,
        )
        if args.labels is None:
            args.labels = True
        if args.labels:
            mosaic_frames = add_bottom_label_strip(
                mosaic_frames,
                widths_per_frame,
                ids_per_frame,
                strip_height=args.label_strip_height,
                label_offset=args.label_offset,
            )
        out_gif = args.output / f"{args.mosaic_name}.gif"
        out_mp4 = args.output / f"{args.mosaic_name}.mp4"
        out_webm = args.output / f"{args.mosaic_name}.webm"

        save_gif(normalize_sizes(mosaic_frames, mode="pad", pad_color=pad_color), out_gif, fps=args.fps)
        save_video_ffmpeg(mosaic_frames, out_mp4, fps=args.fps)
        save_video_ffmpeg(mosaic_frames, out_webm, fps=args.fps)

        print("Done.")
        print(f"Mosaic GIF:  {out_gif}")
        print(f"Mosaic MP4:  {out_mp4}")
        print(f"Mosaic WebM: {out_webm}")
        return

    # Set videos/gifs: each output column is a different transformation (different montage)
    sets_dir = args.output / "sets"
    n_sets = math.ceil(len(montages) / args.set_size) if args.include_incomplete else (len(montages) // args.set_size)

    for s in range(n_sets):
        start = s * args.set_size
        end = min((s + 1) * args.set_size, len(montages))
        if end - start < args.set_size and not args.include_incomplete:
            break

        group_frames = per_montage_frames[start:end]
        frame_count = max(len(f) for f in group_frames)

        composite_frames: List[Image.Image] = []
        for t in range(frame_count):
            slices = [f[t % len(f)] for f in group_frames]
            # normalize each slice list before stacking
            slices = normalize_sizes(slices, mode="pad", pad_color=pad_color)
            if args.set_collate == "replace":
                composite_frames.append(hstack(slices, pad_color=pad_color))
            elif args.set_collate == "rolling":
                # Place transformation i at column (i + t), i.e. shift right by 1 each frame.
                col_w, col_h = slices[0].size
                n = len(slices)
                canvas = Image.new("RGB", ((n + t) * col_w, col_h), pad_color)
                for i, sl in enumerate(slices):
                    canvas.paste(sl, ((i + t) * col_w, 0))
                composite_frames.append(canvas)
            else:
                raise ValueError(f"Unknown --set-collate: {args.set_collate}")

        base = f"set_{s:03d}" if args.set_collate == "replace" else f"set_{s:03d}_rolling"
        composite_frames = composite_frames[3:]
        save_video_ffmpeg(composite_frames, sets_dir / f"{base}.mp4", fps=args.fps)
        save_video_ffmpeg(composite_frames, sets_dir / f"{base}.webm", fps=args.fps)
        if not args.no_set_gif:
            save_gif(normalize_sizes(composite_frames, mode="pad", pad_color=pad_color),
                    sets_dir / f"{base}.gif", fps=args.fps)

    print("Done.")
    print(f"Per-montage GIFs: {args.output/'gifs'}")
    print(f"Set videos/gifs:  {args.output/'sets'}")
    if args.dump_columns:
        print(f"Extracted columns:{args.output/'columns'}")


if __name__ == "__main__":
    main()