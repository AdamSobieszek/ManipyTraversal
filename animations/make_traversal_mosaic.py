#!/usr/bin/env python3
"""
make_traversal_mosaic.py

Takes a directory of "traversal" composite images where each file contains N columns
(typically 6) representing a single transformation from the same starting image (left)
to an ending image (right). For each file it:

1) Splits the composite into columns.
2) Creates a forward+backward (ping-pong) sequence: 0..N-1..1 (no duplicate endpoints).
3) Builds an animation where each of the N output columns is driven by a *different* transformation.
   After completing a full ping-pong cycle, it switches to the next set of N transformations.
4) Exports an animated GIF and animated WebP.

Example:
  python make_traversal_mosaic.py --input ./traversals --pattern "traversal_k_*.jpg" --out mosaic

Outputs:
  mosaic.gif
  mosaic.webp
"""
from __future__ import annotations

import argparse, glob, math, os
from typing import List, Tuple
from PIL import Image

def pingpong(indices: List[int]) -> List[int]:
    if len(indices) <= 1:
        return indices
    return indices + indices[-2:0:-1]  # e.g. 0,1,2,3,4,5,4,3,2,1

def concat_h(images: List[Image.Image], bg=(0,0,0)) -> Image.Image:
    h = max(im.size[1] for im in images)
    total_w = sum(im.size[0] for im in images)
    out = Image.new("RGB", (total_w, h), bg)
    x = 0
    for im in images:
        if im.size[1] != h:
            im = im.resize((im.size[0], h), Image.LANCZOS)
        out.paste(im, (x, 0))
        x += im.size[0]
    return out

def split_into_columns(img: Image.Image, ncols: int) -> List[Image.Image]:
    w, h = img.size
    # Robust split even if width isn't perfectly divisible:
    base = w // ncols
    remainder = w - base * ncols
    cols = []
    x = 0
    for i in range(ncols):
        w_i = base + (1 if i < remainder else 0)
        cols.append(img.crop((x, 0, x + w_i, h)))
        x += w_i

    # Normalize widths by padding to the widest column (avoids seam jitter)
    maxw = max(c.size[0] for c in cols)
    if any(c.size[0] != maxw for c in cols):
        padded = []
        for c in cols:
            if c.size[0] == maxw:
                padded.append(c)
            else:
                canvas = Image.new("RGB", (maxw, h), (0,0,0))
                canvas.paste(c, (0,0))
                padded.append(canvas)
        cols = padded
    return cols

def load_transforms(paths: List[str], ncols: int, target_size: Tuple[int,int] | None) -> Tuple[List[List[Image.Image]], Tuple[int,int]]:
    transforms: List[List[Image.Image]] = []
    col_size: Tuple[int,int] | None = None

    for p in paths:
        img = Image.open(p).convert("RGB")
        if target_size and img.size != target_size:
            img = img.resize(target_size, Image.LANCZOS)

        cols = split_into_columns(img, ncols=ncols)
        # column size after split (normalized)
        csz = cols[0].size

        if col_size is None:
            col_size = csz
        elif csz != col_size:
            cols = [c.resize(col_size, Image.LANCZOS) for c in cols]

        transforms.append(cols)

    assert col_size is not None
    return transforms, col_size

def build_animation(
    transforms: List[List[Image.Image]],
    ncols: int,
    repeat_all_batches: int = 1,
) -> List[Image.Image]:
    stage_indices = pingpong(list(range(ncols)))
    frames: List[Image.Image] = []

    batches = math.ceil(len(transforms) / ncols)

    for _ in range(repeat_all_batches):
        for b in range(batches):
            batch = transforms[b*ncols:(b+1)*ncols]

            # Keep exactly ncols columns in the output at all times.
            # If the final batch is short, wrap within the batch.
            if len(batch) < ncols:
                if len(batch) == 0:
                    continue
                for k in range(ncols - len(batch)):
                    batch.append(batch[k % len(batch)])

            for si in stage_indices:
                frame_cols = [batch[j][si] for j in range(ncols)]
                frames.append(concat_h(frame_cols))

    return frames

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Directory with traversal composites")
    ap.add_argument("--pattern", default="*.png", help='Glob pattern, e.g. "traversal_k_*.jpg"')
    ap.add_argument("--ncols", type=int, default=6, help="Number of columns per composite")
    ap.add_argument("--fps", type=float, default=12.0, help="Frames per second")
    ap.add_argument("--repeat", type=int, default=1, help="Repeat the whole set of batches this many times")
    ap.add_argument("--out", default="mosaic", help="Output basename (no extension)")
    ap.add_argument("--max_width", type=int, default=0, help="Optional downscale: max output width in pixels (0 disables)")
    ap.add_argument("--sort", action="store_true", help="Sort filenames (recommended)")
    args = ap.parse_args()

    paths = glob.glob(os.path.join(args.input, args.pattern))
    if args.sort:
        paths = sorted(paths)

    if not paths:
        raise SystemExit(f"No images matched {args.pattern} in {args.input}")

    # Use the first image as the size reference
    ref_size = Image.open(paths[0]).size

    transforms, _ = load_transforms(paths, ncols=args.ncols, target_size=ref_size)
    frames = build_animation(transforms, ncols=args.ncols, repeat_all_batches=args.repeat)

    # Optional downscale to keep file sizes reasonable
    if args.max_width and frames:
        w0, h0 = frames[0].size
        if w0 > args.max_width:
            scale = args.max_width / float(w0)
            new_size = (int(w0 * scale), int(h0 * scale))
            frames = [f.resize(new_size, Image.LANCZOS) for f in frames]

    duration_ms = int(round(1000.0 / args.fps))
    gif_path = f"{args.out}.gif"
    webp_path = f"{args.out}.webp"

    # GIF
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
        disposal=2,
    )

    # Animated WebP
    frames[0].save(
        webp_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        format="WEBP",
        quality=80,
        method=6,
    )

    print(f"Wrote {gif_path} and {webp_path}")
    print(f"Frames: {len(frames)}  FPS: {args.fps}  Duration: {len(frames)/args.fps:.2f}s")

if __name__ == "__main__":
    main()
