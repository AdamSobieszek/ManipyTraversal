"""
Collate saved PNG frames into a GIF.

Examples:
  python scripts/make_gif_from_frames.py \
    --frames_dir experiments/wip/<exp_dir>/plot_frames/paths__sector_projection \
    --out_gif    experiments/wip/<exp_dir>/plot_frames/paths__sector_projection.gif \
    --fps 6
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

# Ensure repo root is on sys.path so `simulations.*` imports work when running as a script.
# (When executing `python scripts/foo.py`, Python puts `scripts/` on sys.path[0], not the repo root.)
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulations.aux import ImageViz


def _pick_consensus_size(frames_dir: Path, glob_pattern: str) -> tuple[int, int] | None:
    """
    Return the most common (width,height) among matching PNGs, or None if no frames.
    Ties are broken by largest area, then width, then height.
    """
    files = sorted(frames_dir.glob(glob_pattern))
    if not files:
        return None

    sizes: Counter[tuple[int, int]] = Counter()
    # Prefer PIL for fast header-only size reads; fallback to imageio if PIL isn't available.
    try:
        from PIL import Image  # type: ignore

        for f in files:
            with Image.open(f) as im:
                sizes[(int(im.size[0]), int(im.size[1]))] += 1
    except Exception:
        try:
            import imageio.v2 as imageio  # type: ignore

            for f in files:
                im = imageio.imread(f)
                # im shape is (H,W,...) for typical images
                h = int(im.shape[0])
                w = int(im.shape[1])
                sizes[(w, h)] += 1
        except Exception:
            return None

    if not sizes:
        return None

    items = list(sizes.items())
    items.sort(key=lambda kv: (kv[1], kv[0][0] * kv[0][1], kv[0][0], kv[0][1]), reverse=True)
    return items[0][0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames_dir", type=str, required=True, help="Directory containing PNG frames (e.g. .../plot_frames/<tag>)")
    ap.add_argument("--out_gif", type=str, required=True, help="Output GIF path")
    ap.add_argument("--fps", type=float, default=6.0, help="Frames per second")
    ap.add_argument("--glob", type=str, default="*.png", help="Glob pattern for frames (default: *.png)")
    ap.add_argument("--resize_w", type=int, default=0, help="Optional resize width (0 disables)")
    ap.add_argument("--resize_h", type=int, default=0, help="Optional resize height (0 disables)")
    ap.add_argument(
        "--auto_resize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If frames have varying sizes and no explicit --resize_w/--resize_h is given, pick the consensus size and resize all frames (default: enabled).",
    )
    args = ap.parse_args()

    resize = None
    if args.resize_w > 0 and args.resize_h > 0:
        resize = (int(args.resize_w), int(args.resize_h))
    elif args.auto_resize:
        consensus = _pick_consensus_size(Path(args.frames_dir), str(args.glob))
        if consensus is not None:
            resize = consensus

    out = ImageViz.frames_to_gif(
        frames_dir=Path(args.frames_dir),
        out_gif=Path(args.out_gif),
        fps=float(args.fps),
        glob_pattern=str(args.glob),
        resize=resize,
    )
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

