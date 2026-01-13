#!/usr/bin/env python3
"""
Extract image summaries from a TensorBoard event file.

Usage:
  /opt/anaconda3/envs/manip311/bin/python scripts/extract_tb_images.py \
    --event_file /path/to/events.out.tfevents.* \
    --out_dir /path/to/output_dir
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path


def _sanitize_tag(tag: str) -> str:
    # Keep folders, but make each segment filename-safe.
    parts = tag.split("/")
    safe_parts: list[str] = []
    for p in parts:
        p = p.strip().replace(" ", "_")
        p = re.sub(r"[^A-Za-z0-9._-]+", "_", p)
        p = re.sub(r"_+", "_", p).strip("_")
        safe_parts.append(p or "tag")
    return "/".join(safe_parts)


def main() -> int:
    ap = argparse.ArgumentParser(description="Extract image summaries from a TensorBoard events file.")
    ap.add_argument("--event_file", required=True, help="Path to events.out.tfevents.* file")
    ap.add_argument(
        "--out_dir",
        default=None,
        help="Output directory. Defaults to <event_file_dir>/extracted_images",
    )
    ap.add_argument(
        "--size_guidance",
        type=int,
        default=0,
        help="TensorBoard size guidance for images. 0 means 'load all'.",
    )
    args = ap.parse_args()

    event_file = Path(args.event_file).expanduser()
    if not event_file.exists():
        print(f"ERROR: event file not found: {event_file}", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir).expanduser() if args.out_dir else (event_file.parent / "extracted_images")
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except Exception as e:  # noqa: BLE001
        print(
            "ERROR: Could not import TensorBoard's EventAccumulator.\n"
            "Install tensorboard in this environment (or ensure it's available), then retry.\n"
            f"Import error: {e}",
            file=sys.stderr,
        )
        return 3

    size_guidance = {
        "compressedHistograms": 1,
        "images": args.size_guidance,
        "scalars": 1,
        "histograms": 1,
        "tensors": 1,
    }

    ea = EventAccumulator(str(event_file), size_guidance=size_guidance)
    ea.Reload()

    tags = ea.Tags().get("images", [])
    if not tags:
        print(f"No image tags found in: {event_file}")
        return 0

    total = 0
    for tag in tags:
        safe_tag = _sanitize_tag(tag)
        tag_dir = out_dir / safe_tag
        tag_dir.mkdir(parents=True, exist_ok=True)

        img_events = ea.Images(tag)
        for idx, img_ev in enumerate(img_events):
            # Prefer step to keep chronological ordering; include wall time for uniqueness if needed.
            step = getattr(img_ev, "step", None)
            wall_time = getattr(img_ev, "wall_time", None)
            encoded = getattr(img_ev, "encoded_image_string", None)
            if not encoded:
                continue

            # TensorBoard stores encoded PNG/JPEG bytes here; keep original bytes.
            step_part = f"step{step:08d}" if isinstance(step, int) else f"idx{idx:08d}"
            time_part = f"_t{int(wall_time)}" if isinstance(wall_time, (int, float)) else ""
            fname = f"{step_part}{time_part}_{idx:04d}.png"
            out_path = tag_dir / fname

            # Avoid overwriting if multiple events map to same name.
            if out_path.exists():
                base = out_path.stem
                suffix = out_path.suffix
                k = 1
                while (tag_dir / f"{base}_{k}{suffix}").exists():
                    k += 1
                out_path = tag_dir / f"{base}_{k}{suffix}"

            out_path.write_bytes(encoded)
            total += 1

    print(f"Extracted {total} images to: {out_dir}")
    print("Tags:")
    for t in tags:
        print(f"  - {t}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

