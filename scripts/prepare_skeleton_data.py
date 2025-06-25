# -*- coding: utf-8 -*-
"""prepare_skeleton_data.py — scripts.prepare_skeleton_data

概要:
    フォント画像から細線化した骨格画像を生成するユーティリティ。

:author: pumpCurry
:copyright: (c) pumpCurry 2025 / 5r4ce2
:license: MIT
:version: 1.0.74 (PR #34)
:since:   1.0.64 (PR #29)
:last-modified: 2025-06-25 12:00:00 JST+9
:todo:
    - Support batch rendering
"""

from __future__ import annotations

import argparse
import os
from typing import Dict
import torch

from PIL import Image, ImageDraw, ImageFont
import numpy as np
from skimage.filters import gaussian, threshold_otsu
from skimage.morphology import skeletonize, remove_small_objects
from skimage.util import invert
from tqdm import tqdm
import multiprocessing as mp


def render_char_to_png(font_path: str, char: str, out_path: str, size: int = 256) -> Image.Image:
    """Render ``char`` using ``font_path`` and save as PNG.

    Args:
        font_path: TrueType/OpenType font file path.
        char: Character to render.
        out_path: Output PNG path.
        size: Image size in pixels.

    Returns:
        Rendered ``PIL.Image`` instance.
    """
    font = ImageFont.truetype(font_path, int(size * 0.8))
    img = Image.new("L", (size, size), color=255)
    draw = ImageDraw.Draw(img)
    try:
        bbox = draw.textbbox((0, 0), char, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x = (size - w) / 2 - bbox[0]
        y = (size - h) / 2 - bbox[1]
    except AttributeError:
        w, h = draw.textsize(char, font=font)
        ascent, _ = font.getmetrics()
        x = (size - w) / 2
        y = (size - ascent) / 2
    draw.text((x, y), char, font=font, fill=0)
    img.save(out_path)
    return img


def load_char_list_from_file(path: str) -> Dict[int, str]:
    """Load character list from text file.

    Each line may contain a single character or ``U+XXXX`` form.

    Args:
        path: File path.

    Returns:
        Mapping from code point to character.
    """
    chars: Dict[int, str] = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.upper().startswith("U+"):
                try:
                    code = int(line[2:], 16)
                    chars[code] = chr(code)
                except Exception:
                    continue
            else:
                ch = line[0]
                chars[ord(ch)] = ch
    return chars


def create_skeleton_image(
    input_img_path: str,
    output_path: str,
    apply_blur: bool = True,
    pt_path: str | None = None,
) -> None:
    """Create 1px width skeleton image from glyph PNG and save tensor."""
    img = Image.open(input_img_path).convert("L")
    arr = np.array(img)
    if apply_blur:
        arr = gaussian(arr, sigma=0.7)
    thresh = threshold_otsu(arr)
    binary_image = arr < thresh
    skeleton = skeletonize(binary_image)
    skeleton = remove_small_objects(skeleton, min_size=10)
    skeleton_img = Image.fromarray((~skeleton * 255).astype(np.uint8))
    skeleton_img.save(output_path)
    if pt_path:
        tensor = torch.tensor(np.array(skeleton_img), dtype=torch.uint8)
        torch.save(tensor, pt_path, pickle_protocol=4)


def _job_render(args: tuple[str, str, str, int]) -> None:
    """Worker for multiprocessing render."""
    font, ch, path, size = args
    render_char_to_png(font, ch, path, size)


def _job_skeleton(args: tuple[str, str, bool, str]) -> None:
    """Worker for multiprocessing skeleton generation."""
    inp, out, blur, pt = args
    create_skeleton_image(inp, out, apply_blur=blur, pt_path=pt)


def main() -> None:
    """Generate skeleton images for all characters in list."""
    parser = argparse.ArgumentParser(description="Create skeleton data")
    parser.add_argument("--font", type=str, required=True, help="Base font path")
    parser.add_argument("--char_list", type=str, required=True, help="Learning list")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--temp_dir", type=str, default="./temp_render", help="Temporary render dir")
    parser.add_argument("--size", type=int, default=256, help="Render size")
    parser.add_argument(
        "--no_blur",
        action="store_true",
        help="Disable Gaussian blur preprocessing",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.temp_dir, exist_ok=True)

    chars = load_char_list_from_file(args.char_list)
    render_jobs = [
        (args.font, ch, os.path.join(args.temp_dir, f"{code}.png"), args.size)
        for code, ch in chars.items()
    ]
    with mp.Pool() as pool:
        list(
            tqdm(
                pool.imap_unordered(_job_render, render_jobs),
                total=len(render_jobs),
                desc="Rendering base font",
            )
        )

    skeleton_jobs = [
        (
            os.path.join(args.temp_dir, f),
            os.path.join(args.out_dir, f),
            not args.no_blur,
            os.path.join(args.out_dir, os.path.splitext(f)[0] + ".pt"),
        )
        for f in os.listdir(args.temp_dir)
    ]
    with mp.Pool() as pool:
        list(
            tqdm(
                pool.imap_unordered(_job_skeleton, skeleton_jobs),
                total=len(skeleton_jobs),
                desc="Generating skeletons",
            )
        )


if __name__ == "__main__":
    main()
