# -*- coding: utf-8 -*-
"""prepare_data_step1_5.py — scripts.prepare_data_step1_5

概要:
    参考・ターゲット・骨格画像を1つの ``.pt`` ファイルにまとめて保存する前処理スクリプト。

:author: pumpCurry
:copyright: (c) pumpCurry 2025 / 5r4ce2
:license: MIT
:version: 1.0.68 (PR #31)
:since:   1.0.68 (PR #31)
:last-modified: 2025-06-25 10:40:00 JST+9
:todo:
    - Support parallel processing
"""

from __future__ import annotations

import argparse
import os
from typing import Dict

from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np
import torch
from skimage.morphology import skeletonize
from skimage.util import invert
from skimage.filters import threshold_otsu
from tqdm import tqdm


def render_char_to_png(font_path: str, char: str, buffer_or_path: str | os.PathLike | None, size: int = 256) -> Image.Image:
    """Render ``char`` with ``font_path`` to a PIL image."""
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
    if isinstance(buffer_or_path, str):
        img.save(buffer_or_path)
    return img


def load_char_list_from_file(path: str) -> Dict[int, str]:
    """Return mapping of code point to character."""
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


def create_images_and_skeleton(ref_font: str, tgt_font: str, ske_font: str, glyph: str, size: int) -> tuple[Image.Image, Image.Image, Image.Image]:
    """Render three images and skeletonize the base font."""
    ref_img = render_char_to_png(ref_font, glyph, None, size=size)
    tgt_img = render_char_to_png(tgt_font, glyph, None, size=size)
    ske_base = render_char_to_png(ske_font, glyph, None, size=size)
    ske_base = ske_base.filter(ImageFilter.GaussianBlur(radius=1))
    inv = invert(np.array(ske_base.convert("L")))
    thresh = threshold_otsu(inv)
    skeleton = skeletonize(inv > thresh)
    ske_img = Image.fromarray((~skeleton).astype(np.uint8))
    return ref_img, tgt_img, ske_img


def main() -> None:
    """Execute preprocessing of font data."""
    parser = argparse.ArgumentParser(description="Pre-process font data into .pt files")
    parser.add_argument("--ref_font", type=str, required=True)
    parser.add_argument("--target_font", type=str, required=True)
    parser.add_argument("--skeleton_base_font", type=str, required=True)
    parser.add_argument("--char_list", type=str, required=True)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--output_dir", type=str, default="./data/preprocessed")
    args = parser.parse_args()

    out_dir = os.path.join(args.output_dir, str(args.size))
    os.makedirs(out_dir, exist_ok=True)

    char_map = load_char_list_from_file(args.char_list)
    print(f"Processing {len(char_map)} characters for size {args.size}x{args.size}...")
    for code, glyph in tqdm(char_map.items(), desc="Preprocessing"):
        try:
            ref_img, tgt_img, ske_img = create_images_and_skeleton(
                args.ref_font, args.target_font, args.skeleton_base_font, glyph, args.size
            )
            r = torch.from_numpy(np.array(ref_img)).to(torch.uint8).unsqueeze(0)
            t = torch.from_numpy(np.array(tgt_img)).to(torch.uint8).unsqueeze(0)
            k = torch.from_numpy(np.array(ske_img)).to(torch.uint8).unsqueeze(0)
            torch.save({"source": r, "target": t, "skeleton": k}, os.path.join(out_dir, f"{code}.pt"))
        except Exception as exc:
            print(f"[Error] Failed to process U+{code:04X} ({glyph}): {exc}")
    print(f"\nPreprocessing complete. Data saved to: {out_dir}")


if __name__ == "__main__":
    main()
