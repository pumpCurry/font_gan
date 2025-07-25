# -*- coding: utf-8 -*-
"""train_pix2pix_pro.py — train_pix2pix_pro

概要:
    256px 事前学習と 512px プログレッシブ学習を自動で行うスクリプト。

:author: pumpCurry
:copyright: (c) pumpCurry 2025 / 5r4ce2
:license: MIT
:version: 1.0.85 (PR #41)
:since:   1.0.30 (PR #14)
:last-modified: 2025-07-14 02:00:20 JST+9
:todo:
    - Improve configurability via YAML
"""

import os
import glob
import random
import io
import argparse
import pprint
import time
import json
import math
import sys
import unicodedata
from collections.abc import Callable


from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import torchvision.transforms as T
import torchvision.models as models
import torchvision.utils as vutils
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as calc_psnr
from skimage.morphology import skeletonize
from skimage.metrics import structural_similarity as calc_ssim
from scipy.ndimage import binary_dilation, binary_erosion


def set_seed(seed: int = 2025) -> None:
    """Set random seed for reproducible results."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_char_list_from_file(path: str) -> dict[int, str]:
    """Load learning characters from a text file.

    Each line may contain a single character or a code point in the form
    ``U+XXXX``.

    Args:
        path: File path to read.

    Returns:
        Mapping from code point to character.
    """
    chars: dict[int, str] = {}
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


def build_candidate_chars(
    include_file: str | None,
    exclude_file: str | None,
    range_start: int | None,
    range_end: int | None,
) -> dict[int, str]:
    """Construct a candidate character dictionary.

    :param include_file: Text file listing characters or code points to include.
    :param exclude_file: Text file listing characters to remove from the set.
    :param range_start: Unicode code point (hex) for range start.
    :param range_end: Unicode code point (hex) for range end.
    :return: Mapping from code point to character.
    """
    candidates: dict[int, str] = {}
    if include_file and os.path.exists(include_file):
        candidates.update(load_char_list_from_file(include_file))
    if range_start is not None and range_end is not None:
        for code in range(range_start, range_end + 1):
            candidates.setdefault(code, chr(code))
    if exclude_file and os.path.exists(exclude_file):
        excludes = load_char_list_from_file(exclude_file)
        for code in excludes.keys():
            candidates.pop(code, None)
    return candidates


def filter_by_both_fonts(
    candidates: dict[int, str],
    base_font: str,
    ref_font: str,
    size: int = 256,
) -> dict[int, str]:
    """Filter out characters blank in either font.

    :param candidates: Mapping from code point to character.
    :param base_font: Path to the base font file.
    :param ref_font: Path to the reference font file.
    :param size: Rendering size for blank check.
    :return: Filtered mapping available in both fonts.
    """
    filtered: dict[int, str] = {}
    for code, ch in candidates.items():
        if not is_blank_glyph(base_font, ch, size) and not is_blank_glyph(
            ref_font, ch, size
        ):
            filtered[code] = ch
    return filtered


def is_blank_glyph(font_path: str, char: str, size: int = 256, threshold: int = 250) -> bool:
    """Return ``True`` if the font renders ``char`` as almost blank."""
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
    arr = np.array(img)
    return float(np.mean(arr)) > threshold


def build_learning_char_map(font_path: str, candidate_chars: dict[int, str], external_list_path: str | None = None) -> dict[int, str]:
    """Return learning characters using file or auto detection."""
    if external_list_path and os.path.exists(external_list_path):
        print(f"[Info] Learning list file found: {external_list_path}")
        return load_char_list_from_file(external_list_path)

    print(f"[Info] No external list. Auto-detecting from OTF (font: {font_path}) ...")
    learning: dict[int, str] = {}
    for code, ch in candidate_chars.items():
        try:
            if not is_blank_glyph(font_path, ch, size=256):
                learning[code] = ch
        except Exception as exc:
            print(f"  [Warning] Could not render U+{code:04X}: {exc}")
    print(f"[Info] Auto-detect done. {len(learning)}/{len(candidate_chars)} characters selected.")
    return learning


def dump_char_string(
    target_font: str,
    ref_font: str,
    candidate_chars: dict[int, str],
    output_path: str,
    external_list_path: str | None = None,
    size: int = 256,
) -> None:
    """Write available non-space characters to ``output_path`` and exit.

    :param target_font: Base font file path.
    :type target_font: str
    :param ref_font: Reference font file path.
    :type ref_font: str
    :param candidate_chars: Mapping of code point to candidate character.
    :type candidate_chars: dict[int, str]
    :param output_path: Destination text file path.
    :type output_path: str
    :param external_list_path: Optional file listing characters to force use.
    :type external_list_path: str | None
    :param size: Rendering size for blank check.
    :type size: int
    """
    filtered = {c: ch for c, ch in candidate_chars.items() if not ch.isspace()}
    filtered = filter_by_both_fonts(filtered, base_font=target_font, ref_font=ref_font, size=size)
    char_map = build_learning_char_map(target_font, filtered, external_list_path)
    char_str = "".join(char_map.values())
    with open(output_path, "w", encoding="utf-8") as fp:
        fp.write(char_str)
    print(f"[Info] Character string saved to: {output_path}")


def dump_registed_all_char(font_path: str, output_path: str, size: int = 256) -> None:
    """Enumerate all supported characters in ``font_path``.

    A character is considered supported when it has an assigned Unicode name and
    rendering the glyph with ``Pillow`` does not result in an almost blank
    image. The scan range is ``U+0000`` through ``sys.maxunicode``.

    :param font_path: Path to the font file to inspect.
    :type font_path: str
    :param output_path: Destination text file path.
    :type output_path: str
    :param size: Rendering size for blank check.
    :type size: int
    """
    font = ImageFont.truetype(font_path, int(size * 0.8))
    registered: list[str] = []
    for code in range(sys.maxunicode + 1):
        ch = chr(code)
        if ch.isspace():
            continue
        try:
            unicodedata.name(ch)
        except ValueError:
            continue

        img = Image.new("L", (size, size), color=255)
        draw = ImageDraw.Draw(img)
        try:
            bbox = draw.textbbox((0, 0), ch, font=font)
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            x = (size - w) / 2 - bbox[0]
            y = (size - h) / 2 - bbox[1]
        except AttributeError:
            w, h = draw.textsize(ch, font=font)
            ascent, _ = font.getmetrics()
            x = (size - w) / 2
            y = (size - ascent) / 2
        draw.text((x, y), ch, font=font, fill=0)
        arr = np.array(img)
        if float(np.mean(arr)) <= 250:
            registered.append(ch)

    with open(output_path, "w", encoding="utf-8") as fp:
        fp.write("".join(registered))
    print(f"[Info] Registered characters saved to: {output_path}")


def render_char_to_png(font_path: str, char: str, out_path_or_buffer: str | io.BytesIO, size: int = 256) -> Image.Image:
    """Render a single glyph to PNG using the given font.

    Args:
        font_path: Path to the font file.
        char: Character to draw.
        out_path_or_buffer: Output path or ``io.BytesIO``.
        size: Square image size in pixels.

    Returns:
        The rendered ``PIL.Image`` instance.
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
    if isinstance(out_path_or_buffer, str):
        img.save(out_path_or_buffer)
    else:
        img.save(out_path_or_buffer, format="PNG")
        out_path_or_buffer.seek(0)
    return img


def compute_metrics(fake: torch.Tensor, real: torch.Tensor) -> dict:
    """Calculate PSNR and SSIM metrics.

    Args:
        fake: Generated tensor in ``[-1, 1]`` range.
        real: Target tensor in ``[-1, 1]`` range.

    Returns:
        Dictionary with ``psnr`` and ``ssim`` values.
    """
    fake_np = fake.add(1).div(2).clamp(0, 1).cpu().numpy()
    real_np = real.add(1).div(2).clamp(0, 1).cpu().numpy()
    psnr_vals: list[float] = []
    ssim_vals: list[float] = []
    b = fake_np.shape[0]
    for i in range(b):
        f_img = fake_np[i, 0]
        r_img = real_np[i, 0]
        psnr_vals.append(calc_psnr(r_img, f_img, data_range=1.0))
        win = min(7, f_img.shape[0], f_img.shape[1])
        if win % 2 == 0:
            win -= 1
        ssim_vals.append(calc_ssim(r_img, f_img, data_range=1.0, win_size=win))
    return {"psnr": float(np.mean(psnr_vals)), "ssim": float(np.mean(ssim_vals))}


def create_stratified_split(
    data_dir: str, char_codes_all: list[int], val_split_ratio: float, n_bins: int = 5
) -> tuple[list[int], list[int]]:
    """Split codes into train/val using edge area stratification."""
    stats = []
    for code in char_codes_all:
        path = os.path.join(data_dir, f"{code}.pt")
        if not os.path.exists(path):
            continue
        item = torch.load(path, map_location="cpu", mmap=True)
        area = item.get("edge_area")
        if area is None and "skeleton" in item:
            ske = item["skeleton"].float().unsqueeze(0) / 255.0
            area = float(edge_map(ske * 2.0 - 1.0).mean().item())
        stats.append({"code": code, "edge_area": area or 0.0})
    if not stats:
        return char_codes_all, []
    areas = np.array([s["edge_area"] for s in stats])
    bins = np.linspace(areas.min(), areas.max(), n_bins + 1)
    binned = np.digitize(areas, bins)
    train_codes: list[int] = []
    val_codes: list[int] = []
    for idx in range(1, n_bins + 1):
        codes_in_bin = [stats[i]["code"] for i, b in enumerate(binned) if b == idx]
        random.shuffle(codes_in_bin)
        val_size = int(len(codes_in_bin) * val_split_ratio)
        val_codes.extend(codes_in_bin[:val_size])
        train_codes.extend(codes_in_bin[val_size:])
    return train_codes, val_codes


@torch.no_grad()
def validate_epoch(
    generator: nn.Module,
    loader: DataLoader,
    device: torch.device,
    writer: SummaryWriter,
    epoch: int,
    metric_ratio: float = 1.0,
) -> float:
    """Run validation and log metrics.

    Args:
        generator: Trained generator model.
        loader: Validation ``DataLoader``.
        device: Torch device.
        writer: TensorBoard writer.
        epoch: Current epoch number.

    Returns:
        Average PSNR value over the validation set.
    """
    generator.eval()
    psnrs: list[float] = []
    ssims: list[float] = []
    ious: list[float] = []
    widths: list[float] = []
    first = True
    vbar = tqdm(loader, desc=f"Validating Epoch {epoch}")
    max_b = int(len(loader) * metric_ratio)
    for i, (src, real, ske) in enumerate(vbar):
        src, real, ske = src.to(device), real.to(device), ske.to(device)
        fake = generator(src)
        metrics = compute_metrics(fake, real)
        psnrs.append(metrics["psnr"])
        ssims.append(metrics["ssim"])
        if i < max_b:
            emap = edge_map(fake)
            ious.append(edge_iou(fake, ske).item())
            widths.append(float(emap.mean().item()))
        if first:
            grid = vutils.make_grid(
                torch.cat([
                    src.add(1).div(2),
                    fake.add(1).div(2),
                    real.add(1).div(2),
                ]),
                nrow=src.size(0),
            )
            writer.add_image("Validation/Comparison", grid, epoch)
            first = False

    avg_psnr = float(np.mean(psnrs))
    avg_ssim = float(np.mean(ssims))
    avg_iou = float(np.mean(ious)) if ious else 0.0
    avg_w = float(np.mean(widths)) if widths else 0.0
    writer.add_scalar("Validation/PSNR", avg_psnr, epoch)
    writer.add_scalar("Validation/SSIM", avg_ssim, epoch)
    writer.add_scalar("Validation/Edge_IoU", avg_iou, epoch)
    writer.add_scalar("Validation/Mean_Edge_Width", avg_w, epoch)
    print(
        f"Validation Epoch {epoch} -> Avg PSNR: {avg_psnr:.2f}, Avg SSIM: {avg_ssim:.3f}, Avg IoU: {avg_iou:.3f}, Avg W: {avg_w:.3f}"
    )
    return avg_psnr


class VGGPerceptualLoss(nn.Module):
    """VGG を用いた知覚損失。"""

    def __init__(self, layers: tuple[str, ...] = ("relu1_2", "relu2_2", "relu3_3", "relu4_3"), weights: dict | None = None) -> None:
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features.eval()
        self.layers = layers
        self.weights = weights or {l: 1.0 for l in layers}
        self.outputs: dict[str, torch.Tensor] = {}
        self.hooks = {}
        layer_map = {"relu1_2": 3, "relu2_2": 8, "relu3_3": 15, "relu4_3": 22}
        for name, idx in layer_map.items():
            if name in self.layers:
                self.hooks[name] = vgg[idx].register_forward_hook(
                    lambda m, i, o, key=name: self.outputs.update({key: o})
                )
        self.vgg = vgg
        for p in self.vgg.parameters():
            p.requires_grad = False
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return perceptual loss between ``x`` and ``y``."""
        x3 = self.normalize(x.repeat(1, 3, 1, 1))
        y3 = self.normalize(y.repeat(1, 3, 1, 1))
        self.outputs.clear()
        _ = self.vgg(torch.cat([x3, y3], dim=0))
        loss = 0.0
        b = x.size(0)
        for name in self.layers:
            fx = self.outputs[name][:b]
            fy = self.outputs[name][b:]
            loss += self.weights.get(name, 1.0) * F.l1_loss(fx, fy)
        return loss

    def __del__(self) -> None:
        for h in self.hooks.values():
            h.remove()


def morphology_transform(img: Image.Image, min_shift: int = 1, max_shift: int = 2) -> Image.Image:
    """Randomly dilate or erode the glyph to shift stroke width."""
    arr = np.array(img)
    binary = arr < 200
    shift = random.randint(min_shift, max_shift)
    if random.random() > 0.5:
        processed = binary_dilation(binary, iterations=shift)
    else:
        processed = binary_erosion(binary, iterations=shift)
    out = np.where(processed, 0, 255).astype(np.uint8)
    return Image.fromarray(out)

_SOBEL_X = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
_SOBEL_Y = _SOBEL_X.transpose(-1,-2)

def edge_map(x: torch.Tensor) -> torch.Tensor:
    """Return gradient magnitude via Sobel filter."""
    kx = _SOBEL_X.to(x.device)
    ky = _SOBEL_Y.to(x.device)
    ex = F.conv2d(x, kx, padding=1)
    ey = F.conv2d(x, ky, padding=1)
    return torch.sqrt(ex ** 2 + ey ** 2 + 1e-6)

def stroke_loss(fake: torch.Tensor, ske: torch.Tensor) -> torch.Tensor:
    """Approximate stroke consistency loss with edge maps."""
    return F.l1_loss(edge_map(fake), edge_map(ske))


def edge_iou(
    fake: torch.Tensor, ske: torch.Tensor, thr_ratio: float = 0.25
) -> torch.Tensor:
    """Return IoU of edge maps with adaptive thresholds."""
    fe = edge_map(fake)
    se = edge_map(ske)
    thr_f = fe.amax(dim=(-1, -2), keepdim=True) * thr_ratio
    thr_s = se.amax(dim=(-1, -2), keepdim=True) * thr_ratio
    f = fe > thr_f
    s = se > thr_s
    inter = torch.sum(f & s, dim=(1, 2, 3)).float()
    union = torch.sum(f | s, dim=(1, 2, 3)).float()
    return torch.mean(inter / (union + 1e-6))

def add_noise(t: torch.Tensor, std: float = 0.05) -> torch.Tensor:
    """Add Gaussian noise for discriminator input."""
    if std == 0:
        return t
    return t + torch.randn_like(t) * std





class FontPairDataset(Dataset):
    """Provide paired font images as ``torch.utils.data.Dataset``."""

    def __init__(
        self,
        source_dir: str,
        target_dir: str,
        transform_source_pil: Callable[[Image.Image], Image.Image] | None = None,
        transform_target_pil: Callable[[Image.Image], Image.Image] | None = None,
        transform_skel_pil: Callable[[Image.Image], Image.Image] | None = None,
        img_size: int = 256,
        is_stage2: bool = False,
        rehearsal_ratio: float = 0.0,
        rehearsal_source_dir: str | None = None,
        rehearsal_target_dir: str | None = None,
        char_codes: list[int] | None = None,
        skeleton_dir: str | None = None,
    ) -> None:
        """Initialize dataset with optional skeleton directory.

        Args:
            source_dir: Directory containing source font images.
            target_dir: Directory containing target font images.
            transform_source_pil: Optional transform applied to the source image.
            transform_target_pil: Optional transform applied to the target image.
            transform_skel_pil: Optional transform applied to the skeleton image.
            img_size: Output image size.
            is_stage2: Whether stage2 training is active.
            rehearsal_ratio: Ratio of rehearsal samples used in stage2.
            rehearsal_source_dir: Directory for rehearsal source images.
            rehearsal_target_dir: Directory for rehearsal target images.
            char_codes: List of character codes used for filtering.
            skeleton_dir: Directory containing skeleton tensors or images.
        """
        self.skel_paths: list[str] | None = None
        if char_codes is not None:
            self.src_paths = [os.path.join(source_dir, f"{c}.png") for c in char_codes]
            self.tgt_paths = [os.path.join(target_dir, f"{c}.png") for c in char_codes]
            pairs = [
                (s, t)
                for s, t in zip(self.src_paths, self.tgt_paths)
                if os.path.exists(s) and os.path.exists(t)
            ]
            self.src_paths, self.tgt_paths = zip(*pairs) if pairs else ([], [])
        else:
            self.src_paths = sorted(glob.glob(os.path.join(source_dir, "*.png")))
            self.tgt_paths = [os.path.join(target_dir, os.path.basename(p)) for p in self.src_paths]

        if skeleton_dir:
            if char_codes is not None:
                self.skel_paths = [os.path.join(skeleton_dir, f"{c}.pt") for c in char_codes]
            else:
                self.skel_paths = [os.path.join(skeleton_dir, os.path.splitext(os.path.basename(p))[0] + ".pt") for p in self.src_paths]

        if (
            is_stage2
            and rehearsal_ratio > 0
            and rehearsal_source_dir
            and rehearsal_target_dir
        ):
            print(f"[Info] Applying Rehearsal Strategy with ratio: {rehearsal_ratio}")
            num_rehearsal = int(len(self.src_paths) * rehearsal_ratio)
            rehearsal_candidates = sorted(glob.glob(os.path.join(rehearsal_source_dir, "*.png")))
            if rehearsal_candidates:
                chosen = random.sample(rehearsal_candidates, min(num_rehearsal, len(rehearsal_candidates)))
                self.src_paths.extend(chosen)
                self.tgt_paths.extend(
                    [os.path.join(rehearsal_target_dir, os.path.basename(p)) for p in chosen]
                )
                print(f"  Added {len(chosen)} rehearsal samples. Total samples: {len(self.src_paths)}")

        self.tsf_s = transform_source_pil
        self.tsf_t = transform_target_pil
        self.tsf_k = transform_skel_pil
        self.to_tensor = T.ToTensor()
        self.norm = T.Normalize((0.5,), (0.5,))
        self.img_size = img_size

    def __len__(self) -> int:
        return len(self.src_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return input tensor, target tensor and optional skeleton."""
        s_img = Image.open(self.src_paths[index]).convert("L")
        t_img = Image.open(self.tgt_paths[index]).convert("L")
        sk_tensor: torch.Tensor | None = None
        if self.skel_paths:
            sk_path = self.skel_paths[index]
            if sk_path.endswith(".pt") and os.path.exists(sk_path):
                k_uint8 = torch.load(sk_path, map_location="cpu")
                sk_tensor = k_uint8.float().unsqueeze(0) / 255.0
            elif os.path.exists(sk_path):
                sk_img = Image.open(sk_path).convert("L")
            else:
                sk_img = None
        else:
            sk_img = None

        if s_img.size != (self.img_size, self.img_size):
            s_img = s_img.resize((self.img_size, self.img_size), Image.BICUBIC)
        if t_img.size != (self.img_size, self.img_size):
            t_img = t_img.resize((self.img_size, self.img_size), Image.BICUBIC)
        if sk_img and sk_img.size != (self.img_size, self.img_size):
            sk_img = sk_img.resize((self.img_size, self.img_size), Image.BICUBIC)

        if self.tsf_s:
            s_img = self.tsf_s(s_img)
        if self.tsf_t:
            t_img = self.tsf_t(t_img)
        if sk_img and self.tsf_k:
            sk_img = self.tsf_k(sk_img)

        s_tensor = self.norm(self.to_tensor(s_img))
        t_tensor = self.norm(self.to_tensor(t_img))
        if sk_tensor is None and sk_img is not None:
            sk_tensor = self.norm(self.to_tensor(sk_img))
        if sk_tensor is None:
            sk_tensor = torch.zeros(1, self.img_size, self.img_size)

        sk_tensor = torch.where(sk_tensor > 0, 1.0, -1.0)
        s_tensor = torch.cat([sk_tensor, s_tensor], dim=0) if self.skel_paths else s_tensor
        return s_tensor, t_tensor, sk_tensor


class PreprocessedFontDataset(Dataset):
    """Load pre-rendered ``.pt`` samples for fast training."""

    def __init__(self, data_dir: str, char_codes: list[int] | None = None) -> None:
        self.paths = []
        if char_codes is None:
            files = sorted(glob.glob(os.path.join(data_dir, "*.pt")))
            self.paths = files
        else:
            for c in char_codes:
                p = os.path.join(data_dir, f"{c}.pt")
                if os.path.exists(p):
                    self.paths.append(p)
        self.norm = T.Normalize((0.5,), (0.5,))

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # lazy-load with mmap で RAM 抑制
        item = torch.load(self.paths[index], map_location="cpu", mmap=True)
        src = self.norm(item["source"].float() / 255.0)
        tgt = self.norm(item["target"].float() / 255.0)
        ske = self.norm(item["skeleton"].float() / 255.0)
        src = torch.cat([ske, src], dim=0)
        return src, tgt, ske


def get_norm_layer(norm_type: str = "instance") -> type[nn.Module]:
    """Return normalization layer class."""
    if norm_type == "batch":
        return nn.BatchNorm2d
    if norm_type == "instance":
        return nn.InstanceNorm2d
    raise ValueError(norm_type)


class UNetSkipConnectionBlock(nn.Module):
    """A building block of the U-Net generator."""

    def __init__(self, outer_nc: int, inner_nc: int, in_nc: int | None = None, submodule: nn.Module | None = None, outermost: bool = False, innermost: bool = False, norm_layer: type[nn.Module] | None = None, use_dropout: bool = False) -> None:
        super().__init__()
        self.outermost = outermost
        if in_nc is None:
            in_nc = outer_nc
        down = [nn.Conv2d(in_nc, inner_nc, 4, 2, 1, bias=norm_layer is nn.InstanceNorm2d), nn.LeakyReLU(0.2, True)]
        if not innermost:
            if norm_layer:
                down += [norm_layer(inner_nc)]
        up = [nn.ReLU(True), nn.ConvTranspose2d(inner_nc * (1 if innermost else 2), outer_nc, 4, 2, 1, bias=norm_layer is nn.InstanceNorm2d)]
        if norm_layer:
            up += [norm_layer(outer_nc)]
        if use_dropout:
            up += [nn.Dropout(0.5)]
        model = down + ([submodule] if submodule else []) + up
        if outermost:
            model += [nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.outermost:
            return self.model(x)
        return torch.cat([x, self.model(x)], 1)


class UNetGenerator(nn.Module):
    """U-Net generator for pix2pix."""

    def __init__(self, in_nc: int = 1, out_nc: int = 1, num_downs: int = 8, ngf: int = 64, norm_type: str = "instance", use_dropout: bool = False) -> None:
        super().__init__()
        norm_layer = get_norm_layer(norm_type)
        unet: nn.Module = UNetSkipConnectionBlock(ngf * 8, ngf * 8, innermost=True, norm_layer=norm_layer)
        for _ in range(num_downs - 5):
            unet = UNetSkipConnectionBlock(ngf * 8, ngf * 8, submodule=unet, norm_layer=norm_layer, use_dropout=use_dropout)
        unet = UNetSkipConnectionBlock(ngf * 4, ngf * 8, submodule=unet, norm_layer=norm_layer)
        unet = UNetSkipConnectionBlock(ngf * 2, ngf * 4, submodule=unet, norm_layer=norm_layer)
        unet = UNetSkipConnectionBlock(ngf, ngf * 2, submodule=unet, norm_layer=norm_layer)
        self.model = UNetSkipConnectionBlock(out_nc, ngf, in_nc=in_nc, submodule=unet, outermost=True, norm_layer=norm_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def weights_init(module: nn.Module) -> None:
    """Initialize convolutional and normalization layers."""
    classname = module.__class__.__name__
    if "Conv" in classname:
        nn.init.normal_(module.weight, 0.0, 0.02)
    elif "BatchNorm" in classname:
        nn.init.normal_(module.weight, 1.0, 0.02)
        nn.init.constant_(module.bias, 0.0)


class PatchDiscriminator(nn.Module):
    """PatchGAN discriminator."""

    def __init__(self, in_nc: int = 2, ndf: int = 64, n_layers: int = 3, norm_type: str = "instance") -> None:
        super().__init__()
        norm_layer = get_norm_layer(norm_type)
        kw, pad = 4, 1
        seq: list[nn.Module] = [nn.Conv2d(in_nc, ndf, kw, 2, pad), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            seq += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kw, 2, pad, bias=False), norm_layer(ndf * nf_mult), nn.LeakyReLU(0.2, True)]
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        seq += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kw, 1, pad, bias=False), norm_layer(ndf * nf_mult), nn.LeakyReLU(0.2, True)]
        seq += [nn.Conv2d(ndf * nf_mult, 1, kw, 1, pad)]
        self.model = nn.Sequential(*seq)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        return self.model(torch.cat([src, tgt], 1))


def train(config: dict) -> None:
    """Run pix2pix training with the provided configuration."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(config.get("seed", 2025))
    run_dir = os.path.join(
        config["checkpoint_dir"], f"{config['stage_name']}_{time.strftime('%Y%m%d-%H%M%S')}"
    )
    log_dir = os.path.join(run_dir, "logs")
    model_dir = os.path.join(run_dir, "models")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    writer.add_text("Config", f"<pre>{pprint.pformat(config)}</pre>", 0)
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as fp:
        json.dump(config, fp, indent=4, ensure_ascii=False)
    if config.get("preprocessed_dir"):
        files = sorted(glob.glob(os.path.join(config["preprocessed_dir"], "*.pt")))
        all_codes = [int(os.path.splitext(os.path.basename(p))[0]) for p in files]
    else:
        os.makedirs(config["source_data_dir"], exist_ok=True)
        os.makedirs(config["target_data_dir"], exist_ok=True)
        char_map = build_learning_char_map(
            font_path=config["target_font_path"],
            candidate_chars=config["candidate_chars"],
            external_list_path=config.get("learning_list_file"),
        )
        for code, glyph in char_map.items():
            render_char_to_png(
                config["ref_font_path"],
                glyph,
                os.path.join(config["source_data_dir"], f"{code}.png"),
                size=config["img_size"],
            )
            render_char_to_png(
                config["target_font_path"],
                glyph,
                os.path.join(config["target_data_dir"], f"{code}.png"),
                size=config["img_size"],
            )
        all_codes = list(char_map.keys())

    random.shuffle(all_codes)
    if config.get("preprocessed_dir"):
        train_codes, val_codes = create_stratified_split(
            config["preprocessed_dir"], all_codes, config.get("val_split_ratio", 0.1)
        )
    else:
        val_size = int(len(all_codes) * config.get("val_split_ratio", 0.1))
        val_codes = all_codes[:val_size]
        train_codes = all_codes[val_size:]
    assert not set(train_codes) & set(val_codes), "Overlap detected between train and validation sets"
    print(f"[Info] Data split -> Train: {len(train_codes)}, Validation: {len(val_codes)}")

    tf_s = T.Compose(
        [
            T.RandomApply([lambda img: morphology_transform(img)], p=0.2),
            T.RandomAffine(
                degrees=1.5,
                translate=(0.02, 0.02),
                scale=(0.98, 1.02),
                fill=255,
            ),
        ]
    )
    tf_t = T.Compose([T.RandomApply([lambda img: morphology_transform(img)], p=0.1)])
    if config.get("preprocessed_dir"):
        train_ds = PreprocessedFontDataset(config["preprocessed_dir"], train_codes)
        val_ds = PreprocessedFontDataset(config["preprocessed_dir"], val_codes)
        in_nc = 2
    else:
        train_ds = FontPairDataset(
            config["source_data_dir"],
            config["target_data_dir"],
            transform_source_pil=tf_s,
            transform_target_pil=tf_t,
            transform_skel_pil=None,
            img_size=config["img_size"],
            is_stage2=config["stage_name"] == "progressive_512",
            rehearsal_ratio=config.get("rehearsal_ratio", 0.0),
            rehearsal_source_dir=config.get("rehearsal_source_dir"),
            rehearsal_target_dir=config.get("rehearsal_target_dir"),
            char_codes=train_codes,
            skeleton_dir=config.get("skeleton_dir"),
        )
        val_ds = FontPairDataset(
            config["source_data_dir"],
            config["target_data_dir"],
            transform_skel_pil=None,
            img_size=config["img_size"],
            char_codes=val_codes,
            skeleton_dir=config.get("skeleton_dir"),
        )
        in_nc = 2 if config.get("skeleton_dir") else 1
    dl = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config.get("num_workers", 0),
        pin_memory=True,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config.get("num_workers", 0),
        pin_memory=True,
    )
    in_nc = 2 if config.get("skeleton_dir") else 1
    G = UNetGenerator(in_nc=in_nc, num_downs=config["num_unet_downs"], norm_type=config["norm_type"]).to(device)
    if config.get("use_compile") and hasattr(torch, "compile"):
        print("[Info] Compiling Generator with torch.compile")
        G = torch.compile(G)
    D = PatchDiscriminator(in_nc=in_nc + 1, n_layers=config["d_n_layers"], norm_type=config["norm_type"]).to(device)
    if config.get("load_G_path"):
        G.load_state_dict(torch.load(config["load_G_path"], map_location=device))
    else:
        G.apply(weights_init)

    if config.get("load_D_path"):
        D.load_state_dict(torch.load(config["load_D_path"], map_location=device))
    else:
        D.apply(weights_init)
    if config.get("freeze_G_layers", 0) > 0:
        for i, layer in enumerate(G.model.children()):
            if i < config["freeze_G_layers"]:
                for p in layer.parameters():
                    p.requires_grad = False
    cgan = nn.BCEWithLogitsLoss()
    cl1 = nn.L1Loss()
    cperc = VGGPerceptualLoss().to(device)
    opt_g = optim.Adam(filter(lambda p: p.requires_grad, G.parameters()), lr=config["lr_G"], betas=(0.5, 0.999))
    opt_d = optim.Adam(D.parameters(), lr=config["lr_D"], betas=(0.5, 0.999))
    if config.get("scheduler_type", "cosine") == "plateau":
        sched_g = ReduceLROnPlateau(
            opt_g, mode="max", patience=config.get("plateau_patience", 10), factor=0.5
        )
        sched_d = ReduceLROnPlateau(
            opt_d, mode="max", patience=config.get("plateau_patience", 10), factor=0.5
        )
    else:
        sched_g = CosineAnnealingLR(opt_g, T_max=config["epochs"], eta_min=1e-7)
        sched_d = CosineAnnealingLR(opt_d, T_max=config["epochs"], eta_min=1e-7)
    scaler = torch.cuda.amp.GradScaler(enabled=config["use_amp"])
    step = 0
    best_psnr = 0.0
    start_epoch = 1
    if config.get("resume_ckpt"):
        print(f"[Info] Resuming from {config['resume_ckpt']}")
        ckpt = torch.load(config["resume_ckpt"], map_location=device)
        G.load_state_dict(ckpt["G_state_dict"])
        D.load_state_dict(ckpt["D_state_dict"])
        opt_g.load_state_dict(ckpt["opt_G"])
        opt_d.load_state_dict(ckpt["opt_D"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_psnr = ckpt.get("best_psnr", 0.0)
        step = ckpt.get("global_step", 0)
    epochs_no_improve = 0
    for epoch in range(start_epoch, config["epochs"] + 1):
        G.train()
        D.train()
        base_sw = 0.0
        if config.get("stroke_lambda", 0) > 0:
            base_sw = config["stroke_lambda"] * 0.5 * (
                1 + math.cos(math.pi * epoch / config["epochs"])
            )
        pbar = tqdm(dl, desc=f"Epoch {epoch}/{config['epochs']}")
        for i, (src, real, ske) in enumerate(pbar):
            src, real, ske = src.to(device), real.to(device), ske.to(device)
            cur_std = config.get("d_noise_std", 0.05) * max(0.0, 1 - epoch / config["epochs"])
            opt_g.zero_grad()
            with torch.cuda.amp.autocast(enabled=config["use_amp"]):
                fake = G(src)
                l_gan = cgan(
                    D(add_noise(src, cur_std), add_noise(fake, cur_std)),
                    torch.ones_like(D(src, fake)),
                )
                l_l1 = cl1(fake, real) * config["l1_lambda"]
                l_p = cperc(fake, real) * config["perceptual_lambda"]
                if config.get("stroke_lambda", 0) > 0:
                    area = torch.mean(edge_map(ske)).item()
                    w_stroke = base_sw * (area / config["mean_edge_area"])
                    l_sk = stroke_loss(fake, ske) * w_stroke
                else:
                    l_sk = torch.tensor(0.0, device=device)
                l_g = (l_gan + l_l1 + l_p + l_sk) / config["accum_steps"]
            scaler.scale(l_g).backward()
            if (i + 1) % config["accum_steps"] == 0:
                scaler.unscale_(opt_g)
                torch.nn.utils.clip_grad_norm_(
                    G.parameters(), config.get("clip_grad_norm", 1.0)
                )
                scaler.step(opt_g)
                scaler.update()
                opt_g.zero_grad()
            opt_d.zero_grad()
            with torch.cuda.amp.autocast(enabled=config["use_amp"]):
                l_dr = cgan(
                    D(add_noise(src, cur_std), add_noise(real, cur_std)),
                    torch.ones_like(D(src, real)),
                )
                l_df = cgan(
                    D(add_noise(src, cur_std), add_noise(fake.detach(), cur_std)),
                    torch.zeros_like(D(src, fake.detach())),
                )
                l_d = ((l_dr + l_df) * 0.5) / config["accum_steps"]
            scaler.scale(l_d).backward()
            if (i + 1) % config["accum_steps"] == 0:
                scaler.unscale_(opt_d)
                torch.nn.utils.clip_grad_norm_(
                    D.parameters(), config.get("clip_grad_norm", 1.0)
                )
                scaler.step(opt_d)
                scaler.update()
                opt_d.zero_grad()
            if step % config["log_freq"] == 0:
                mets = compute_metrics(fake, real)
                post = {
                    "L_D": f"{l_d.item()*config['accum_steps']:.3f}",
                    "L_G": f"{(l_gan + l_l1 + l_p).item():.3f}",
                    "PSNR": f"{mets['psnr']:.2f}",
                    "SSIM": f"{mets['ssim']:.3f}",
                }
                if config.get("stroke_lambda", 0) > 0:
                    post["L_G_stroke"] = f"{l_sk.item():.3f}"
                pbar.set_postfix(post)
                writer.add_scalar("Loss/D", l_d.item() * config["accum_steps"], step)
                writer.add_scalar("Loss/G_GAN", l_gan.item(), step)
                writer.add_scalar("Loss/G_L1", l_l1.item(), step)
                writer.add_scalar("Loss/G_Perceptual", l_p.item(), step)
                if config.get("stroke_lambda", 0) > 0:
                    writer.add_scalar("Loss/G_Stroke", l_sk.item(), step)
                writer.add_scalar("Metrics/PSNR", mets["psnr"], step)
                writer.add_scalar("Metrics/SSIM", mets["ssim"], step)
                writer.add_scalar(
                    "Memory/Allocated (GB)",
                    torch.cuda.memory_allocated(0) / 1024 ** 3,
                    step,
                )
            step += 1
        avg_psnr = validate_epoch(
            G, val_dl, device, writer, epoch, config.get("val_metric_ratio", 1.0)
        )
        if isinstance(sched_g, ReduceLROnPlateau):
            sched_g.step(avg_psnr)
            sched_d.step(avg_psnr)
        else:
            sched_g.step()
            sched_d.step()
        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            epochs_no_improve = 0
            torch.save(
                {
                    "epoch": epoch,
                    "global_step": step,
                    "best_psnr": best_psnr,
                    "G_state_dict": G.state_dict(),
                    "D_state_dict": D.state_dict(),
                    "opt_G": opt_g.state_dict(),
                    "opt_D": opt_d.state_dict(),
                    "sch_G": sched_g.state_dict(),
                    "sch_D": sched_d.state_dict(),
                },
                os.path.join(model_dir, "ckpt_best.pth"),
            )
            print(f"\u2728 New best model saved with PSNR: {best_psnr:.2f}")
        else:
            epochs_no_improve += 1
            if (
                config.get("early_stop_patience")
                and epochs_no_improve >= config["early_stop_patience"]
            ):
                print(
                    f"Early stopping triggered after {epochs_no_improve} epochs with no improvement."
                )
                break

        if epoch % config["save_epoch_freq"] == 0 or epoch == config["epochs"]:
            torch.save(
                {
                    "epoch": epoch,
                    "global_step": step,
                    "best_psnr": best_psnr,
                    "G_state_dict": G.state_dict(),
                    "D_state_dict": D.state_dict(),
                    "opt_G": opt_g.state_dict(),
                    "opt_D": opt_d.state_dict(),
                    "sch_G": sched_g.state_dict(),
                    "sch_D": sched_d.state_dict(),
                },
                os.path.join(model_dir, f"ckpt_ep{epoch}.pth"),
            )
    writer.close()


def main() -> None:
    """Parse command line arguments and execute training."""
    parser = argparse.ArgumentParser(description="Advanced Font Generation")
    parser.add_argument(
        "--stage",
        type=str,
        required=True,
        choices=["s1_256", "s2_512"],
        help="Training stage to run.",
    )
    parser.add_argument("--ref_font", type=str, required=True, help="Reference font path")
    parser.add_argument("--target_font", type=str, required=True, help="Target font path")
    parser.add_argument("--char_list", type=str, default=None, help="External character list")
    parser.add_argument(
        "--candidate_char_file",
        type=str,
        default=None,
        help="Candidate character file for auto detection",
    )
    parser.add_argument("--include_chars", type=str, default=None, help="Characters to always include")
    parser.add_argument("--exclude_chars", type=str, default=None, help="Characters to exclude")
    parser.add_argument(
        "--range_start",
        type=lambda s: int(s, 16),
        default=None,
        help="Range start code point in hex (e.g. 3040)",
    )
    parser.add_argument(
        "--range_end",
        type=lambda s: int(s, 16),
        default=None,
        help="Range end code point in hex (e.g. 309F)",
    )
    parser.add_argument(
        "--data_dir_root",
        type=str,
        default="./data",
        help="Root directory for training images",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints/gd_highway_pro",
        help="Directory to save checkpoints",
    )
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size override")
    parser.add_argument("--skeleton_dir", type=str, default=None, help="Directory of skeleton images")
    parser.add_argument("--preprocessed_dir", type=str, default=None, help="Directory of preprocessed .pt files")
    parser.add_argument(
        "--stroke_lambda",
        type=float,
        default=0.0,
        help="Weight for stroke consistency loss",
    )
    parser.add_argument(
        "--use_compile",
        action="store_true",
        help="Use torch.compile if available",
    )
    parser.add_argument(
        "--resume_ckpt",
        type=str,
        default=None,
        help="Checkpoint path to resume training",
    )
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=None,
        help="Epochs to wait for improvement before stopping",
    )
    parser.add_argument(
        "--scheduler_type",
        choices=["cosine", "plateau"],
        default="cosine",
        help="Type of LR scheduler",
    )
    parser.add_argument(
        "--val_metric_ratio",
        type=float,
        default=1.0,
        help="Fraction of validation batches for heavy metrics",
    )
    parser.add_argument(
        "--dump_char_string",
        type=str,
        default=None,
        help="Output detected characters to a file and exit",
    )
    parser.add_argument(
        "--dump_registed_all_char",
        type=str,
        default=None,
        help="Output all nonblank registered characters in the font and exit",
    )

    args = parser.parse_args()
    print(f"{os.path.basename(__file__)} launched.")

    candidate_chars = build_candidate_chars(
        include_file=args.include_chars or args.candidate_char_file,
        exclude_file=args.exclude_chars,
        range_start=args.range_start,
        range_end=args.range_end,
    )
    if not candidate_chars:
        if args.candidate_char_file and os.path.exists(args.candidate_char_file):
            candidate_chars = load_char_list_from_file(args.candidate_char_file)
        else:
            candidate_chars = {
                ord(c): c
                for c in (
                    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                    "abcdefghijklmnopqrstuvwxyz"
                    "0123456789"
                    "あいうえおかきくけこさしすせそ"
                    "たちつてと"
                    "なにぬねのはひふへほ"
                    "まみむめもやゆよらりるれろわをん"
                    "道高速路"
                )
            }

    candidate_chars = filter_by_both_fonts(
        candidate_chars,
        base_font=args.target_font,
        ref_font=args.ref_font,
    )

    if args.dump_char_string:
        dump_char_string(
            target_font=args.target_font,
            ref_font=args.ref_font,
            candidate_chars=candidate_chars,
            output_path=args.dump_char_string,
            external_list_path=args.char_list,
        )
        return

    if args.dump_registed_all_char:
        dump_registed_all_char(
            font_path=args.target_font,
            output_path=args.dump_registed_all_char,
        )
        return

    base_config = {
        "norm_type": "instance",
        "l1_lambda": 100.0,
        "perceptual_lambda": 1.0,
        "use_amp": True,
        "clip_grad_norm": 1.0,
        "log_freq": 100,
        "save_epoch_freq": 10,
        "val_split_ratio": 0.1,
        "num_workers": 0,
        "ref_font_path": args.ref_font,
        "target_font_path": args.target_font,
        "candidate_chars": candidate_chars,
        "learning_list_file": args.char_list,
        "checkpoint_dir": args.checkpoint_dir,
        "skeleton_dir": args.skeleton_dir,
        "preprocessed_dir": args.preprocessed_dir,
        "stroke_lambda": args.stroke_lambda,
        "use_compile": args.use_compile,
        "resume_ckpt": args.resume_ckpt,
        "early_stop_patience": args.early_stop_patience,
        "scheduler_type": args.scheduler_type,
        "plateau_patience": args.early_stop_patience,
        "val_metric_ratio": args.val_metric_ratio,
        "mean_edge_area": 0.07,
        "d_noise_std": 0.05,
    }

    if args.stage == "s1_256":
        config = base_config.copy()
        config.update(
            {
                "stage_name": "pretrain_256",
                "img_size": 256,
                "num_unet_downs": 8,
                "d_n_layers": 3,
                "epochs": 150,
                "batch_size": args.batch_size or 8,
                "lr_G": 2e-4,
                "lr_D": 2e-4,
                "accum_steps": 2,
                "source_data_dir": os.path.join(args.data_dir_root, "train_s1/source"),
                "target_data_dir": os.path.join(args.data_dir_root, "train_s1/target"),
            }
        )
        train(config)
    else:
        last_epoch_s1 = 150
        config = base_config.copy()
        config.update(
            {
                "stage_name": "progressive_512",
                "img_size": 512,
                "num_unet_downs": 9,
                "d_n_layers": 4,
                "epochs": 100,
                "batch_size": args.batch_size or 2,
                "lr_G": 1e-4,
                "lr_D": 1e-4,
                "accum_steps": 8,
                "source_data_dir": os.path.join(args.data_dir_root, "train_s2/source"),
                "target_data_dir": os.path.join(args.data_dir_root, "train_s2/target"),
                "load_G_path": os.path.join(
                    args.checkpoint_dir, f"G_pretrain_256_ep{last_epoch_s1}.pth"
                ),
                "freeze_G_layers": 2,
                "rehearsal_ratio": 0.1,
                "rehearsal_source_dir": os.path.join(args.data_dir_root, "train_s1/source"),
                "rehearsal_target_dir": os.path.join(args.data_dir_root, "train_s1/target"),
            }
        )
        train(config)


if __name__ == "__main__":
    main()
