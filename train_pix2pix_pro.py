# -*- coding: utf-8 -*-
"""train_pix2pix_pro.py — train_pix2pix_pro

概要:
    256px 事前学習と 512px プログレッシブ学習を自動で行うスクリプト。

:author: pumpCurry
:copyright: (c) pumpCurry 2025 / 5r4ce2
:license: MIT
:version: 1.0.30 (PR #14)
:since:   1.0.30 (PR #14)
:last-modified: 2025-06-22 15:26:00 JST+9
:todo:
    - Improve configurability via YAML
"""

import os
import glob
import random
import time
import io

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.transforms as T
import torchvision.models as models
from skimage.metrics import peak_signal_noise_ratio as calc_psnr
from skimage.metrics import structural_similarity as calc_ssim
from scipy.ndimage import binary_dilation, binary_erosion


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


class FontPairDataset(Dataset):
    """Provide paired font images as ``torch.utils.data.Dataset``."""

    def __init__(self, source_dir: str, target_dir: str, transform_source_pil: callable | None = None, transform_target_pil: callable | None = None, img_size: int = 256) -> None:
        self.src = sorted(glob.glob(os.path.join(source_dir, "*.png")))
        self.tgt = [os.path.join(target_dir, os.path.basename(p)) for p in self.src]
        self.tsf_s = transform_source_pil
        self.tsf_t = transform_target_pil
        self.to_tensor = T.ToTensor()
        self.norm = T.Normalize((0.5,), (0.5,))
        self.img_size = img_size

    def __len__(self) -> int:
        return len(self.src)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        s = Image.open(self.src[index]).convert("L")
        t = Image.open(self.tgt[index]).convert("L")
        if s.size != (self.img_size, self.img_size):
            s = s.resize((self.img_size, self.img_size), Image.BICUBIC)
        if t.size != (self.img_size, self.img_size):
            t = t.resize((self.img_size, self.img_size), Image.BICUBIC)
        if self.tsf_s:
            s = self.tsf_s(s)
        if self.tsf_t:
            t = self.tsf_t(t)
        s = self.norm(self.to_tensor(s))
        t = self.norm(self.to_tensor(t))
        return s, t


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
    writer = SummaryWriter(log_dir=os.path.join(config["checkpoint_dir"], "logs"))
    os.makedirs(config["source_data_dir"], exist_ok=True)
    os.makedirs(config["target_data_dir"], exist_ok=True)
    char_map = config["common_chars_for_stage1"] if "256" in config["stage_name"] else config["specific_chars_for_stage2"]
    for code, glyph in char_map.items():
        render_char_to_png(config["ref_font_path"], glyph, os.path.join(config["source_data_dir"], f"{code}.png"), size=config["img_size"])
        render_char_to_png(config["target_font_path"], glyph, os.path.join(config["target_data_dir"], f"{code}.png"), size=config["img_size"])
    tf_s = T.Compose([lambda img: morphology_transform(img, 1, 1), T.RandomAffine(degrees=1.5, translate=(0.02, 0.02), scale=(0.98, 1.02), fill=255)])
    tf_t = T.Compose([lambda img: morphology_transform(img, 1, 1)])
    ds = FontPairDataset(config["source_data_dir"], config["target_data_dir"], transform_source_pil=tf_s, transform_target_pil=tf_t, img_size=config["img_size"])
    dl = DataLoader(ds, batch_size=config["batch_size"], shuffle=True, num_workers=4)
    G = UNetGenerator(num_downs=config["num_unet_downs"], norm_type=config["norm_type"]).to(device)
    D = PatchDiscriminator(n_layers=config["d_n_layers"], norm_type=config["norm_type"]).to(device)
    if config.get("load_G_path"):
        G.load_state_dict(torch.load(config["load_G_path"], map_location=device))
    if config.get("load_D_path"):
        D.load_state_dict(torch.load(config["load_D_path"], map_location=device))
    if config.get("freeze_G_layers", 0) > 0:
        for i, layer in enumerate(G.model.children()):
            if i < config["freeze_G_layers"]:
                for p in layer.parameters():
                    p.requires_grad = False
    G.apply(weights_init)
    D.apply(weights_init)
    cgan = nn.BCEWithLogitsLoss()
    cl1 = nn.L1Loss()
    cperc = VGGPerceptualLoss().to(device)
    opt_g = optim.Adam(filter(lambda p: p.requires_grad, G.parameters()), lr=config["lr_G"], betas=(0.5, 0.999))
    opt_d = optim.Adam(D.parameters(), lr=config["lr_D"], betas=(0.5, 0.999))
    sched_g = CosineAnnealingLR(opt_g, T_max=config["epochs"], eta_min=1e-7)
    sched_d = CosineAnnealingLR(opt_d, T_max=config["epochs"], eta_min=1e-7)
    scaler = torch.cuda.amp.GradScaler(enabled=config["use_amp"])
    step = 0
    for epoch in range(1, config["epochs"] + 1):
        G.train()
        D.train()
        for i, (src, real) in enumerate(dl):
            src, real = src.to(device), real.to(device)
            opt_g.zero_grad()
            with torch.cuda.amp.autocast(enabled=config["use_amp"]):
                fake = G(src)
                l_gan = cgan(D(src, fake), torch.ones_like(D(src, fake)))
                l_l1 = cl1(fake, real) * config["l1_lambda"]
                l_p = cperc(fake, real) * config["perceptual_lambda"]
                l_g = (l_gan + l_l1 + l_p) / config["accum_steps"]
            scaler.scale(l_g).backward()
            if (i + 1) % config["accum_steps"] == 0:
                scaler.step(opt_g)
                scaler.update()
                opt_g.zero_grad()
            opt_d.zero_grad()
            with torch.cuda.amp.autocast(enabled=config["use_amp"]):
                l_dr = cgan(D(src, real), torch.ones_like(D(src, real)))
                l_df = cgan(D(src, fake.detach()), torch.zeros_like(D(src, fake.detach())))
                l_d = ((l_dr + l_df) * 0.5) / config["accum_steps"]
            scaler.scale(l_d).backward()
            if (i + 1) % config["accum_steps"] == 0:
                scaler.step(opt_d)
                scaler.update()
                opt_d.zero_grad()
            if step % config["log_freq"] == 0:
                mets = compute_metrics(fake, real)
                print(
                    f"Epoch[{epoch}/{config['epochs']}] Step {step} L_D:{l_d.item()*config['accum_steps']:.4f} "
                    f"L_G:{(l_gan + l_l1 + l_p).item():.4f} PSNR:{mets['psnr']:.2f} SSIM:{mets['ssim']:.3f}"
                )
                writer.add_scalar("Loss/D", l_d.item() * config["accum_steps"], step)
                writer.add_scalar("Loss/G_GAN", l_gan.item(), step)
                writer.add_scalar("Loss/G_L1", l_l1.item(), step)
                writer.add_scalar("Loss/G_Perceptual", l_p.item(), step)
                writer.add_scalar("Metrics/PSNR", mets["psnr"], step)
                writer.add_scalar("Metrics/SSIM", mets["ssim"], step)
            step += 1
        sched_g.step()
        sched_d.step()
        if epoch % config["save_epoch_freq"] == 0:
            torch.save(G.state_dict(), os.path.join(config["checkpoint_dir"], f"G_{config['stage_name']}_ep{epoch}.pth"))
            torch.save(D.state_dict(), os.path.join(config["checkpoint_dir"], f"D_{config['stage_name']}_ep{epoch}.pth"))
    writer.close()


if __name__ == "__main__":
    common_chars = {ord(c): c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをん道高速路"}
    specific_chars = {ord("壱"): "壱", ord("弐"): "弐", ord("参"): "参"}
    config1 = {
        "stage_name": "pretrain_256",
        "img_size": 256,
        "num_unet_downs": 8,
        "d_n_layers": 3,
        "norm_type": "instance",
        "epochs": 150,
        "batch_size": 8,
        "lr_G": 2e-4,
        "lr_D": 2e-4,
        "l1_lambda": 100.0,
        "perceptual_lambda": 1.0,
        "use_amp": True,
        "accum_steps": 2,
        "log_freq": 200,
        "save_epoch_freq": 10,
        "checkpoint_dir": "./checkpoints/gd_highway_pro",
        "source_data_dir": "./data/train_s1/source",
        "target_data_dir": "./data/train_s1/target",
        "ref_font_path": "path/to/reference_font.otf",
        "target_font_path": "path/to/GD-HighwayGothicJA.otf",
        "common_chars_for_stage1": common_chars,
        "specific_chars_for_stage2": specific_chars,
    }
    train(config1)
    config2 = config1.copy()
    config2.update(
        {
            "stage_name": "progressive_512",
            "img_size": 512,
            "num_unet_downs": 9,
            "d_n_layers": 4,
            "epochs": 100,
            "batch_size": 2,
            "accum_steps": 4,
            "lr_G": 1e-4,
            "lr_D": 1e-4,
            "source_data_dir": "./data/train_s2/source",
            "target_data_dir": "./data/train_s2/target",
            "load_G_path": "./checkpoints/gd_highway_pro/G_pretrain_256_ep150.pth",
            "freeze_G_layers": 2,
            "specific_chars_for_stage2": common_chars,
        }
    )
    train(config2)
