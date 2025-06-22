# -*- coding: utf-8 -*-
"""train_pix2pix.py — train_pix2pix

概要:
    pix2pix ベースのフォント補完モデルを訓練・推論するスクリプト。

:author: pumpCurry
:copyright: (c) pumpCurry 2025 / 5r4ce2
:license: MIT
:version: 1.0.13 (PR #6)
:since:   1.0.13 (PR #6)
:last-modified: 2025-06-22 12:16:29 JST+9
:todo:
    - Refactor training loop for CLI usage
"""

import io
import os
import glob
from typing import Callable, Tuple, Dict

from PIL import Image, ImageDraw, ImageFont
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


def render_char_to_png(font_path: str, char: str, out_path_or_buffer: str | io.BytesIO, size: int = 256) -> Image.Image:
    """指定したフォントで1文字を描画し PNG へ保存する。

    :param font_path: 使用するフォントファイルパス
    :type font_path: str
    :param char: 描画する文字
    :type char: str
    :param out_path_or_buffer: 出力先パスまたは ``io.BytesIO``
    :type out_path_or_buffer: str | io.BytesIO
    :param size: 出力画像の一辺ピクセル数
    :type size: int
    :return: 描画済み ``PIL.Image``
    :rtype: Image.Image
    """
    font = ImageFont.truetype(font_path, int(size * 0.8))
    img = Image.new("L", (size, size), color=255)
    draw = ImageDraw.Draw(img)
    try:
        bbox = draw.textbbox((0, 0), char, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x = (size - text_width) / 2 - bbox[0]
        y = (size - text_height) / 2 - bbox[1]
    except AttributeError:
        text_width, text_height = draw.textsize(char, font=font)
        ascent, _ = font.getmetrics()
        x = (size - text_width) / 2
        y = (size - ascent) / 2 - (size * 0.05)
    draw.text((x, y), char, font=font, fill=0)

    if isinstance(out_path_or_buffer, str):
        img.save(out_path_or_buffer)
    elif isinstance(out_path_or_buffer, io.BytesIO):
        img.save(out_path_or_buffer, format="PNG")
        out_path_or_buffer.seek(0)
    return img


class FontPairDataset(Dataset):
    """参考フォント画像とターゲットフォント画像のペアを提供する。"""

    def __init__(self, source_dir: str, target_dir: str, transform: Callable | None = None) -> None:
        """ディレクトリから画像ペアを読み込む。

        :param source_dir: 参考フォント画像ディレクトリ
        :type source_dir: str
        :param target_dir: 目標フォント画像ディレクトリ
        :type target_dir: str
        :param transform: 前処理変換
        :type transform: Callable | None
        """
        src = sorted(glob.glob(os.path.join(source_dir, "*.png")))
        tgt = []
        for p in src:
            tp = os.path.join(target_dir, os.path.basename(p))
            if os.path.exists(tp):
                tgt.append(tp)
            else:
                print(f"Warning: Target image {tp} not found for source {p}. Skipping.")
        valid_src = [p for p in src if os.path.join(target_dir, os.path.basename(p)) in tgt]
        self.src_paths = valid_src
        self.tgt_paths = [os.path.join(target_dir, os.path.basename(p)) for p in valid_src]
        self.transform = transform or T.Compose([
            T.ToTensor(),
            T.Normalize((0.5,), (0.5,)),
        ])

    def __len__(self) -> int:
        """データ数を返す。"""
        return len(self.src_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """インデックスの画像ペアを取得する。"""
        src_path = self.src_paths[idx]
        tgt_path = self.tgt_paths[idx]
        try:
            src = Image.open(src_path).convert("L")
            tgt = Image.open(tgt_path).convert("L")
        except FileNotFoundError as exc:
            raise RuntimeError(f"Image pair not found: {src_path}, {tgt_path}") from exc
        return self.transform(src), self.transform(tgt)


class UNetGenerator(nn.Module):
    """U-Net 形式のジェネレータ。"""

    def __init__(self, in_ch: int = 1, out_ch: int = 1, ngf: int = 64) -> None:
        super().__init__()
        self.down1 = nn.Sequential(nn.Conv2d(in_ch, ngf, 4, 2, 1, bias=False), nn.LeakyReLU(0.2))
        self.down2 = nn.Sequential(nn.Conv2d(ngf, ngf * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf * 2), nn.LeakyReLU(0.2))
        self.down3 = nn.Sequential(nn.Conv2d(ngf * 2, ngf * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf * 4), nn.LeakyReLU(0.2))
        self.down4 = nn.Sequential(nn.Conv2d(ngf * 4, ngf * 8, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf * 8), nn.LeakyReLU(0.2))
        self.down5 = nn.Sequential(nn.Conv2d(ngf * 8, ngf * 8, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf * 8), nn.LeakyReLU(0.2))
        self.down6 = nn.Sequential(nn.Conv2d(ngf * 8, ngf * 8, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf * 8), nn.LeakyReLU(0.2))
        self.down7 = nn.Sequential(nn.Conv2d(ngf * 8, ngf * 8, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf * 8), nn.LeakyReLU(0.2))
        self.down8 = nn.Sequential(nn.Conv2d(ngf * 8, ngf * 8, 4, 2, 1, bias=False), nn.ReLU(True))

        self.up1 = nn.Sequential(nn.ConvTranspose2d(ngf * 8, ngf * 8, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf * 8), nn.ReLU(True))
        self.up2 = nn.Sequential(nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf * 8), nn.ReLU(True))
        self.up3 = nn.Sequential(nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf * 8), nn.ReLU(True))
        self.up4 = nn.Sequential(nn.ConvTranspose2d(ngf * 16, ngf * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf * 4), nn.ReLU(True))
        self.up5 = nn.Sequential(nn.ConvTranspose2d(ngf * 8, ngf * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf * 2), nn.ReLU(True))
        self.up6 = nn.Sequential(nn.ConvTranspose2d(ngf * 4, ngf, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf), nn.ReLU(True))
        self.up7 = nn.Sequential(nn.ConvTranspose2d(ngf * 2, out_ch, 4, 2, 1), nn.Tanh())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)

        u1 = self.up1(d8)
        u2 = self.up2(torch.cat([u1, d7], 1))
        u3 = self.up3(torch.cat([u2, d6], 1))
        u4 = self.up4(torch.cat([u3, d5], 1))
        u5 = self.up5(torch.cat([u4, d4], 1))
        u6 = self.up6(torch.cat([u5, d3], 1))
        final = self.up7(torch.cat([u6, d1], 1))
        return final


class PatchDiscriminator(nn.Module):
    """PatchGAN 識別器。"""

    def __init__(self, in_ch: int = 2, ndf: int = 64, n_layers: int = 3) -> None:
        super().__init__()
        layers = [nn.Conv2d(in_ch, ndf, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True)]
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            layers += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        layers += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 4, 1, 1, bias=False),
            nn.BatchNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * nf_mult, 1, 4, 1, 1),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """参考画像と生成画像のペアから真偽を判定する。"""
        x = torch.cat([src, tgt], dim=1)
        return self.model(x)


def train(
    target_font_path: str,
    ref_font_path: str,
    chars_to_render: Dict[int, str],
    epochs: int = 200,
    batch_size: int = 4,
    lr: float = 2e-4,
    l1_lambda: int = 100,
    checkpoint_dir: str = "checkpoints",
    source_data_dir: str = "data/train/source",
    target_data_dir: str = "data/train/target",
) -> None:
    """学習ループを実行する。"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(source_data_dir, exist_ok=True)
    os.makedirs(target_data_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    for code, glyph in chars_to_render.items():
        render_char_to_png(ref_font_path, glyph, os.path.join(source_data_dir, f"{code}.png"))
        render_char_to_png(target_font_path, glyph, os.path.join(target_data_dir, f"{code}.png"))

    ds = FontPairDataset(source_data_dir, target_data_dir)
    if len(ds) == 0:
        print("Dataset is empty")
        return
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=max(os.cpu_count() // 2, 1), pin_memory=True)

    G = UNetGenerator().to(device)
    D = PatchDiscriminator().to(device)

    def weights_init(m: nn.Module) -> None:
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find("BatchNorm") != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    G.apply(weights_init)
    D.apply(weights_init)

    opt_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

    criterion_GAN = nn.BCEWithLogitsLoss()
    criterion_L1 = nn.L1Loss()

    for epoch in range(1, epochs + 1):
        G.train()
        D.train()
        for src, real in dl:
            src, real = src.to(device), real.to(device)

            opt_D.zero_grad()
            fake = G(src)
            real_pred = D(src, real)
            fake_pred = D(src, fake.detach())
            loss_D_real = criterion_GAN(real_pred, torch.ones_like(real_pred))
            loss_D_fake = criterion_GAN(fake_pred, torch.zeros_like(fake_pred))
            loss_D = (loss_D_real + loss_D_fake) * 0.5
            loss_D.backward()
            opt_D.step()

            opt_G.zero_grad()
            fake = G(src)
            fake_pred_for_g = D(src, fake)
            loss_G_GAN = criterion_GAN(fake_pred_for_g, torch.ones_like(fake_pred_for_g))
            loss_G_L1 = criterion_L1(fake, real) * l1_lambda
            loss_G = loss_G_GAN + loss_G_L1
            loss_G.backward()
            opt_G.step()

        print(f"Epoch {epoch:03d} loss_D:{loss_D.item():.4f} loss_G:{loss_G.item():.4f}")
        if epoch % 10 == 0 or epoch == epochs:
            torch.save(G.state_dict(), os.path.join(checkpoint_dir, f"G_epoch{epoch:03d}.pth"))
            torch.save(D.state_dict(), os.path.join(checkpoint_dir, f"D_epoch{epoch:03d}.pth"))
            G.eval()
            with torch.no_grad():
                sample_src, _ = next(iter(dl))
                sample = G(sample_src.to(device)).cpu().squeeze(0)
                out = ((sample * 0.5 + 0.5) * 255).clamp(0, 255).numpy().astype("uint8")[0]
                Image.fromarray(out).save(os.path.join(checkpoint_dir, f"sample_epoch{epoch:03d}.png"))


def inference(
    gen_checkpoint: str,
    chars_to_generate: Dict[int, str],
    ref_font_path: str,
    out_dir: str,
    batch_size: int = 4,
) -> None:
    """学習済みモデルを用いて文字画像を生成する。"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    G = UNetGenerator().to(device)
    G.load_state_dict(torch.load(gen_checkpoint, map_location=device))
    G.eval()

    os.makedirs(out_dir, exist_ok=True)
    transform = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])

    char_list = list(chars_to_generate.items())
    for i in range(0, len(char_list), batch_size):
        batch_items = char_list[i:i + batch_size]
        tensors = []
        file_names = []
        for code, glyph in batch_items:
            buf = io.BytesIO()
            render_char_to_png(ref_font_path, glyph, buf)
            img = Image.open(buf).convert("L")
            tensors.append(transform(img))
            file_names.append(os.path.join(out_dir, f"{code}.png"))
        if not tensors:
            continue
        x = torch.stack(tensors).to(device)
        with torch.no_grad():
            y_batch = G(x).cpu()
        for j in range(y_batch.size(0)):
            y = y_batch[j].squeeze(0)
            out = ((y * 0.5 + 0.5) * 255).clamp(0, 255).numpy().astype("uint8")
            if out.ndim == 3 and out.shape[0] == 1:
                out = out[0]
            Image.fromarray(out).save(file_names[j])


if __name__ == "__main__":
    TARGET_FONT_PATH = "path/to/GD-HighwayGothicJA.otf"
    REFERENCE_FONT_PATH = "path/to/reference_font.otf"
    OUTPUT_DIR_TRAIN = "data_updated"
    CHECKPOINT_DIR = "checkpoints_gd_highwaygothic"
    GENERATED_FONT_DIR = "output_gd_highwaygothic"
    NUM_EPOCHS = 200
    BATCH_SIZE = 4
    LEARNING_RATE = 0.0002
    L1_LAMBDA = 100

    common_chars_for_training = {
        ord("あ"): "あ",
        ord("い"): "い",
    }

    if not common_chars_for_training:
        raise ValueError("Training characters not specified")
    if not os.path.exists(TARGET_FONT_PATH) or not os.path.exists(REFERENCE_FONT_PATH):
        raise FileNotFoundError("Font file not found")

    train(
        target_font_path=TARGET_FONT_PATH,
        ref_font_path=REFERENCE_FONT_PATH,
        chars_to_render=common_chars_for_training,
        epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE,
        l1_lambda=L1_LAMBDA,
        checkpoint_dir=CHECKPOINT_DIR,
        source_data_dir=os.path.join(OUTPUT_DIR_TRAIN, "train", "source"),
        target_data_dir=os.path.join(OUTPUT_DIR_TRAIN, "train", "target"),
    )

    missing_chars_to_generate = {
        ord("琉"): "琉",
    }
    latest_checkpoint = os.path.join(CHECKPOINT_DIR, f"G_epoch{NUM_EPOCHS:03d}.pth")
    if os.path.exists(latest_checkpoint) and missing_chars_to_generate:
        inference(
            gen_checkpoint=latest_checkpoint,
            chars_to_generate=missing_chars_to_generate,
            ref_font_path=REFERENCE_FONT_PATH,
            out_dir=GENERATED_FONT_DIR,
            batch_size=BATCH_SIZE,
        )

