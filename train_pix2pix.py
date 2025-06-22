# -*- coding: utf-8 -*-
"""train_pix2pix.py — train_pix2pix

概要:
    pix2pix ベースのフォント補完モデルを訓練・推論するスクリプト。

:author: pumpCurry
:copyright: (c) pumpCurry 2025 / 5r4ce2
:license: MIT
:version: 1.0.10 (PR #5)
:since:   1.0.10 (PR #5)
:last-modified: 2025-06-22 12:16:29 JST+9
:todo:
    - Refactor training loop for CLI usage
"""

import os
import glob
from typing import Callable, Tuple

from PIL import Image, ImageDraw, ImageFont
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

# 1. フォントから PNG を生成するユーティリティ

def render_char_to_png(font_path: str, char: str, out_path: str, size: int = 256) -> None:
    """指定したフォントで1文字を描画しPNGとして保存する。

    :param font_path: 使用するフォントファイルのパス
    :type font_path: str
    :param char: 描画する1文字
    :type char: str
    :param out_path: 保存するPNGファイルパス
    :type out_path: str
    :param size: 出力画像サイズ
    :type size: int
    :return: ``None``
    :rtype: None
    :example:
        >>> render_char_to_png("NotoSans.otf", "あ", "output.png")
    """
    font = ImageFont.truetype(font_path, size)
    img = Image.new("L", (size, size), color=255)
    draw = ImageDraw.Draw(img)
    w, h = draw.textsize(char, font=font)
    draw.text(((size - w) / 2, (size - h) / 2), char, font=font, fill=0)
    img.save(out_path)


# 2. Dataset：ペア画像を読み込み
class FontPairDataset(Dataset):
    """参考画像とターゲット画像のペアを扱うデータセット。"""

    def __init__(self, source_dir: str, target_dir: str, transform=None) -> None:
        """データセットを初期化する。

        :param source_dir: 参考フォント画像のディレクトリ
        :type source_dir: str
        :param target_dir: 目標フォント画像のディレクトリ
        :type target_dir: str
        :param transform: 画像前処理関数
        :type transform: Callable | None
        :return: ``None``
        :rtype: None
        :example:
            >>> ds = FontPairDataset('src', 'tgt')
        """
        self.src_paths = sorted(glob.glob(os.path.join(source_dir, "*.png")))
        self.tgt_paths = [os.path.join(target_dir, os.path.basename(p)) for p in self.src_paths]
        self.transform = transform or T.Compose([
            T.ToTensor(),
            T.Normalize((0.5,), (0.5,)),
        ])

    def __len__(self) -> int:
        """データセットのサイズを返す。

        :return: 画像ペアの数
        :rtype: int
        """

        return len(self.src_paths)

    def __getitem__(self, idx: int):
        """インデックス指定でペア画像を取得する。

        :param idx: 取得するインデックス
        :type idx: int
        :return: 変換後の参考画像とターゲット画像のペア
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """

        src = Image.open(self.src_paths[idx]).convert("L")
        tgt = Image.open(self.tgt_paths[idx]).convert("L")
        return self.transform(src), self.transform(tgt)


# 3. モデル定義
# 3.1 U-Net Generator
class UNetGenerator(nn.Module):
    """U-Net 風のジェネレータネットワーク。"""

    def __init__(self, in_ch: int = 1, out_ch: int = 1, ngf: int = 64) -> None:
        """レイヤーを初期化する。

        :param in_ch: 入力チャンネル数
        :type in_ch: int
        :param out_ch: 出力チャンネル数
        :type out_ch: int
        :param ngf: 基本フィルタ数
        :type ngf: int
        :return: ``None``
        :rtype: None
        :example:
            >>> net = UNetGenerator()
        """

        super().__init__()
        # エンコーダ部
        self.down1 = nn.Sequential(nn.Conv2d(in_ch, ngf, 4, 2, 1), nn.LeakyReLU(0.2))
        self.down2 = nn.Sequential(
            nn.Conv2d(ngf, ngf * 2, 4, 2, 1), nn.BatchNorm2d(ngf * 2), nn.LeakyReLU(0.2)
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf * 4, 4, 2, 1), nn.BatchNorm2d(ngf * 4), nn.LeakyReLU(0.2)
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(ngf * 4, ngf * 8, 4, 2, 1), nn.BatchNorm2d(ngf * 8), nn.LeakyReLU(0.2)
        )
        # デコーダ部
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1), nn.BatchNorm2d(ngf * 4), nn.ReLU()
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8, ngf * 2, 4, 2, 1), nn.BatchNorm2d(ngf * 2), nn.ReLU()
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf, 4, 2, 1), nn.BatchNorm2d(ngf), nn.ReLU()
        )
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, out_ch, 4, 2, 1), nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """入力テンソルから画像を生成する。

        :param x: 入力画像テンソル
        :type x: torch.Tensor
        :return: 生成された画像テンソル
        :rtype: torch.Tensor
        """

        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        u1 = self.up1(d4)
        u2 = self.up2(torch.cat([u1, d3], dim=1))
        u3 = self.up3(torch.cat([u2, d2], dim=1))
        u4 = self.up4(torch.cat([u3, d1], dim=1))
        return u4


# 3.2 PatchGAN Discriminator
class PatchDiscriminator(nn.Module):
    """PatchGAN 風の識別器。"""

    def __init__(self, in_ch: int = 2, ndf: int = 64) -> None:
        """レイヤーを初期化する。

        :param in_ch: 入力チャンネル数
        :type in_ch: int
        :param ndf: ベースフィルタ数
        :type ndf: int
        :return: ``None``
        :rtype: None
        :example:
            >>> disc = PatchDiscriminator()
        """

        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_ch, ndf, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2),
            nn.Conv2d(ndf * 4, 1, 4, 1, 1),
        )

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """画像ペアを入力し真偽判定用のパッチマップを返す。

        :param src: 参考画像テンソル
        :type src: torch.Tensor
        :param tgt: 目標画像テンソル
        :type tgt: torch.Tensor
        :return: 判定結果のパッチマップ
        :rtype: torch.Tensor
        """

        x = torch.cat([src, tgt], dim=1)
        return self.model(x)


# 4. 学習ループ

def train() -> None:
    """学習ループを実行する。

    :return: ``None``
    :rtype: None
    :example:
        >>> train()
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = FontPairDataset("data/train/source", "data/train/target")
    dl = DataLoader(ds, batch_size=4, shuffle=True, num_workers=2)

    G = UNetGenerator().to(device)
    D = PatchDiscriminator().to(device)

    opt_G = optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

    criterion_GAN = nn.BCEWithLogitsLoss()
    criterion_L1 = nn.L1Loss()

    for epoch in range(1, 201):
        for src, real in dl:
            src, real = src.to(device), real.to(device)
            # --- Discriminator update ---
            fake = G(src)
            real_pred = D(src, real)
            fake_pred = D(src, fake.detach())
            loss_D = (
                criterion_GAN(real_pred, torch.ones_like(real_pred))
                + criterion_GAN(fake_pred, torch.zeros_like(fake_pred))
            ) * 0.5
            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

            # --- Generator update ---
            fake_pred = D(src, fake)
            loss_G = criterion_GAN(fake_pred, torch.ones_like(fake_pred)) + 100 * criterion_L1(
                fake, real
            )
            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

        print(f"Epoch {epoch:03d}  loss_D:{loss_D.item():.4f}  loss_G:{loss_G.item():.4f}")
        if epoch % 10 == 0:
            torch.save(G.state_dict(), f"checkpoints/G_{epoch:03d}.pth")


# 5. 推論スクリプト

def inference(gen_checkpoint: str, chars: list[str], ref_font_path: str, out_dir: str) -> None:
    """学習済みモデルを用いて文字画像を生成する。

    :param gen_checkpoint: 生成器のチェックポイントパス
    :type gen_checkpoint: str
    :param chars: 生成する文字のリスト
    :type chars: list[str]
    :param ref_font_path: 参考フォントファイルのパス
    :type ref_font_path: str
    :param out_dir: 生成画像の出力ディレクトリ
    :type out_dir: str
    :return: ``None``
    :rtype: None
    :example:
        >>> inference('G_200.pth', ['あ'], 'ref.otf', 'out')
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    G = UNetGenerator().to(device)
    G.load_state_dict(torch.load(gen_checkpoint, map_location=device))
    G.eval()

    os.makedirs(out_dir, exist_ok=True)
    transform = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])

    for c in chars:
        tmp = f"tmp_{ord(c)}.png"
        render_char_to_png(ref_font_path, c, tmp)
        img = Image.open(tmp).convert("L")
        x = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            y = G(x).cpu().squeeze(0)
        out = ((y * 0.5 + 0.5) * 255).clamp(0, 255).numpy().astype("uint8")[0]
        Image.fromarray(out).save(os.path.join(out_dir, f"{ord(c)}.png"))
        os.remove(tmp)


def main() -> None:
    """エントリーポイント。

    :return: ``None``
    :rtype: None
    :example:
        >>> main()
    """

    train()

    # Example usage of inference:
    # missing_chars = ["あ", "い", "う"]
    # inference("checkpoints/G_200.pth", missing_chars, "reference_font.otf", "output")


if __name__ == "__main__":
    main()
