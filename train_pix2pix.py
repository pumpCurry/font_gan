# -*- coding: utf-8 -*-
"""train_pix2pix.py — train_pix2pix

概要:
    pix2pix ベースのフォント補完モデルを訓練・推論するスクリプト。

:author: pumpCurry
:copyright: (c) pumpCurry 2025 / 5r4ce2
:license: MIT
:version: 1.0.26 (PR #11)
:since:   1.0.15 (PR #7)
:last-modified: 2025-06-22 23:28:26 JST+9

:todo:
    - Refactor training loop for CLI usage
"""

import io
import os
import glob
import random
from typing import Callable, Tuple, Dict

from PIL import Image, ImageDraw, ImageFont, ImageFilter
from skimage.metrics import peak_signal_noise_ratio as calc_psnr
from skimage.metrics import structural_similarity as calc_ssim
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models
import torch.nn.functional as F


class RandomStrokeWidth:
    """ランダムに文字の太さを変える前処理。"""

    def __init__(self, radius: int = 1, p: float = 0.5) -> None:
        self.radius = radius
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            if random.random() < 0.5:
                return img.filter(ImageFilter.MaxFilter(self.radius))
            return img.filter(ImageFilter.MinFilter(self.radius))
        return img


class AddGaussianNoise:
    """Tensor にガウシアンノイズを加える。"""

    def __init__(self, mean: float = 0.0, std: float = 0.02) -> None:
        self.mean = mean
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(tensor) * self.std + self.mean
        return tensor + noise


def morphology_shift(img: Image.Image, min_shift: int = 1, max_shift: int = 2) -> Image.Image:
    """Apply small dilation or erosion to change stroke width."""
    img_np = np.array(img)
    threshold = 200
    binary = img_np < threshold
    shift = random.randint(min_shift, max_shift)
    if random.random() > 0.5:
        processed = binary_dilation(binary, iterations=shift)
    else:
        processed = binary_erosion(binary, iterations=shift)
    final_np = np.where(processed, 0, 255).astype(np.uint8)
    return Image.fromarray(final_np)


def get_norm_layer(norm_type: str = "batch") -> type[nn.Module]:
    """Return normalization layer class based on type."""
    if norm_type == "batch":
        return nn.BatchNorm2d
    if norm_type == "instance":
        return nn.InstanceNorm2d
    raise NotImplementedError(f"norm layer {norm_type} not supported")


class VGGPerceptualLoss(nn.Module):
    """VGG16 を用いた知覚損失。"""

    def __init__(
        self,
        layers: tuple[str, ...] = ("relu1_2", "relu2_2", "relu3_3", "relu4_3"),
        weights: Dict[str, float] | None = None,
        resize: bool = True,
    ) -> None:
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features.eval()
        self.layers = layers
        self.weights = weights or {l: 1.0 for l in layers}
        self.resize = resize
        self.hooks: Dict[str, torch.utils.hooks.RemovableHandle] = {}
        self.outputs: Dict[str, torch.Tensor] = {}
        layer_map = {"relu1_2": 3, "relu2_2": 8, "relu3_3": 15, "relu4_3": 22}
        for name, idx in layer_map.items():
            if name in layers:
                self.hooks[name] = vgg[idx].register_forward_hook(
                    lambda _m, _i, out, key=name: self.outputs.update({key: out})
                )
        self.vgg = vgg
        for p in self.vgg.parameters():
            p.requires_grad = False
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
            y = y.repeat(1, 3, 1, 1)
        x = (x - self.mean.to(x.device)) / self.std.to(x.device)
        y = (y - self.mean.to(y.device)) / self.std.to(y.device)
        if self.resize:
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
            y = F.interpolate(y, size=(224, 224), mode="bilinear", align_corners=False)
        self.outputs.clear()
        _ = self.vgg(torch.cat([x, y], dim=0))
        loss = 0.0
        for name in self.layers:
            feat = self.outputs[name]
            loss += self.weights[name] * F.l1_loss(feat[:b], feat[b:])
        return loss


def compute_metrics(fake: torch.Tensor, real: torch.Tensor) -> Dict[str, float]:
    """PSNR と SSIM を計算するユーティリティ関数。"""

    fake_np = fake.mul(0.5).add(0.5).clamp(0, 1).cpu().numpy()
    real_np = real.mul(0.5).add(0.5).clamp(0, 1).cpu().numpy()
    psnr_list: list[float] = []
    ssim_list: list[float] = []
    for i in range(fake_np.shape[0]):
        psnr_list.append(
            float(calc_psnr(real_np[i, 0], fake_np[i, 0], data_range=1.0))
        )
        ssim_list.append(
            float(calc_ssim(real_np[i, 0], fake_np[i, 0], data_range=1.0))
        )
    return {
        "psnr": sum(psnr_list) / len(psnr_list),
        "ssim": sum(ssim_list) / len(ssim_list),
    }

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

    def __init__(
        self,
        source_dir: str,
        target_dir: str,
        transform: Callable | None = None,
        augment: bool = False,
        augment_source_only: bool = False,
        img_size: int = 256,
        transform_source: Callable | None = None,
        transform_target: Callable | None = None,
    ) -> None:
        """ディレクトリから画像ペアを読み込む。

        :param source_dir: 参考フォント画像ディレクトリ
        :type source_dir: str
        :param target_dir: 目標フォント画像ディレクトリ
        :type target_dir: str
        :param transform: 前処理変換
        :type transform: Callable | None
        :param augment: データ増強を有効にするかどうか
        :type augment: bool
        :param augment_source_only: 参考フォントのみ増強を適用するかどうか
        :type augment_source_only: bool
        :param img_size: 画像サイズを揃えるピクセル数
        :type img_size: int
        :param transform_source: 参考フォント専用前処理
        :type transform_source: Callable | None
        :param transform_target: ターゲットフォント専用前処理
        :type transform_target: Callable | None
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
        self.img_size = img_size
        if transform_source or transform_target:
            self.src_transform = transform_source or T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])
            self.tgt_transform = transform_target or T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])
        elif transform:
            self.src_transform = transform
            self.tgt_transform = transform
        else:
            src_transforms: list[Callable] = []
            tgt_transforms: list[Callable] = []
            if augment:
                src_transforms.extend(
                    [
                        T.RandomAffine(degrees=2, translate=(0.02, 0.02), scale=(0.95, 1.05)),
                        T.GaussianBlur(3, sigma=(0.1, 1.0)),
                        T.RandomApply([T.Lambda(lambda img: morphology_shift(img, 1, 1))], p=0.2),
                    ]
                )
                if augment_source_only:
                    src_transforms.append(RandomStrokeWidth(radius=1, p=0.5))
                    tgt_transforms.append(
                        T.RandomApply([T.Lambda(lambda img: morphology_shift(img, 1, 1))], p=0.1)
                    )
                else:
                    common_sw = RandomStrokeWidth(radius=1, p=0.5)
                    src_transforms.append(common_sw)
                    tgt_transforms.append(common_sw)
                    tgt_transforms.append(
                        T.RandomApply([T.Lambda(lambda img: morphology_shift(img, 1, 1))], p=0.1)
                    )
            common = [
                T.ToTensor(),
                AddGaussianNoise(),
                T.Normalize((0.5,), (0.5,)),
            ]
            self.src_transform = T.Compose(src_transforms + common)
            if augment_source_only:
                self.tgt_transform = T.Compose(tgt_transforms + common)
            else:
                self.tgt_transform = T.Compose(tgt_transforms + common)

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
        if src.size != (self.img_size, self.img_size):
            src = src.resize((self.img_size, self.img_size), Image.BICUBIC)
        if tgt.size != (self.img_size, self.img_size):
            tgt = tgt.resize((self.img_size, self.img_size), Image.BICUBIC)
        return self.src_transform(src), self.tgt_transform(tgt)


class UNetGenerator(nn.Module):
    """U-Net 形式のジェネレータ。"""

    def __init__(
        self,
        in_ch: int = 1,
        out_ch: int = 1,
        ngf: int = 64,
        num_downs: int = 8,
        norm_type: str = "batch",
    ) -> None:
        super().__init__()
        norm_layer = get_norm_layer(norm_type)
        downs: list[nn.Module] = []
        ch = ngf
        downs.append(nn.Sequential(nn.Conv2d(in_ch, ch, 4, 2, 1, bias=False), nn.LeakyReLU(0.2)))
        for i in range(1, num_downs):
            in_c = ch
            ch = min(ngf * 2 ** i, ngf * 8)
            downs.append(
                nn.Sequential(
                    nn.Conv2d(in_c, ch, 4, 2, 1, bias=False),
                    norm_layer(ch),
                    nn.LeakyReLU(0.2),
                )
            )
        self.downs = nn.ModuleList(downs)

        ups: list[nn.Module] = []
        for i in range(num_downs - 1):
            in_c = ch * 2 if i != 0 else ch
            ch = min(max(in_c // 2, ngf), ngf * 8) if i < num_downs - 2 else ngf
            ups.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_c, ch, 4, 2, 1, bias=False),
                    norm_layer(ch),
                    nn.ReLU(True),
                )
            )
        ups.append(nn.Sequential(nn.ConvTranspose2d(ch * 2, out_ch, 4, 2, 1), nn.Tanh()))
        self.ups = nn.ModuleList(ups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        downs_out: list[torch.Tensor] = []
        cur = x
        for mod in self.downs:
            cur = mod(cur)
            downs_out.append(cur)
        for i, mod in enumerate(self.ups[:-1]):
            skip = downs_out[-(i + 2)] if i + 2 <= len(downs_out) else downs_out[0]
            cur = mod(torch.cat([cur, skip], dim=1))
        final = self.ups[-1](torch.cat([cur, downs_out[0]], dim=1))
        return final


class PatchDiscriminator(nn.Module):
    """PatchGAN 識別器。"""

    def __init__(
        self,
        in_ch: int = 2,
        ndf: int = 64,
        n_layers: int = 3,
        norm_type: str = "batch",
    ) -> None:
        super().__init__()
        norm_layer = get_norm_layer(norm_type)
        layers = [nn.Conv2d(in_ch, ndf, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True)]
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            layers += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 4, 2, 1, bias=False),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        layers += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 4, 1, 1, bias=False),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * nf_mult, 1, 4, 1, 1),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """参考画像と生成画像のペアから真偽を判定する。"""
        x = torch.cat([src, tgt], dim=1)
        return self.model(x)


def freeze_generator_layers(generator: UNetGenerator, num_layers: int) -> None:
    """ジェネレータの最初の ``num_layers`` 層の学習を凍結する。"""

    layers = list(generator.downs)
    for l in layers[:num_layers]:
        for p in l.parameters():
            p.requires_grad = False


def train(
    target_font_path: str,
    ref_font_path: str,
    chars_to_render: Dict[int, str],
    epochs: int = 200,
    batch_size: int = 4,
    lr: float = 2e-4,
    l1_lambda: int = 100,
    perceptual_lambda: float = 0.0,
    use_perceptual_loss: bool = False,
    img_size: int = 256,
    num_downs: int = 8,
    use_amp: bool = False,
    checkpoint_dir: str = "checkpoints",
    source_data_dir: str = "data/train/source",
    target_data_dir: str = "data/train/target",
    pretrained_G_path: str | None = None,
    pretrained_D_path: str | None = None,
    augment: bool = False,
    augment_source_only: bool = False,
    freeze_layers: int = 0,
    norm_type: str = "batch",
) -> None:
    """学習ループを実行する。

    :param target_font_path: 目標フォントファイルパス
    :type target_font_path: str
    :param ref_font_path: 参考フォントファイルパス
    :type ref_font_path: str
    :param chars_to_render: 学習に用いる文字マップ
    :type chars_to_render: Dict[int, str]
    :param epochs: 学習エポック数
    :type epochs: int
    :param batch_size: ミニバッチサイズ
    :type batch_size: int
    :param lr: 学習率
    :type lr: float
    :param l1_lambda: L1 損失の重み
    :type l1_lambda: int
    :param perceptual_lambda: 知覚損失の重み
    :type perceptual_lambda: float
    :param use_perceptual_loss: VGG 知覚損失を使用するか
    :type use_perceptual_loss: bool
    :param img_size: 画像レンダリングサイズ
    :type img_size: int
    :param num_downs: U-Net のダウンサンプル回数
    :type num_downs: int
    :param use_amp: Mixed Precision 学習を行うか
    :type use_amp: bool
    :param checkpoint_dir: チェックポイント保存先
    :type checkpoint_dir: str
    :param source_data_dir: 参考フォント画像ディレクトリ
    :type source_data_dir: str
    :param target_data_dir: ターゲットフォント画像ディレクトリ
    :type target_data_dir: str
    :param pretrained_G_path: 事前学習済みGenerator
    :type pretrained_G_path: str | None
    :param pretrained_D_path: 事前学習済みDiscriminator
    :type pretrained_D_path: str | None
    :param augment: データ増強を有効にするか
    :type augment: bool
    :param augment_source_only: 参考フォントのみ増強を適用するか
    :type augment_source_only: bool
    :param freeze_layers: Stage2 で凍結するジェネレータ層数
    :type freeze_layers: int
    :param norm_type: 正規化レイヤー種別 ("batch" or "instance")
    :type norm_type: str
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(source_data_dir, exist_ok=True)
    os.makedirs(target_data_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    for code, glyph in chars_to_render.items():
        render_char_to_png(
            ref_font_path,
            glyph,
            os.path.join(source_data_dir, f"{code}.png"),
            size=img_size,
        )
        render_char_to_png(
            target_font_path,
            glyph,
            os.path.join(target_data_dir, f"{code}.png"),
            size=img_size,
        )

    ds = FontPairDataset(
        source_data_dir,
        target_data_dir,
        augment=augment,
        augment_source_only=augment_source_only,
        img_size=img_size,
    )
    if len(ds) == 0:
        print("Dataset is empty")
        return
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=max(os.cpu_count() // 2, 1), pin_memory=True)

    G = UNetGenerator(num_downs=num_downs, norm_type=norm_type).to(device)
    d_layers = 3 if img_size <= 256 else 4
    D = PatchDiscriminator(n_layers=d_layers, norm_type=norm_type).to(device)

    def weights_init(m: nn.Module) -> None:
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find("BatchNorm") != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    if pretrained_G_path and os.path.exists(pretrained_G_path):
        G.load_state_dict(torch.load(pretrained_G_path, map_location=device))
    else:
        G.apply(weights_init)
    if freeze_layers > 0:
        freeze_generator_layers(G, freeze_layers)

    if pretrained_D_path and os.path.exists(pretrained_D_path):
        D.load_state_dict(torch.load(pretrained_D_path, map_location=device))
    else:
        D.apply(weights_init)

    opt_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    scheduler_G = optim.lr_scheduler.CosineAnnealingLR(opt_G, T_max=epochs, eta_min=1e-6)
    scheduler_D = optim.lr_scheduler.CosineAnnealingLR(opt_D, T_max=epochs, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device == "cuda"))

    criterion_GAN = nn.BCEWithLogitsLoss()
    criterion_L1 = nn.L1Loss()
    perceptual = VGGPerceptualLoss().to(device) if use_perceptual_loss else None

    for epoch in range(1, epochs + 1):
        G.train()
        D.train()
        psnr_sum = 0.0
        ssim_sum = 0.0
        batch_count = 0
        for src, real in dl:
            src, real = src.to(device), real.to(device)

            opt_D.zero_grad()
            with torch.cuda.amp.autocast(enabled=(use_amp and device == "cuda")):
                fake = G(src)
                real_pred = D(src, real)
                fake_pred = D(src, fake.detach())
                loss_D_real = criterion_GAN(real_pred, torch.ones_like(real_pred))
                loss_D_fake = criterion_GAN(fake_pred, torch.zeros_like(fake_pred))
                loss_D = (loss_D_real + loss_D_fake) * 0.5
            scaler.scale(loss_D).backward()
            scaler.step(opt_D)
            scaler.update()

            opt_G.zero_grad()
            with torch.cuda.amp.autocast(enabled=(use_amp and device == "cuda")):
                fake = G(src)
                fake_pred_for_g = D(src, fake)
                loss_G_GAN = criterion_GAN(fake_pred_for_g, torch.ones_like(fake_pred_for_g))
                loss_G_L1 = criterion_L1(fake, real) * l1_lambda
                if perceptual is not None:
                    loss_G_per = perceptual(fake, real) * perceptual_lambda
                    loss_G = loss_G_GAN + loss_G_L1 + loss_G_per
                else:
                    loss_G = loss_G_GAN + loss_G_L1
            scaler.scale(loss_G).backward()
            scaler.step(opt_G)
            scaler.update()

            metrics = compute_metrics(fake.detach(), real)
            psnr_sum += metrics["psnr"]
            ssim_sum += metrics["ssim"]
            batch_count += 1

        print(
            f"Epoch {epoch:03d} loss_D:{loss_D.item():.4f} loss_G:{loss_G.item():.4f}"
            f" PSNR:{psnr_sum/batch_count:.2f} SSIM:{ssim_sum/batch_count:.4f}"
        )
        scheduler_G.step()
        scheduler_D.step()
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
    img_size: int = 256,
    num_downs: int = 8,
) -> None:
    """学習済みモデルを用いて文字画像を生成する。

    :param gen_checkpoint: Generator のチェックポイント
    :type gen_checkpoint: str
    :param chars_to_generate: 生成する文字マップ
    :type chars_to_generate: Dict[int, str]
    :param ref_font_path: 参考フォントパス
    :type ref_font_path: str
    :param out_dir: 出力ディレクトリ
    :type out_dir: str
    :param batch_size: 推論バッチサイズ
    :type batch_size: int
    :param img_size: 画像サイズ
    :type img_size: int
    :param num_downs: Generator のダウンサンプル回数
    :type num_downs: int
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    G = UNetGenerator(num_downs=num_downs).to(device)
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
            render_char_to_png(ref_font_path, glyph, buf, size=img_size)
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


def stagewise_train(
    target_font_path: str,
    ref_font_path: str,
    all_chars: Dict[int, str],
    fine_tune_chars: Dict[int, str] | None = None,
    stage1_epochs: int = 200,
    stage2_epochs: int = 20,
    stage1_lr: float = 2e-4,
    stage2_lr: float = 1e-5,
    checkpoint_dir: str = "checkpoints",
    rehearsal_ratio: float = 0.0,
    freeze_layers: int = 0,
    stage1_img_size: int = 256,
    stage2_img_size: int | None = None,
    stage1_num_downs: int = 8,
    stage2_num_downs: int | None = None,
    norm_type: str = "batch",
    **kwargs,
) -> None:
    """事前学習と微調整を続けて行うヘルパー関数。

    :param rehearsal_ratio: Stage2 で混合する既存文字の割合
    :type rehearsal_ratio: float
    :param freeze_layers: Stage2 学習で凍結するジェネレータ層数
    :type freeze_layers: int
    :param stage1_img_size: Stage1 の画像サイズ
    :type stage1_img_size: int
    :param stage2_img_size: Stage2 の画像サイズ (未指定なら Stage1 と同じ)
    :type stage2_img_size: int | None
    :param stage1_num_downs: Stage1 のU-Netダウンサンプル回数
    :type stage1_num_downs: int
    :param stage2_num_downs: Stage2 のU-Netダウンサンプル回数
    :type stage2_num_downs: int | None
    :param norm_type: 正規化レイヤー種別
    :type norm_type: str
    """

    train(
        target_font_path=target_font_path,
        ref_font_path=ref_font_path,
        chars_to_render=all_chars,
        epochs=stage1_epochs,
        lr=stage1_lr,
        checkpoint_dir=checkpoint_dir,
        img_size=stage1_img_size,
        num_downs=stage1_num_downs,
        norm_type=kwargs.get("norm_type", "batch"),
        **kwargs,
    )

    if not fine_tune_chars:
        return

    g_ckpt = os.path.join(checkpoint_dir, f"G_epoch{stage1_epochs:03d}.pth")
    d_ckpt = os.path.join(checkpoint_dir, f"D_epoch{stage1_epochs:03d}.pth")
    if rehearsal_ratio > 0:
        rehearse_n = max(1, int(len(all_chars) * rehearsal_ratio))
        rehearse_chars = dict(random.sample(list(all_chars.items()), rehearse_n))
        fine_tune_chars = {**fine_tune_chars, **rehearse_chars}

    train(
        target_font_path=target_font_path,
        ref_font_path=ref_font_path,
        chars_to_render=fine_tune_chars,
        epochs=stage2_epochs,
        lr=stage2_lr,
        checkpoint_dir=checkpoint_dir,
        pretrained_G_path=g_ckpt,
        pretrained_D_path=d_ckpt,
        augment_source_only=True,
        freeze_layers=freeze_layers,
        img_size=stage2_img_size or stage1_img_size,
        num_downs=stage2_num_downs or stage1_num_downs,
        norm_type=kwargs.get("norm_type", "batch"),
        **kwargs,
    )


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

    missing_chars_to_generate = {
        ord("琉"): "琉",
    }

    stagewise_train(
        target_font_path=TARGET_FONT_PATH,
        ref_font_path=REFERENCE_FONT_PATH,
        all_chars=common_chars_for_training,
        fine_tune_chars=missing_chars_to_generate,
        stage1_epochs=NUM_EPOCHS,
        stage2_epochs=50,
        stage1_lr=LEARNING_RATE,
        stage2_lr=LEARNING_RATE * 0.1,
        checkpoint_dir=CHECKPOINT_DIR,
        batch_size=BATCH_SIZE,
        l1_lambda=L1_LAMBDA,
        use_perceptual_loss=True,
        source_data_dir=os.path.join(OUTPUT_DIR_TRAIN, "train", "source"),
        target_data_dir=os.path.join(OUTPUT_DIR_TRAIN, "train", "target"),
        augment=True,
        rehearsal_ratio=0.1,
        freeze_layers=2,
        perceptual_lambda=0.1,
        img_size=256,
    )

    latest_checkpoint = os.path.join(CHECKPOINT_DIR, f"G_epoch{NUM_EPOCHS+50:03d}.pth")
    if os.path.exists(latest_checkpoint):
        inference(
            gen_checkpoint=latest_checkpoint,
            chars_to_generate=missing_chars_to_generate,
            ref_font_path=REFERENCE_FONT_PATH,
            out_dir=GENERATED_FONT_DIR,
            batch_size=BATCH_SIZE,
        )

