# train_pix2pix.py
import os
import glob
from PIL import Image, ImageDraw, ImageFont
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

# 1. フォントから PNG を生成するユーティリティ

def render_char_to_png(font_path, char, out_path, size=256):
    font = ImageFont.truetype(font_path, size)
    img = Image.new("L", (size, size), color=255)
    draw = ImageDraw.Draw(img)
    w, h = draw.textsize(char, font=font)
    draw.text(((size-w)/2, (size-h)/2), char, font=font, fill=0)
    img.save(out_path)


# 2. Dataset：ペア画像を読み込み
class FontPairDataset(Dataset):
    def __init__(self, source_dir, target_dir, transform=None):
        self.src_paths = sorted(glob.glob(os.path.join(source_dir, "*.png")))
        self.tgt_paths = [os.path.join(target_dir, os.path.basename(p)) for p in self.src_paths]
        self.transform = transform or T.Compose([
            T.ToTensor(),
            T.Normalize((0.5,), (0.5,)),
        ])

    def __len__(self):
        return len(self.src_paths)

    def __getitem__(self, idx):
        src = Image.open(self.src_paths[idx]).convert("L")
        tgt = Image.open(self.tgt_paths[idx]).convert("L")
        return self.transform(src), self.transform(tgt)


# 3. モデル定義
# 3.1 U-Net Generator
class UNetGenerator(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, ngf=64):
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

    def forward(self, x):
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
    def __init__(self, in_ch=2, ndf=64):
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

    def forward(self, src, tgt):
        x = torch.cat([src, tgt], dim=1)
        return self.model(x)


# 4. 学習ループ

def train():
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

def inference(gen_checkpoint, chars, ref_font_path, out_dir):
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


if __name__ == "__main__":
    # 学習
    train()

    # 推論例
    # missing = ["あ", "い", "う"]  # 補完したい文字リスト
    # inference("checkpoints/G_200.pth", missing, "reference_font.otf", "output")
