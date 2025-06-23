# 環境構築ガイド

このページでは本プロジェクトを動かすための詳細な環境構築手順を説明します。GPU ドライバの準備から Python ライブラリのインストール、ディレクトリ構造の作成までを順を追って紹介します。

## 1. システム要件

- **OS**: Ubuntu 20.04 LTS を推奨。Windows では WSL2 上での利用を想定しています。
- **GPU**: CUDA 対応の NVIDIA GPU (メモリ 8GB 以上)。
- **ドライバと CUDA**: `nvidia-smi` で確認できる最新ドライバを導入し、CUDA 11.x もしくは 12.x をインストールしてください。
- **ディスク容量**: フォント画像やチェックポイントを保存するため数 GB の空き容量が必要です。

## 2. Conda 環境の作成

1. Miniconda もしくは Anaconda をインストールします。
2. 新しい環境を作成し Python 3.9 を指定します。

```bash
conda create -n font_gan python=3.9 -y
conda activate font_gan
```

## 3. GPU 動作確認

```bash
nvidia-smi
```

CUDA バージョンや GPU が表示されれば準備完了です。

## 4. PyTorch のインストール

自身の CUDA バージョンに合わせて PyTorch をインストールします。以下は CUDA 11.8 を使用する例です。

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

インストール後、以下のスクリプトで GPU が利用可能か確認できます。

```python
import torch
print(torch.__version__, torch.cuda.is_available())
```

## 5. 追加ライブラリ

画像処理や評価指標、TensorBoard などのライブラリをインストールします。

```bash
pip install Pillow scipy opencv-python
pip install scikit-image
pip install tensorboard
# Optional
pip install optuna
pip install fontforge
```

## 6. ディレクトリ構造

学習用フォルダは以下のように配置します。`checkpoints/` や `output/` はスクリプト実行時に自動生成されます。

```
font_gan/
├── train_pix2pix.py
├── checkpoints/
├── data/
│   ├── train_s1/{source,target}/
│   └── train_s2/{source,target}/
├── output/
└── fonts/
    ├── GD-HighwayGothicJA.otf
    └── reference_font.otf
```

## 7. TensorBoard の起動

学習中のログは次のコマンドで確認できます。

```bash
tensorboard --logdir ./checkpoints/gd_highway_pro/logs
```

ブラウザで `http://localhost:6006/` を開くと損失やサンプル画像を閲覧できます。

## 8. 学習と推論

Conda 環境を有効化し、`train_pix2pix.py` を実行すると学習が始まります。Stage1(256px) から Stage2(512px) へ自動で移行し、生成結果は `output/` に保存されます。

```bash
conda activate font_gan
python train_pix2pix.py
```

学習後は生成された PNG を FontForge 等でトレースし、フォントファイルへ組み込んでください。
引き続き [チュートリアル](tutorial.md) で実行例を確認するとスムーズです。

