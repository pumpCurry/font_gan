# 環境構築ガイド（pip + venv 版）

このドキュメントでは `conda` を利用しない場合のセットアップ手順を説明します。標準の `venv` と `pip` を使って環境を構築する方法です。

また、Windows 環境向けの手順は [installation_windows.md](installation_windows.md) を参照してください。
## 1. システム要件

- **OS**: Ubuntu 20.04 LTS を推奨。Windows では WSL2 上での利用を想定しています。
- **GPU**: CUDA 対応の NVIDIA GPU (メモリ 8GB 以上)。
- **ドライバと CUDA**: `nvidia-smi` で確認できる最新ドライバを導入し、CUDA 11.x もしくは 12.x をインストールしてください。
- **ディスク容量**: フォント画像やチェックポイントを保存するため数 GB の空き容量が必要です。

## 2. Python と仮想環境の準備

1. Python 3.9 以上をインストールします。
2. プロジェクト用の仮想環境を作成し、有効化します。

```bash
python3 -m venv font_gan_venv
source font_gan_venv/bin/activate
```
これらの手順は `scripts/setup.sh` を実行することで自動化できます。

## 3. GPU 動作確認

```bash
nvidia-smi
```

`nvidia-smi: command not found` と表示される場合は GPU ドライバがインストールされていません。CPU で実行するか、NVIDIA 公式サイトからドライバを入れてください。
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
上記コードを `scripts/check_gpu.py` として保存して実行できます。

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
## 7. 初回準備ファイル

最初に `fonts/` ディレクトリに補完対象フォントと参考フォントを配置します。既存画像を利用する場合は `data/train_s1` や `data/train_s2` 以下に PNG を置きます。必要に応じて `config.json` や学習文字リストもここに準備してください。

## 8. TensorBoard の起動

学習中のログは次のコマンドで確認できます。

```bash
tensorboard --logdir ./checkpoints/gd_highway_pro/logs
```

ブラウザで `http://localhost:6006/` を開くと損失やサンプル画像を閲覧できます。

## 9. 学習と推論

仮想環境を有効化し、`train_pix2pix.py` を実行すると学習が始まります。Stage1(256px) から Stage2(512px) へ自動で移行し、生成結果は `output/` に保存されます。

```bash
source font_gan_venv/bin/activate
python train_pix2pix.py
```

学習後は生成された PNG を FontForge 等でトレースし、フォントファイルへ組み込んでください。
チュートリアルは [tutorial.md](tutorial.md) を参照してください。
