# font_gan

フォントを GAN で生成するための実験リポジトリです。`docs/` 以下に学習手順や環境構
築の詳細をまとめています。ここでは最小構成での実行例を示します。

## 環境セットアップ

1. [環境構築ガイド](docs/installation.md) に従い Conda 環境を作成します。
2. ライブラリをインストール後、以下のコマンドで GPU が利用可能か確認します。

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## 基本的な学習

フォントファイルを `fonts/` に配置したら、次のコマンドで 256px から 512px まで連続
学習できます。

```bash
python train_pix2pix_pro.py \
  --stage s1_256 \
  --ref_font ./fonts/reference_font.otf \
  --target_font ./fonts/target_font.otf

python train_pix2pix_pro.py \
  --stage s2_512 \
  --ref_font ./fonts/reference_font.otf \
  --target_font ./fonts/target_font.otf \
  --checkpoint_dir ./checkpoints/pro_run
```

## 推論

学習済みモデルを使って新しいグリフを生成するには次のようにします。

```python
from train_pix2pix import inference
chars = {ord("あ"): "あ"}
inference("checkpoints/pro_run/G_epoch200.pth", chars, "fonts/reference_font.otf", "output")
```

チュートリアル形式での説明は [docs/tutorial.md](docs/tutorial.md) を参照してくださ
い。
