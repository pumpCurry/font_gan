# チュートリアル

このチュートリアルでは、環境構築から学習、推論までの基本的な流れをまとめます。
より詳細なオプションは各ドキュメントを参照してください。

## 1. 環境を整える

1. [環境構築ガイド](installation.md) に従い Conda 環境を作成します。
2. 必要なライブラリをインストール後、以下のコマンドで GPU が認識されていることを確認します。

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

`True` が表示されれば準備完了です。

## 2. フォントを配置する

`fonts/` ディレクトリに補完対象のフォント (`target_font.otf`) と参考フォント (`reference_font.otf`) を置きます。
学習文字リストは `--include_chars` でファイル指定するか、フォント内の非空白グリフを自動抽出できます。

## 3. 学習を実行する

まずは 256px で事前学習を行い、その後 512px へ微調整する例を示します。

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

学習中は `checkpoints/` 以下にモデルやログが保存されます。`tensorboard --logdir` で進捗を確認できます。

## 4. 生成を試す

学習したモデルを使ってグリフを生成するには `inference` 関数を利用します。

```python
from train_pix2pix import inference

chars = {ord("あ"): "あ"}
inference("checkpoints/pro_run/G_epoch200.pth", chars, "fonts/reference_font.otf", "output")
```

`output/` には Unicode 番号をファイル名とした PNG が出力されます。必要に応じて FontForge などでトレースしてフォントへ取り込みます。

