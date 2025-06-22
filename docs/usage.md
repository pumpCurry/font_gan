# 使用方法

このページでは `train_pix2pix.py` を利用した学習と推論の流れをまとめます。

## データ準備

学習スクリプトはフォントファイルと文字一覧を指定すると学習用PNGを自動生成します。既存の画像を利用する場合は `data/train/source/` と `data/train/target/` に同名ファイルを配置してください。ファイル名は Unicode 番号を使うと便利です。

## 学習

### 単一フェーズでの学習

```bash
python train_pix2pix.py
```

10エポックごとに `checkpoints/` へモデルが保存されます。

### 2 段階学習

```python
from train_pix2pix import stagewise_train
stagewise_train(
    target_font,
    reference_font,
    all_chars,
    fine_tune_chars,
    augment=True,
    perceptual_lambda=0.1,
    rehearsal_ratio=0.1,
    freeze_layers=2,
)
```

まず全文字で事前学習を行い、その後不足文字のみを低学習率で微調整します。`rehearsal_ratio` を指定すると既存文字の一部も混在させて忘却を防止できます。`freeze_layers` でジェネレータ前半を固定し、`perceptual_lambda` により Perceptual Loss の強さを調節します。`augment=True` を指定すると学習時にアフィン変換やノイズ付与が行われます。

## 推論

```python
from train_pix2pix import inference
inference("checkpoints/G_epoch200.pth", {ord("あ"): "あ"}, "reference_font.otf", "output")
```

`output/` には生成されたグリフのPNGが保存されます。

