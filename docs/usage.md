# 使用方法

このページでは `train_pix2pix.py` を利用した学習と推論の流れをまとめます。

## データ準備

学習スクリプトでは指定したフォントから必要な文字を自動描画して保存できます。既存の画像を使う場合は `data/train/source/` と `data/train/target/` に同名ファイルで配置します。ファイル名は Unicode 番号を使うと便利です。

## 学習

```bash
python train_pix2pix.py
```

- 10エポックごとに `checkpoints/` フォルダへモデルが保存されます。

## 推論

```python
inference("checkpoints/G_epoch200.pth", {ord("あ"): "あ"}, "reference_font.otf", "output")
```

- `output/` には生成されたグリフのPNGが保存されます。
