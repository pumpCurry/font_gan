# 使用方法

このページでは `train_pix2pix.py` を利用した学習と推論の流れをまとめます。

## データ準備

1. `render_char_to_png()` を用いて参考フォントとGD-高速道路ゴシックJAフォントの文字をそれぞれPNG化します。
2. `data/train/source/` と `data/train/target/` に同名ファイルで配置します。ファイル名は `65.png` のようにUnicode番号を使うと便利です。

## 学習

```bash
python train_pix2pix.py
```

- 10エポックごとに `checkpoints/` フォルダへモデルが保存されます。

## 推論

```python
inference("checkpoints/G_200.pth", ["あ", "い"], "reference_font.otf", "output")
```

- `output/` には生成されたグリフのPNGが保存されます。
