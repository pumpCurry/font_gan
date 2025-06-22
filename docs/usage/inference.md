# 推論手順

学習済みモデルを使って新しいグリフを生成する方法を示します。

```python
from train_pix2pix import inference

chars = ["あ", "い", "う"]
inference("checkpoints/G_200.pth", chars, "reference_font.otf", "output")
```

`output/` に Unicode 番号をファイル名とした PNG が生成されます。
