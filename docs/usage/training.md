# 学習手順

このページでは `train_pix2pix.py` を用いた学習方法を説明します。

## データ準備

フォントファイルと文字リストを指定すると学習用PNGが自動生成されます。既存画像を使う場合は `data/train/source/` と `data/train/target/` に配置してください。

## 基本的な学習

```bash
python train_pix2pix.py
```

エポックごとに進捗が表示され、10エポックごとに `checkpoints/` へモデルが保存されます。

## 2 段階学習

データ量が限られている場合は `stagewise_train` を用いることで精度を高められます。

```python
from train_pix2pix import stagewise_train
stagewise_train(target_font, ref_font, all_chars, fine_chars, augment=True)
```

最初に全文字で学習した後、指定文字のみを低学習率で微調整します。`augment=True` を指定するとアフィン変換やノイズ付与などの前処理が行われます。

