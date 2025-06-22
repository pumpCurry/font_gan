# 学習手順

このページでは `train_pix2pix.py` を用いた学習方法を説明します。

## データ準備

スクリプトではフォントファイルと文字リストを与えると学習用 PNG を自動生成します。既に画像を用意してある場合は `data/train/source/` と `data/train/target/` に配置してください。

## 学習の実行

```bash
python train_pix2pix.py
```

エポックごとに学習状況が出力され、10 エポックごとに `checkpoints/` にモデルが保存されます。
