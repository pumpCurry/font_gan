# フォント補完用 GAN

このリポジトリには、GD-高速道路ゴシックJA フォントを参考フォントから学習し、新規グリフを生成するための pix2pix ベースの実装 `train_pix2pix.py` が含まれます。以下ではデータ準備から学習、推論までの手順を簡単にまとめます。

## 1. データ準備

1. **画像化**: 参考フォントと GD-高速道路ゴシックJA フォントをそれぞれ画像化します。`render_char_to_png()` を利用して 256×256 の白黒 PNG を作成します。
2. **ディレクトリ構造**:
   - `data/train/source/` … 参考フォント由来の PNG
   - `data/train/target/` … 対応する GD-高速道路ゴシックJA の PNG

同名ファイル同士がペアになるよう配置してください。ファイル名には `65.png` のように Unicode 番号を使うと後の処理が分かりやすくなります。

## 2. 学習

`train_pix2pix.py` の `train()` 関数が学習ループを担当します。`FontPairDataset` でペア画像を読み込み、U-Net 風ジェネレータと PatchGAN 風識別器を用いて通常の pix2pix 学習を行います。

```python
# データセットの作成例
 ds = FontPairDataset("data/train/source", "data/train/target")
 dl = DataLoader(ds, batch_size=4, shuffle=True, num_workers=2)
```

```python
# 学習ループ
 for epoch in range(1, 201):
     for src, real in dl:
         # Discriminator update
         fake = G(src)
         real_pred = D(src, real)
         fake_pred = D(src, fake.detach())
         ...
         # Generator update
         fake_pred = D(src, fake)
         loss_G = criterion_GAN(fake_pred, torch.ones_like(fake_pred)) + 100 * criterion_L1(fake, real)
         ...
```
上記処理は `train()` 内部で実行されており、10 エポックごとに `checkpoints/G_XXX.pth` が保存されます【F:train_pix2pix.py†L105-L146】。

学習は以下のコマンドで開始します。

```bash
python train_pix2pix.py
```

## 3. 推論

学習後は `inference()` を使って参考フォントから新しい文字を生成します。

```python
inference("checkpoints/G_200.pth", chars, "reference_font.otf", "output")
```
`chars` には補完したい文字のリストを渡します。出力結果は `output/` フォルダに PNG として保存されます【F:train_pix2pix.py†L150-L168】。

## 4. 生成結果の活用

生成された PNG からベクターフォントへ変換する場合は、FontForge の自動トレース機能などを組み合わせます。さらに品質を向上させたい場合は、モデルの深層化やストローク一貫性を考慮した損失の追加等を検討してください。

## 関連ドキュメント

- [使用方法の詳細](usage.md)
- [検討過程のメモ](process.md)


