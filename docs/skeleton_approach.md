# 骨格入力とスタイル分離の検討

ここではフォント画像を細線化した骨格と、参考フォント画像の2チャネルで学習する
簡易手法をまとめます。骨格情報を明示的に与えることで高解像度でもストローク形状
の再現性を高めることが目的です。

## データ準備

`scripts/prepare_skeleton_data.py` を用いてベースフォントの骨格画像を生成します。
Gaussian ブラーと Otsu 二値化を経て skeletonize した後、細かなノイズを除去して
1px の中心線画像を得ます。

```bash
python scripts/prepare_skeleton_data.py \
  --font fonts/base.otf \
  --char_list chars.txt \
  --out_dir data/skeleton

生成された骨格は PNG とともに `.pt` 形式のテンソルも保存され、学習時に高速に読み込めます。
```

## 学習方法

`train_pix2pix_pro.py` の ``--skeleton_dir`` オプションに骨格画像ディレクトリを
指定すると、Generator への入力が ``[骨格, 参考フォント]`` の2チャネルになります。
データセットクラス ``FontPairDataset`` は骨格画像も読み込み、必要に応じて前処理を
適用してから結合します。

```bash
python train_pix2pix_pro.py \
  --stage s1_256 \
  --ref_font fonts/ref.otf \
  --target_font fonts/target.otf \
  --skeleton_dir data/skeleton
```

この設定で生成品質を確認し、効果が高ければスタイルベクトルを用いた AdaIN などへ
発展させます。
