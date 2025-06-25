# 学習手順

このページでは `train_pix2pix.py` を用いた学習方法を説明します。

## データ準備

フォントファイルと文字リストを指定すると学習用PNGが自動生成されます。既存画像を使う場合は `data/train/source/` と `data/train/target/` に配置してください。

## 基本的な学習

```bash
python train_pix2pix.py
```
エポックごとに進捗が表示され、10エポックごとに `checkpoints/` へモデルが保存されます。
`img_size` や `num_downs` を調整することで 512px までの高解像度モデルが学習可能です。
GPU メモリが厳しい場合は `use_amp=True` を指定すると混合精度学習となります。
`log_memory=True` を渡すと各エポック後に GPU 使用量が出力されます。

## 2 段階学習

データ量が限られている場合は `stagewise_train` を用いることで精度を高められます。

```python
from train_pix2pix import stagewise_train
stagewise_train(
    target_font,
    ref_font,
    all_chars,
    fine_chars,
    augment=True,
    perceptual_lambda=0.1,
    rehearsal_ratio=0.1,
    freeze_layers=2,
    norm_type="instance",
)
```

最初に全文字で学習した後、指定文字のみを低学習率で微調整します。`rehearsal_ratio` を指定すると既存文字の一部も混在させ、忘却を防ぎます。`freeze_layers` でジェネレータの前半層を固定でき、`perceptual_lambda` を設定すると Perceptual Loss が有効になります。`augment=True` を指定するとアフィン変換やノイズ付与などの前処理が行われます。
`norm_type` には `batch` と `instance` が選択でき、スタイル転送用途では InstanceNorm を推奨します。

## 設定ファイルからの学習

複数のハイパーパラメータを YAML にまとめ、`train_from_config` 関数から読み込むことができます。

設定ファイルの詳細は [設定ファイルによる実行](../configuration.md) を参照してください。特に `learning_list_file` を使うと学習文字を外部ファイルで管理できます。

```bash
python -c "from train_pix2pix import train_from_config; train_from_config('conf.yaml')"
```


## 自動2段階学習スクリプト

`train_pix2pix_pro.py` を実行すると、256px での事前学習と 512px での微調整を連続で行えます。主要な設定はコマンドライン引数から変更でき、実行中は `tqdm` による進捗バーが表示されます。

```bash
python train_pix2pix_pro.py \
  --stage s1_256 \
  --ref_font ./fonts/reference_font.otf \
  --target_font ./fonts/GD-HighwayGothicJA.otf \
  --skeleton_dir ./data/skeleton

``--skeleton_dir`` には ``prepare_skeleton_data.py`` で生成した骨格画像のディレクトリを指定します。指定しない場合は1チャネル入力となります。

python train_pix2pix_pro.py \
  --stage s2_512 \
  --ref_font ./fonts/reference_font.otf \
  --target_font ./fonts/GD-HighwayGothicJA.otf \
  --checkpoint_dir ./checkpoints/gd_highway_pro
```

学習対象文字を細かく指定したい場合は ``--include_chars`` ``--exclude_chars``
``--range_start`` ``--range_end`` を併用します。

```bash
python train_pix2pix_pro.py \
  --stage s1_256 \
  --include_chars chars.txt --exclude_chars skip.txt \
  --range_start 3040 --range_end 309F \
  --ref_font ./fonts/ref.otf \
  --target_font ./fonts/GD-HighwayGothicJA.otf
```

