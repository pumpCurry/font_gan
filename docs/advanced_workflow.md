# 高度なワークフロー

ここでは `prepare_data_step1_5.py` と `calculate_stats.py`、 `train_pix2pix_pro.py` を用いたステップ1.5 の実行方法をまとめます。

## 1. データ前処理

```bash
python scripts/prepare_data_step1_5.py \
  --ref_font ./fonts/ref.otf \
  --target_font ./fonts/target.otf \
  --skeleton_base_font ./fonts/base.otf \
  --char_list ./chars.txt \
  --size 256 \
  --output_dir ./data/preprocessed
```

各文字の参照画像・目標画像・骨格画像が `U+XXXX.pt` として保存され、エッジ面積も記録されます。

## 2. 統計値の計算

```bash
python scripts/calculate_stats.py --preprocessed_dir ./data/preprocessed/256
```

出力された `Mean Edge Area` の値を `train_pix2pix_pro.py` の引数 `--mean_edge_area` または設定ファイルに追記します。

## 3. 学習の実行

```bash
python train_pix2pix_pro.py \
  --stage s1_256 \
  --preprocessed_dir ./data/preprocessed/256 \
  --mean_edge_area 0.07 \
  --use_compile
```

層化サンプリングにより検証データを選び、`torch.compile` が利用可能ならモデルを JIT コンパイルして高速化を試みます。
