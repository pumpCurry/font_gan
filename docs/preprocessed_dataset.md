# 高速前処理データセット

`prepare_data_step1_5.py` を使うと、参考フォント、ターゲットフォント、骨格画像を 1 つの `.pt` ファイルにまとめて保存できます。学習時に PNG を逐次読み込む必要がなくなり、I/O と CPU の負荷を大幅に削減できます。

## 使い方

```bash
python scripts/prepare_data_step1_5.py \
  --ref_font ./fonts/reference_font.otf \
  --target_font ./fonts/GD-HighwayGothicJA.otf \
  --skeleton_base_font ./fonts/base_gothic.otf \
  --char_list ./chars.txt \
  --size 256 \
  --output_dir ./data/preprocessed
```

各文字ごとに `U+XXXX.pt` が出力されます。このディレクトリを `train_pix2pix_pro.py` の `--preprocessed_dir` に指定すると、高速ローダー `PreprocessedFontDataset` が使用されます。 このローダーは ``torch.load(..., mmap=True)`` を利用してメモリマップ読み込みを行うため、大規模データでも RAM 消費を抑えられます。
