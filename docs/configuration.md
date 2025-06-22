# 設定ファイルによる実行

`train_from_config` を使うと YAML ファイルに学習設定をまとめて記述できます。
以下はサンプルです。

```yaml
target_font_path: path/to/target.otf
ref_font_path: path/to/ref.otf
chars_to_render:
  12354: "あ"
  12356: "い"
epochs: 200
batch_size: 4
lr: 0.0002
use_perceptual_loss: true
log_memory: true
```

```bash
python -c "from train_pix2pix import train_from_config; train_from_config('conf.yaml')"
```
