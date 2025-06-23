# 設定ファイルによる実行

`train_from_config` を使うと YAML ファイルに学習設定をまとめて記述できます。
ここでは主要なキーと文字リストの指定方法を説明します。

## 基本構造

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

各項目の意味は次の通りです。

- `target_font_path` / `ref_font_path` — 学習対象と参考フォントのパス。
- `chars_to_render` — 学習に利用する文字マップ。キーは Unicode
  コードポイント、値は実際の文字です。代わりに `learning_list_file`
  を指定すると外部テキストから読み込めます。
- `epochs` — 学習エポック数。
- `batch_size` — ミニバッチサイズ。
- `lr` — 学習率。
- `use_perceptual_loss` — VGG 知覚損失を有効にするか。
- `log_memory` — エポックごとに GPU メモリ使用量を表示。

```bash
python -c "from train_pix2pix import train_from_config; train_from_config('conf.yaml')"
```

## 文字リストを別ファイルにまとめる

`learning_list_file` キーを用いると、学習対象文字をテキストファイルから
読み込めます。ファイルは 1 行に 1 文字、または `U+XXXX` 形式でコードポイント
を記述します。

```text
あ
い
U+6C34
```

YAML では次のように指定します。

```yaml
target_font_path: path/to/target.otf
ref_font_path: path/to/ref.otf
learning_list_file: chars.txt
epochs: 200
```

この方法を使うと、複雑な辞書を直接記述せずに学習文字を管理できます。
