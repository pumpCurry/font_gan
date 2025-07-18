# 学習対象文字リストの管理

`train_pix2pix_pro.py` では学習に使用する文字のリストを外部ファイルから読み込むか、
フォントファイルを解析して自動生成する2通りの方法を選べます。

## 外部ファイルを使用する場合

`learning_list.txt` のようなテキストファイルを用意し、1行に1文字または
`U+3042` のようなコードポイントを記述します。このファイルへのパスを
`learning_list_file` に指定すると、記載された文字だけが学習対象となります。

```
あ
い
U+6C34
```

## 自動判定を利用する場合

`learning_list_file` を指定しないときは、`candidate_chars` に含まれる文字を
順にフォントへ描画し、ほぼ白紙となるグリフを除外して学習リストを作成します。
既存フォントに空のグリフがある場合でも自動的にスキップされるため、
手動メンテナンスの手間が減ります。

この仕組みにより、コード内に文字列をベタ書きせず柔軟に学習文字を管理できます。

## include/exclude 方式と範囲指定

バージョン 1.0.50 からは ``--include_chars`` と ``--exclude_chars`` に
テキストファイルを指定することで、学習候補を追加・削除できます。
さらに ``--range_start`` ``--range_end`` を組み合わせると、
Unicode コードポイント範囲を直接指定できます。これらで得られた候補は
ベースフォントと参考フォントの両方に存在するかを確認し、片方でも
空白グリフの場合は除外されます。

```bash
python train_pix2pix_pro.py --stage s1_256 \
  --include_chars chars.txt --exclude_chars skip.txt \
  --range_start 3040 --range_end 309F \
  --ref_font ./fonts/ref.otf --target_font ./fonts/base.otf
```

``chars.txt`` に必ず学習したい文字を、``skip.txt`` に除外したい文字を列挙します。

## 文字一覧だけを抽出したい場合

``--dump_char_string`` を指定すると、候補文字をフィルタリングした結果を
1 行の文字列としてファイルへ保存して終了します。空白文字や空白グリフは
自動的に除外されます。

```bash
python train_pix2pix_pro.py --stage s1_256 \
  --ref_font ./fonts/ref.otf --target_font ./fonts/base.otf \
  --dump_char_string output_chars.txt
```

``output_chars.txt`` には抽出された文字が連続した文字列として書き出されます。

## フォント収録済み文字をすべて列挙する場合

``--dump_registed_all_char`` を使うと、指定フォントに登録されている文字を
Unicode 全域にわたり走査して抽出します。未定義コードポイントや描画しても
白紙となるグリフは除外されます。

```bash
python train_pix2pix_pro.py --stage s1_256 \
  --target_font ./fonts/base.otf \
  --dump_registed_all_char all_chars.txt
```

結果は ``all_chars.txt`` に 1 行の文字列として保存されます。
