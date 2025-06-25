# フォント補完用 GAN

このリポジトリは pix2pix を利用したフォント補完モデルを提供します。参考フォントのグリフ画像から GD-高速道路ゴシックJA の欠損文字を生成することを目的としています。

中心となるスクリプトは `train_pix2pix.py` で、フォントファイルと文字リストを与えるだけで学習用画像の生成からモデル学習までを自動で行います。二段階学習や多層 Perceptual Loss、512px までの高解像度学習、混合精度学習に対応しています。さらに PSNR/SSIM による定量評価と CosineAnnealingLR による学習率スケジュールも実装しました。最新バージョンでは SciPy を用いた太さ変化の増強や InstanceNorm 切り替えも可能です。`tqdm` で進捗バーを表示し、重み初期化やリハーサル戦略を改善した `train_pix2pix_pro.py` を使うと、TensorBoard に比較画像も保存しながら 256px 事前学習と 512px 微調整を自動実行できます。主要設定はコマンドライン引数から変更できます。

バージョン 1.0.47 では検証データによる PSNR/SSIM 評価に加え、細線化・太線化の確率的増強を ``RandomApply`` で実装しました。`pin_memory` を有効にしたデータローダと、設定可能な勾配クリッピングで学習の安定性を向上させています。すべての実行設定は ``config.json`` として保存されます。

バージョン 1.0.50 では学習文字の指定方法が拡張されました。``--include_chars`` ``--exclude_chars`` ``--range_start`` ``--range_end`` を組み合わせることで、任意の文字セットを簡単に構築できます。ベースフォントと参考フォントのどちらかが空白グリフを返す文字は自動的に除外されます。

バージョン 1.0.60 では細線化した骨格画像を入力に加える ``--skeleton_dir`` オプションを導入しました。`scripts/prepare_skeleton_data.py` で生成した骨格画像を参考フォント画像と結合して学習させることで、ストローク構造の再現性を高めています。

バージョン 1.0.62 では骨格生成時に Gaussian ブラーと Otsu 二値化を適用し、
小さなノイズを除去した上で skeletonize するよう改良しました。データセット
``FontPairDataset`` は骨格画像用の前処理を受け取れるようになり、2チャネル入力
による簡易検証を進めやすくなっています。

バージョン 1.0.64 では Discriminator の入力チャンネルを修正し、生成画像の細線化結果と骨格画像との L1 損失 ``stroke_lambda`` を追加しました。
バージョン 1.0.65 では骨格画像を.pt 形式で保存し、Sobel フィルタを用いた GPU ストローク損失と Discriminator 入力ノイズを追加しました。骨格損失の重みはコサインスケジュールで減衰します。
``prepare_skeleton_data.py`` は ``--no_blur`` オプションで前処理を制御できます。
バージョン 1.0.68 では画像ペアと骨格を一括保存する ``prepare_data_step1_5.py`` を追加し、``--preprocessed_dir`` から高速読み込みが可能になりました。評価指標に Edge IoU を導入しています。

バージョン 1.0.70 では ``PreprocessedFontDataset`` がメモリマップ読み込みに対応し、骨格損失の重みをエッジ面積で正規化します。Discriminator 入力ノイズをエポックに応じて減衰させ、検証指標 ``Mean_Edge_Width`` を追加しました。

バージョン 1.0.72 では ``prepare_data_step1_5.py`` が各サンプルのエッジ面積を保存し、``calculate_stats.py`` で平均値を取得できます。学習時には層化サンプリングで検証データを選び、``--use_compile`` オプションで ``torch.compile`` を試せます。

## ドキュメント構成

- [環境構築ガイド](installation.md)
- [pip + venv での環境構築](installation_venv.md)
- [Windows 版セットアップ](installation_windows.md)
- [チュートリアル](tutorial.md)
- [使用方法の概要](usage.md)
- [学習手順](usage/training.md)
- [推論手順](usage/inference.md)
- [学習戦略](training_strategy.md)
- [設定ファイルによる実行](configuration.md)
- [検討過程のメモ](process.md)
- [大解像度画像利用の検討](high_resolution.md)
- [コード概要](code_overview.md)
- [実装の概要](technical_details.md)
- [骨格入力とスタイル分離](skeleton_approach.md)
- [学習文字リストの管理](character_list.md)
- [実験再現性とデバッグ](reproducibility.md)
- [高速前処理データセット](preprocessed_dataset.md)
- [高度なワークフロー](advanced_workflow.md)

