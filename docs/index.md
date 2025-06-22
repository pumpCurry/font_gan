# フォント補完用 GAN

このリポジトリは pix2pix を利用したフォント補完モデルを提供します。参考フォントのグリフ画像から GD-高速道路ゴシックJA の欠損文字を生成することを目的としています。

中心となるスクリプトは `train_pix2pix.py` で、フォントファイルと文字リストを与えるだけで学習用画像の生成からモデル学習までを自動で行います。二段階学習や多層 Perceptual Loss、512px までの高解像度学習、混合精度学習に対応しています。さらに PSNR/SSIM による定量評価と CosineAnnealingLR による学習率スケジュールも実装しました。最新バージョンでは SciPy を用いた太さ変化の増強や InstanceNorm 切り替えも可能です。`tqdm` で進捗バーを表示し、重み初期化やリハーサル戦略を改善した `train_pix2pix_pro.py` を使うと、TensorBoard に比較画像も保存しながら 256px 事前学習と 512px 微調整を自動実行できます。主要設定はコマンドライン引数から変更できます。

## ドキュメント構成

- [環境構築ガイド](installation.md)
- [使用方法の概要](usage.md)
- [学習手順](usage/training.md)
- [推論手順](usage/inference.md)
- [学習戦略](training_strategy.md)
- [設定ファイルによる実行](configuration.md)
- [検討過程のメモ](process.md)
- [大解像度画像利用の検討](high_resolution.md)
- [コード概要](code_overview.md)
- [実装の概要](technical_details.md)
- [学習文字リストの管理](character_list.md)

