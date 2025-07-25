# コード概要

このページでは `train_pix2pix_pro.py` の主な機能と構成要素を概説します。

## 主要コンポーネント

- **PNG 生成**: `render_char_to_png` 関数は指定フォントのグリフを画像として保存します。学習用データセット生成の要となります。
- **評価指標**: `compute_metrics` で PSNR と SSIM を計算し、生成品質を定量評価します。
- **Perceptual Loss**: `VGGPerceptualLoss` クラスは VGG16 の中間層を利用した特徴損失を計算し、細部の再現性を高めます。
- **データ拡張**: `morphology_transform` は膨張・収縮処理による太さ変化を実現します。`torchvision.transforms` と組み合わせて参照文字とターゲット文字へ別々の前処理を施します。
- **データセット**: `FontPairDataset` がフォント画像ペアを読み込み、学習時には正規化とリサイズを行います。
- **モデル構成**: U-Net ベースのジェネレータと PatchGAN 識別器を実装。正規化層は `BatchNorm` または `InstanceNorm` を選択可能です。
- **学習ループ**: 自動混合精度 (`torch.cuda.amp`) と勾配累積を用いてメモリ消費を抑えつつ学習します。`tqdm` を使って進捗を表示し、各エポックでは TensorBoard に比較画像も保存されます。学習率は `CosineAnnealingLR` で徐々に減衰させます。

## 2 段階学習設定

スクリプトは `argparse` を用いた CLI として提供され、`--stage` オプションで 256px 学習 (`s1_256`) と 512px 学習 (`s2_512`) を切り替えられます。Stage2 では Stage1 の重みを読み込んで微調整を行います。
