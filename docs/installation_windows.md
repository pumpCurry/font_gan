# Windows 版セットアップガイド

このドキュメントでは Windows 環境で `pip` と `venv` を用いて本プロジェクトを動作させる方法を説明します。WSL2 ではなく、純粋な Windows 上に Python がインストールされていることを想定しています。

## 1. 必要なソフトウェア

- Python 3.9 以上
- NVIDIA GPU と最新ドライバ (CUDA 11.x または 12.x 対応)
- Git (オプション)

## 2. 仮想環境の作成とライブラリのインストール

同梱の `scripts/setup_windows.bat` を実行すると、仮想環境の作成から主要ライブラリのインストールまで自動で行われます。PowerShell から次のように実行してください。

```powershell
cd path\to\font_gan
./scripts/setup_windows.bat
```

上記では CUDA 11.8 用の PyTorch を導入しています。CUDA バージョンが異なる場合はスクリプト内の URL を適宜変更してください。

## 3. GPU の確認

ドライバが正しくインストールされていれば、`nvidia-smi` コマンドで GPU の情報が表示されます。`'nvidia-smi' is not recognized` と表示された場合はドライバが不足しています。NVIDIA 公式サイトから Windows 用ドライバをインストールしてください。

環境が整ったら、以下のスクリプトで PyTorch が GPU を認識するかを確認します。

```bash
python scripts/check_gpu.py
```

`True` が表示されれば GPU が利用可能です。`False` のときは CUDA のパスやドライバの設定を見直してください。

## 4. ディレクトリ構造と初回準備

基本的なフォルダ構成は Linux 版と共通です。`fonts/` に補完対象フォント (`GD-HighwayGothicJA.otf` など) と参考フォントを配置し、`data/` 以下に学習用画像を用意します。詳しくは [installation_venv.md](installation_venv.md) の "初回準備ファイル" を参照してください。
