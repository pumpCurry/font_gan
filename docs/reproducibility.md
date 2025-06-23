# 実験再現性とデバッグ

`train_pix2pix_pro.py` で安定した学習を行うためのポイントをまとめます。

## 乱数シードの固定

`set_seed()` を呼び出すことで PyTorch、NumPy、random のシードを統一し、
再現性を確保します。

```python
set_seed(2025)
```

## 実行設定の保存

`SummaryWriter` へ `writer.add_text('Config', pprint.pformat(config))`
を最初に記録すると、後からパラメータを確認しやすくなります。
同時に ``config.json`` を保存しておくと、実験ごとの設定を完全に再現できます。

## データセット取得

`FontPairDataset.__getitem__` は画像リサイズ、変換、正規化を行って
`(source, target)` を返します。以前のサンプルでは `pass` のままでしたが、
現在は下記のように実装されています。

```python
s = Image.open(self.src_paths[i]).convert('L')
# ...
return s, t
```

## 例外処理

フォント読み込みや文字描画に失敗した際は `try`/`except` で警告を出して
スクリプトが停止しないようにします。

## 設定ファイルの活用

複数実験を管理する場合は YAML など外部設定ファイルにまとめ、
コマンドライン引数で読み込む方法が便利です。

## DataLoader のワーカー

開発中は `num_workers=0` で開始し、安定したら 4 などに増やすと
予期せぬワーカー終了を避けられます。

## GPU メモリの監視

`nvidia-smi` のログや `torch.cuda.memory_allocated()` を定期的に記録すると
OOM 対策に役立ちます。

## ベストモデルの自動保存

検証データで計算した PSNR がこれまでの最高値を更新したときに `G_best.pth` が書き出されます。途中で性能が低下しても、常に最良モデルを保持できるため後処理が楽になります。

## 勾配クリッピング

勾配爆発を防ぐため、オプティマイザのステップ前に `clip_grad_norm_` を適用しています。これにより大きな学習率でも安定して収束しやすくなります。
