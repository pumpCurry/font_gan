# font-GAN Python Docstring 仕様書

## 1. 全体概要

* **対象**
  `font-GAN font creation tool` の Pythonモジュール群
* **目的**
  各ファイル・モジュール・クラス・関数に統一的かつ詳細なドックストリングを付与し、

  * モジュール／関数の目的と機能
  * 引数・戻り値の型と説明
  * バージョン管理情報
  * メンテナンス情報（作成者、ライセンスなど）
    を明確にドキュメント化する

---

## 2. ファイル先頭ドックストリングテンプレート

各 `.py` ファイルの最上部に必ず以下を記述してください。

```python
# -*- coding: utf-8 -*-
"""
{ファイル名}.py — {モジュール名}

概要:
    {このモジュールが提供する大まかな機能}

:author: pumpCurry
:copyright: (c) pumpCurry 2025 / 5r4ce2
:license: MIT
:version: 1.0.{コミット数} (PR #{PR番号})
:since:   1.0.{コミット数} (PR #{PR番号})
:last-modified: YYYY-MM-DD HH:MM:SS JST+9
:todo:
    - {未完了タスクの箇条書き}
"""
```

* **プレースホルダ説明**

  * `{ファイル名}.py`：実際のファイル名
  * `{モジュール名}`：パッケージ階層を含むPythonモジュール名
  * `{コミット数}`：`git rev-list --count main` 等で取得する累積コミット数
  * `{PR番号}`：該当プルリクエスト番号（未割当時は「最新PR番号+1」）
  * `:last-modified:`：ISO形式（JST+9）で最終更新日時
  * `:todo:`：開発で未対応の項目を箇条書き

---

## 3. モジュールレベルのドックストリング

モジュール内で最初に書くドックストリングは、上記ファイル先頭ドックストリングを兼ねます。以降、追加で説明が必要な場合のみ、モジュール概要や依存関係を追記します。

---

## 4. 関数／メソッドレベルのドックストリング

各公開関数・メソッドの直後に、GoogleスタイルまたはreSTスタイルで以下を必ず記述してください（※プロジェクト統一を推奨）。

### 4.1 reSTスタイル例

```python
def create_gan_model(config: dict, pretrained: bool = False) -> torch.nn.Module:
    """
    GANベースのフォント生成モデルを構築する。

    :param config: モデル設定辞書
    :type config: dict
    :param pretrained: 学習済み重みをロードする場合はTrue
    :type pretrained: bool
    :return: 構築されたPyTorchモデル
    :rtype: torch.nn.Module
    :raises ValueError: configの必須キーが欠如している場合
    :example:
        >>> model = create_gan_model({'layers': 5}, pretrained=True)
    """
```

### 4.2 Googleスタイル例

```python
def merge_fonts(base_font: str, target_glyphs: List[str]) -> Font:
    """ベースフォントに対して、指定グリフを生成済みフォントからマージする。

    Args:
        base_font (str): 元フォントのファイルパス
        target_glyphs (List[str]): 生成済みグリフ名のリスト

    Returns:
        Font: マージ後のフォントオブジェクト

    Raises:
        FileNotFoundError: base_fontが存在しない場合
        RuntimeError: マージ処理に失敗した場合

    Example:
        >>> merged = merge_fonts("NotoSans.otf", ["A", "B", "C"])
    """
```

* **必須タグ**

  * `:param`／`Args:`／`:type`：引数ごとに型と説明
  * `:return:`／`Returns:`／`:rtype:`：戻り値型と説明
  * `:raises`／`Raises:`：例外条件
  * `:example:`／`Example:`：呼び出しサンプル（推奨）

---

## 5. クラスレベルのドックストリング

```python
class FontGenerator:
    """
    font-GANのフォント生成器クラス。

    Attributes:
        latent_dim (int): 潜在ベクトルの次元数
        device (torch.device): 計算デバイス
    """

    def __init__(self, latent_dim: int, device: torch.device):
        """
        インスタンスを初期化する。

        :param latent_dim: 潜在ベクトルの次元数
        :type latent_dim: int
        :param device: モデルを配置するデバイス
        :type device: torch.device
        """
```

* **Attributes** セクションではインスタンス変数を列挙

---

## 6. 定数・列挙型のドックストリング

```python
#: モデル保存時の拡張子
MODEL_EXT: Final[str] = ".pt"

class FontStyle(Enum):
    """フォントスタイルの列挙型"""

    REGULAR = auto()
    BOLD = auto()
    ITALIC = auto()
```

* 定数にはモジュール変数直後に `#:` を使って簡潔にコメント
* `Enum` や `NamedTuple` 等もクラスドックストリングを記述

---

## 7. プライベート関数／メソッド

先頭にアンダースコアを付け、`@private` タグは不要ですが、簡潔に説明を残してください。

```python
def _load_checkpoint(path: str) -> dict:
    """モデルチェックポイントを読み込む内部関数。

    :param path: チェックポイントファイルパス
    :type path: str
    :return: チェックポイント辞書
    :rtype: dict
    """
```

---

## 8. バージョニング運用

* **新規ファイル追加時**

  * ファイル先頭の `:version:` `:since:` は必ず最新コミット数・PR番号で埋める
* **既存ファイル更新時**

  * `:version:` を更新
  * 新規追加した関数／クラスには必ずドックストリングを追加
* **CIチェック**

  * `pydocstyle` や `flake8-docstrings` を導入し、未記載をエラー化推奨

---

以上を遵守することで、font-GANのPythonコード全体にわたり、一貫性あるドキュメント品質とメンテナンス性を担保できます。
