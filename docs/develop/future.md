# 3dpmon v2 フロントエンドデバッグ手順

ステップ7（E2E テスト＆リリース準備）までを完了したが、ブラウザ上で真っ白な画面しか表示されない場合の確認事項をまとめます。レンダリングパイプラインが正しく動作していれば、以下のような画面構成が現れるはずです。

## ✅ 本来の画面構成イメージ

1. **タイトルバー（Header）**
   - 左端にハンバーガーメニューアイコン
   - 中央にアプリ名「3dpmon v2」またはログインステータス
   - 右端にサイドメニューアイコン（将来追加）
2. **タブバー**（タイトルバー直下）
   - `Printer 1` など接続先ごとにタブが追加される（スクロール可能）
3. **サイドメニュー**（通常は非表示）
   - ハンバーガーメニュークリックでスライドイン
   - 「接続一覧」「設定」「テーマ切替」「About」を配置
4. **カード群（Dashboard 本体）**
   - **CameraCard**：リアルタイム映像プレビュー
   - **HeadPreviewCard**：位置表示用 Canvas/3D ビュー
   - **StatusCard**：印刷状態や温度など
   - **ControlPanelCard**：手動操作ボタン
   - **CurrentPrintCard**：進捗バーと残り時間
   - **TempGraphCard**：温度グラフ＋ファン速度軸
   - **MachineInfoCard**：ファームウェア情報
   - **HistoryFileCard**：ジョブ一覧とファイルブラウザ
   - **SettingsCard**：JSON インポートなど各種設定
5. **共通 UI**
   - 各カード右上にメニューハンドルと閉じるボタン
   - ドラッグ・スケール操作対応
   - ダーク/ライトテーマ切替可能

---

## ⚙️ デバッグチェックリスト

1. **起動パイプラインの確認**
   - `npm run dev` 実行後、ターミナルに次の表示が出るか確認する。
     ```
     VITE vX.Y.Z  ready in N ms
     ➜  Local: http://localhost:5173/
     ```
2. **`startup.js` の呼び出し**
   ```js
   // src/startup.js
   import { App } from './core/App.js';
   new App('#app-root');
   ```
   が記述されているか。
3. **`App.js` から `DashboardManager` への接続**
   ```js
   export class App {
     constructor(selector) {
       this.root = document.querySelector(selector);
       this.cm = new ConnectionManager(bus);
       this.db = new DashboardManager(bus, this.cm);
       this.db.render();  // TitleBar とカードを描画
     }
   }
   ```
   が正しく呼ばれているか。
4. **ブラウザの Console / Network タブ**
   - JavaScript エラーが出ていないか。
   - `/src/core/App.js` や `/src/cards/Bar_Title.js` が 200 で返るか。
5. **`#app-root` 要素の中身**
   - DevTools の Elements タブで `<div id="app-root">` 内に `<header>` や `<main>` が生成されているか確認。

いずれかが欠けている場合、フロントエンドが正しく初期化されていない可能性があります。上記を一つずつ検証することで、レンダリング停止箇所を特定できます。
