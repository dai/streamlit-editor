# LLMを活用したドキュメントエディタ: DSPy & LangChain統合によるインテリジェントなライティング (OpenRouter/OpenAI/Deepseek/Gemini/Github/Ollama)

**マルチLLM統合によるインテリジェントなライティングアシスタントで、コンテンツ作成と編集を強化します。**

[![Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://doc-editor.streamlit.app)

[![App Screen Recording](https://github.com/user-attachments/assets/9adbdbc6-2138-4d63-8e0e-83dd8cb03d7f)](https://github.com/user-attachments/assets/f37c4dd5-423c-4406-a08d-51f67942ac7b)

DSPyのLLMオーケストレーションとLangChainのドキュメント処理を活用して、前例のない効率でコンテンツを作成、洗練、管理します。技術ライター、コンテンツクリエーター、インテリジェントなドキュメント編集を求める知識労働者に最適です。

## 📚 目次
- [LLMを活用したドキュメントエディタ: DSPy \& LangChain統合によるインテリジェントなライティング (OpenRouter/OpenAI/Deepseek/Gemini/Github/Ollama)](#llmを活用したドキュメントエディタ-dspy--langchain統合によるインテリジェントなライティング-openrouteropenaideepseekgeminigithubollama)
  - [📚 目次](#-目次)
  - [🚀 クイックスタート](#-クイックスタート)
  - [✨ インテリジェントなドキュメントワークフロー](#-インテリジェントなドキュメントワークフロー)
    - [1. コンテンツ作成フェーズ](#1-コンテンツ作成フェーズ)
    - [2. AIコラボレーションフェーズ](#2-aiコラボレーションフェーズ)
    - [3. 最終化と管理](#3-最終化と管理)
  - [⚙️ システムアーキテクチャ](#️-システムアーキテクチャ)
  - [🔧 技術スタック](#-技術スタック)
  - [📄 ライセンス](#-ライセンス)

## 🚀 クイックスタート

ライブデモをすぐに試す:
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://doc-editor.streamlit.app)

1. リポジトリをクローン:
```
git clone https://github.com/clchinkc/streamlit-editor.git
python -m venv venv
source venv/bin/activate  # Unix/MacOS
# .\venv\Scripts\activate  # Windows
```

2. 依存関係をインストール:
```
pip install -r requirements.txt
```

3. Streamlitのシークレットを設定:
```bash
mkdir -p .streamlit
touch .streamlit/secrets.toml
```

以下を`.streamlit/secrets.toml`に追加:
```toml
# APIキー (少なくとも1つ必要)
[openrouter]
OPENROUTER_API_KEY = "your_openrouter_api_key"
OPENROUTER_MODEL = "your_openrouter_model"

[openai]
OPENAI_API_KEY = "your_openai_api_key"

[deepseek]
DEEPSEEK_API_KEY = "your_deepseek_api_key"

[gemini]
GEMINI_API_KEY = "your_gemini_api_key"

[github]
GITHUB_TOKEN = "your_github_token"

[ollama]
OLLAMA_MODEL = "your_ollama_model"
```

4. (Ollamaを使用する場合) Ollamaをセットアップ:

まず、[Ollama](https://ollama.com/download)をインストールします。

次に、指定されたモデルでOllamaサーバーを起動:
```
ollama run your_ollama_model
```

5. アプリケーションを起動:
```
streamlit run streamlit_editor.py
```
アプリはhttp://localhost:8501で実行されます。

## ✨ インテリジェントなドキュメントワークフロー

**統合機能とユーザープロセス**

### 1. コンテンツ作成フェーズ
- **マルチフォーマット編集スイート**
  - ✍️ デュアルモードエディタ (エディタ + マークダウンプレビュー)
  - 📥 ファイル取り込み: ドラッグアンドドロップで`.md`/`.txt`サポート
  - 📤 エクスポートの柔軟性: マークダウンのダウンロードまたはクリップボードコピー
  
- **構造ツール**
  - 🗂️ LangChainを活用したドキュメントチャンク化
  - 📚 セクションレベルの編集

### 2. AIコラボレーションフェーズ
- **コンテキスト対応アシスタンス**
  - 🤖 DSPyを活用したフィードバック提案 (一般的なものまたは参照テキストに特化したもの)
  - 📑 自動セクション要約
  - 🧩 LLM駆動のコンテンツ再生成
  - 📝 AI生成の変更をレビューおよび比較

- **品質管理**
  - 🔍 ソース参照付きのセマンティックフィードバック追跡
  - 📊 リアルタイムフィードバックダッシュボード
  - ✅ 差分ビューでAI提案を受け入れる/拒否する

### 3. 最終化と管理
- **出力最適化**
  - 🧮 バッチ操作処理による一括編集

- **高度なオーケストレーション**
  - 🚦 DSPy管理の提案パイプライン
  - 📜 バージョン履歴追跡
  - 🌐 マルチモーダルプレビュー (生データ + レンダリングビュー)
  - 📄 要約のページネーション

## ⚙️ システムアーキテクチャ

[![](https://mermaid.ink/img/pako:eNqdVV1r2zAU_SvCpWODmGUrpKkfBknsjMEKZW4oDL8o9o0tYusaSW4a2v73XVv-iLvuYdGTPu45urrnSHp2YkzA8Zxdjoc448qwez-SjNrlJbvlQrIVFiVKkEbb-Y0G9TFydtzbcbeiAXtQwoCKnE82IEiEwSEEaMhCo4AXOfU2P_rAB1R7XfIY-tiEG77lGoYlQnIDPcQP74599FbV-dVT7E5hDFrjkMVPLtNVRgF9eC7kfphmYUnpnOa9Bki2PN73gBiLoj53v8LCozZQ9IiwKgquhoR2IgeX56ZbYIFMhRyyD55KVGY4LR5kjjxp59ktJlVuo3sJViglxEagPCk_c91vLx0LVwoPrhJpZqwS-nOtgI6cl1aKU1lGUM0fgS0qg01PM4M1qK_9G5XGUPIKsHvaNgXV7NXo8GEo8BsZRui4Mrb-tCc8GfYoeE3Rlm2QegQqeCpi9h0kKLJEs2cnzFjAEaqSCbJNmXSQN6frlBplh-WRfIfqXcQ_6nGUMfsFO8Jk8Fft38c0dgGrvbWA7iphhyc-CM2R_JuygzAZOxBdfWFdXUIsdlSVGHNUrUM0hUInN-2Rexc30yC4mU60UbgH7-Lq6qrtuweRmMz7Wj5NGgrvYtq0U6YheUs2v14FwfJMskZVy7NeL2erL2fyDL7qyBbT68WZZL1xLJfvL6a-fyZXZ6g2rWkwm6_OpGqfhVbC-Xp5Mz-TqXk0LE8wC2br_ymUM3EKUAUXCX0TzzVr5JgMCnqoPOomXO0jJ5KvFMfpKQnpIjieURVMHIVVmjlk9VzTqGouoS94qnjRz5Zc_kYcxtD49tb-Ss3n9PoHw9cvYQ?type=png)](https://mermaid.live/edit#pako:eNqdVV1r2zAU_SvCpWODmGUrpKkfBknsjMEKZW4oDL8o9o0tYusaSW4a2v73XVv-iLvuYdGTPu45urrnSHp2YkzA8Zxdjoc448qwez-SjNrlJbvlQrIVFiVKkEbb-Y0G9TFydtzbcbeiAXtQwoCKnE82IEiEwSEEaMhCo4AXOfU2P_rAB1R7XfIY-tiEG77lGoYlQnIDPcQP74599FbV-dVT7E5hDFrjkMVPLtNVRgF9eC7kfphmYUnpnOa9Bki2PN73gBiLoj53v8LCozZQ9IiwKgquhoR2IgeX56ZbYIFMhRyyD55KVGY4LR5kjjxp59ktJlVuo3sJViglxEagPCk_c91vLx0LVwoPrhJpZqwS-nOtgI6cl1aKU1lGUM0fgS0qg01PM4M1qK_9G5XGUPIKsHvaNgXV7NXo8GEo8BsZRui4Mrb-tCc8GfYoeE3Rlm2QegQqeCpi9h0kKLJEs2cnzFjAEaqSCbJNmXSQN6frlBplh-WRfIfqXcQ_6nGUMfsFO8Jk8Fft38c0dgGrvbWA7iphhyc-CM2R_JuygzAZOxBdfWFdXUIsdlSVGHNUrUM0hUInN-2Rexc30yC4mU60UbgH7-Lq6qrtuweRmMz7Wj5NGgrvYtq0U6YheUs2v14FwfJMskZVy7NeL2erL2fyDL7qyBbT68WZZL1xLJfvL6a-fyZXZ6g2rWkwm6_OpGqfhVbC-Xp5Mz-TqXk0LE8wC2br_ymUM3EKUAUXCX0TzzVr5JgMCnqoPOomXO0jJ5KvFMfpKQnpIjieURVMHIVVmjlk9VzTqGouoS94qnjRz5Zc_kYcxtD49tb-Ss3n9PoHw9cvYQ)

## 🔧 技術スタック

| コンポーネント       | 技術        | 目的                    |
|----------------|-------------------|----------------------------|
| AIフレームワーク   | DSPy             | LLM操作管理  |
| テキスト処理| LangChain        | ドキュメントチャンク化          |
| UIフレームワーク   | Streamlit        | Webインターフェース              |
| ビジュアライゼーション  | Streamlit Mermaid| ドキュメントフローダイアグラム     |

## 📄 ライセンス

MITライセンス - 詳細は[LICENSE](LICENSE)を参照してください。
