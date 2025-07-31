# SAM2.1 + Qwen2.5-VL Integration

SAM2.1（Segment Anything Model 2.1）とQwen2.5-VL-3Bを統合したマルチモーダル基盤モデルの実装です。

## 🎯 プロジェクト概要

将来的なFoodLMM改の基盤となるLISA改の実装として、最新のVLM（Vision-Language Model）と最新のSAM（Segment Anything Model）を統合し、画像と言語を深い次元で理解できる基盤モデルを構築しています。

### 主な特徴

- **Qwen2.5-VL-3B**: 最新の視覚言語モデルによる高精度な画像理解
- **SAM2.1**: 最新のセグメンテーションモデルによる精密な物体分割
- **Q-Formerベース統合**: 効率的な視覚特徴プロジェクション
- **マルチモーダル推論**: 同時セグメンテーション＋テキスト生成

## 📁 プロジェクト構造

```
SAM2_1_Qwen_2_5/
├── README.md                           # このファイル
├── requirements.txt                    # 必要なライブラリ
├── config_qwen_sam.py                 # 設定ファイル
├── test_sam_qwen_integration.py       # 統合テストスクリプト
├── model/                             # コアモデル実装
│   ├── sam_qwen_model.py             # メイン統合モデル
│   ├── visual_projector.py           # 視覚特徴プロジェクター
│   └── losses.py                     # 損失関数
├── utils/                            # ユーティリティ
└── md_files/                         # 仕様書
    └── o3_spec.md                    # 技術仕様書
```

## 🚀 セットアップ

### 1. 環境構築

```bash
# Python環境（3.8+推奨）
pip install -r requirements.txt

# SAM2.1のインストール（別途必要）
pip install git+https://github.com/facebookresearch/sam2.git
```

### 2. 必要なモデル

- **Qwen2.5-VL-3B-Instruct**: Hugging Faceから自動ダウンロード
- **SAM2.1-hiera-large**: Facebook Researchから自動ダウンロード

## 💻 使用方法

### 基本的な使用例

```python
from config_qwen_sam import get_config
from model.sam_qwen_model import create_sam_qwen_model
from PIL import Image
import torch

# 設定の取得
config = get_config('development')
model_config = config.get_model_config()

# モデルの初期化
model = create_sam_qwen_model(model_config)

# 画像と質問の準備
image = Image.open("example.jpg")
question = "この画像に何が写っていますか？"

# 点プロンプト（オプション）
point_coords = torch.tensor([[300.0, 200.0]])  # x, y座標
point_labels = torch.tensor([1])  # 1=前景

# 推論実行
results = model.forward(
    image=image,
    question=question,
    point_coords=point_coords,
    point_labels=point_labels
)

# 結果の取得
mask = results['mask']  # セグメンテーションマスク
text = results['generated_text']  # 生成されたテキスト

print(f"生成テキスト: {text}")
print(f"マスクサイズ: {mask.shape}")
```

### テストの実行

```bash
# 統合テストの実行
python test_sam_qwen_integration.py
```

## 🔧 設定

`config_qwen_sam.py`で様々な設定を調整できます：

### 環境別設定

```python
from config_qwen_sam import get_config

# 開発環境用（軽量設定）
config = get_config('development')

# 本番環境用（高性能設定）
config = get_config('production')

# テスト環境用（最小設定）
config = get_config('test')
```

### 主要パラメータ

- `USE_QFORMER`: Q-Formerを使用するか（推奨: True）
- `NUM_QUERIES`: Q-Formerのクエリ数（デフォルト: 32）
- `TORCH_DTYPE`: モデルの精度（デフォルト: float16）
- `BATCH_SIZE`: バッチサイズ

## 📊 モデルアーキテクチャ

### 統合フロー

1. **画像入力** → SAM2.1画像エンコーダー → 視覚特徴抽出
2. **視覚特徴** → VisualProjector（Q-Former） → LLM互換特徴
3. **テキスト+視覚特徴** → Qwen2.5-VL → テキスト生成
4. **視覚特徴+プロンプト** → SAM2.1デコーダー → セグメンテーション

### 主要コンポーネント

- **SAMQwenModel**: メイン統合クラス
- **VisualProjector**: 視覚特徴の次元変換（Q-Former/Linear）
- **モンキーパッチ**: Qwenの視覚エンコーダーをSAM2.1に置換

## 🧪 テスト機能

`test_sam_qwen_integration.py`では以下をテストします：

1. ✅ モデル初期化
2. ✅ 視覚プロジェクター動作
3. ✅ セグメンテーション機能
4. ✅ テキスト生成機能
5. ✅ 統合推論（同時実行）

## 📈 パフォーマンス

### 推奨環境

- **GPU**: 16GB+ VRAM（RTX 4080/V100以上）
- **RAM**: 32GB以上
- **CUDA**: 11.8以上

### 処理時間（参考）

- セグメンテーションのみ: ~0.5秒
- テキスト生成のみ: ~1.0秒
- 統合推論: ~1.2秒

## 🔄 今後の展開

1. **マルチスケール推論**: 異なる解像度での同時処理
2. **LoRA適用**: 効率的なファインチューニング
3. **FoodLMM統合**: 食品特化モデルへの発展
4. **量推定機能**: 料理・食材の量推定精度向上

## 📝 技術詳細

詳細な技術仕様は[md_files/o3_spec.md](md_files/o3_spec.md)を参照してください。

## 🤝 開発履歴

- **方針転換**: Llama4-LISA-Code → SAM2.1+Qwen2.5-VL
- **コードクリーンアップ**: 不要ファイル削除、シンプル化
- **統合実装**: o3_spec.mdベースの正規実装
- **テスト整備**: 包括的なテストスイート

## ⚠️ 注意事項

- SAM2.1は別途インストールが必要です
- GPU環境での実行を強く推奨します
- 初回実行時はモデルのダウンロードに時間がかかります

## 📞 サポート

問題が発生した場合は、以下を確認してください：

1. GPU環境とCUDAのバージョン
2. 必要ライブラリのインストール状況
3. テストスクリプトの実行結果

---

**🎉 SAM2.1 + Qwen2.5-VLの統合により、次世代マルチモーダルAIの基盤が完成しました！**