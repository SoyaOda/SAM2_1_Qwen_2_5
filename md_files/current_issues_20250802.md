# SAM2.1 + Qwen2.5-VL統合モデル - 現状の問題点まとめ
*作成日: 2025年8月2日*

## 📋 概要

SAM2.1とQwen2.5-VLを統合したモデルの実装において、以下の問題が発生しています。
本ドキュメントは、`option_A_spec.md`と`o3_spec6.md`の方針に基づく実装での課題を整理します。

## 🔴 主要な問題点

### 1. 次元の不一致問題

#### 1.1 SAM2.1の出力次元
- **仕様書（o3_spec6.md）の記載**: 1280次元
- **実際の実装**: 256次元（FPNによる統一化）
- **発生箇所**: `model/sam_qwen_model.py` L498

```python
# 現在の実装
sam_feat_dim = 256  # FPN d_model (sam2_hiera_l.yaml)

# o3_spec6.mdの期待
# SAM ViT出力（1280次元）→ 線形層（1280→3584）→ LLMデコーダ
```

#### 1.2 Qwen2.5-VLのhidden_size
- **仕様書（o3_spec6.md）の記載**: 3584次元
- **実際（3Bモデル）**: 2048次元
- **発生箇所**: `model/sam_qwen_model.py` L494

```python
# 実際の値
qwen_hidden_size = self.qwen_model.config.hidden_size  # 2048 (3Bモデル)
```

### 2. SAM mask_decoderのサイズ不一致

#### 2.1 エラー内容
```
[❗ SAM mask_decoder呼び出しエラー: The size of tensor a (14) must match the size of tensor b (64) at non-singleton dimension 3
```

#### 2.2 原因
- SAM2.1の`vision_features`: 14×14×256
- mask_decoderが期待する`image_pe`: 64×64×256
- **発生箇所**: `model/sam_qwen_model.py` L766-790

#### 2.3 現在の対処
- 簡易実装へのフォールバック（本来のSAM2.1 APIを使用せず）

### 3. seg_hidden値の範囲問題

#### 3.1 症状
- seg_hidden（<SEG>トークンの隠れ状態）の値が非常に大きい
- 例: `mean=0.016907, std=3.914062, min=-54.781250, max=49.718750`
- **発生箇所**: `model/sam_qwen_model.py` L653-668

#### 3.2 影響
- 初期エポックではNaNが発生していた（現在は緩和）
- 学習の不安定性の原因

### 4. 学習効果の低さ

#### 4.1 症状
- 5エポック後の改善率: わずか1.0%
- Loss: 1.0806 → 1.0700
- IoU/Dice: 0.062/0.116（変化なし）

#### 4.2 原因の可能性
- SAMとQwenが「並行動作」している（統合されていない）
- option_A_spec.mdの「QwenビジョンエンコーダをSAM ViTで完全置換」が未実現

## 🟡 部分的に解決された問題

### 1. NaN問題
- **以前**: 2番目のサンプルでNaN発生
- **現在**: 5エポック完走可能
- **対策済み**:
  - 学習率を1e-5に削減
  - 重み初期化の改善（Kaiming初期化 + スケーリング）
  - seg_hiddenのクリッピング（-10〜10）

### 2. <SEG>トークン位置検出
- **以前**: 常に最後のトークンを使用
- **現在**: 実際の<SEG>トークン位置を検索
- **実装箇所**: `model/sam_qwen_model.py` L638-650

## 🔧 技術的詳細

### 1. モデルアーキテクチャの現状

```
[現在の実装]
画像 → SAM2.1 encoder → 256次元特徴
     ↓
画像 → Qwen Vision → Qwen LLM → テキスト生成
                            ↓
                        <SEG>トークン → SAM decoder → マスク

[option_A_spec.mdが要求する実装]
画像 → SAM2.1 encoder → 1280次元特徴 → 投影層 → Qwen LLM → テキスト生成
                                                    ↓
                                              <SEG>トークン → SAM decoder → マスク
```

### 2. 実際の次元（調査結果）

| コンポーネント | 仕様書記載 | 実際の値 |
|------------|---------|--------|
| SAM2.1 Hiera backbone | 1280次元 | 不明（FPN前） |
| SAM2.1 FPN出力 | - | 256次元 |
| Qwen2.5-VL-3B hidden | 3584次元 | 2048次元 |
| Qwen Vision hidden | - | 1280次元 |

### 3. デバッグ出力例

```
[DEBUG] backbone_out keys: dict_keys(['vision_features', 'vision_pos_enc', 'backbone_fpn'])
[DEBUG] vision_features shape: torch.Size([1, 256, 14, 14])
[DEBUG] backbone_fpn length: 3
[DEBUG] backbone_fpn[0] shape: torch.Size([1, 256, 56, 56])
[DEBUG] backbone_fpn[1] shape: torch.Size([1, 256, 28, 28])
[DEBUG] backbone_fpn[2] shape: torch.Size([1, 256, 14, 14])
[DEBUG] Using vision_features with shape torch.Size([1, 196, 256])
[DEBUG] Found <SEG> token at position 227 (after 196 vision tokens)
[DEBUG] seg_hidden initial stats: mean=0.016907, std=3.914062, min=-54.781250, max=49.718750
```

## 📝 必要な調査事項

1. **SAM2.1 Hiera-Lの実際の出力次元**
   - FPN前のbackbone出力は本当に1280次元なのか？
   - 256次元への変換はどこで行われているのか？

2. **Qwen2.5-VLの次元マッピング**
   - 3Bモデル（2048次元）と7B/72Bモデル（3584次元）の違い
   - o3_spec6.mdは7B/72Bモデルを前提としている可能性

3. **SAM2.1 mask_decoderの正しい使用方法**
   - 14×14の特徴マップをどのように64×64にアップサンプリングするか
   - 公式実装でのベストプラクティス

## 🚀 推奨される次のステップ

1. **次元の確認と修正**
   - SAM2.1 Hieraの実際の出力次元を確認
   - モデルサイズに応じた適切な投影層の設計

2. **option_A_spec.mdの完全実装**
   - QwenビジョンエンコーダをSAM ViTで完全置換
   - 並行動作から統合動作への移行

3. **SAM2.1 mask_decoder APIの正しい実装**
   - サイズ不一致の解決
   - 簡易実装からの脱却

## 📁 関連ファイル

- 仕様書: `md_files/current_spec/option_A_spec.md`, `md_files/current_spec/o3_spec6.md`
- 実装: `model/sam_qwen_model.py`
- テスト: `train_minimal_lora.py`, `debug_nan_issue.py`
- 調査: `check_model_dimensions.py`