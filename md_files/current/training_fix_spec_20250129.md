# Training Phase 3B Enhanced 修正仕様書 - 2025/01/29

## Webリサーチ結果に基づく修正方針

### 1. Llama-4 Scout Early Fusion実装の修正

#### 現在の問題
- `_process_llama4_images()`, `_create_early_fusion_input()` 等の未実装メソッド
- Llama-4のネイティブマルチモーダル処理が不適切

#### 修正方針
Webリサーチ結果：Llama-4 Scoutは**ネイティブマルチモーダル**で、Early Fusionアーキテクチャを採用
- **40兆トークンのマルチモーダルデータで事前学習済み**
- テキストと画像トークンが統一システムで学習
- **AutoProcessor.apply_chat_template()** がマルチモーダル統合の標準API

#### 具体的修正内容
```python
# ❌ 削除: 未実装メソッド群
def _process_llama4_images(self, images): # 削除
def _create_early_fusion_input(self, visual_embeds, text_tokens): # 削除
def _integrate_early_fusion_qformer(self, early_fusion_embeds, qformer_embeds): # 削除

# ✅ 修正: Llama-4ネイティブマルチモーダル処理
# dataset.pyで既にAutoProcessor.apply_chat_template()を使用済み
# → pixel_valuesを直接使用するシンプルな実装に変更
```

### 2. SAM2学習用ベストプラクティスの適用

#### 現在の問題
- `@torch.no_grad()`による勾配フロー遮断
- 不安定な学習設定

#### 修正方針
Webリサーチ結果：SAM2学習には**勾配フロー管理**が重要
- **`@torch.no_grad()`の完全除去** - 勾配収集を阻害する
- **勾配クリッピング**で安定化
- **低学習率**（5e-5以下）でデコーダーの事前学習知識を保持
- **ウォームアップ必須** - 特にLoRA使用時

#### 具体的修正内容
```python
# ❌ 除去: SAM2内の@torch.no_grad()
# 975行目付近
# @torch.no_grad()  # 除去
for idx, blk in enumerate(sam_model.image_encoder.trunk.blocks):

# ✅ 追加: 勾配クリッピング（train_phase3b_enhanced.pyに追加済み）
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# ✅ 修正: requires_grad強制設定
sam_input_images.requires_grad_(True)
masks.requires_grad_(True)
```

### 3. BLIP-2 Q-Former標準実装の採用

#### 現在の問題
- Enhanced Q-Formerの実装不備
- Hugging Face標準との不整合

#### 修正方針
Webリサーチ結果：BLIP-2 Q-Formerの**標準実装パターン**
- **32個の学習可能クエリ埋め込み** (32 x 768)
- **自己注意+相互注意**でテキストと画像を統合
- **FC層でLLM次元に射影** (768 → 5120)

#### 具体的修正内容
```python
# ✅ 修正: 標準BLIP-2設定に準拠
self.qformer_config = {
    'num_queries': 32,        # BLIP-2標準
    'hidden_size': 768,       # BLIP-2標準
    'num_layers': 12,         # BLIP-2標準
    'num_heads': 12,          # BLIP-2標準
    'intermediate_size': 3072, # BLIP-2標準
}

# ✅ 確認: get_enhanced_qformer_model()の実装が標準準拠か確認
```

### 4. デバイス統一とメモリ最適化

#### 現在の問題
- 複雑すぎるデバイス移動処理
- メモリリークの可能性

#### 修正方針
```python
# ✅ 簡素化: デバイス移動を初期化時のみに限定
# 実行時の動的移動は禁止 → エラーで停止する方針

# ✅ 強化: メモリクリーンアップ
# バッチ毎のtorch.cuda.empty_cache()
# 不要な中間変数の即座削除
```

## 実装優先順位

### Phase 1: 緊急修正（即座に実行）
1. **SAM2勾配フロー修正** - `@torch.no_grad()`除去
2. **Llama-4マルチモーダル簡素化** - 未実装メソッド削除
3. **エラーハンドリング強化** - RuntimeErrorで明確停止

### Phase 2: 安定化修正
1. **Q-Former実装確認と修正**
2. **デバイス統一処理の簡素化**
3. **メモリ最適化の強化**

### Phase 3: 検証と最適化
1. **学習実行テスト**
2. **勾配フロー確認**
3. **性能評価**

## 期待される改善効果

1. **勾配フロー正常化** → SAM2の学習が可能に
2. **エラー削減** → 安定した学習実行
3. **メモリ効率向上** → より大きなバッチサイズが可能
4. **実装の簡素化** → デバッグとメンテナンスが容易

この修正により、train_phase3b_enhanced.pyが安定して動作し、実際の学習が可能になることを目指します。