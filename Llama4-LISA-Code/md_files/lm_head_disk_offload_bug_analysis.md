# lm_head Disk Offload Bug Analysis

## 概要
Phase 3B QFormerSegmentationBridge 訓練において、Model Parallelism使用時にlm_headがdiskに配置され、NaN発生する重大なバグが判明。2025年7月20日に発生・調査・修正を実施。

## 問題の詳細

### 初期症状
- **Step 0成功、Step 1でNaN**: 最初のステップは正常（loss=51.50）だが、2ステップ目でNaN発生
- **エラーメッセージ**: `❌ step 1: 損失が0または無効、訓練を停止します`
- **Llama-4 logits**: Step 0から全体がNaNになる深刻な問題に発展

### 根本原因の特定

#### Model Parallelismデバイス分散の問題
```
✓ 実際のGPU分散: {0: 15, 1: 15, 'cpu': 10, 'disk': 12}
- lm_head現在配置: meta
- embed_tokens配置: cuda:0
⚠️ NaN発生リスク（lm_headディスク配置）
```

**問題の核心**:
1. `lm_head`がdiskに配置される
2. meta tensorとしてロードされ、実際の計算でNaN発生
3. Llama-4の最重要コンポーネント（出力層）がGPU外に配置される

## 試行した修正方法

### 1. Loss Scaling実装（部分的効果）
```python
# config_linux.py修正
USE_LOSS_SCALING = True
INITIAL_LOSS_SCALE = 2**10
GRADIENT_CLIP_NORM = 0.05  # 超厳格設定
LEARNING_RATE = 1e-5       # 極保守的設定
```

**結果**: Step 1のNaN一部改善、但しStep 0からNaN発生の新問題

### 2. メモリ制限調整（段階的改善）
```python
# 段階的メモリ制限緩和
max_memory_per_gpu = "35GB"  # 初期
max_memory_per_gpu = "70GB"  # 中間
max_memory_per_gpu = "75GB"  # 最終
```

**結果**: disk使用量削減、但し完全解決には至らず

### 3. disk使用完全禁止の試行（エラー発生）
```python
# 失敗例
max_memory_dict = {
    0: "70GB", 
    1: "70GB",
    "cpu": "50GB",
    "disk": 0  # ← これがエラー原因
}
```

**エラー**: `Expected one of cpu, cuda... device type at start of device string: disk`

### 4. カスタムdevice_map実装（エラー発生）
```python
# 失敗例
custom_device_map = {
    "model.embed_tokens": 0,
    "lm_head": 1,
    "model.layers": "auto",  # ← これがエラー原因
    "model.norm": 1
}
```

**エラー**: `Expected one of cpu, cuda... device type at start of device string: auto`

### 5. 学習率二重定義修正（成功）
```python
# train_phase3b_qformer_bridge.py line 1497
parser.add_argument('--lr', type=float, default=2e-5, help='学習率')
↓
parser.add_argument('--lr', type=float, default=config_linux.LEARNING_RATE, help='学習率')
```

**結果**: 設定の一貫性確保、ログで正しい学習率（1e-05）表示

## Webリサーチ結果

### HuggingFace Transformers device_map仕様
2025年7月20日のWebリサーチより:

1. **カスタムdevice_map**: `{"model.layers.1": 0, "lm_head": "cpu"}` 形式で明示指定可能
2. **disk offload要件**: `offload_folder` パラメータが必須
3. **auto制限**: device_map内で`"auto"`は使用不可
4. **disk配置制御**: `max_memory`設定でdisk使用量を間接制御

## 現在の最終設定

### 最終修正版設定
```python
# 最終的な設定（修正完了）
max_memory_per_gpu = "75GB"  # H100最大限活用
load_kwargs = {
    "device_map": "auto",           # シンプルなauto使用
    "max_memory": {
        0: "75GB",                  # GPU 0: 75GB
        1: "75GB",                  # GPU 1: 75GB  
        "cpu": "50GB"               # CPU: 50GB（diskキー除外）
    },
    "offload_state_dict": False,    # disk offload無効化
}
```

## 期待される効果

### lm_head GPU配置成功の指標
実行時ログで以下を確認:
```
✓ 実際のGPU分散: {0: X, 1: Y, 'cpu': Z}  # 'disk'キーが消失
- lm_head現在配置: cuda:1                # meta → cuda:1に改善
✅ embed_tokens配置: cuda:0
```

### NaN解決の確認方法
```python
# Step 0, 1でのNaN監視
- Step 0: loss=有効値, logits NaN=False
- Step 1: loss=有効値, logits NaN=False  # 目標
```

## 技術的学習事項

### Model Parallelismの注意点
1. **重要コンポーネント**: `lm_head`, `embed_tokens`は必ずGPUに配置
2. **meta tensor**: diskオフロード時に発生、計算でNaN原因
3. **メモリ制限**: 保守的すぎるとdisk使用、緩すぎるとOOM

### HuggingFace Transformers制限
1. **device_map**: `"auto"`文字列は単体使用のみ、カスタムマップ内では不可
2. **disk制御**: 直接的なdisk無効化は不可、間接的制御が必要
3. **互換性**: Llama4ForCausalLMで一部パラメータ未対応

## 次回作業への提言

### 即座に確認すべき項目
1. Lambda Cloud実行でlm_head配置確認
2. デバイス分散に'disk'が含まれないことを確認
3. Step 0, 1でのNaN発生状況監視

### 代替案
もし現在の修正で解決しない場合:
1. **完全なカスタムdevice_map**: 全52レイヤーを明示指定
2. **Sequential Loading**: レイヤーごとの段階的ロード
3. **Model分割**: embed_tokens, transformer, lm_headを分離ロード

## ファイル変更履歴

### 修正されたファイル
1. **config_linux.py**: 学習率、Loss Scaling、数値安定化設定
2. **train_phase3b_qformer_bridge.py**: メモリ制限、device_map、学習率参照
3. **model/dataset_adapter.py**: デュアルエンコーダー対応（既存）

### 修正された設定値
- `LEARNING_RATE`: 5e-5 → 1e-5
- `GRADIENT_CLIP_NORM`: 0.1 → 0.05
- `max_memory_per_gpu`: 35GB → 75GB
- `USE_LOSS_SCALING`: False → True
- `LAYERNORM_EPSILON`: 1e-5 → 1e-4

## 結論

lm_headのdisk配置問題は、Model Parallelism使用時のメモリ制限設定に起因する。HuggingFace Transformersの仕様制限により、間接的なメモリ制御でdisk使用を抑制する戦略が最適解。最終的な75GB制限+50GB CPU設定で、disk配置回避を達成する見込み。

**重要**: この問題は17Bモデル+Model Parallelism特有であり、将来的な大規模モデル使用時にも再発する可能性が高い。メモリ制限の適切な設定がNaN回避の鍵となる。