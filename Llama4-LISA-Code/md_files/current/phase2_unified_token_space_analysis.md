# Phase 2: 統一トークン空間実装分析
## 📅 作成日: 2025年1月20日

## 🔍 調査結果サマリー

### 現在の実装状況
1. **Phase 1完了**: 軽量[SEG]トークン実装済み（28.00%性能維持）
2. **アーキテクチャ**: Llama-4-Scout-17B + Q-Former + SAM2
3. **トークン次元**:
   - Llama-4: 5120次元
   - Q-Former: 768次元  
   - SAM2: 256次元

### Webリサーチ結果

#### Sa2VA (2025)の統一トークン空間
- **コアコンセプト**: LLMの柔軟なトークン長処理を活用し、全モダリティを追加設計なしでビジュアルトークンとして扱う
- **実装方式**: テキスト・画像・動画トークンを連結し、attention層とMoE expert層で共同処理
- **[SEG]トークン**: LLMの隠れ状態をSAM2の空間-時間プロンプトとして使用（2層線形変換）

#### Llama-4のネイティブマルチモーダル設計
- **Early Fusion**: 全モダリティのトークンを最初から統一配列として処理
- **クロスモーダルアテンション**: 最初のself-attention層から、テキストトークンが画像パッチトークンに直接attend可能
- **MoE統合**: 各トークンが適切なexpertを見つける（128 routed experts + 1 shared expert）

#### LLaVAのアプローチ比較
- **フル実装**: 画像あたり~500トークン、全てをLLM全体で処理
- **部分実装（LLaVA-Mini）**: 
  - モダリティ事前融合で視覚情報をテキストトークンに事前統合
  - 視覚トークンを1トークンまで圧縮可能
  - FLOPs 77%削減、レイテンシ40ms以下

## 💡 実装方針の結論

### 推奨: ハイブリッドアプローチ（部分統一 + 動的拡張）

#### 理由
1. **現実的な実装**:
   - 現在のQ-Formerベースアーキテクチャと親和性が高い
   - メモリ効率的（2x H100環境に適合）
   - 段階的な実装・検証が可能

2. **性能とのバランス**:
   - LLaVA-Miniの実証結果: 1トークンでも576トークンのLLaVA-v1.5を上回る
   - 必要に応じて動的にトークン数を拡張可能
   - Phase 3Bの28%改善を維持しながら追加改善

3. **技術的優位性**:
   - Q-Formerが既にクロスモーダル情報を32クエリに圧縮
   - Sa2VAの[SEG]トークン機構と組み合わせることで効率的な統一空間を実現
   - Llama-4のMoE構造を活用した専門的処理

## 📐 Phase 2実装設計

### アーキテクチャ概要
```
入力画像 → Q-Former(32クエリ) → 圧縮統一トークン(1-8) → Llama-4 MoE
     ↓                              ↓
SAM2画像 → [SEG]トークン生成 → 動的SAM制御
```

### 主要コンポーネント

#### 1. 適応的トークン圧縮器
```python
class AdaptiveTokenCompressor(nn.Module):
    """
    Q-Formerクエリを1-8個の統一トークンに適応的に圧縮
    タスク複雑度に応じて動的にトークン数を調整
    """
    def __init__(self, config):
        # 学習可能な圧縮クエリ（最大8個）
        self.compression_queries = nn.Parameter(
            torch.randn(8, config.qformer_dim)
        )
        # 複雑度予測器
        self.complexity_predictor = nn.Linear(
            config.qformer_dim, 1
        )
```

#### 2. クロスモーダル統一層
```python
class CrossModalUnifier(nn.Module):
    """
    視覚・テキスト・[SEG]トークンを統一空間で処理
    Llama-4のearly fusionアプローチを採用
    """
    def __init__(self, config):
        # モダリティ別投影層
        self.projectors = nn.ModuleDict({
            'vision': nn.Linear(768, 5120),  # Q-Former → Llama
            'seg': nn.Identity(),  # 既にLlama次元
            'text': nn.Identity()  # 既にLlama次元
        })
        # 2D正弦波位置エンコーディング（空間情報保持）
        self.spatial_pos_encoder = SinusoidalPositionalEncoding2D()
```

#### 3. MoE統合アダプター
```python
class MoEIntegrationAdapter(nn.Module):
    """
    統一トークンをLlama-4のMoE層に効率的にルーティング
    視覚トークン専用のexpertを活用
    """
    def __init__(self, config):
        # トークンタイプ埋め込み
        self.token_type_embeddings = nn.Embedding(3, 5120)
        # ルーティング最適化層
        self.routing_optimizer = nn.Linear(5120, 128)
```

### 実装の利点

1. **メモリ効率**:
   - 最悪ケースでも8トークン（従来の500トークンから大幅削減）
   - GPU RAM使用量を最小限に抑制

2. **性能維持**:
   - 事前融合により視覚情報をテキストトークンに統合
   - 動的トークン数調整で複雑なタスクにも対応

3. **拡張性**:
   - 将来的なフル統一への移行パス確保
   - 新しいモダリティの追加が容易

## 📊 期待される成果

### パフォーマンス予測
- **基本性能**: Phase 1の28%改善を維持
- **追加改善**: 10-12%（統一空間による効率化）
- **推論速度**: 30-50%高速化（トークン削減効果）

### リスク評価
- **技術リスク**: 中（LLaVA-Miniで実証済みアプローチ）
- **実装リスク**: 低（段階的実装可能）
- **性能リスク**: 低（動的拡張でフォールバック可能）

## 🚀 実装計画

### Week 1: コア実装
1. AdaptiveTokenCompressorの実装
2. CrossModalUnifierの実装  
3. 基本的な統合テスト

### Week 2: 最適化と評価
1. MoEIntegrationAdapterの実装
2. 動的トークン数調整の最適化
3. 性能評価とチューニング

## 📝 結論

**部分統一空間の実装を推奨**します。理由：

1. **実証済みの効果**: LLaVA-Miniが1トークンでフル実装を上回る
2. **現実的な実装**: 現在のアーキテクチャと高い親和性
3. **柔軟性**: 必要に応じて動的にトークン数を拡張可能
4. **将来性**: フル統一への移行パスを確保

フル統一は理論的には理想的ですが、現在の2x H100環境では非現実的であり、部分統一で十分な効果が期待できます。