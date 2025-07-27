# マルチモーダル統合修正仕様書

## 現在の実装状況（2025年7月20日更新）

### ✅ 実装完了項目

#### 1. HybridDatasetのデュアルエンコーダー対応
- **状態**: ✅ 完了
- `utils/dataset.py`の`__getitem__`メソッドで既に実装済み
- `pixel_values`: Llama-4用画像データ (448x448) - 592行目
- `sam_pixel_values`: SAM2用画像データ (1024x1024) - 593行目

#### 2. test_phase3b_integration_real.pyのデュアルエンコーダー対応
- **状態**: ✅ 完了
- `prepare_real_test_data`メソッドで両方の画像を準備（609-616行目）
- `adapt_dataset_for_qformer`で適切に変換（1018行目）

#### 3. QFormerSegmentationBridgeのデュアルエンコーダー対応
- **状態**: ✅ 完了
- `forward`メソッドで`sam_images`パラメータをサポート（650行目）
- デュアルエンコーダーモードの判定ロジック実装（670-674行目）
- SAM2処理で適切な画像を選択（917-923行目）

#### 4. dataset_adapter.pyの実装
- **状態**: ✅ 完了
- デュアルエンコーダー構成のマッピング処理実装
- `configure_dual_encoder`関数でconfig設定

## 修正方針

### 1. デュアルエンコーダーアーキテクチャの採用理由
- **Llama-4**: ネイティブマルチモーダル機能を活用した高レベル理解
- **SAM2**: 高精度セグメンテーション専用エンコーダー
- **Q-Former**: 両方の特徴を統合し、タスク関連情報を抽出

### 2. 実装修正点

#### A. HybridDataset修正
```python
# 現在の実装（シングルエンコーダー）
return {
    'sam_pixel_values': sam_image,  # SAM2用のみ
    'input_ids': input_ids,
    'attention_mask': attention_mask,
    'labels': labels
}

# 修正後（デュアルエンコーダー）
return {
    'pixel_values': llama_image,      # Llama-4用画像（追加）
    'sam_pixel_values': sam_image,    # SAM2用画像
    'input_ids': input_ids,
    'attention_mask': attention_mask,
    'labels': labels
}
```

#### B. QFormerSegmentationBridge修正
- Llama-4のネイティブマルチモーダル機能を活用
- Q-Formerで両方のエンコーダーからの特徴を統合
- Early Fusion戦略の完全実装

## 技術的詳細

### 1. 画像処理フロー
1. **入力画像**
   - 元画像を2つの異なる解像度で処理
   
2. **Llama-4用処理** (448x448)
   - `preprocess_llama_image()`関数を使用
   - Llama-4プロセッサーによる正規化
   
3. **SAM2用処理** (1024x1024)
   - `preprocess_sam_image()`関数を使用
   - SAM2準拠の正規化（pixel_mean/pixel_std）

### 2. デュアルエンコーダー統合
- **Early Fusion**: Llama-4で画像とテキストを早期に統合
- **Q-Former Cross-Modal**: Q-Formerで両エンコーダーの特徴を融合
- **SAM Feature Extraction**: SAM2から高品質なビジュアル特徴を抽出

## 期待される効果
1. **精度向上**: 各エンコーダーの専門性を活かした高精度処理
2. **情報の豊富化**: 複数の視点からの特徴抽出
3. **タスク特化**: セグメンテーションタスクに最適化された処理

## 現在の課題と解決策

### 潜在的な問題
1. **設定の不整合**: `use_dual_encoder`フラグが正しく設定されているか確認が必要
2. **データフローの検証**: 実際のトレーニング/推論時にデータが正しく流れているか
3. **メモリ効率**: 両方の画像を処理することによるメモリ使用量の増加

### 推奨事項
1. **デバッグログの追加**: データフローを追跡するためのログを追加
2. **ユニットテストの作成**: 各コンポーネントの動作を個別に検証
3. **メモリプロファイリング**: GPU/CPUメモリ使用量の監視

## 実装状況サマリー
- **HybridDataset**: ✅ デュアルエンコーダー対応完了
- **QFormerSegmentationBridge**: ✅ デュアルエンコーダー対応完了
- **test_phase3b_integration_real.py**: ✅ テストデータ作成対応完了
- **dataset_adapter.py**: ✅ アダプター実装完了

すべての主要コンポーネントでデュアルエンコーダー対応が完了しています。

## 参考資料
- phase3_moe_optimization_strategy.md
- SAM2+MLE論文: "Customize SAM for Multi-Modal Semantic Segmentation with Mixture of LoRA Experts"
- Llama-4公式ドキュメント（ネイティブマルチモーダル機能）