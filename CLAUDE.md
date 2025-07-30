

odasoya@odasounaris-MBP llama4-lisa-gemini-ver-o3-modification % ssh -i ~/.ssh/lambda_cloud_key ubuntu@192.222.54.200 "cd /lambda/nfs/llama4-lisa-project-fs-central-texas/code/LISA-Gemma-Linux && source ../../venvs/lisa_gemma_venv/bin/activate && export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 && export CUDA_LAUNCH_BLOCKING=1 && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && PYTHONUNBUFFERED=1 python -u test_phase3b_enhanced_multiscale_lora.py 2>&1"
✅ Lambda Cloud パス検出: /lambda/nfs/llama4-lisa-project-fs-central-texas
✅ HuggingFace公式BLIP-2 Q-Former利用可能
✅ Meta公式SAM2利用可能 (Python 3.10+)
✅ Llama4ForConditionalGeneration利用可能（2025年正規実装）
✅ Llama4ForConditionalGeneration利用可能

🚀 Enhanced マルチスケール + LoRA テスト開始...

✅ GPU利用可能: NVIDIA H100 80GB HBM3
  - GPU数: 8

🧠 Llama-4初期化...
Loading checkpoint shards: 100%|██████████| 50/50 [00:36<00:00,  1.38it/s]
✅ Llama-4初期化完了 (2025年正規実装)

🚀 Enhanced Model初期化（マルチスケール + LoRA有効）...
⚠️ SAM2モデルのcompileモードを無効化しました

================================================================================
🚀 Enhanced QFormerSegmentationBridge初期化開始
================================================================================

🧠 Llama-4セットアップ...
✅ 共有Llama-4インスタンス使用

🔍 Enhanced Q-Former初期化...
🔧 Enhanced Q-Former初期化中...
  - クエリ数: 32
  - 隠れ層サイズ: 768
  - LLM隠れ層サイズ: 5120
  - エンコーダ隠れ層サイズ: 256 (SAM2互換)
✅ Enhanced Q-Former初期化完了
  - パラメータ数: 98,876,544
✅ Enhanced Q-Former初期化完了
  - テキスト入力対応: 有効
  - 学習モード: ITC/ITM/ITG対応

🔀 視覚・言語分離機構初期化...
✅ モダリティ特殊トークン追加: 4個
✅ DualModalityOutput初期化完了
  - LLM隠れ層: 5120
  - 語彙サイズ: 128256
  - 視覚出力次元: 1024
✅ 視覚・言語分離機構初期化完了

🎯 SAM2マルチスケール初期化...
🔧 SAM2を明示的デバイスで初期化: cuda:0
🔄 Meta公式SAM2を使用: facebook/sam2-hiera-large
  - データ型統一: torch.bfloat16
  - デバッグモード: True
  ✅ TensorFloat-32有効化 (Ampere GPU最適化)
  🔧 SAM2Wrapper設定:
    - target_dtype: torch.bfloat16
    - debug_mode: True
    - vos_optimized: True
    - compile_model: False
    - memory_pathways: 3
    - mixed_precision: True
🔄 Meta公式SAM2初期化中...
  - モデルID: facebook/sam2-hiera-large
  - デバイス: cuda
  - 自動取得: HuggingFace Hub
🔧 SAM2用CUDA初期化...
  ✅ SAM2用CUDA初期化完了: GPU 0/8
  - 2025年最適化: HuggingFace優先・torch.compile対応
✅ 2025年VOS最適化SAM2初期化成功
  - HuggingFaceモデル: facebook/sam2-hiera-large
✅ 重みファイル自動取得完了
✅ MultiScaleFeatureExtractor初期化
  - 抽出ステージ: [3, 6, 9, 12]
✅ AuxiliaryDecoder初期化
  - 入力チャネル: 768
  - 隠れチャネル: 256
✅ EnhancedMaskDecoder初期化
  - メインデコーダー: SAM2 MaskDecoder
  - 補助デコーダー: マルチスケール対応
✅ MultiScaleSegmentationHead初期化完了
✅ マルチスケールセグメンテーションヘッド有効
🔧 高解像度Auxデコーダーモジュール初期化...
✅ 高解像度Auxデコーダーモジュール初期化完了

🔧 LoRAエキスパート初期化...
🔍 LoRA注入デバッグ:
  - enable_multiscale: True
  - segmentation_head type: <class 'model.multiscale_decoder.MultiScaleSegmentationHead'>
  - segmentation_head has image_encoder: True
  - LoRA注入対象: MultiScaleSegmentationHead.image_encoder
  - target_encoder type: <class 'sam2.modeling.backbones.image_encoder.ImageEncoder'>
🔍 target_encoder内のLinearモジュール調査:
  - 発見されたLinearモジュール数: 195
  - 最初の5個: ['trunk.blocks.0.attn.qkv', 'trunk.blocks.0.attn.proj', 'trunk.blocks.0.mlp.layers.0', 'trunk.blocks.0.mlp.layers.1', 'trunk.blocks.1.attn.qkv']
🔍 config SAM2_TARGET_MODULES: ['trunk.blocks.*.attn.qkv', 'trunk.blocks.*.attn.proj', 'trunk.blocks.*.mlp.layers.0', 'trunk.blocks.*.mlp.layers.1', 'blocks.*.attn.qkv', 'blocks.*.attn.proj', 'blocks.*.mlp.layers.0', 'blocks.*.mlp.layers.1', 'attn.qkv', 'attn.proj', 'mlp.layers.0', 'mlp.layers.1']
  - マッチしたモジュール数: 192個（例: ['trunk.blocks.0.attn.qkv', 'trunk.blocks.1.attn.qkv', 'trunk.blocks.2.attn.qkv']...）
🔧 LoRA注入: target_modules修正
  - 修正前: ['trunk.blocks.*.attn.qkv', 'trunk.blocks.*.attn.proj', 'trunk.blocks.*.mlp.layers.0', 'trunk.blocks.*.mlp.layers.1', 'blocks.*.attn.qkv', 'blocks.*.attn.proj', 'blocks.*.mlp.layers.0', 'blocks.*.mlp.layers.1', 'attn.qkv', 'attn.proj', 'mlp.layers.0', 'mlp.layers.1']
  - 修正後: ['trunk.blocks.*.attn.qkv', 'trunk.blocks.*.attn.proj', 'trunk.blocks.*.mlp.layers.0', 'trunk.blocks.*.mlp.layers.1', 'blocks.*.attn.qkv', 'blocks.*.attn.proj', 'blocks.*.mlp.layers.0', 'blocks.*.mlp.layers.1', 'attn.qkv', 'attn.proj', 'mlp.layers.0', 'mlp.layers.1']
🔍 LoRA注入開始デバッグ:
  - 対象モデル: ImageEncoder
  - ターゲットモジュール: ['trunk.blocks.*.attn.qkv', 'trunk.blocks.*.attn.proj', 'trunk.blocks.*.mlp.layers.0', 'trunk.blocks.*.mlp.layers.1', 'blocks.*.attn.qkv', 'blocks.*.attn.proj', 'blocks.*.mlp.layers.0', 'blocks.*.mlp.layers.1', 'attn.qkv', 'attn.proj', 'mlp.layers.0', 'mlp.layers.1']
  - ランク: 16, アルファ: 32
  - MoE使用: True, エキスパート数: 2
  - GPU メモリ (注入前): Allocated=25.07GB, Reserved=25.08GB
  - 全Linearモジュール数: 195
  - LoRA注入対象数: 192
  🔧 LoRA注入開始: 192個のモジュール
  🔧 LoRA注入進行: 0% (1/192)
  🔧 LoRA注入進行: 25% (49/192)
  🔧 LoRA注入進行: 50% (97/192)
  🔧 LoRA注入進行: 75% (145/192)
  🔧 LoRA注入進行: 100% (192/192)
  - GPU メモリ (注入後): Allocated=25.25GB, Reserved=25.26GB
✅ LoRA注入完了: 192/192個のモジュール
✅ LoRAエキスパート注入完了
  - エキスパート数: 2
  - ランク: 16

📊 損失関数初期化...
✅ Stage 1 複合損失初期化完了
  - Focal Tversky (最高性能): 1.0
  - Lovász-Softmax (IoU): 0.5
  - Q-Former (マルチモーダル): 0.5
  - SAM2プロンプト: 0.0
✅ 損失関数初期化完了 (Stage 1)

🔧 デバイス配置確認中...
主要デバイス確認: 直接アクセス (Model Parallelism: 主要デバイス cuda:0)
📦 全コンポーネントを主要デバイスに配置...
📦 SAM2深層コンポーネント統一配置...
✅ SAM2深層コンポーネント統一完了
✅ CUDA default device設定: cuda:0 (index: 0)
🔍 全コンポーネントデバイス確認...
✅ 全コンポーネントデバイス確認完了: 基準デバイス cuda:0
✅ デバイス配置統一完了: cuda:0

✅ Enhanced QFormerSegmentationBridge初期化完了
  - テキスト入力: 対応
  - 視覚・言語分離: 有効
  - LoRA適応: 有効
  - マルチスケール: 有効
  - 訓練ステージ: 1
✅ Enhanced Model初期化完了

🔧 LoRA統計（Web調査準拠PEFT標準）:
  - LoRAモジュール数: 384
  - LoRAパラメータ数: 13,976,064
  - LoRAパラメータ詳細（上位5個）:
    - segmentation_head.image_encoder.trunk.blocks.0.attn.qkv.experts.0.lora_A: [16, 144] (2,304個)
    - segmentation_head.image_encoder.trunk.blocks.0.attn.qkv.experts.0.lora_B: [432, 16] (6,912個)
    - segmentation_head.image_encoder.trunk.blocks.0.attn.qkv.experts.1.lora_A: [16, 144] (2,304個)
    - segmentation_head.image_encoder.trunk.blocks.0.attn.qkv.experts.1.lora_B: [432, 16] (6,912個)
    - segmentation_head.image_encoder.trunk.blocks.0.attn.proj.experts.0.lora_A: [16, 144] (2,304個)

📦 テストデータ作成...
  - Llama-4画像形状: torch.Size([1, 3, 448, 448])
  - SAM2画像形状: torch.Size([1, 3, 1024, 1024])
  - テキスト: ['Segment the main object in this image']
  - ラベル形状: torch.Size([1, 1024, 1024])
  - GT正例ピクセル数: 70681.0

🧪 単一スケールForward処理テスト...
🔍 forward入力時 labels型: type=<class 'torch.Tensor'>, shape=torch.Size([1, 1024, 1024])
🔍 Llama-4: 標準画像形式 torch.Size([1, 3, 448, 448])
✅ Llama-4: 標準形式使用 torch.Size([1, 3, 448, 448])
🔍 SAM2マルチスケール: 高解像度画像準備 torch.Size([1, 3, 1024, 1024])
🔧 マルチモーダル統合デバイス: cuda:0
  ✅ 入力テンソル0既に統一済み: cuda:0
  ✅ 入力テンソル1既に統一済み: cuda:0

🚀 Llama-4ネイティブマルチモーダル処理開始
✅ dataset.py処理済みpixel_values使用完了

🎯 SAM2マルチスケール特徴抽出開始
🔍 SAM2デバイス状況詳細確認:
⚠️ SAM2 predictor.modelにアクセスできません
🔍 o3準拠マルチスケール特徴抽出開始
🔍 SAM2データ型デバッグ:
  - sam_input_bfloat16: torch.bfloat16, device=cuda:0
  - image_encoder device: cuda:0
  - image_encoder dtype: torch.bfloat16
  - bias trunk.patch_embed.proj.bias: torch.bfloat16
🔍 SAM2 FPN neck経由の特徴抽出を実行...
  🔍 LoRAExpertMoE形状サマリー: 入力=torch.Size([1024, 8, 8, 144]), 出力=torch.Size([2, 1024, 8, 8, 432]), 重み=torch.Size([1024, 8, 8, 2])
  ⚠️ 想定外routing_weights形状検出: torch.Size([1024, 8, 8, 2]) (後続の同様警告は省略)
✅ FPN特徴抽出成功:
  - FPN特徴数: 4
  - メイン特徴: torch.Size([1, 256, 256, 256])
  - 位置エンコーディング数: 4
✅ o3マルチスケール成功:
  - feat_high: None
  - feat_mid: None
  - feat_global: torch.Size([1, 256, 256, 256])

🔍 BLIP-2準拠Q-Former統合処理開始
🔄 Q-Formerデバイス状況:
  - 画像特徴: device=cuda:0, dtype=torch.bfloat16
  - Q-Former: device=cuda:0, dtype=torch.bfloat16
  ✅ Q-Formerデバイス統一済み: cuda:0
✅ BLIP-2準拠Q-Former処理完了: mode=itg
🔍 勾配フロー追跡: Q-Former出力後
  🔍 [qformer_llm_embeds] shape=torch.Size([1, 32, 5120]), dtype=torch.bfloat16, device=cuda:0
      requires_grad=True, grad_fn=<ViewBackward0 object at 0x7610f4bcd5a0>
      ✅ 勾配フロー正常: qformer_llm_embeds
  🔍 [qformer_query_embeds] shape=torch.Size([1, 32, 768]), dtype=torch.bfloat16, device=cuda:0
      requires_grad=True, grad_fn=<SliceBackward0 object at 0x7610f4bcd5a0>
      ✅ 勾配フロー正常: qformer_query_embeds
  🔍 [qformer_sam_prompts] shape=torch.Size([1, 32, 256]), dtype=torch.bfloat16, device=cuda:0
      requires_grad=True, grad_fn=<ViewBackward0 object at 0x7610f4bcd5a0>
      ✅ 勾配フロー正常: qformer_sam_prompts

🧠 Llama-4統合マルチモーダル処理開始
🔍 Q-Former視覚埋め込み: torch.Size([1, 32, 5120])
🔍 Q-Former視覚埋め込み準備: torch.Size([1, 32, 5120])
✅ Llama-4統合入力準備完了: torch.Size([1, 44, 5120])
✅ Llama-4マルチモーダル処理完了

🔀 視覚・言語特徴分離機構実行
✅ 特徴分離完了: visual=torch.Size([1, 32, 5120]), text=None

🎯 o3準拠マルチスケールセグメンテーション開始
🎯 SAM2バッチ処理開始: batch_size=1
🔍 勾配フロー追跡: SAM2入力前
  🔍 [sam_input_images] shape=torch.Size([1, 3, 1024, 1024]), dtype=torch.bfloat16, device=cuda:0
      requires_grad=True, grad_fn=None
      ⚠️  勾配フロー途切れ: sam_input_images (requires_grad=True, grad_fn=None)
  🔍 [sam_prompts] shape=torch.Size([1, 32, 256]), dtype=torch.bfloat16, device=cuda:0
      requires_grad=True, grad_fn=<ViewBackward0 object at 0x7610f4a88100>
      ✅ 勾配フロー正常: sam_prompts
🔧 SAM2処理モード: multiscale
🔍 SAM2ゼロマスク原因調査:
  - sam_input_images統計: min=-3.328125, max=3.328125, mean=-0.000207
  - sam_prompts統計: min=-0.925781, max=0.972656, mean=-0.030151
  - SAM2モデル訓練モード（修正前）: False
  - SAM2 image_encoder訓練モード（修正前）: False
  - SAM2 mask_decoder訓練モード（修正前）: False
🔧 SAM2訓練モード強制有効化中...
  🔧 SAM2 image_encoder.forwardのno_grad無効化完了
  🔧 SAM2 mask_decoder.forwardのno_grad無効化完了
🔧 SAM2全体BFloat16統一実行中...
  ✅ SAM2モデル全体BFloat16変換完了
  ✅ SAM2モデル訓練モード（修正後）: True
  ✅ SAM2 image_encoder訓練モード（修正後）: True
  ✅ SAM2 mask_decoder訓練モード（修正後）: True
  📊 バッチ 1/1 処理中...
    🔧 根本修正: バッチ次元保持 shape=torch.Size([1, 3, 1024, 1024])
  🔧 SAM2 tensor-only学習モード: numpy変換完全回避
    🔍 image_tensor形状: torch.Size([1, 3, 1024, 1024])
    🔧 SAM2 image_encoder出力タイプ: <class 'dict'>
    ✅ メイン埋め込み取得 (vision_features): torch.Size([1, 256, 64, 64]), torch.float32
    🔧 dtype統一処理: target_dtype=torch.bfloat16
      encoded_features: torch.float32 -> torch.bfloat16
    🔍 mask_decoder入力確認:
      - encoded_features: torch.Size([1, 256, 64, 64]), torch.bfloat16
      - batch_prompts: torch.Size([32, 256]), torch.bfloat16
      - sparse_embeddings: torch.Size([1, 32, 256])
      - dense_embeddings: torch.Size([1, 256, 64, 64])
      - image_pe: torch.Size([1, 256, 64, 64]), torch.bfloat16
    🔧 Web調査準拠high_res_features構築開始
    🔧 ハイブリッド特徴統合開始: Q-Former(256ch) + SAM2 backbone直接特徴抽出
        ✅ 戦略2成功: feat_s0=torch.Size([1, 32, 256, 256]), feat_s1=torch.Size([1, 64, 128, 128])
    ✅ SAM2 mask_decoder成功: masks=torch.Size([1, 3, 256, 256]), iou=torch.Size([1, 3])
    ✅ SAM2 tensor-only出力:
      - masks: torch.Size([1, 3, 256, 256]), torch.bfloat16, device: cuda:0
      - iou_predictions: torch.Size([1, 3]), torch.bfloat16
      - 勾配フロー維持: True
    SAM2出力統計: masks: torch.Size([1, 3, 256, 256]), torch.bfloat16, device: cuda:0
    iou_predictions: torch.Size([3]), 平均IoU: 0.000
      🔍 Visual context入力形状:
        - base_masks: torch.Size([1, 3, 256, 256])
        - visual_context: torch.Size([1, 32, 5120])
      ✅ 形状正規化完了: masks=torch.Size([3, 256, 256]), context=torch.Size([1, 32, 5120])
📊 マスク統合: 1 samples
    Batch 0: torch.Size([3, 256, 256])
🔧 SAM2出力テンソルrequires_grad=True設定完了
✅ 統合後のマスク形状: torch.Size([1, 3, 256, 256])
✅ 統合後のIoU形状: torch.Size([1, 3])

❌ 単一スケールエラー: local variable 'outputs' referenced before assignment
Traceback (most recent call last):
  File "/lambda/nfs/llama4-lisa-project-fs-central-texas/code/LISA-Gemma-Linux/test_phase3b_enhanced_multiscale_lora.py", line 214, in test_enhanced_multiscale_lora
    outputs = model(
  File "/lambda/nfs/llama4-lisa-project-fs-central-texas/venvs/lisa_gemma_venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/lambda/nfs/llama4-lisa-project-fs-central-texas/venvs/lisa_gemma_venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
  File "/lambda/nfs/llama4-lisa-project-fs-central-texas/code/LISA-Gemma-Linux/model/enhanced_llama4_qformer_sam2.py", line 1606, in forward
    outputs['masks'] = masks
UnboundLocalError: local variable 'outputs' referenced before assignment
odasoya@odasounaris-MBP llama4-lisa-gemini-ver-o3-modification % 
