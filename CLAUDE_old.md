# CLAUDE.md

日本語で応答すること！
This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

オリジナルLISA（オリジナルレポジトリ：./Original-LISA-Code/）のVLMをLlavaからgemma-3-4b-it（2025年6月時点でのGoogleのVLMの最新モデル）に変更して、学習するプロジェクト（./Gemma_LISA-Code/）を実装し、train.pyまで完了していたが、そのプロジェクトからさらにVLMをLlama-4-Scout-17B-16E-Instructに変更して学習するように修正している。

This repository contains LISA-Gemma-Linux, a computer vision project that implements Large Language Instructed Segmentation Assistant (LISA) with multiple model backends including Gemma-3 and Llama-4. The project focuses on multimodal learning combining vision and language for image segmentation tasks.

## Development Rules & Guidelines

### Lambda Cloud Development Workflow
Lambda Cloud環境での実行を行うので、ローカルファイルの修正を行うたびに、以下のコマンド例を参考に、lambda上に転送し、lambda上で実行すること

```bash
# File transfer to Lambda Cloud
rsync -avz --progress --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' --exclude='lambda_results' --exclude='verification_output' --exclude='vis_output' --exclude='.gitignore' -e "ssh -i ~/.ssh/lambda_cloud_key" ./ ubuntu@<ip address>:/lambda/nfs/lisa-gemma-project-fs/code/LISA-Gemma-Linux/

# Remote execution
ssh -i ~/.ssh/lambda_cloud_key ubuntu@<ip address> "cd /lambda/nfs/lisa-gemma-project-fs/code/LISA-Gemma-Linux && source ../../venvs/lisa_gemma_venv/bin/activate && CUDA_VISIBLE_DEVICES=0,1, ... PYTHONUNBUFFERED=1 python -u <script>.py 2>&1"

or 

ssh -i ~/.ssh/lambda_cloud_key ubuntu@<ip address> "cd /lambda/nfs/llama4-lisa-project-fs-central-texas/code/LISA-Gemma-Linux && source ../../venvs/lisa_gemma_venv/bin/activate && CUDA_VISIBLE_DEVICES=0,1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 python -u <script>.py 2>&1"
```

### Core Development Principles

1. **Reference Existing Implementations**: 最も重要な方針として、本プロジェクトは既存の実際に実行できたオリジナルLISAとGemma-LISAにおいて、VLMをLlama-4-Scout-17B-16E-Instructに変更したプロジェクトである。原則としてGemma-LISAプロジェクト（./Gemma_LISA-Code/）とオリジナルLISA（./Original-LISA-Code/）を参考に進めて

※現状以下のテストスクリプトが走ることが確認されているので、積極的に参照すること

A. データセット初期化や一部前処理
・verify_dataset_integrity.py
・verify_input_formatting.py

B. モデル関連
・verify_loss_and_gradients.py
・overfit_llama4_lisa_batch.py

C. トレーニング関連
・train_llama4_lisa_single_process.py

※公式で推奨されてる方法とかはWebになさそう？もしなければカスタムコードでいいが、公式が準備してくれているメソッドなどがあればそちらを使うこと！

2. **Web Research for Official Methods (2025年最新情報ベース)**: 
   - Llama-4-Scout-17B-16E-Instructの使い方については、積極的にWebリサーチを行い、overfit_llama4_lisa_batch.pyの成功パターンをベースに、PEFT device_map preservationなどの既知の問題対策を積極的に参照して実装すること
   - DeepSpeed ZeROとPEFTの組み合わせについては、LlamaFactoryやNVIDIA NeMoの成功事例を参考にすること
   - SAMの使い方については、積極的にWebリサーチを行い、公式や非公式の実際に動くコードを積極的に参照して、カスタムコードでなく用意されている方法でシンプルに実装すること

3. **Error Handling Strategy**: フォールバック的なコードはエラーを隠蔽するので、エラーを出して止め、一つ一つデバッグするように実装すること

4. **Reference Integration Models**: 統合モデル(llama4_lisa.pyやその参照元ファイル)についてはgemma_lisa.pyやOriginal-LISA-Code（オリジナルのLISAでLlavaとSAMの統合モデルプロジェクト）を参考にして

### Code Quality Standards

- Always run tests before making significant changes
- Use the unified configuration system in `config_linux.py`
- Follow existing code style and naming conventions
- Add detailed docstrings for complex functions
- Never commit sensitive information (API keys, tokens)

## Implementation Status & Roadmap

### 🏆 **Phase 1: Model Parallelism Baseline (COMPLETED)**
- ✅ llama4_lisa.pyでLlama-4-Scout-17B-16E-InstructとSAMの統合モデルの実装完了
- ✅ Web調査ベースのPEFT device_map preservation実装完了
- ✅ overfit_llama4_lisa_batch.pyで109Bモデル(17B active)の学習能力検証成功
- ✅ Model Parallelismでの安定学習ベースライン確立

### 🔄 **Phase 2: DeepSpeed ZeRO-2 Integration (NEXT)**
- 🛠️ NCCL/TCP "Broken pipe"エラーの根本解決
- 🛠️ DeepSpeed ZeRO-2 + device_map preservationの組み合わせ
- 🛠️ 中規模データセットでの分散学習検証

### 🎯 **Phase 3: Large-Scale Distributed Training (FUTURE)**
- 🕰️ DeepSpeed ZeRO-3 + LoRAで大規模分散学習
- 🕰️ NVIDIA NeMo統合 or LlamaFactory採用検討
- 🕰️ 最終プロダクションシステム確立

## Architecture Components

### Model Implementations
- **LISA-Gemma3**: Integration with Google's Gemma-3-4B-IT model using dual-stream data pipeline
- **LISA-Llama4**: Integration with Meta's Llama-4-Scout-17B-16E-Instruct (109B total, 17B active per token) 
- **Segment Anything Model (SAM)**: SAM ViT-H for segmentation tasks
- **Dual-stream Architecture**: Separate image processing pipelines for Llama4 (336x336) and SAM (1024x1024)

### Core Components
- **MLP Projector**: Bridges between language model hidden states and SAM prompt embeddings
- **LoRA Configuration**: Parameter-efficient fine-tuning with device_map preservation (Web研究ベース)
- **HybridDataset**: Unified dataset handling for multiple data sources (ReasonSeg, VQA, ReferSeg, SemSeg)
- **Model Parallelism**: HuggingFace device_map="auto" for 109B model across 8 A100 GPUs
- **PEFT Device Map Preservation**: Web調査に基づくget_peft_model後のdevice_map復元ロジック

## Common Development Commands

### Training Commands

#### **Phase 1: Model Parallelism (Current - Stable)**
```bash
# Quick validation test
python launch_simple_training.py --exp_name quick_test --steps_per_epoch 10 --epochs 1

# Small-scale training (recommended for current phase)
python launch_simple_training.py --exp_name lisa_small_scale --epochs 5

# Direct single process execution
python train_llama4_lisa_single_process.py --exp_name lisa_mp --batch_size 1 --epochs 3
```

#### **Phase 2: DeepSpeed ZeRO-2 (Development)**
```bash
# Medium-scale distributed training (when Phase 2 ready)
python launch_training.py --exp_name lisa_zero2 --zero_stage 2 --batch_size 1
```

#### **Phase 3: DeepSpeed ZeRO-3 (Future)**
```bash
# Large-scale distributed training (future implementation)
python train_llama4_deepspeed.py --deepspeed_config ds_config_llama4_zero3.json
```

### Testing & Verification Commands

#### **Core Verification (Always run before training)**
```bash
# Basic model functionality
python test_llama4_standalone.py          # Basic model test
python test_llama4_lisa_standalone.py     # Full LISA integration test

# Learning capability verification (CRITICAL - use this as baseline)
python overfit_llama4_lisa_batch.py       # ✅ PROVEN: Model Parallelism + device_map preservation

# Gradient flow verification
python verify_llama4_lisa_gradients.py    # Gradient flow verification
```

#### **Dataset & Environment Verification**
```bash
# Dataset verification
python verify_dataset_integrity.py        # Dataset health check
python verify_model_architecture.py       # Model architecture check

# Environment verification
python verify_config_and_setup.py         # Full environment check
```

#### **Performance Benchmarking**
```bash
# Inference pipeline test
python test_llama4_inference_pipeline.py  # Inference pipeline test
```

### Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Check configuration
python config_linux.py                    # Verify paths and settings

# Environment verification
python verify_config_and_setup.py         # Full environment check

# Lambda Cloud setup (A100 80GB × 8GPU)
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_DEVICE_ORDER=PCI_BUS_ID
```

## Web Research Insights (2025年最新)

### **Llama-4-Scout-17B-16E-Instruct の特性**
- **総パラメータ**: 109B (16 experts MoE)
- **アクティブパラメータ**: 17B per token
- **コンテキスト長**: 10M tokens
- **推奨量子化**: INT4 (single H100 可能)

### **知られている課題**
- H100 80GB でもLoRA学習時にOOM発生報告あり
- Multi-GPU LoRA学習でPEFTライブラリにバグあり
- Model Parallelismのみでは大規模データ学習は非現実的

### **成功報告のある手法**
- **LlamaFactory + DeepSpeed ZeRO-3**: 8 × L20 48G GPUs
- **NVIDIA NeMo**: 公式Llama-4 LoRA/PEFT サポート
- **Flash Attention + Gradient Checkpointing**: 必須の最適化

### **推奨される次期実装**
1. **DeepSpeed ZeRO-2復活** + device_map preservation
2. **段階的スケールアップ**: 小→中→大規模データ
3. **NVIDIA NeMo統合検討**: プロダクション環境向け



### Model-Specific Settings
- **Llama-4**: Uses `meta-llama/Llama-4-Scout-17B-16E-Instruct` with 5120 hidden size
- **SAM**: Uses ViT-H checkpoint with 256 prompt embedding dimension

