"""
Gemma3-LISA フルデータセット学習スクリプト (DeepSpeed対応版)

このスクリプトはGemma3とSAMを組み合わせたLISAモデルを、
すべてのトレーニングデータセットを使用してDeepSpeedで学習するためのものです。
テストレベルの学習が完了した後、本格的な学習を行うために使用します。

主な特徴:
- すべてのデータセット (sem_seg, refer_seg, vqa, reason_seg) を使用
- LoRAを使用したパラメータ効率の良い学習
- DeepSpeedによる分散学習のサポート
- 勾配チェックポイントによるメモリ効率の向上
"""

import argparse
import os
import shutil
import sys
import time
from functools import partial
import logging
import pickle
import json

try:
    import deepspeed
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False

import numpy as np
import torch
import tqdm
import transformers
from peft import LoraConfig, get_peft_model
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    AutoProcessor, 
    AutoTokenizer, 
    TrainingArguments, 
    default_data_collator,
    set_seed
)

from model import LISAForCausalLM
from utils.dataset import HybridDataset, ValDataset, collate_fn
from utils.utils import (
    DEFAULT_IM_END_TOKEN, 
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    AverageMeter, 
    ProgressMeter, 
    Summary, 
    dict_to_cuda,
    intersectionAndUnionGPU,
    IMAGE_TOKEN_INDEX
)

from torchvision.transforms import Compose, ToTensor, RandomResizedCrop, Normalize, InterpolationMode

# 初期化メッセージとログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# GPU/CPU環境の確認
if torch.cuda.is_available():
    device_count = torch.cuda.device_count()
    device_names = [torch.cuda.get_device_name(i) for i in range(device_count)]
    logger.info(f"=== GPU環境で実行: 利用可能なGPU {device_count}台 ===")
    for i, name in enumerate(device_names):
        logger.info(f"GPU {i}: {name}")
    logger.info(f"CUDA バージョン: {torch.version.cuda}")
else:
    logger.info("=== CPU環境で実行: GPUが利用できません ===")
    logger.info(f"PyTorch バージョン: {torch.__version__}")

def parse_args(args):
    parser = argparse.ArgumentParser(description="LISA-Gemma3 Model Training (Full Dataset)")
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")
    parser.add_argument(
        "--version", default="google/gemma-3-4b-it", type=str,
        help="Gemma3モデルのパス"
    )
    parser.add_argument("--vis_save_path", default="./vis_output", type=str)
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument(
        "--vision_tower", default="sigLIP", type=str,
        help="Gemma3用のビジョンエンコーダ (CLIP or sigLIP)"
    )

    parser.add_argument(
        "--dataset", default="sem_seg||refer_seg||vqa||reason_seg", type=str
    )
    parser.add_argument("--sample_rates", default="9,3,3,1", type=str)
    parser.add_argument(
        "--sem_seg_data",
        default="ade20k||cocostuff||mapillary",
        type=str,
    )
    parser.add_argument(
        "--refer_seg_data", default="refcoco||refcoco+||refcocog", type=str
    )
    parser.add_argument("--vqa_data", default="llava_instruct_150k", type=str)
    parser.add_argument("--reason_seg_data", default="reason_seg/ReasonSeg", type=str)
    parser.add_argument("--val_dataset", default="ReasonSeg|val", type=str)
    parser.add_argument("--dataset_dir", default="H:\\download\\LISA-dataset\\dataset", type=str)
    parser.add_argument("--log_base_dir", default="./runs", type=str)
    parser.add_argument("--exp_name", default="lisa-gemma3-full", type=str)
    # フルデータセット用にエポック数を調整
    parser.add_argument("--epochs", default=3, type=int)
    # フルデータセット用にステップ数を増加
    parser.add_argument("--steps_per_epoch", default=1000, type=int)
    parser.add_argument(
        "--batch_size", default=2, type=int, help="batch size per device per step"
    )
    # 勾配蓄積ステップ数を調整
    parser.add_argument(
        "--grad_accumulation_steps",
        default=8,
        type=int,
    )
    parser.add_argument("--val_batch_size", default=1, type=int)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--lr", default=0.0003, type=float)
    parser.add_argument("--ce_loss_weight", default=1.0, type=float)
    parser.add_argument("--dice_loss_weight", default=0.5, type=float)
    parser.add_argument("--bce_loss_weight", default=2.0, type=float)
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--lora_target_modules", default="q_proj,k_proj,v_proj", type=str)
    parser.add_argument("--explanatory", default=0.1, type=float)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.95, type=float)
    parser.add_argument("--num_classes_per_sample", default=3, type=int)
    parser.add_argument("--exclude_val", action="store_true", default=False)
    parser.add_argument("--no_eval", action="store_true", default=False)
    parser.add_argument("--eval_only", action="store_true", default=False)
    parser.add_argument("--vision_pretrained", default="/mnt/c/Users/oda/foodlmm-llama/weights/sam_vit_h_4b8939.pth", type=str)
    parser.add_argument("--out_dim", default=256, type=int)
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--print_freq", default=1, type=int)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--train_mask_decoder", action="store_true", default=True)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--auto_resume", action="store_true", default=True)
    parser.add_argument(
        "--conv_type",
        default="gemma_v1",
        type=str,
        choices=["gemma_v1", "llava_v1", "llava_llama_2"],
    )
    parser.add_argument("--deepspeed_config", default="ds_config.json", type=str)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--debug_samples", type=int, default=None)
    parser.add_argument("--debug_dataset", type=str, default="sem_seg")
    parser.add_argument("--use_cached_data", action="store_true", default=False,
                        help="キャッシュされたデータセットを使用（存在する場合）")
    parser.add_argument("--save_cached_data", action="store_true", default=False,
                        help="処理済みデータセットをキャッシュとして保存")
    
    # カスタムパスオプション（パスのトラブルシューティング向け）
    parser.add_argument("--custom_dataset_paths", action="store_true", help="カスタムデータセットパスを使用する")
    parser.add_argument("--sem_seg_base_dir", type=str, default="sem_seg", help="セマンティックセグメンテーションデータセットのベースディレクトリ")
    parser.add_argument("--refer_seg_base_dir", type=str, default="refer_seg", help="参照セグメンテーションデータセットのベースディレクトリ")
    parser.add_argument("--vqa_base_dir", type=str, default="llava_dataset", help="VQAデータセットのベースディレクトリ")
    parser.add_argument("--reason_seg_base_dir", type=str, default="reason_seg", help="理由付きセグメンテーションデータセットのベースディレクトリ")
    
    return parser.parse_args(args)

def initialize_model_and_tokenizer(args):
    """
    モデルとトークナイザーの初期化
    
    Args:
        args: コマンドライン引数
        
    Returns:
        model: 初期化されたモデル
        tokenizer: 初期化されたトークナイザー
    """
    # トークナイザーの初期化
    logger.info(f"トークナイザーを初期化: {args.version}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=1024,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.unk_token
    num_added_tokens = tokenizer.add_tokens("[SEG]")
    seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    logger.info(f"[SEG]トークンのインデックス: {seg_token_idx}")

    # Gemma3の画像トークンを追加
    gemma_image_token = "<start_of_image>"
    num_added = tokenizer.add_tokens([gemma_image_token], special_tokens=True)
    if num_added > 0:
        gemma_image_token_id = tokenizer.convert_tokens_to_ids(gemma_image_token)
        logger.info(f"Gemma3画像トークン '{gemma_image_token}' を追加しました。ID: {gemma_image_token_id}")
    else:
        logger.info(f"Gemma3画像トークン '{gemma_image_token}' は既に存在しています")
    
    # 標準の画像トークンも追加
    if args.use_mm_start_end:
        tokenizer.add_tokens(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IMAGE_TOKEN], 
            special_tokens=True
        )
        logger.info("標準画像トークンを追加しました")

    # モデル引数の設定
    model_args = {
        "train_mask_decoder": args.train_mask_decoder,
        "out_dim": args.out_dim,
        "ce_loss_weight": args.ce_loss_weight,
        "dice_loss_weight": args.dice_loss_weight,
        "bce_loss_weight": args.bce_loss_weight,
        "seg_token_idx": seg_token_idx,
        "vision_pretrained": args.vision_pretrained,
        "use_mm_start_end": args.use_mm_start_end,
    }
    
    # 精度設定
    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half
    
    logger.info("モデルを初期化しています...")
    
    # 量子化設定
    if args.load_in_8bit or args.load_in_4bit:
        from transformers import BitsAndBytesConfig
        
        if args.load_in_8bit:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        model_args["quantization_config"] = quantization_config
    
    # モデル読み込み
    try:
        model = LISAForCausalLM.from_pretrained(
            args.version, 
            torch_dtype=torch_dtype, 
            low_cpu_mem_usage=True, 
            **model_args
        )
        
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.bos_token_id = tokenizer.bos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        
        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable()
            
        # トークナイザーでモデルの埋め込み層をリサイズ
        model.resize_token_embeddings(len(tokenizer))
        
        # モデルのサイズを確認
        model_size = sum(p.numel() for p in model.parameters())
        logger.info(f"モデルのパラメータ数: {model_size:,}")
        
        # モデルのデバイス情報を表示
        device_info = next(model.parameters()).device
        logger.info(f"現在のモデルデバイス: {device_info}")
        
        # LISA モジュールを初期化
        logger.info("LISA モジュールを初期化しています...")
        model.initialize_lisa_modules(args)
        
        if not args.eval_only:
            # LoRAの設定
            lora_r = args.lora_r
            if lora_r > 0:
                logger.info(f"LoRAを適用しています (r={lora_r}, alpha={args.lora_alpha})...")
                
                def find_linear_layers(model, lora_target_modules):
                    cls = torch.nn.Linear
                    lora_module_names = set()
                    for name, module in model.named_modules():
                        if (
                            isinstance(module, cls)
                            and all(
                                [
                                    x not in name
                                    for x in [
                                        "visual_model",
                                        "vision_tower",
                                        "mm_projector",
                                        "text_hidden_fcs",
                                    ]
                                ]
                            )
                            and any([x in name for x in lora_target_modules])
                        ):
                            lora_module_names.add(name)
                    return sorted(list(lora_module_names))

                lora_target_modules = find_linear_layers(model, args.lora_target_modules.split(","))
                logger.info(f"LoRA適用レイヤー: {lora_target_modules}")
                
                peft_config = LoraConfig(
                    r=args.lora_r,
                    lora_alpha=args.lora_alpha,
                    target_modules=lora_target_modules,
                    lora_dropout=args.lora_dropout,
                    bias="none",
                    task_type="CAUSAL_LM",
                )
                model = get_peft_model(model, peft_config)
                model.print_trainable_parameters()
                
    except Exception as e:
        logger.exception(f"モデルの初期化中にエラーが発生しました: {e}")
        raise e
    
    return model, tokenizer

# 画像トークンのインデックスを取得する関数
def get_image_token_index(tokenizer):
    """
    トークナイザーから画像トークンのインデックスを取得します
    
    Args:
        tokenizer: トークナイザー
        
    Returns:
        int: 画像トークンのインデックス
    """
    # Gemma3の画像トークンを優先
    try:
        gemma_image_token_id = tokenizer.convert_tokens_to_ids("<start_of_image>")
        if gemma_image_token_id is not None and gemma_image_token_id != tokenizer.unk_token_id:
            logger.info(f"Gemma画像トークン '<start_of_image>' のIDを使用: {gemma_image_token_id}")
            return gemma_image_token_id
    except:
        pass
        
    # 標準の画像トークンを試す
    try:
        image_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)
        if image_token_id is not None and image_token_id != tokenizer.unk_token_id:
            logger.info(f"標準画像トークン '{DEFAULT_IMAGE_TOKEN}' のIDを使用: {image_token_id}")
            return image_token_id
    except:
        pass
    
    # デフォルト値を返す
    logger.info(f"画像トークンIDが見つからないため、デフォルト値を使用: {IMAGE_TOKEN_INDEX}")
    return IMAGE_TOKEN_INDEX

def parse_dataset_string(dataset_str):
    """
    データセット文字列をパースしてリストに変換
    
    Args:
        dataset_str: データセット文字列（カンマまたはパイプで区切られた文字列）
        
    Returns:
        dataset_list: データセット名のリスト
    """
    if "||" in dataset_str:
        return dataset_str.split("||")
    elif "," in dataset_str:
        return dataset_str.split(",")
    else:
        return [dataset_str]  # 単一データセットの場合

def main(args):
    """
    メイン関数
    
    Args:
        args: コマンドライン引数
    """
    # 引数の解析
    args = parse_args(args)
    args.log_dir = os.path.join(args.log_base_dir, args.exp_name)
    
    # メインプロセスで実行している場合
    if args.local_rank == 0:
        os.makedirs(args.log_dir, exist_ok=True)
        writer = SummaryWriter(args.log_dir)
    else:
        writer = None

    # シード値を設定して再現性を確保
    set_seed(42)
    
    # デバイス情報の詳細表示
    logger.info("=== トレーニング環境詳細 ===")
    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        logger.info(f"現在のアクティブデバイス: GPU {current_device} ({torch.cuda.get_device_name(current_device)})")
        logger.info(f"GPU メモリ使用量: {torch.cuda.memory_allocated(current_device)/1024**3:.2f} GB (割り当て済み)")
        logger.info(f"GPU メモリ予約量: {torch.cuda.memory_reserved(current_device)/1024**3:.2f} GB (予約済み)")
        logger.info(f"CUDA バージョン: {torch.version.cuda}")
        
        # 利用可能な最大メモリも表示（可能な場合）
        try:
            gpu_properties = torch.cuda.get_device_properties(current_device)
            total_memory = gpu_properties.total_memory / 1024**3
            logger.info(f"GPU 総メモリ: {total_memory:.2f} GB")
        except:
            pass
    else:
        logger.info("CPU環境での実行: GPU機能は利用できません")
        # CPUスレッド数など、可能であれば表示
        import multiprocessing
        logger.info(f"CPU コア数: {multiprocessing.cpu_count()}")
    logger.info("==========================")

    try:
        # モデルとトークナイザーの初期化
        logger.info("モデルとトークナイザーを初期化中...")
        model, tokenizer = initialize_model_and_tokenizer(args)

        # モデルを勾配チェックポインティングモードに設定
        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable()
            logger.info("勾配チェックポイントが有効化されました")

        # データセットの初期化
        logger.info("データセットをロード中...")
        world_size = torch.cuda.device_count()
        args.distributed = world_size > 1

        # キャッシュからロードするか確認
        dataset_cache_path = None
        if args.use_cached_data:
            cache_dir = os.path.join("./dataset_cache")
            os.makedirs(cache_dir, exist_ok=True)
            
            # キャッシュファイル名を生成
            cache_filename = f"dataset_cache_{args.dataset.replace('||', '_').replace(',', '_')}.pkl"
            if args.debug:
                cache_filename = f"debug_{args.debug_dataset}_{args.debug_samples}_{cache_filename}"
            
            dataset_cache_path = os.path.join(cache_dir, cache_filename)
            
            # キャッシュが存在する場合はロード
            if os.path.exists(dataset_cache_path):
                logger.info(f"キャッシュからデータセットをロード中: {dataset_cache_path}")
                try:
                    with open(dataset_cache_path, 'rb') as f:
                        train_dataset = pickle.load(f)
                    logger.info(f"キャッシュからデータセットをロードしました: {len(train_dataset)} サンプル")
                except Exception as e:
                    logger.warning(f"キャッシュからのロード中にエラーが発生しました: {e}")
                    train_dataset = None
            else:
                train_dataset = None
        else:
            train_dataset = None
        
        # データセットがまだロードされていない場合は作成
        if train_dataset is None:
            logger.info("データセットを初期化中...")
            
            # サンプルレートの解析
            sample_rates = None
            if args.sample_rates:
                sample_rates = [float(x) for x in parse_dataset_string(args.sample_rates)]
            
            # データセットの初期化
            train_dataset = HybridDataset(
                args.dataset_dir,
                tokenizer,
                args.vision_tower,
                samples_per_epoch=args.batch_size 
                * args.grad_accumulation_steps 
                * args.steps_per_epoch 
                * world_size,
                precision=args.precision,
                image_size=args.image_size,
                num_classes_per_sample=args.num_classes_per_sample,
                exclude_val=args.exclude_val,
                dataset=args.dataset,
                sample_rate=sample_rates,
                sem_seg_data=args.sem_seg_data,
                refer_seg_data=args.refer_seg_data,
                vqa_data=args.vqa_data,
                reason_seg_data=args.reason_seg_data,
                explanatory=args.explanatory,
                custom_paths=args.custom_dataset_paths,
                sem_seg_base_dir=args.sem_seg_base_dir if args.custom_dataset_paths else "sem_seg",
                refer_seg_base_dir=args.refer_seg_base_dir if args.custom_dataset_paths else "refer_seg",
                vqa_base_dir=getattr(args, "vqa_base_dir", "llava_dataset") if args.custom_dataset_paths else "llava_dataset",
                reason_seg_base_dir=args.reason_seg_base_dir if args.custom_dataset_paths else "reason_seg",
                debug=args.debug,
                debug_samples=args.debug_samples if args.debug else None,
            )
            logger.info(f"トレーニングデータセットのサイズ: {len(train_dataset)}")
            
            # キャッシュに保存
            if args.save_cached_data and dataset_cache_path:
                logger.info(f"データセットをキャッシュに保存中: {dataset_cache_path}")
                try:
                    with open(dataset_cache_path, 'wb') as f:
                        pickle.dump(train_dataset, f)
                    logger.info("データセットをキャッシュに保存しました")
                except Exception as e:
                    logger.warning(f"キャッシュへの保存中にエラーが発生しました: {e}")
        
        # 検証データセットの作成（指定がある場合）
        val_dataset = None
        if args.val_dataset and not args.no_eval:
            logger.info(f"検証データセットの初期化: {args.val_dataset}")
            val_dataset = ValDataset(
                args.dataset_dir, 
                tokenizer, 
                args.vision_tower, 
                args.val_dataset, 
                args.image_size
            )
            logger.info(f"検証データセットのサイズ: {len(val_dataset)}")
            
        # DeepSpeedの設定を読み込み
        if os.path.exists(args.deepspeed_config):
            with open(args.deepspeed_config, 'r') as f:
                ds_config = json.load(f)
                
            # コマンドライン引数で指定された値で上書き
            ds_config["train_micro_batch_size_per_gpu"] = args.batch_size
            ds_config["gradient_accumulation_steps"] = args.grad_accumulation_steps
            ds_config["optimizer"]["params"]["lr"] = args.lr
            ds_config["optimizer"]["params"]["betas"] = [args.beta1, args.beta2]
            ds_config["scheduler"]["params"]["warmup_max_lr"] = args.lr
            ds_config["scheduler"]["params"]["total_num_steps"] = args.epochs * args.steps_per_epoch
            ds_config["fp16"]["enabled"] = args.precision == "fp16"
            ds_config["bf16"]["enabled"] = args.precision == "bf16"
            
            logger.info(f"DeepSpeed設定を読み込みました: {args.deepspeed_config}")
        else:
            # 設定ファイルがない場合はデフォルト設定を使用
            logger.warning(f"DeepSpeed設定ファイルが見つかりません: {args.deepspeed_config}")
            logger.info("デフォルトのDeepSpeed設定を使用します")
            
            ds_config = {
                "train_micro_batch_size_per_gpu": args.batch_size,
                "gradient_accumulation_steps": args.grad_accumulation_steps,
                "optimizer": {
                    "type": "AdamW",
                    "params": {
                        "lr": args.lr,
                        "weight_decay": 0.0,
                        "betas": [args.beta1, args.beta2],
                    },
                },
                "scheduler": {
                    "type": "WarmupDecayLR",
                    "params": {
                        "total_num_steps": args.epochs * args.steps_per_epoch,
                        "warmup_min_lr": 0,
                        "warmup_max_lr": args.lr,
                        "warmup_num_steps": 100,
                        "warmup_type": "linear",
                    },
                },
                "fp16": {
                    "enabled": args.precision == "fp16",
                },
                "bf16": {
                    "enabled": args.precision == "bf16",
                },
                "gradient_clipping": 1.0,
                "zero_optimization": {
                    "stage": 2,
                    "contiguous_gradients": True,
                    "overlap_comm": True,
                    "reduce_scatter": True,
                    "reduce_bucket_size": 5e8,
                    "allgather_bucket_size": 5e8,
                },
            }
        
        # DeepSpeedでモデルを初期化
        model_engine, optimizer, train_loader, scheduler = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            training_data=train_dataset,
            collate_fn=partial(
                collate_fn,
                tokenizer=tokenizer,
                conv_type=args.conv_type,
                use_mm_start_end=args.use_mm_start_end,
                local_rank=args.local_rank,
            ),
            config=ds_config,
        )
        
        logger.info("DeepSpeedでモデルを初期化しました")

        # チェックポイントの復元
        if args.auto_resume and len(args.resume) == 0:
            resume = os.path.join(args.log_dir, "ckpt_model")
            if os.path.exists(resume):
                args.resume = resume

        if args.resume:
            logger.info(f"チェックポイントを読み込みます: {args.resume}")
            load_path, client_state = model_engine.load_checkpoint(args.resume)
            with open(os.path.join(args.resume, "latest"), "r") as f:
                ckpt_dir = f.readlines()[0].strip()
            args.start_epoch = (
                int(ckpt_dir.replace("global_step", "")) // args.steps_per_epoch
            )
            logger.info(
                f"チェックポイントを読み込みました: {args.resume}, エポック {args.start_epoch} から開始します"
            )

        # 検証データセットのデータローダー初期化
        if val_dataset is not None:
            assert args.val_batch_size == 1
            val_sampler = torch.utils.data.distributed.DistributedSampler(
                val_dataset, shuffle=False, drop_last=False
            )
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=args.val_batch_size,
                shuffle=False,
                num_workers=args.workers,
                pin_memory=False,
                sampler=val_sampler,
                collate_fn=partial(
                    collate_fn,
                    tokenizer=tokenizer,
                    conv_type=args.conv_type,
                    use_mm_start_end=args.use_mm_start_end,
                    local_rank=args.local_rank,
                ),
            )
        else:
            val_loader = None

        # 学習開始前の設定
        train_iter = iter(train_loader)
        best_score, cur_ciou = 0.0, 0.0

        # 評価のみの場合
        if args.eval_only:
            if val_loader is not None:
                logger.info("評価を開始します...")
                giou, ciou = validate(val_loader, model_engine, 0, writer, args)
                logger.info(f"評価結果: gIoU={giou:.4f}, cIoU={ciou:.4f}")
            else:
                logger.warning("検証データセットが指定されていないため、評価をスキップします")
            return

        # トレーニングループ
        logger.info("フルデータセットでのトレーニングを開始します...")
        for epoch in range(args.start_epoch, args.epochs):
            # 1エポック分トレーニング
            train_iter = train(
                train_loader,
                model_engine,
                epoch,
                scheduler,
                writer,
                train_iter,
                args,
            )

            # 評価が有効な場合は検証を実行
            if not args.no_eval and val_loader is not None:
                giou, ciou = validate(val_loader, model_engine, epoch, writer, args)
                is_best = giou > best_score
                best_score = max(giou, best_score)
                cur_ciou = ciou if is_best else cur_ciou
                
                if args.local_rank == 0:
                    logger.info(f"検証結果: エポック {epoch}, gIoU={giou:.4f}, cIoU={ciou:.4f}")
                    logger.info(f"ベストスコア: gIoU={best_score:.4f}, cIoU={cur_ciou:.4f}")

            # モデルを保存
            if args.no_eval or is_best:
                save_dir = os.path.join(args.log_dir, "ckpt_model")
                if args.local_rank == 0:
                    torch.save(
                        {"epoch": epoch},
                        os.path.join(
                            args.log_dir,
                            f"meta_log_giou{best_score:.3f}_ciou{cur_ciou:.3f}.pth"
                        ),
                    )
                    if os.path.exists(save_dir):
                        shutil.rmtree(save_dir)
                # 分散環境での同期を待つ
                if args.distributed:
                    torch.distributed.barrier()
                # モデルを保存
                model_engine.save_checkpoint(save_dir)
                logger.info(f"モデルチェックポイントを保存しました: {save_dir}")

        logger.info("トレーニングが完了しました!")
        
        # トレーニング終了時に最終チェックポイントを保存
        final_save_dir = os.path.join(args.log_dir, "ckpt_model_final")
        if args.local_rank == 0:
            if os.path.exists(final_save_dir):
                shutil.rmtree(final_save_dir)
        # 分散環境での同期を待つ
        if args.distributed:
            torch.distributed.barrier()
        # 最終モデルを保存
        model_engine.save_checkpoint(final_save_dir)
        logger.info(f"最終モデルチェックポイントを保存しました: {final_save_dir}")
        
        # DeepSpeedチェックポイントをpytorch_model.binに変換（メインプロセスのみ）
        if args.local_rank == 0:
            try:
                logger.info("DeepSpeedチェックポイントをPyTorchモデルに変換しています...")
                # カレントディレクトリを保存
                current_dir = os.getcwd()
                # チェックポイントディレクトリに移動
                os.chdir(final_save_dir)
                
                # deepspeed.ops.zero_to_fp32.pyを実行
                convert_cmd = f"python -m deepspeed.ops.zero_to_fp32 . ../pytorch_model.bin"
                os.system(convert_cmd)
                
                # 元のディレクトリに戻る
                os.chdir(current_dir)
                
                # 変換が成功したか確認
                pytorch_model_path = os.path.join(args.log_dir, "pytorch_model.bin")
                if os.path.exists(pytorch_model_path):
                    logger.info(f"PyTorchモデルへの変換が完了しました: {pytorch_model_path}")
                else:
                    logger.warning("PyTorchモデルへの変換が失敗した可能性があります")
            except Exception as e:
                logger.error(f"PyTorchモデルへの変換中にエラーが発生しました: {e}")
                
        if args.local_rank == 0:
            logger.info("=== トレーニングプロセスの完了 ===")
            logger.info(f"モデル: {args.version}")
            logger.info(f"実験名: {args.exp_name}")
            logger.info(f"保存先: {args.log_dir}")
            logger.info(f"最終性能: gIoU={best_score:.4f}, cIoU={cur_ciou:.4f}")
        
    except Exception as e:
        logger.error(f"トレーニング中にエラーが発生しました: {e}")
        import traceback
        logger.error(traceback.format_exc())
        logger.error("トレーニングを中止します。")
        return


def train(
    train_loader,
    model,
    epoch,
    scheduler,
    writer,
    train_iter,
    args,
):
    """
    1エポック分のトレーニングを実行
    
    Args:
        train_loader: トレーニングデータのデータローダー
        model: DeepSpeedで初期化されたモデル
        epoch: 現在のエポック
        scheduler: 学習率スケジューラ
        writer: TensorBoardのSummaryWriter
        train_iter: データローダーのイテレータ
        args: コマンドライン引数
        
    Returns:
        train_iter: 更新されたデータローダーのイテレータ
    """
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    ce_losses = AverageMeter("CeLoss", ":.4f")
    mask_bce_losses = AverageMeter("MaskBCELoss", ":.4f")
    mask_dice_losses = AverageMeter("MaskDICELoss", ":.4f")
    mask_losses = AverageMeter("MaskLoss", ":.4f")

    progress = ProgressMeter(
        args.steps_per_epoch,
        [
            batch_time,
            losses,
            ce_losses,
            mask_losses,
            mask_bce_losses,
            mask_dice_losses,
        ],
        prefix=f"エポック: [{epoch}]",
    )

    # トレーニングモードに設定
    model.train()
    end = time.time()
    
    # 1エポック分のステップを実行
    for global_step in range(args.steps_per_epoch):
        for i in range(args.grad_accumulation_steps):
            try:
                input_dict = next(train_iter)
            except:
                train_iter = iter(train_loader)
                input_dict = next(train_iter)

            # データロード時間を記録
            data_time.update(time.time() - end)
            
            # データをGPUに転送
            input_dict = dict_to_cuda(input_dict)

            # 精度に応じてデータ型を変換
            if args.precision == "fp16":
                input_dict["images"] = input_dict["images"].half()
                input_dict["images_clip"] = input_dict["images_clip"].half()
            elif args.precision == "bf16":
                input_dict["images"] = input_dict["images"].bfloat16()
                input_dict["images_clip"] = input_dict["images_clip"].bfloat16()
            else:
                input_dict["images"] = input_dict["images"].float()
                input_dict["images_clip"] = input_dict["images_clip"].float()

            # モデルの順伝播
            output_dict = model(**input_dict)

            # 損失の抽出
            loss = output_dict["loss"]
            ce_loss = output_dict["ce_loss"]
            mask_bce_loss = output_dict["mask_bce_loss"]
            mask_dice_loss = output_dict["mask_dice_loss"]
            mask_loss = output_dict["mask_loss"]

            # スケーリングされた損失で勾配計算
            model.backward(loss)
            
            # 最後の蓄積ステップでのみパラメータ更新
            if (i + 1) % args.grad_accumulation_steps == 0:
                model.step()
                
                # ロギング
                losses.update(loss.item(), input_dict["images"].size(0))
                ce_losses.update(ce_loss.item(), input_dict["images"].size(0))
                mask_bce_losses.update(mask_bce_loss.item(), input_dict["images"].size(0))
                mask_dice_losses.update(mask_dice_loss.item(), input_dict["images"].size(0))
                mask_losses.update(mask_loss.item(), input_dict["images"].size(0))
                
                # バッチ処理時間を更新
                batch_time.update(time.time() - end)
                end = time.time()
                
                # 進捗表示（メインプロセスのみ）
                if args.local_rank == 0 and global_step % args.print_freq == 0:
                    progress.display(global_step)
                    
                    # TensorBoardにログを記録
                    if writer is not None:
                        writer.add_scalar("train/loss", losses.val, epoch * args.steps_per_epoch + global_step)
                        writer.add_scalar("train/ce_loss", ce_losses.val, epoch * args.steps_per_epoch + global_step)
                        writer.add_scalar("train/mask_bce_loss", mask_bce_losses.val, epoch * args.steps_per_epoch + global_step)
                        writer.add_scalar("train/mask_dice_loss", mask_dice_losses.val, epoch * args.steps_per_epoch + global_step)
                        writer.add_scalar("train/mask_loss", mask_losses.val, epoch * args.steps_per_epoch + global_step)
                        writer.add_scalar("train/lr", scheduler.get_last_lr()[0], epoch * args.steps_per_epoch + global_step)

    return train_iter


def validate(val_loader, model, epoch, writer, args):
    """
    検証データセットで評価を実行
    
    Args:
        val_loader: 検証データのデータローダー
        model: DeepSpeedで初期化されたモデル
        epoch: 現在のエポック
        writer: TensorBoardのSummaryWriter
        args: コマンドライン引数
        
    Returns:
        giou: グローバルIoUスコア
        ciou: クラスIoUスコア
    """
    # 評価モードに設定
    model.eval()
    
    batch_time = AverageMeter("Time", ":6.3f")
    intersection_meter = AverageMeter("Intersec", ":6.3f")
    union_meter = AverageMeter("Union", ":6.3f")
    target_meter = AverageMeter("Target", ":6.3f")
    
    progress = ProgressMeter(
        len(val_loader), [batch_time, intersection_meter, union_meter, target_meter], prefix="検証: "
    )
    
    intersection_sum = 0
    union_sum = 0
    target_sum = 0
    
    # 勾配計算なしで評価を実行
    with torch.no_grad():
        end = time.time()
        for i, input_dict in enumerate(val_loader):
            # データをGPUに転送
            input_dict = dict_to_cuda(input_dict)
            
            # 精度に応じてデータ型を変換
            if args.precision == "fp16":
                input_dict["images"] = input_dict["images"].half()
                input_dict["images_clip"] = input_dict["images_clip"].half()
            elif args.precision == "bf16":
                input_dict["images"] = input_dict["images"].bfloat16()
                input_dict["images_clip"] = input_dict["images_clip"].bfloat16()
            else:
                input_dict["images"] = input_dict["images"].float()
                input_dict["images_clip"] = input_dict["images_clip"].float()
                
            # モデルの予測
            output_dict = model(**input_dict)
            
            # 予測マスクと正解マスクを抽出
            pred_masks = output_dict["pred_masks"]
            masks = input_dict["masks"]
            
            # IoU計算
            intersection, union, target = intersectionAndUnionGPU(
                pred_masks.squeeze(1).contiguous(),
                masks.contiguous(),
                2,  # バイナリセグメンテーション（背景とクラス）
                ignore_index=255,
            )
            
            # 結果を集計
            intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
            intersection_sum += intersection
            union_sum += union
            target_sum += target
            
            # バッチごとのIoUを計算
            intersection_meter.update(np.nansum(intersection))
            union_meter.update(np.nansum(union))
            target_meter.update(np.nansum(target))
            
            # バッチ処理時間を更新
            batch_time.update(time.time() - end)
            end = time.time()
            
            # 進捗表示（メインプロセスのみ）
            if args.local_rank == 0 and i % args.print_freq == 0:
                progress.display(i)
                
    # グローバルIoUとクラスIoUを計算
    giou = intersection_sum / (union_sum + 1e-10)
    ciou = intersection_sum / (target_sum + 1e-10)
    giou, ciou = np.nanmean(giou), np.nanmean(ciou)
    
    # メインプロセスのみログ記録
    if args.local_rank == 0:
        logger.info(f"検証結果: gIoU={giou:.4f}, cIoU={ciou:.4f}")
        
        # TensorBoardにログを記録
        if writer is not None:
            writer.add_scalar("val/giou", giou, epoch)
            writer.add_scalar("val/ciou", ciou, epoch)
    
    return giou, ciou


if __name__ == "__main__":
    main(sys.argv[1:])
