"""
Gemma3-LISA フルデータセット学習スクリプト

このスクリプトはGemma3とSAMを組み合わせたLISAモデルを、
すべてのトレーニングデータセットを使用して学習するためのものです。
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
from utils.trainer import GemmaLISATrainer
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
        "--dataset", default="sem_seg,refer_seg,vqa,reason_seg", type=str
    )
    parser.add_argument("--sample_rates", default=None, type=str)
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
    parser.add_argument("--dataset_dir", default="/mnt/h/download/LISA-dataset/dataset", type=str)
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

# 必要な関数を追加
def parse_dataset_string(dataset_str):
    """カンマまたは||で区切られたデータセット文字列を分割"""
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
        # モデルのロード
        logger.info("モデルをロード中...")
        
        # トークナイザーとモデルの初期化
        model, tokenizer = initialize_model_and_tokenizer(args)
        
        # モデルをGPUに明示的に移動
        if torch.cuda.is_available():
            logger.info("モデルをGPUに移動します...")
            # 量子化モデルの場合はすでにGPUに配置されている可能性があるため、確認する
            device_info = next(model.parameters()).device
            if device_info == torch.device('cpu'):
                model = model.cuda()
                logger.info(f"モデルをGPUに移動しました: {next(model.parameters()).device}")
            else:
                logger.info(f"モデルは既にGPUに配置されています: {device_info}")
        else:
            logger.info("GPUが利用できないため、CPUモードで実行します")
                
        # トレーニング用画像変換
        if not args.eval_only:
            train_transform = Compose([
                ToTensor(),
                RandomResizedCrop(
                    size=(896, 896),  # Gemma3の推奨サイズ (SiGLIP ViT用)
                    scale=(0.9, 1.0),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                          std=[0.26862954, 0.26130258, 0.27577711]),
            ])
            
            # 変換処理を保存
            setattr(args, "vis_processors", type("obj", (), {"train": train_transform}))
            
        # データセットの初期化
        logger.info("データセットをロード中...")
        
        # 初期状態では何もロードされていない
        train_dataset = None
        
        # キャッシュからロードするか確認
        dataset_cache_path = None
        if args.use_cached_data:
            cache_dir = os.path.join("./dataset_cache")
            os.makedirs(cache_dir, exist_ok=True)
            
            # キャッシュファイル名を生成
            cache_filename = f"dataset_cache_{args.dataset.replace('||', '_')}.pkl"
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
                    # 必要なフィールドが存在するか確認
                    required_attrs = ['tokenizer', 'seg_token_idx', 'image_token_idx']
                    if all(hasattr(train_dataset, attr) for attr in required_attrs):
                        logger.info("キャッシュからロードしたデータセットは有効です")
                    else:
                        logger.warning("キャッシュからロードしたデータセットは無効です。再作成します。")
                        train_dataset = None
                except Exception as e:
                    logger.warning(f"キャッシュからのロード中にエラーが発生しました: {e}")
                    train_dataset = None
        
        # デバッグモードの設定
        if args.debug and train_dataset is None:
            logger.info(f"==== デバッグモード有効: サンプル数制限={args.debug_samples} ====")
            
            # マルチデータセットデバッグモードのチェック
            if args.debug_dataset == "multi" or args.debug_dataset == "all":
                # 元のデータセット設定を保持したまま、サンプル数とエポック数だけ調整
                logger.info(f"マルチデータセットデバッグモード: {args.dataset}")
                # サンプルレートは元の設定を維持
                args.steps_per_epoch = min(args.steps_per_epoch, 10)  # 少ないステップ数
                args.val_dataset = "ReasonSeg|val"  # 検証データセットも設定
            # 単一データセットモード
            elif args.debug_dataset == "ade20k":
                args.dataset = "sem_seg"
                args.sem_seg_data = "ade20k"
                args.sample_rates = "1"  # 単一データセット用
                args.steps_per_epoch = min(args.steps_per_epoch, 10)  # 少ないステップ数
                args.val_dataset = "ReasonSeg|val"  # 検証データセットも設定
            elif args.debug_dataset == "cocostuff":
                args.dataset = "sem_seg"
                args.sem_seg_data = "cocostuff"
                args.sample_rates = "1"  # 単一データセット用
                args.steps_per_epoch = min(args.steps_per_epoch, 10)  # 少ないステップ数
                args.val_dataset = "ReasonSeg|val"  # 検証データセットも設定
            elif args.debug_dataset == "sem_seg":
                # すべてのセマンティックセグメンテーションデータセットを使用
                args.dataset = "sem_seg"
                # 直接ルートディレクトリにあるセマンティックセグメンテーションデータセットを指定
                if "," in args.sem_seg_data:
                    # カンマ区切りの場合はそのまま使用
                    dataset_count = len(args.sem_seg_data.split(","))
                else:
                    # ||区切りの場合
                    dataset_count = len(args.sem_seg_data.split("||"))
                # データセット数と同じ数のサンプルレートを設定
                args.sample_rates = ",".join(["1"] * dataset_count)
                args.steps_per_epoch = min(args.steps_per_epoch, 10)  # 少ないステップ数
                args.val_dataset = "ReasonSeg|val"  # 検証データセットも設定
            else:
                # その他のデータセット
                args.dataset = "sem_seg"
                args.sem_seg_data = args.debug_dataset
                args.sample_rates = "1"  # 単一データセット用
                args.steps_per_epoch = min(args.steps_per_epoch, 10)  # 少ないステップ数
                args.val_dataset = "ReasonSeg|val"  # 検証データセットも設定
            
            logger.info(f"デバッグモード設定: dataset={args.dataset}, steps_per_epoch={args.steps_per_epoch}")
        
        # データセットがまだロードされていない場合は作成
        if train_dataset is None:
            # デバッグ用かフル用かのメッセージ
            if args.debug:
                logger.info(f"デバッグデータセットを初期化: {args.sem_seg_data} (サンプル数: {args.debug_samples})")
            else:
                logger.info(f"フルデータセットを初期化: {args.dataset}")
            
            # データセットパスの検証
            logger.info("データセットパスの検証を行います...")
            
            # カスタムパスオプションが有効な場合、ベースディレクトリを更新
            sem_seg_base = args.sem_seg_base_dir if args.custom_dataset_paths else "sem_seg"
            refer_seg_base = args.refer_seg_base_dir if args.custom_dataset_paths else "refer_seg"
            vqa_base = getattr(args, "vqa_base_dir", "llava_dataset") if args.custom_dataset_paths else "llava_dataset"
            reason_seg_base = args.reason_seg_base_dir if args.custom_dataset_paths else "reason_seg"
            
            # データセット指定文字列の解析
            # カンマとパイプ区切りの両方に対応
            dataset_types = parse_dataset_string(args.dataset)
            
            # サンプルレートの検証
            if args.sample_rates:
                sample_rates = parse_dataset_string(args.sample_rates)
                if len(sample_rates) != len(dataset_types):
                    logger.warning(f"サンプルレート数 ({len(sample_rates)}) がデータセット数 ({len(dataset_types)}) と一致しません")
                    logger.warning(f"データセット数に合わせて均等なサンプルレートを使用します")
                    args.sample_rates = ",".join(["1"] * len(dataset_types))
                    logger.info(f"調整後のサンプルレート: {args.sample_rates}")
            else:
                # sample_ratesが指定されていない場合は、均等なレートを設定
                args.sample_rates = ",".join(["1"] * len(dataset_types))
                logger.info(f"サンプルレートが未指定です。均等なレートを適用: {args.sample_rates}")
            
            logger.info(f"使用するデータセットパス設定:")
            logger.info(f"- データセット種類: {dataset_types}")
            logger.info(f"- ベースディレクトリ: {args.dataset_dir}")
            logger.info(f"- セマンティックセグメンテーションベース: {sem_seg_base}")
            logger.info(f"- 参照セグメンテーションベース: {refer_seg_base}")
            logger.info(f"- VQAベース: {vqa_base}")
            logger.info(f"- 理由付きセグメンテーションベース: {reason_seg_base}")
            
            # セマンティックセグメンテーションデータセットの検証
            if "sem_seg" in dataset_types:
                logger.info(f"セマンティックセグメンテーションデータセット: {args.sem_seg_data}")
                for ds in parse_dataset_string(args.sem_seg_data):
                    # パスの形式に応じて適切なパスを構築
                    if "/" in ds:  # 完全なパスが指定されている場合
                        ds_path = os.path.join(args.dataset_dir, ds)
                    else:  # データセット名のみが指定されている場合（直接ルートディレクトリに存在）
                        # 直接データセットディレクトリを探す
                        ds_path = os.path.join(args.dataset_dir, ds)
                        
                    if os.path.exists(ds_path):
                        logger.info(f"セマンティックセグメンテーションデータセット '{ds}' のパスが存在します: {ds_path}")
                    else:
                        logger.warning(f"セマンティックセグメンテーションデータセット '{ds}' のパスが見つかりません: {ds_path}")
                        
            # 参照セグメンテーションデータセットの検証
            if "refer_seg" in dataset_types:
                # 必要なディレクトリが存在するか検証
                refer_base_dir = os.path.join(args.dataset_dir, refer_seg_base)
                refer_images_dir = os.path.join(refer_base_dir, "images")
                mscoco_dir = os.path.join(refer_images_dir, "mscoco", "images")
                saiapr_dir = os.path.join(refer_images_dir, "saiapr_tc-12")
                
                logger.info(f"参照セグメンテーションデータセット: {args.refer_seg_data}")
                logger.info(f"参照セグメンテーションベースディレクトリ: {refer_base_dir}")
                
                # カンマ区切りと||区切りの両方に対応
                refer_seg_datasets = parse_dataset_string(args.refer_seg_data)
                logger.info(f"検出された参照セグメンテーションデータセット: {refer_seg_datasets}")
                
                if os.path.exists(refer_base_dir):
                    logger.info(f"参照セグメンテーションベースディレクトリが存在します: {refer_base_dir}")
                    
                    if os.path.exists(refer_images_dir):
                        logger.info(f"参照セグメンテーション 'images' ディレクトリが存在します: {refer_images_dir}")
                        
                        if os.path.exists(mscoco_dir):
                            logger.info(f"mscoco画像ディレクトリが存在します: {mscoco_dir}")
                        else:
                            logger.warning(f"mscoco画像ディレクトリが見つかりません: {mscoco_dir}")
                            
                        if os.path.exists(saiapr_dir):
                            logger.info(f"saiapr_tc-12ディレクトリが存在します: {saiapr_dir}")
                        else:
                            logger.warning(f"saiapr_tc-12ディレクトリが見つかりません: {saiapr_dir}")
                    else:
                        logger.warning(f"参照セグメンテーション 'images' ディレクトリが見つかりません: {refer_images_dir}")
                else:
                    logger.warning(f"参照セグメンテーションベースディレクトリが見つかりません: {refer_base_dir}")
            
            # VQAデータセットの検証
            if "vqa" in dataset_types:
                vqa_dir = os.path.join(args.dataset_dir, vqa_base)
                logger.info(f"VQAデータセット: {args.vqa_data}")
                logger.info(f"VQAディレクトリ: {vqa_dir}")
                
                # カンマ区切りと||区切りの両方に対応
                vqa_datasets = parse_dataset_string(args.vqa_data)
                logger.info(f"検出されたVQAデータセット: {vqa_datasets}")
                
                if os.path.exists(vqa_dir):
                    logger.info(f"VQAデータセットディレクトリが存在します: {vqa_dir}")
                else:
                    logger.warning(f"VQAデータセットディレクトリが見つかりません: {vqa_dir}")
            
            # ReasonSegデータセットの検証
            if "reason_seg" in dataset_types:
                # パスの形式に応じて適切なパスを構築
                reason_seg_datasets = parse_dataset_string(args.reason_seg_data)
                logger.info(f"検出された理由付きセグメンテーションデータセット: {reason_seg_datasets}")
                
                for reason_seg_dataset in reason_seg_datasets:
                    if "/" in reason_seg_dataset:  # 完全なパスが指定されている場合
                        reason_seg_parts = reason_seg_dataset.split("/")
                        if len(reason_seg_parts) == 2:
                            reason_seg_dir = os.path.join(args.dataset_dir, reason_seg_dataset)
                        else:
                            reason_seg_dir = os.path.join(args.dataset_dir, reason_seg_dataset)
                    else:  # データセット名のみが指定されている場合
                        reason_seg_dir = os.path.join(args.dataset_dir, reason_seg_base, reason_seg_dataset)
                    
                    reason_seg_train = os.path.join(reason_seg_dir, "train")
                    
                    logger.info(f"理由付きセグメンテーションデータセット: {reason_seg_dataset}")
                    logger.info(f"理由付きセグメンテーションディレクトリ: {reason_seg_dir}")
                    
                    if os.path.exists(os.path.dirname(reason_seg_dir)):
                        logger.info(f"ReasonSegの親ディレクトリが存在します: {os.path.dirname(reason_seg_dir)}")
                        
                        if os.path.exists(reason_seg_dir):
                            logger.info(f"ReasonSegディレクトリが存在します: {reason_seg_dir}")
                            
                            if os.path.exists(reason_seg_train):
                                logger.info(f"ReasonSeg trainディレクトリが存在します: {reason_seg_train}")
                            else:
                                logger.warning(f"ReasonSeg trainディレクトリが見つかりません: {reason_seg_train}")
                        else:
                            logger.warning(f"ReasonSegディレクトリが見つかりません: {reason_seg_dir}")
                    else:
                        logger.warning(f"ReasonSegの親ディレクトリが見つかりません: {os.path.dirname(reason_seg_dir)}")
                    
            # 少なくとも1つのデータセットが有効かチェック
            valid_datasets = False
            # セマンティックセグメンテーションは直接パスをチェック
            if "sem_seg" in dataset_types:
                for ds in parse_dataset_string(args.sem_seg_data):
                    # 直接ルートディレクトリにあるデータセットをチェック
                    ds_path = os.path.join(args.dataset_dir, ds)
                    if os.path.exists(ds_path):
                        valid_datasets = True
                        logger.info(f"有効なセマンティックセグメンテーションデータセット: {ds_path}")
                        break
            if "refer_seg" in dataset_types and os.path.exists(os.path.join(args.dataset_dir, refer_seg_base)):
                valid_datasets = True
            if "vqa" in dataset_types and os.path.exists(os.path.join(args.dataset_dir, vqa_base)):
                valid_datasets = True
            if "reason_seg" in dataset_types and os.path.exists(os.path.dirname(reason_seg_dir)):
                valid_datasets = True
                
            if not valid_datasets:
                logger.error("有効なデータセットが見つかりません。少なくとも1つのデータセットが必要です。")
                logger.error(f"指定されたデータセットディレクトリ: {args.dataset_dir}")
                logger.error(f"指定されたデータセット種類: {args.dataset}")
                
                # データセットディレクトリの内容を表示
                logger.info("データセットディレクトリの内容:")
                try:
                    for item in os.listdir(args.dataset_dir):
                        item_path = os.path.join(args.dataset_dir, item)
                        if os.path.isdir(item_path):
                            logger.info(f"- ディレクトリ: {item}")
                        else:
                            logger.info(f"- ファイル: {item}")
                except Exception as e:
                    logger.error(f"データセットディレクトリのリスト取得中にエラー: {e}")
                
                raise RuntimeError("有効なデータセットが見つかりません。少なくとも1つの有効なデータセットが必要です。")
            
            # データセットの初期化（エラーハンドリングを追加）
            try:
                # カスタムパスオプションが有効な場合、データセットのパスを調整
                if args.custom_dataset_paths:
                    sem_seg_data_adjusted = []
                    for ds in parse_dataset_string(args.sem_seg_data):
                        if "/" not in ds:
                            # データセット名のみの場合、直接ルートディレクトリにあるデータセットを使用
                            sem_seg_data_adjusted.append(ds)
                        else:
                            # すでにパスが含まれている場合はそのまま
                            sem_seg_data_adjusted.append(ds)
                    
                    sem_seg_data_final = "||".join(sem_seg_data_adjusted)
                    logger.info(f"調整後のセマンティックセグメンテーションデータセット: {sem_seg_data_final}")
                else:
                    # カスタムパスが無効な場合でも、直接ルートディレクトリを使用
                    # カンマ区切りを||区切りに変換
                    if "," in args.sem_seg_data and "||" not in args.sem_seg_data:
                        sem_seg_data_final = "||".join(parse_dataset_string(args.sem_seg_data))
                    else:
                        sem_seg_data_final = args.sem_seg_data
                    logger.info(f"使用するセマンティックセグメンテーションデータセット: {sem_seg_data_final}")
                
                # reasonセグメンテーションデータセットの調整
                if args.custom_dataset_paths and "reason_seg" in dataset_types:
                    reason_seg_data_adjusted = []
                    for reason_seg_dataset in parse_dataset_string(args.reason_seg_data):
                        if "/" not in reason_seg_dataset:
                            reason_seg_data_adjusted.append(f"{reason_seg_base}/{reason_seg_dataset}")
                        else:
                            reason_seg_data_adjusted.append(reason_seg_dataset)
                    
                    # 区切り文字の標準化
                    reason_seg_data_final = "||".join(reason_seg_data_adjusted)
                    logger.info(f"調整後の理由付きセグメンテーションデータセット: {reason_seg_data_final}")
                else:
                    # カンマ区切りを||区切りに標準化
                    if "," in args.reason_seg_data and "||" not in args.reason_seg_data:
                        reason_seg_data_final = "||".join(parse_dataset_string(args.reason_seg_data))
                    else:
                        reason_seg_data_final = args.reason_seg_data
                
                # refer_segとvqaデータセットのカンマ区切りを||区切りに標準化
                refer_seg_data_final = args.refer_seg_data
                if "," in args.refer_seg_data and "||" not in args.refer_seg_data:
                    refer_seg_data_final = "||".join(parse_dataset_string(args.refer_seg_data))

                vqa_data_final = args.vqa_data
                if "," in args.vqa_data and "||" not in args.vqa_data:
                    vqa_data_final = "||".join(parse_dataset_string(args.vqa_data))
                
                train_dataset = HybridDataset(
                    tokenizer=tokenizer,
                    model_name=args.version,  # 明示的にGemma3モデル名を設定
                    vision_tower=args.vision_tower,
                    sam_checkpoint=args.vision_pretrained,
                    base_image_dir=args.dataset_dir,
                    samples_per_epoch=args.steps_per_epoch * args.batch_size * args.grad_accumulation_steps,
                    sample_rates=args.sample_rates,
                    dataset=dataset_types,  # 既にパースされたデータセットリストを使用
                    sem_seg_data=sem_seg_data_final,
                    refer_seg_data=refer_seg_data_final,
                    vqa_data=vqa_data_final,
                    reason_seg_data=reason_seg_data_final,
                    seg_token_idx=tokenizer("[SEG]", add_special_tokens=False).input_ids[0],
                    image_token_idx=get_image_token_index(tokenizer),
                    debug=args.debug,
                    debug_samples=args.debug_samples if args.debug else None,
                )
                logger.info(f"トレーニングデータセットのサイズ: {len(train_dataset)}")
            except Exception as e:
                logger.error(f"データセットの初期化中にエラーが発生しました: {e}")
                import traceback
                logger.error(traceback.format_exc())
                
                # デバッグモード時は問題を明確にするためのヒントを表示
                if args.debug:
                    logger.info("デバッグモードのトラブルシューティング:")
                    logger.info("1. データセットディレクトリが正しいか確認してください")
                    logger.info("2. --dataset_dir オプションで正しいパスを指定してください")
                    logger.info("3. 指定したデータセットタイプとデータが存在することを確認してください")
                    logger.info("4. 別のデータセットまたはデバッグサンプル数で試してください")
                    
                    # 特定のエラーの場合、ヒントを表示
                    if "すべてのデータセットが無効です" in str(e):
                        logger.info("5. 指定されたデータセット種類が存在しない可能性があります。別のデータセットを試してください")
                        logger.info("例: --debug_dataset ade20k または --debug_dataset cocostuff")
                
                raise RuntimeError(f"データセットの初期化に失敗しました: {e}") from e
            
            # キャッシュに保存
            if args.save_cached_data and dataset_cache_path:
                logger.info(f"データセットをキャッシュに保存中: {dataset_cache_path}")
                try:
                    with open(dataset_cache_path, 'wb') as f:
                        pickle.dump(train_dataset, f)
                    logger.info("データセットをキャッシュに保存しました")
                except Exception as e:
                    logger.warning(f"キャッシュへの保存中にエラーが発生しました: {e}")
        
        else:
            train_dataset = None
        
        # 検証データセットの作成（指定がある場合）
        if args.val_dataset and not args.no_eval:
            logger.info(f"検証データセットの初期化: {args.val_dataset}")
            # 一時的にコメントアウト - 評価データセットは現在Trainer APIと互換性がありません
            # val_dataset = ValDataset(
            #     args.dataset_dir, tokenizer, args.version, args.val_dataset, args.image_size, transform=val_transform
            # )
            # logger.info(f"検証データセットのサイズ: {len(val_dataset)}")
            
            # 評価データセットの問題を解決するまで一時的にNoneに設定
            val_dataset = None
            logger.warning("評価データセットが一時的に無効化されています。")
        else:
            val_dataset = None
        
        # データセットが正しく初期化されたか確認
        if not args.eval_only and (train_dataset is None or len(train_dataset) == 0):
            raise RuntimeError("トレーニングデータセットが初期化できませんでした。データがない可能性があります。")
            
        # データローダーの作成
        logger.info("データローダーを初期化中...")
        
        # TrainingArgumentsの設定
        deepspeed_config = None
        # DeepSpeedの使用を無効化（明示的に無効化）
        if False and DEEPSPEED_AVAILABLE and os.path.exists(args.deepspeed_config) and args.deepspeed_config != "None":
            deepspeed_config = args.deepspeed_config
            logger.info(f"DeepSpeed設定が読み込まれました: {deepspeed_config}")
        else:
            logger.info("DeepSpeedは無効化されています")
    
        training_args = TrainingArguments(
            output_dir=args.log_dir,
            overwrite_output_dir=True,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.val_batch_size,
            gradient_accumulation_steps=args.grad_accumulation_steps,
            learning_rate=args.lr,
            weight_decay=0.0,
            max_grad_norm=1.0,
            warmup_ratio=0.03,
            lr_scheduler_type="cosine",
            logging_dir=os.path.join(args.log_dir, "logs"),
            logging_steps=1,
            save_steps=args.steps_per_epoch // 2,
            save_total_limit=3,
            # 一時的に評価を無効化
            eval_strategy="no",  # "steps" if val_dataset is not None else "no",
            eval_steps=None,  # args.steps_per_epoch // 2 if val_dataset is not None else None,
            fp16=args.precision == "fp16",
            bf16=args.precision == "bf16",
            dataloader_num_workers=args.workers,
            local_rank=args.local_rank,
            remove_unused_columns=False,  # 画像データを保持するため
            report_to=["tensorboard"],
            label_names=["labels", "mask_labels"],  # モデルへの入力として渡す特別なキー
            deepspeed=deepspeed_config,
        )
        
        # カスタムTrainerのインスタンス化
        trainer = GemmaLISATrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=partial(collate_fn, tokenizer=tokenizer),
            processing_class=tokenizer,  # tokenizer の代わりに processing_class を使用
            # カスタム損失の重み
            ce_loss_weight=args.ce_loss_weight,
            bce_loss_weight=args.bce_loss_weight,
            dice_loss_weight=args.dice_loss_weight,
        )
        
        # 学習またはモデルのロード
        # チェックポイントの復元
        if args.resume:
            if os.path.isfile(args.resume):
                logger.info(f"チェックポイントを読み込みます: {args.resume}")
                # Trainer.train() に前回のチェックポイントを渡して続きから学習
                trainer_path = args.resume
            else:
                logger.warning(f"チェックポイントが見つかりません: {args.resume}")
                trainer_path = None
        elif args.auto_resume:
            # 自動的に最新のチェックポイントを探す
            latest = trainer._get_checkpoint_path()
            if latest:
                logger.info(f"最新のチェックポイントを読み込みます: {latest}")
                trainer_path = latest
            else:
                logger.info("チェックポイントが見つからないため、初めから学習します")
                trainer_path = None
        else:
            trainer_path = None
            
        # 評価モードまたはトレーニングモードを実行
        if args.eval_only:
            if val_dataset:
                logger.info("評価を開始します...")
                metrics = trainer.evaluate(eval_dataset=val_dataset)
                logger.info(f"評価結果: {metrics}")
            else:
                logger.warning("検証データセットが指定されていないため、評価をスキップします")
        else:
            logger.info("フルデータセットでのトレーニングを開始します...")
            
            # トレーニング開始直前のデバイス・メモリ情報を表示
            train_device = next(model.parameters()).device
            logger.info(f"トレーニング実行デバイス: {train_device}")
            
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    logger.info(f"GPU {i} メモリ状態: "
                               f"割り当て済み={torch.cuda.memory_allocated(i)/1024**3:.2f}GB, "
                               f"予約済み={torch.cuda.memory_reserved(i)/1024**3:.2f}GB")
            
            # トレーニング実行
            try:
                trainer.train(resume_from_checkpoint=trainer_path)
            except RuntimeError as e:
                # チェックポイント読み込みエラーを検出
                if "PytorchStreamReader failed locating file" in str(e) or "Unexpected key" in str(e):
                    logger.warning(f"チェックポイントの読み込みに失敗しました: {e}")
                    logger.warning("チェックポイントが壊れているか不完全です。初めからトレーニングを再開します。")
                    
                    # チェックポイントディレクトリが壊れているかチェック
                    if trainer_path and os.path.exists(trainer_path):
                        try:
                            # バックアップディレクトリを作成
                            backup_dir = os.path.join(os.path.dirname(trainer_path), "broken_checkpoints")
                            os.makedirs(backup_dir, exist_ok=True)
                            
                            # 壊れたチェックポイントを移動
                            checkpoint_name = os.path.basename(trainer_path)
                            backup_path = os.path.join(backup_dir, f"{checkpoint_name}_{int(time.time())}")
                            logger.info(f"壊れたチェックポイントを {backup_path} に移動します")
                            
                            # ディレクトリまたはファイルを移動
                            if os.path.isdir(trainer_path):
                                shutil.move(trainer_path, backup_path)
                            else:
                                shutil.copy2(trainer_path, backup_path)
                                os.remove(trainer_path)
                        except Exception as move_err:
                            logger.warning(f"壊れたチェックポイントの移動中にエラーが発生しました: {move_err}")
                    
                    # 初めからトレーニングを開始
                    logger.info("初めからトレーニングを開始します...")
                    trainer.train()
                else:
                    # 他のランタイムエラーは再発生させる
                    raise e
            
            # 最終モデルを保存
            trainer.save_model()
            logger.info(f"最終モデルを保存しました: {args.log_dir}")

    except Exception as e:
        logger.error(f"トレーニング中にエラーが発生しました: {e}")
        import traceback
        logger.error(traceback.format_exc())
        logger.error("トレーニングを中止します。")
        return

if __name__ == "__main__":
    main(sys.argv[1:])
