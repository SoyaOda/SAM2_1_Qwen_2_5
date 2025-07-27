# import argparse
import glob
import os
import random
import pickle

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask
from transformers import CLIPImageProcessor, AutoProcessor

# Gemma3のモジュールをインポート
try:
    from model.gemma3 import conversation as gemma_conversation_lib
    from model.gemma3.mm_utils import GemmaImageProcessor
    from model.gemma3.constants import (DEFAULT_IMAGE_TOKEN as GEMMA_IMAGE_TOKEN,
                                      DEFAULT_IM_START_TOKEN as GEMMA_IM_START_TOKEN,
                                      DEFAULT_IM_END_TOKEN as GEMMA_IM_END_TOKEN,
                                      SYSTEM_PROMPT_GEMMA)
    GEMMA_AVAILABLE = True
except ImportError:
    GEMMA_AVAILABLE = False

# LLaVA関連モジュールをインポート（互換性のため）
try:
    from model.llava import conversation as conversation_lib
    from model.llava.constants import (DEFAULT_IMAGE_TOKEN, IGNORE_INDEX,
                                     IMAGE_TOKEN_INDEX)
    from model.llava.mm_utils import tokenizer_image_token
    LLAVA_AVAILABLE = True
except ImportError:
    LLAVA_AVAILABLE = False
    # LLaVAがない場合のフォールバック定義
    from .constants import DEFAULT_IMAGE_TOKEN, IGNORE_INDEX, IMAGE_TOKEN_INDEX
    
    def tokenizer_image_token(prompt, tokenizer, return_tensors=None):
        """LLaVAのtokenizer_image_token関数を模倣"""
        input_ids = tokenizer(prompt, return_tensors=return_tensors).input_ids
        return input_ids

from model.segment_anything.utils.transforms import ResizeLongestSide

from .constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, SYSTEM_PROMPT
from .conversation import get_default_conv_template
from .data_processing import get_mask_from_json
from .reason_seg_dataset import ReasonSegDataset
from .refer import REFER
from .refer_seg_dataset import ReferSegDataset
from .sem_seg_dataset import SemSegDataset
from .vqa_dataset import VQADataset

# Gemma3のシステムプロンプトを使用 - 定数ファイルに移動
# SYSTEM_PROMPT定義を削除（constants.pyに移動済み）
    
# Gemma3とLLaVAで共通に使用される画像トークン - 定数ファイルに移動
# 以下の定義を削除（constants.pyに移動済み）
# DEFAULT_IMAGE_TOKEN = "<image>"
# DEFAULT_IM_START_TOKEN = "<im_start>"  
# DEFAULT_IM_END_TOKEN = "<im_end>"

def tokenizer_image_token(text, tokenizer, return_tensors=None):
    """テキスト内の画像トークンを処理し、Gemma3互換のトークナイズを行います"""
    
    # Gemma3の画像トークンID（設定されている場合）を取得
    image_token_id = None
    if hasattr(tokenizer, "additional_special_tokens"):
        for idx, token in enumerate(tokenizer.additional_special_tokens):
            if token in [DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN, "<start_of_image>"]:
                image_token_id = tokenizer.additional_special_tokens_ids[idx]
                print(f"画像トークン '{token}' のIDを検出: {image_token_id}")
                break
    
    # 画像トークンが見つからない場合の処理
    if image_token_id is None:
        # 回避策: <start_of_image>をボキャブラリに追加
        print("警告: Gemma3画像トークンが見つかりません。<start_of_image>トークンを追加します")
        image_token = "<start_of_image>"
        num_added = tokenizer.add_tokens([image_token], special_tokens=True)
        if num_added > 0:
            image_token_id = tokenizer.convert_tokens_to_ids(image_token)
            print(f"画像トークン '{image_token}' を追加しました。ID: {image_token_id}")
        else:
            # それでも失敗した場合はpad_tokenを使用
            image_token_id = tokenizer.pad_token_id
            print(f"警告: 画像トークンの追加に失敗しました。pad_token_id {image_token_id} を使用します")
    
    # 通常のテキストをトークナイズ
    token_ids = []
    
    # まず、テキスト全体をトークナイズ
    tokenized = tokenizer(text, return_tensors=return_tensors, add_special_tokens=False)
    raw_ids = tokenized.input_ids[0] if return_tensors else tokenized.input_ids
    
    # 画像トークンが元のテキストに含まれているか確認
    contains_image_token = DEFAULT_IMAGE_TOKEN in text or (
        DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN) in text
    
    if contains_image_token:
        # テキスト内の画像トークンを適切なGemma3画像トークンIDに置き換える
        # Gemma3は画像トークンの存在を検出するために、特定のIDが必要
        if return_tensors:
            # テンソル形式の場合は特殊トークンIDをテンソルに挿入
            modified_ids = []
            for id_tensor in raw_ids:
                # デフォルトの画像トークンをGemma3の画像トークンIDに置き換え
                if (DEFAULT_IMAGE_TOKEN in text) and (id_tensor == tokenizer.pad_token_id):
                    modified_ids.append(image_token_id)
                else:
                    modified_ids.append(id_tensor.item())
            token_ids = torch.tensor([modified_ids], dtype=torch.long)
        else:
            # 通常のIDリストを生成
            for i, text_chunk in enumerate(text.split(DEFAULT_IMAGE_TOKEN)):
                # テキストチャンクをトークナイズ
                if i > 0:
                    # 画像トークンを挿入
                    token_ids.append(image_token_id)
                
                # テキストチャンクのトークンを追加
                chunk_ids = tokenizer(text_chunk, add_special_tokens=False).input_ids
                token_ids.extend(chunk_ids)
    else:
        # 画像トークンが含まれていない場合は、通常通りトークナイズ
        token_ids = raw_ids
    
    return token_ids

def collate_fn(
    batch, tokenizer=None, conv_type="gemma_v1", use_mm_start_end=True, local_rank=-1
):
    """バッチを整形するcollate関数"""
    image_path_list = []
    images_list = []  # SAM用高解像度画像
    images_gemma_list = []  # Gemma3視覚モデル用画像
    conversation_list = []
    masks_list = []
    label_list = []
    resize_list = []
    questions_list = []
    sampled_classes_list = []
    offset_list = [0]
    cnt = 0
    inferences = []
    
    # バッチ内の各アイテムから値を取り出し適切なリストに追加
    for item in batch:
        # バッチアイテムの長さを確認して適切にアンパック
        if len(item) == 11:
            # 11個の値を返すデータセット用
            (
                image_path,
                images,
                images_gemma,
                conversations,
                masks,
                label,
                resize,
                questions,
                sampled_classes,
                inference,
                extra_value,  # 11番目の値は無視
            ) = item
        elif len(item) == 10:
            # 元のLISAと同じフォーマット (10個の値)
            (
                image_path,
                images,
                images_gemma,
                conversations,
                masks,
                label,
                resize,
                questions,
                sampled_classes,
                inference,
            ) = item
        elif len(item) == 9:
            # 9個の値だけ返す一部のデータセット用
            (
                image_path,
                images,
                images_gemma,
                conversations,
                masks,
                label,
                resize,
                questions,
                sampled_classes,
            ) = item
            inference = False  # デフォルト値
        else:
            print(f"警告: 予期しないアイテム形式です (長さ {len(item)})")
            continue
        
        # 会話リストが空または無効な場合はスキップ
        if not conversations or len(conversations) == 0:
            print(f"警告: 空の会話リストです。このバッチアイテムをスキップします。")
            continue
        
        # 各値を適切なリストに格納
        image_path_list.append(image_path)
        images_list.append(images)
        images_gemma_list.append(images_gemma)
        conversation_list.extend(conversations)
        label_list.append(label)
        masks_list.append(masks.float())
        resize_list.append(resize)
        questions_list.append(questions)
        sampled_classes_list.append(sampled_classes)
        cnt += len(conversations)
        offset_list.append(cnt)
        inferences.append(inference)
    
    # すべてのアイテムがスキップされた場合は空のバッチを返す
    if len(conversation_list) == 0:
        print("警告: 有効な会話データがないため、空のバッチを返します")
        # 最小限の辞書を返す
        return {
            "image_paths": [],
            "images": torch.zeros(0, 3, 1024, 1024),
            "pixel_values": torch.zeros(0, 3, 224, 224),
            "input_ids": torch.zeros(0, 1, dtype=torch.long),
            "labels": torch.zeros(0, 1, dtype=torch.long),
            "attention_masks": torch.zeros(0, 1, dtype=torch.bool),
            "masks_list": [],
            "label_list": [],
            "resize_list": [],
            "offset": torch.LongTensor([0, 0]),
            "questions_list": [],
            "sampled_classes_list": [],
            "inference": False,
            "conversation_list": [],
        }

    # 画像トークンの置き換え処理
    if use_mm_start_end:
        for i in range(len(conversation_list)):
            replace_token = DEFAULT_IMAGE_TOKEN
            replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            conversation_list[i] = conversation_list[i].replace(DEFAULT_IMAGE_TOKEN, replace_token)
    
    # テキストのトークナイズ
    input_ids = []
    
    # まず各会話をトークナイズして最大長を確認
    conversation_tokens = []
    
    if tokenizer is None:
        print("警告: トークナイザーがNoneです。処理をスキップします。")
        return {
            "image_paths": image_path_list,
            "images": torch.zeros(1, 3, 1024, 1024),
            "pixel_values": torch.zeros(1, 3, 224, 224),
            "input_ids": torch.zeros(1, 1, dtype=torch.long),
            "labels": torch.zeros(1, 1, dtype=torch.long),
            "attention_masks": torch.zeros(1, 1, dtype=torch.bool),
            "masks_list": masks_list,
            "label_list": label_list,
            "resize_list": resize_list,
            "offset": torch.LongTensor(offset_list),
            "questions_list": questions_list,
            "sampled_classes_list": sampled_classes_list,
            "inference": inferences[0] if inferences else False,
            "conversation_list": conversation_list,
        }
    
    # 各会話をトークン化
    for i, prompt in enumerate(conversation_list):
        try:
            # トークナイズ（テンソル変換なし）
            tokens = tokenizer_image_token(prompt, tokenizer, return_tensors=None)
            conversation_tokens.append(tokens)
        except Exception as e:
            print(f"トークナイズエラー（会話 {i}）: {e}, プロンプト: {prompt[:50]}...")
            # エラー時はデフォルトのトークンを使用
            conversation_tokens.append([tokenizer.bos_token_id] if tokenizer.bos_token_id else [0])
    
    # トークンが得られなかった場合のフォールバック
    if not conversation_tokens:
        print("警告: 有効なトークンが作成できませんでした。デフォルト値を使用します。")
        # 最小限の辞書を返す
        return {
            "image_paths": image_path_list,
            "images": torch.stack(images_list) if images_list else torch.zeros(1, 3, 1024, 1024),
            "pixel_values": torch.stack(images_gemma_list) if images_gemma_list else torch.zeros(1, 3, 224, 224),
            "input_ids": torch.zeros(1, 1, dtype=torch.long),
            "labels": torch.zeros(1, 1, dtype=torch.long),
            "attention_masks": torch.zeros(1, 1, dtype=torch.bool),
            "masks_list": masks_list,
            "label_list": label_list,
            "resize_list": resize_list,
            "offset": torch.LongTensor(offset_list),
            "questions_list": questions_list,
            "sampled_classes_list": sampled_classes_list,
            "inference": inferences[0] if inferences else False,
            "conversation_list": conversation_list,
        }
    
    # 各会話をPyTorchテンソルに変換
    max_length = max(len(tokens) for tokens in conversation_tokens)
    for tokens in conversation_tokens:
        # トークンをテンソルに変換
        token_tensor = torch.tensor(tokens, dtype=torch.long)
        input_ids.append(token_tensor)
    
    # パディングを適用して全てのシーケンスを同じ長さに
    try:
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
        )
    except RuntimeError as e:
        print(f"パディングエラー: {e}")
        # 手動でパディングを適用
        padded_ids = []
        for ids in input_ids:
            if len(ids) < max_length:
                padding = torch.full(
                    (max_length - len(ids),), 
                    tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0, 
                    dtype=torch.long
                )
                padded = torch.cat([ids, padding], dim=0)
            else:
                padded = ids[:max_length]  # 切り詰め
            padded_ids.append(padded)
        
        if padded_ids:
            input_ids = torch.stack(padded_ids, dim=0)
        else:
            # 空のリストの場合のフォールバック
            input_ids = torch.zeros(1, 1, dtype=torch.long)
    
    # この時点でinput_idsが有効かチェック
    if input_ids.numel() == 0:
        print("警告: input_idsが空です。デフォルト値を使用します。")
        input_ids = torch.zeros(1, 1, dtype=torch.long)
    
    attention_masks = input_ids.ne(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)

    # ターゲットラベルの作成
    targets = input_ids.clone()
    
    # ラベルの作成（教師強制用）
    # Gemma3の会話形式に合わせてセパレータとロールを設定
    if conv_type == "gemma_v1":
        sep = "\n\nAssistant: "
    else:
        sep = "[/INST] "
    
    for conversation, target in zip(conversation_list, targets):
        total_len = int(target.ne(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0).sum())
        
        # システムプロンプト部分とユーザークエリ部分はIGNORE_INDEXに設定
        # Gemma3の会話形式に応じて調整
        # Gemma3のプロンプト形式: "System: {system}\n\nUser: {query}\n\nAssistant: {response}"
        user_assistant_sep = sep
        system_user_sep = "\n\nUser: "
        
        try:
            if system_user_sep in conversation and user_assistant_sep in conversation:
                # システムプロンプトの開始からアシスタント応答の開始までをIGNORE_INDEXに
                assistant_start = conversation.find(user_assistant_sep) + len(user_assistant_sep)
                
                # トークン位置に変換
                assistant_token_start = len(tokenizer(conversation[:assistant_start], add_special_tokens=False).input_ids)
                
                # システム・ユーザー部分はIGNORE_INDEXに設定
                if assistant_token_start < len(target):
                    target[:assistant_token_start] = IGNORE_INDEX
        except Exception as e:
            print(f"ラベル処理エラー: {e}")
        
        # パディング部分もIGNORE_INDEXに設定
        target[total_len:] = IGNORE_INDEX
    
    # 長いシーケンスのトランケーション
    if all(not inf for inf in inferences):
        truncate_len = tokenizer.model_max_length - 255  # 画像トークン用の余裕

        if input_ids.shape[1] > truncate_len:
            input_ids = input_ids[:, :truncate_len]
            targets = targets[:, :truncate_len]
            attention_masks = attention_masks[:, :truncate_len]

    # SAM用画像とGemma用画像の形状を確認し整形
    valid_sam_images = []
    valid_gemma_images = []
    
    for sam_img, gemma_img in zip(images_list, images_gemma_list):
        # 両方のテンソルが有効であることを確認
        if sam_img is not None and isinstance(sam_img, torch.Tensor) and sam_img.dim() == 3:
            if gemma_img is not None and isinstance(gemma_img, torch.Tensor) and gemma_img.dim() == 3:
                valid_sam_images.append(sam_img)
                valid_gemma_images.append(gemma_img)
            else:
                # Gemma画像が無効な場合はゼロテンソルで置き換え
                print(f"警告: 無効なGemma画像形状: {type(gemma_img)}")
                valid_sam_images.append(sam_img)
                valid_gemma_images.append(torch.zeros(3, 224, 224))
        else:
            # どちらも無効な場合はスキップ
            print(f"警告: 無効な画像ペア、このサンプルをスキップします")
    
    # 有効な画像がない場合は、デフォルトのゼロテンソルを作成
    if not valid_sam_images:
        valid_sam_images = [torch.zeros(3, 1024, 1024)]
        valid_gemma_images = [torch.zeros(3, 224, 224)]
    
    # 有効な画像のみでバッチを構成
    try:
        images_tensor = torch.stack(valid_sam_images, dim=0)
        pixel_values_tensor = torch.stack(valid_gemma_images, dim=0)
    except RuntimeError as e:
        print(f"画像スタックエラー: {e}")
        # サイズが不一致の場合、最初の有効なサイズに合わせる
        if valid_sam_images and valid_gemma_images:
            target_sam_shape = valid_sam_images[0].shape
            target_gemma_shape = valid_gemma_images[0].shape
            
            # すべての画像を同じサイズに調整
            adjusted_sam_images = []
            adjusted_gemma_images = []
            
            for sam_img, gemma_img in zip(valid_sam_images, valid_gemma_images):
                if sam_img.shape != target_sam_shape:
                    # リサイズまたはパディングで調整
                    sam_img = F.interpolate(sam_img.unsqueeze(0), size=target_sam_shape[1:], mode='bilinear').squeeze(0)
                adjusted_sam_images.append(sam_img)
                
                if gemma_img.shape != target_gemma_shape:
                    gemma_img = F.interpolate(gemma_img.unsqueeze(0), size=target_gemma_shape[1:], mode='bilinear').squeeze(0)
                adjusted_gemma_images.append(gemma_img)
            
            images_tensor = torch.stack(adjusted_sam_images, dim=0)
            pixel_values_tensor = torch.stack(adjusted_gemma_images, dim=0)
        else:
            # 最終手段: デフォルト値で埋める
            images_tensor = torch.zeros(len(image_path_list), 3, 1024, 1024)
            pixel_values_tensor = torch.zeros(len(image_path_list), 3, 224, 224)

    return {
        "image_paths": image_path_list,
        "images": images_tensor,
        "pixel_values": pixel_values_tensor,  # Gemma3用の画像テンソル
        "input_ids": input_ids,
        "labels": targets,
        "attention_masks": attention_masks,
        "masks_list": masks_list,
        "label_list": label_list,
        "resize_list": resize_list,
        "offset": torch.LongTensor(offset_list),
        "questions_list": questions_list,
        "sampled_classes_list": sampled_classes_list,
        "inference": inferences[0] if inferences else False,
        "conversation_list": conversation_list,
    }

# サンプルレートの処理
def parse_sample_rates(sample_rates_str, dataset_list):
    """
    サンプルレートをパースして正規化した配列を返す
    
    Args:
        sample_rates_str: カンマ区切りのサンプルレート文字列、またはリスト
        dataset_list: データセットのリスト
        
    Returns:
        正規化されたサンプルレート配列
    """
    # データセットが空の場合はエラー
    if not dataset_list or len(dataset_list) == 0:
        print("警告: データセットリストが空です。デフォルトのサンプルレートを返します。")
        return [1.0]
    
    # 入力がNoneの場合は均等なサンプルレートを返す
    if sample_rates_str is None:
        sample_rates = [1.0 / len(dataset_list)] * len(dataset_list)
        print("サンプルレートが指定されていません。均等なサンプルレートを使用します。")
        return sample_rates
    
    # 文字列の場合は区切り文字で分割
    if isinstance(sample_rates_str, str):
        try:
            if "||" in sample_rates_str:
                sample_rates = [float(rate) for rate in sample_rates_str.split("||")]
            elif "," in sample_rates_str:
                sample_rates = [float(rate) for rate in sample_rates_str.split(",")]
            else:
                # 単一値の場合
                sample_rates = [float(sample_rates_str)]
        except ValueError as e:
            print(f"警告: サンプルレート '{sample_rates_str}' を解析できませんでした: {e}")
            print("均等なサンプルレートを使用します。")
            return [1.0 / len(dataset_list)] * len(dataset_list)
    else:
        # 既にリストまたは配列の場合
        try:
            sample_rates = [float(rate) for rate in sample_rates_str]
        except (TypeError, ValueError) as e:
            print(f"警告: サンプルレート {sample_rates_str} をリストに変換できませんでした: {e}")
            print("均等なサンプルレートを使用します。")
            return [1.0 / len(dataset_list)] * len(dataset_list)
    
    # サンプルレート数とデータセット数が一致するか確認
    if len(sample_rates) != len(dataset_list):
        print(f"警告: サンプルレート数 ({len(sample_rates)}) がデータセット数 ({len(dataset_list)}) と一致しません。")
        print(f"データセット: {dataset_list}")
        print(f"サンプルレート: {sample_rates}")
        
        if len(sample_rates) > len(dataset_list):
            # 余分なレートは削除
            sample_rates = sample_rates[:len(dataset_list)]
            print(f"余分なサンプルレートを削除しました: {sample_rates}")
        else:
            # 足りないレートは均等に追加
            additional_rates = [sample_rates[-1] if sample_rates else 1.0] * (len(dataset_list) - len(sample_rates))
            sample_rates.extend(additional_rates)
            print(f"不足しているサンプルレートを追加しました: {sample_rates}")
    
    # サンプルレートの正規化
    sample_rates = np.array(sample_rates)
    # すべてのレートが0の場合は均等に設定
    if np.sum(sample_rates) == 0:
        sample_rates = np.ones_like(sample_rates) / len(sample_rates)
        print("すべてのサンプルレートが0でした。均等なレートを使用します。")
    else:
        sample_rates = sample_rates / np.sum(sample_rates)
    
    # デバッグ情報の表示
    print(f"データセット数: {len(dataset_list)}")
    print(f"正規化後のサンプルレート: {sample_rates}")
    
    return sample_rates

class HybridDataset(torch.utils.data.Dataset):
    """Gemma3モデル用のハイブリッドデータセット"""
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        args=None,  # argsオブジェクトまたは個別のパラメータで設定可能
        base_image_dir=None,
        tokenizer=None,
        model_name=None,  # Gemma3モデル名を指定
        vision_tower=None,  # 後方互換性のため
        sam_checkpoint=None,  # SAMモデルのチェックポイントパス
        samples_per_epoch=500 * 8 * 2 * 10,
        precision: str = "fp32",
        image_size: int = 224,
        num_classes_per_sample: int = 3,
        exclude_val=False,
        dataset="sem_seg||refer_seg||vqa||reason_seg",
        sample_rate=[9, 3, 3, 1],
        sample_rates=None,  # オリジナルLISAとの互換性のために追加
        sem_seg_data="ade20k||cocostuff||partimagenet||pascal_part||paco_lvis||mapillary",
        refer_seg_data="refclef||refcoco||refcoco+||refcocog",
        vqa_data="llava_instruct_150k",
        reason_seg_data="ReasonSeg|train",
        explanatory=0.1,
        debug_mode=False,  # デバッグモードフラグ
        debug_samples=10,  # デバッグモードで使用するサンプル数
        transform=None,  # カスタム変換を許可
        train_transform=None,  # 訓練用変換
        seg_token_idx=None,  # [SEG]トークンのインデックス
        image_token_idx=None,  # 画像トークンのインデックス
        debug=None,  # debug_modeの別名
    ):
        # argsオブジェクトが渡された場合はそこから値を抽出
        if args is not None:
            base_image_dir = getattr(args, "dataset_dir", base_image_dir)
            tokenizer = None  # argsから取得しない
            precision = getattr(args, "precision", precision)
            image_size = getattr(args, "image_size", image_size)
            num_classes_per_sample = getattr(args, "num_classes_per_sample", num_classes_per_sample)
            exclude_val = getattr(args, "exclude_val", exclude_val)
            dataset = getattr(args, "dataset", dataset)
            sample_rates = getattr(args, "sample_rates", sample_rates)
            sem_seg_data = getattr(args, "sem_seg_data", sem_seg_data)
            refer_seg_data = getattr(args, "refer_seg_data", refer_seg_data)
            vqa_data = getattr(args, "vqa_data", vqa_data)
            reason_seg_data = getattr(args, "reason_seg_data", reason_seg_data)
            explanatory = getattr(args, "explanatory", explanatory)
            debug_mode = getattr(args, "debug", debug_mode)
            debug_samples = getattr(args, "debug_samples", debug_samples)
            # steps_per_epochがあれば、samples_per_epochを計算する
            if hasattr(args, "steps_per_epoch") and hasattr(args, "batch_size") and hasattr(args, "grad_accumulation_steps"):
                samples_per_epoch = args.steps_per_epoch * args.batch_size * args.grad_accumulation_steps
            
        # パラメータの検証と処理
        self.base_image_dir = base_image_dir
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.vision_tower = vision_tower
        self.sam_checkpoint = sam_checkpoint
        self.precision = precision
        self.samples_per_epoch = samples_per_epoch
        self.image_size = image_size
        self.num_classes_per_sample = num_classes_per_sample
        self.exclude_val = exclude_val
        self.debug_mode = debug or debug_mode  # どちらかがTrueならデバッグモード
        self.debug_samples = debug_samples
        self.seg_token_idx = seg_token_idx
        self.image_token_idx = image_token_idx
        self.explanatory = explanatory
        
        if transform is not None:
            self.transform = transform
        elif train_transform is not None:
            self.transform = train_transform
        else:
            # デフォルトの変換を設定
            self.transform = None
            
        # データセットの分割とサンプルレートの処理
        # カンマ区切りとパイプ区切りの両方に対応し、内部では常に`||`形式に変換
        if isinstance(dataset, str):
            if "||" in dataset:
                self.dataset = dataset.split("||")
            elif "," in dataset:
                self.dataset = dataset.split(",")
            else:
                self.dataset = [dataset] if dataset else []
        elif isinstance(dataset, list):
            # 既にリストの場合はそのまま使用
            self.dataset = dataset
        else:
            # それ以外の場合は空リスト
            print(f"警告: 不明なデータセット型 {type(dataset)}。空リストを使用します。")
            self.dataset = []
            
        # データセットリストが空の場合はエラー
        if not self.dataset:
            raise ValueError("データセットリストが空です。少なくとも1つのデータセットタイプを指定してください。")
            
        # サンプルレートの処理
        if sample_rates is not None:
            self.sample_rates = parse_sample_rates(sample_rates, self.dataset)
        else:
            self.sample_rates = parse_sample_rates(sample_rate, self.dataset)
        
        # データセット名を統一（,区切りを||区切りに変換）
        sem_seg_data_normalized = "||".join(sem_seg_data.split(",")) if isinstance(sem_seg_data, str) and "," in sem_seg_data else sem_seg_data
        refer_seg_data_normalized = "||".join(refer_seg_data.split(",")) if isinstance(refer_seg_data, str) and "," in refer_seg_data else refer_seg_data
        vqa_data_normalized = "||".join(vqa_data.split(",")) if isinstance(vqa_data, str) and "," in vqa_data else vqa_data
        reason_seg_data_normalized = "||".join(reason_seg_data.split(",")) if isinstance(reason_seg_data, str) and "," in reason_seg_data else reason_seg_data
        
        # デバッグ情報：初期化されるデータセットタイプを表示
        print(f"初期化されるデータセットタイプ: {self.dataset}")
        
        # Gemma3のプロセッサを初期化
        self.processor = None
        self.image_processor = None
        
        # モデル名からプロセッサを取得
        from transformers import AutoProcessor
        
        try:
            # 明示的にモデル名を指定（Noneの場合はデフォルト値を使用）
            if model_name is None:
                processor_model_name = "google/gemma-3-4b-it"
                print(f"警告: model_nameがNoneです。デフォルト値 '{processor_model_name}' を使用します")
            else:
                processor_model_name = "google/gemma-3-4b-it" if "gemma-3" in model_name else model_name
            
            self.processor = AutoProcessor.from_pretrained(
                processor_model_name, 
                trust_remote_code=True
            )
            print(f"Gemma3プロセッサを初期化: {processor_model_name}")
            
            # Gemma3の画像処理ユーティリティを使用
            from model.gemma3.mm_utils import GemmaImageProcessor
            self.image_processor = GemmaImageProcessor(self.processor)
            print("GemmaImageProcessorの初期化に成功しました")
                
        except Exception as e:
            # プロセッサ初期化エラーは明示的に報告して終了
            error_msg = f"プロセッサの初期化に失敗しました: {e}"
            print(error_msg)
            print("Hint: Hugging Faceにログインしていることを確認してください。")
            
            raise RuntimeError(error_msg) from e
        
        # SAM用の画像処理
        self.sam_transform = ResizeLongestSide(self.img_size)

        self.all_datasets = []
        loaded_datasets = []
        
        # 各データセットタイプの初期化
        for dataset_type in self.dataset:
            try:
                if dataset_type == "sem_seg":
                    print(f"セマンティックセグメンテーションデータセット '{dataset_type}' を初期化中...")
                    self.all_datasets.append(
                        SemSegDataset(
                            base_image_dir=self.base_image_dir,
                            tokenizer=self.tokenizer,
                            model_name=self.model_name,  # vision_towerの代わりにmodel_nameを渡す
                            samples_per_epoch=self.samples_per_epoch,
                            precision=self.precision,
                            image_size=self.image_size,
                            num_classes_per_sample=self.num_classes_per_sample,
                            exclude_val=self.exclude_val,
                            sem_seg_data=sem_seg_data_normalized,
                            debug_mode=self.debug_mode,  # デバッグモードのフラグを渡す
                            debug_samples=self.debug_samples,  # デバッグサンプル数を渡す
                            processor=self.processor,  # 共通のプロセッサを渡す
                            image_processor=self.image_processor,  # 画像プロセッサを渡す
                        )
                    )
                    loaded_datasets.append(dataset_type)
                    print(f"セマンティックセグメンテーションデータセット '{dataset_type}' の初期化に成功しました")
                elif dataset_type == "refer_seg":
                    print(f"参照セグメンテーションデータセット '{dataset_type}' を初期化中...")
                    try:
                        refer_dataset = ReferSegDataset(
                            base_image_dir=self.base_image_dir,
                            tokenizer=self.tokenizer,
                            model_name=self.model_name,
                            samples_per_epoch=self.samples_per_epoch,
                            precision=self.precision,
                            image_size=self.image_size,
                            num_classes_per_sample=self.num_classes_per_sample,
                            exclude_val=self.exclude_val,
                            refer_seg_data=refer_seg_data_normalized,
                            processor=self.processor,  # 共通のプロセッサを渡す
                            image_processor=self.image_processor,  # 画像プロセッサを渡す
                        )
                        self.all_datasets.append(refer_dataset)
                        loaded_datasets.append(dataset_type)
                        print(f"参照セグメンテーションデータセット '{dataset_type}' の初期化に成功しました")
                    except Exception as e:
                        print(f"参照セグメンテーションデータセットの初期化中にエラーが発生しました: {e}")
                        # エラーを再発生させる - エラーハンドリングを無効化
                        raise
                elif dataset_type == "vqa":
                    print(f"VQAデータセット '{dataset_type}' を初期化中...")
                    try:
                        vqa_dataset = VQADataset(
                            base_image_dir=self.base_image_dir,
                            tokenizer=self.tokenizer,
                            model_name=self.model_name,
                            samples_per_epoch=self.samples_per_epoch,
                            precision=self.precision,
                            image_size=self.image_size,
                            num_classes_per_sample=self.num_classes_per_sample,
                            exclude_val=self.exclude_val,
                            vqa_data=vqa_data_normalized,
                            processor=self.processor,  # 共通のプロセッサを渡す
                            image_processor=self.image_processor,  # 画像プロセッサを渡す
                        )
                        self.all_datasets.append(vqa_dataset)
                        loaded_datasets.append(dataset_type)
                        print(f"VQAデータセット '{dataset_type}' の初期化に成功しました")
                    except Exception as e:
                        print(f"VQAデータセットの初期化中にエラーが発生しました: {e}")
                        # エラーを再発生させる - エラーハンドリングを無効化
                        raise
                elif dataset_type == "reason_seg":
                    print(f"理由付きセグメンテーションデータセット '{dataset_type}' を初期化中...")
                    try:
                        reason_dataset = ReasonSegDataset(
                            base_image_dir=self.base_image_dir,
                            tokenizer=self.tokenizer,
                            model_name=self.model_name,
                            samples_per_epoch=self.samples_per_epoch,
                            precision=self.precision,
                            image_size=self.image_size,
                            num_classes_per_sample=self.num_classes_per_sample,
                            exclude_val=self.exclude_val,
                            reason_seg_data=reason_seg_data_normalized,
                            explanatory=self.explanatory,
                            processor=self.processor,  # 共通のプロセッサを渡す
                            image_processor=self.image_processor,  # 画像プロセッサを渡す
                        )
                        self.all_datasets.append(reason_dataset)
                        loaded_datasets.append(dataset_type)
                        print(f"理由付きセグメンテーションデータセット '{dataset_type}' の初期化に成功しました")
                    except Exception as e:
                        print(f"理由付きセグメンテーションデータセットの初期化中にエラーが発生しました: {e}")
                        # エラーを再発生させる - エラーハンドリングを無効化
                        raise
                else:
                    print(f"警告: 不明なデータセットタイプ '{dataset_type}' がスキップされました")
            except Exception as e:
                import traceback
                error_trace = traceback.format_exc()
                print(f"エラー: データセット '{dataset_type}' の初期化中に問題が発生しました:")
                print(f"{str(e)}")
                print(f"詳細なエラートレース:")
                print(error_trace)
                # エラーを再発生させる - エラーハンドリングを無効化
                raise RuntimeError(f"データセット '{dataset_type}' の初期化中にエラーが発生しました: {e}") from e
        
        # 初期化に成功したデータセットを報告
        print(f"初期化に成功したデータセット: {loaded_datasets}")
        
        # データセットが1つも読み込めなかった場合はエラーを発生
        if not self.all_datasets:
            raise RuntimeError(f"すべてのデータセットの初期化に失敗しました。")
            
        # 読み込まれたデータセットが指定されたデータセットと異なる場合は警告
        if len(loaded_datasets) != len(self.dataset):
            missing = set(self.dataset) - set(loaded_datasets)
            print(f"警告: 以下のデータセットが読み込まれませんでした: {missing}")
            
            # データセットリストと対応するサンプルレートを更新
            self.dataset = loaded_datasets
            self.sample_rates = parse_sample_rates(None, loaded_datasets)
            print(f"サンプルレートを調整しました: {self.sample_rates}")

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        # ランダムにサブデータセットを選択
        ind = np.random.choice(list(range(len(self.dataset))), p=self.sample_rates)
        data = self.all_datasets[ind]
        
        # データセットからデータを取得
        batch_data = data[0]
        
        # 画像変換処理
        if len(batch_data) >= 2:
            original_image = batch_data[1]  # 画像データは通常2番目の要素
            
            # 1. SAM用の高解像度画像変換 (常に適用)
            if isinstance(original_image, torch.Tensor):
                # テンソルをnumpyに変換 (C,H,W) -> (H,W,C)
                # 勾配計算が必要なテンソルでも安全に変換できるようにdetach()を追加
                image_numpy = original_image.permute(1, 2, 0).detach().numpy()
                
                # SAM変換を適用
                sam_image = self.sam_transform.apply_image(image_numpy)
                sam_tensor = torch.from_numpy(sam_image).permute(2, 0, 1).contiguous()
                
                # 2. Gemma3用の変換 (設定されている場合のみ適用)
                if self.transform is not None:
                    # Gemma用の変換を適用
                    gemma_tensor = self.transform(image_numpy)
                    
                    # バッチデータの更新 (SAM用とGemma用の両方を設定)
                    batch_data = list(batch_data)
                    batch_data[1] = sam_tensor  # SAM用
                    # 3番目の要素がGemma用の画像データであると仮定
                    if len(batch_data) > 2:
                        batch_data[2] = gemma_tensor  # Gemma用
                    batch_data = tuple(batch_data)
        
        inference = False
        
        # バッチデータとinferenceフラグを返す
        if isinstance(batch_data, tuple):
            return *batch_data, inference
        else:
            return batch_data, inference

    def __getstate__(self):
        """ピックル化のためのオブジェクト状態を返す"""
        state = self.__dict__.copy()
        
        # ピックル化できない可能性のあるオブジェクトを除外
        state['processor'] = None
        state['image_processor'] = None
        
        # トークナイザーの設定は保持するが、オブジェクト自体は除外
        if hasattr(self, 'tokenizer') and self.tokenizer is not None:
            # トークナイザの設定情報だけ保存
            state['tokenizer_class'] = self.tokenizer.__class__.__name__
            state['tokenizer_name'] = self.tokenizer.name_or_path if hasattr(self.tokenizer, 'name_or_path') else None
            state['tokenizer'] = None
        
        # 他のピックル化できない可能性のあるオブジェクトを処理
        for key in list(state.keys()):
            if 'transform' in key and state[key] is not None:
                try:
                    # テスト的にピックル化を試みる
                    pickle.dumps(state[key])
                except:
                    # ピックル化できない場合は除外
                    state[key] = None
                    print(f"警告: {key}はピックル化できないため、キャッシュから除外します")
        
        return state
    
    def __setstate__(self, state):
        """ピックル化されたオブジェクト状態を復元する"""
        self.__dict__.update(state)
        
        # トークナイザーを再作成
        if hasattr(self, 'tokenizer_name') and self.tokenizer_name:
            from transformers import AutoTokenizer
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
                print(f"トークナイザーを再作成しました: {self.tokenizer_name}")
            except Exception as e:
                print(f"トークナイザーの再作成に失敗しました: {e}")
                self.tokenizer = None
        
        # その他の必要なオブジェクトを再初期化
        # SAM変換の再初期化
        from model.segment_anything.utils.transforms import ResizeLongestSide
        self.sam_transform = ResizeLongestSide(self.img_size)
        
        # プロセッサとイメージプロセッサの再初期化が必要な場合は追加


class ValDataset(torch.utils.data.Dataset):
    """評価用データセット"""
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        model_name,  # Gemma3モデル名
        val_dataset,
        image_size=1024,
        transform=None,  # カスタム変換を許可
    ):
        self.base_image_dir = base_image_dir
        self.transform = transform  # Gemma3用のカスタム変換を保存
        self.sam_transform = ResizeLongestSide(image_size)  # SAM用の変換
        self.image_processor = None  # 互換性のために追加
        self.image_size = image_size  # 画像サイズを保存
        self.tokenizer = tokenizer
        
        # val_datasetが文字列であることを確認
        if not isinstance(val_dataset, str):
            raise ValueError(f"val_datasetは文字列でなければなりません。現在の型: {type(val_dataset)}")
        
        splits = val_dataset.split("|")
        if len(splits) == 2:
            ds, split = splits
            images = glob.glob(
                os.path.join(self.base_image_dir, "reason_seg", ds, split, "*.jpg")
            )
            self.images = images
            self.data_type = "reason_seg"
        elif len(splits) == 3:
            ds, splitBy, split = splits
            refer_api = REFER(self.base_image_dir, ds, splitBy)
            ref_ids_val = refer_api.getRefIds(split=split)
            images_ids_val = refer_api.getImgIds(ref_ids=ref_ids_val)
            refs_val = refer_api.loadRefs(ref_ids=ref_ids_val)
            refer_seg_ds = {}
            refer_seg_ds["images"] = []
            loaded_images = refer_api.loadImgs(image_ids=images_ids_val)
            for item in loaded_images:
                item = item.copy()
                if ds == "refclef":
                    item["file_name"] = os.path.join(
                        base_image_dir, "images/saiapr_tc-12", item["file_name"]
                    )
                elif ds in ["refcoco", "refcoco+", "refcocog", "grefcoco"]:
                    item["file_name"] = os.path.join(
                        base_image_dir,
                        "images/mscoco/images/train2014",
                        item["file_name"],
                    )
                refer_seg_ds["images"].append(item)
            refer_seg_ds["annotations"] = refer_api.Anns  # anns_val

            img2refs = {}
            for ref in refs_val:
                image_id = ref["image_id"]
                img2refs[image_id] = img2refs.get(image_id, []) + [
                    ref,
                ]
            refer_seg_ds["img2refs"] = img2refs
            self.refer_seg_ds = refer_seg_ds
            self.data_type = "refer_seg"

        self.ds = ds
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.sam_transform = ResizeLongestSide(image_size)  # SAM用の変換
        # self.transformはGemma3用のカスタム変換として維持
        
        # Gemma3のプロセッサを初期化
        try:
            # モデル名からプロセッサを取得
            from transformers import AutoProcessor
            
            # 明示的にモデル名を指定（Noneの場合はデフォルト値を使用）
            if model_name is None:
                processor_model_name = "google/gemma-3-4b-it"
                print(f"警告: 検証用model_nameがNoneです。デフォルト値 '{processor_model_name}' を使用します")
            else:
                processor_model_name = "google/gemma-3-4b-it" if "gemma-3" in model_name else model_name
            
            self.processor = AutoProcessor.from_pretrained(
                processor_model_name, 
                trust_remote_code=True
            )
            print(f"検証用Gemma3プロセッサを初期化: {processor_model_name}")
            
            # Gemma3の画像処理ユーティリティを使用
            from model.gemma3.mm_utils import GemmaImageProcessor
            self.image_processor = GemmaImageProcessor(self.processor)
                
        except Exception as e:
            # フォールバック処理を削除し、エラーを再発生させる
            error_msg = f"検証用プロセッサの初期化に失敗しました: {e}"
            print(error_msg)
            print("Hint: Hugging Faceにログインしていることを確認してください。")
            
            raise RuntimeError(error_msg) from e

    def __len__(self):
        if self.data_type == "refer_seg":
            return len(self.refer_seg_ds["images"])
        else:
            return len(self.images)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """SAM用にピクセル値を正規化しパディングする"""
        # 正規化
        x = (x - self.pixel_mean) / self.pixel_std

        # パディング
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def __getitem__(self, idx):
        """データセットから要素を取得"""
        if self.data_type == "refer_seg":
            refer_seg_ds = self.refer_seg_ds
            images = refer_seg_ds["images"]
            annotations = refer_seg_ds["annotations"]
            img2refs = refer_seg_ds["img2refs"]

            image_info = images[idx]
            image_path = image_info["file_name"]
            image_id = image_info["id"]

            refs = img2refs[image_id]
            if len(refs) == 0:
                raise ValueError(f"画像 {image_id} に対する参照がありません")

            sents = []
            ann_ids = []
            for ref in refs:
                for sent in ref["sentences"]:
                    sents.append(sent["sent"].strip().lower())
                    ann_ids.append(ref["ann_id"])

            sampled_sents = sents
            sampled_ann_ids = ann_ids
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            is_sentence = False
        else:
            image_path = self.images[idx]
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            json_path = image_path.replace(".jpg", ".json")
            mask_json, sampled_sents, is_sentence = get_mask_from_json(json_path, image)
            sampled_sents = [sampled_sents[0]]

        # 画像の前処理
        # 1. SAM用の高解像度画像変換
        image_sam_np = self.sam_transform.apply_image(image)
        image_sam = self.preprocess(torch.from_numpy(image_sam_np).permute(2, 0, 1).contiguous())
        
        # 2. Gemma3用の画像変換
        if self.transform is not None:
            # Gemma3用のカスタム変換がある場合はそれを使用
            images_gemma = self.transform(image)
            print(f"Gemma3用カスタム変換後の画像サイズ: {images_gemma.shape}")
        else:
            # 標準的なリサイズと正規化
            h, w = image.shape[:2]
            size = 896  # Gemma3の推奨サイズ
            image_gemma = cv2.resize(image, (size, size), interpolation=cv2.INTER_CUBIC)
            image_gemma = torch.from_numpy(image_gemma).permute(2, 0, 1).float() / 255.0
            # 標準的な正規化値を適用
            mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(-1, 1, 1)
            std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(-1, 1, 1)
            images_gemma = (image_gemma - mean) / std
            
        # Gemma3用の会話形式を作成
        conversations = []
        template = get_default_conv_template("gemma_v1")  # Gemma3用テンプレート
        
        i = 0
        while i < len(sampled_sents):
            template.messages = []
            text = sampled_sents[i].strip()
            
            # システムプロンプトを設定
            template.system = SYSTEM_PROMPT
            
            if is_sentence:
                # 文章ベースのセグメンテーション
                query = f"{DEFAULT_IMAGE_TOKEN}\n{text} Please output segmentation mask."
                template.append_message(template.roles[0], query)
                template.append_message(template.roles[1], "[SEG].")
            else:
                # 物体ベースのセグメンテーション
                query = f"{DEFAULT_IMAGE_TOKEN}\nWhat is {text} in this image? Please output segmentation mask."
                template.append_message(template.roles[0], query)
                template.append_message(template.roles[1], "[SEG].")
            
            conversations.append(template.get_prompt())
            i += 1

        # SAM用の高解像度画像処理（常に従来の処理方法を使用）
        # SAMモデルには一定の入力形式が必要なため、カスタム変換は適用しない
        image_sam = self.preprocess(torch.from_numpy(image_sam_np).permute(2, 0, 1).contiguous())
        resize = image.shape[:2]  # 元画像のサイズ情報を保持

        # マスクの処理
        if self.data_type == "refer_seg":
            masks = []
            for i, ann_id in enumerate(sampled_ann_ids):
                ann = annotations[ann_id]
                if len(ann["segmentation"]) == 0 and sampled_sents[i] != "":
                    m = np.zeros((image_info["height"], image_info["width"], 1))
                else:
                    if type(ann["segmentation"][0]) == list:  # polygon
                        rle = mask.frPyObjects(
                            ann["segmentation"],
                            image_info["height"],
                            image_info["width"],
                        )
                    else:
                        rle = ann["segmentation"]
                        for i in range(len(rle)):
                            if not isinstance(rle[i]["counts"], bytes):
                                rle[i]["counts"] = rle[i]["counts"].encode()
                    m = mask.decode(rle)
                m = np.sum(m, axis=2)  # 複数のバイナリマップを合成
                m = m.astype(np.uint8)  # np.uint8に変換
                masks.append(m)
        else:
            masks = [mask_json]

        masks = np.stack(masks, axis=0)
        masks = torch.from_numpy(masks)
        labels = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label
        inference = True

        return (
            image_path,
            image_sam,
            images_gemma,
            conversations,
            masks,
            labels,
            resize,
            None,
            None,
            inference,
        )
