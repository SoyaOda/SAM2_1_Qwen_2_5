# utils/data_processing.py
import glob
import json
import os

import cv2
import numpy as np
import torch
from typing import Dict, List, Any

class DataCollatorForSupervisedDataset:
    """
    Llama-4多モーダルモデル対応のデータコレーター
    バッチ処理とパディングを担当
    """
    def __init__(self, tokenizer, pad_to_multiple_of=None):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
        
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        バッチデータの処理
        """
        # 入力データの抽出
        input_ids_list = []
        attention_mask_list = []
        labels_list = []
        images_sam_list = []
        images_llama_list = []
        masks_list = []
        
        for item in batch:
            if 'input_ids' in item:
                input_ids_list.append(item['input_ids'])
            if 'attention_mask' in item:
                attention_mask_list.append(item['attention_mask'])
            if 'labels' in item:
                labels_list.append(item['labels'])
            if 'images_for_sam' in item:
                images_sam_list.append(item['images_for_sam'])
            if 'images_for_llama' in item:
                images_llama_list.append(item['images_for_llama'])
            if 'ground_truth_mask' in item:
                masks_list.append(item['ground_truth_mask'])
        
        # パディング処理
        batch_output = {}
        
        # テキストのパディング
        if input_ids_list:
            padded = self.tokenizer.pad(
                {'input_ids': input_ids_list},
                padding=True,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors='pt'
            )
            batch_output['input_ids'] = padded['input_ids']
            
            # attention_maskの処理
            if attention_mask_list:
                batch_output['attention_mask'] = padded.get('attention_mask')
            
            # labelsのパディング（IGNORE_INDEX=-100）
            if labels_list:
                max_len = batch_output['input_ids'].size(1)
                padded_labels = []
                for labels in labels_list:
                    if len(labels) < max_len:
                        # -100でパディング
                        padded = torch.cat([
                            labels,
                            torch.full((max_len - len(labels),), -100, dtype=labels.dtype)
                        ])
                        padded_labels.append(padded)
                    else:
                        padded_labels.append(labels[:max_len])
                batch_output['labels'] = torch.stack(padded_labels)
        
        # 画像のスタック
        if images_sam_list:
            batch_output['images_for_sam'] = torch.stack(images_sam_list)
        if images_llama_list:
            batch_output['images_for_llama'] = torch.stack(images_llama_list)
        if masks_list:
            batch_output['masks'] = torch.stack(masks_list)
        
        return batch_output

def get_mask_from_json(json_path, img):
    """
    JSONファイルからマスクを生成する関数
    Original-LISA-Codeから移植
    """
    try:
        with open(json_path, "r") as r:
            anno = json.loads(r.read())
    except:
        with open(json_path, "r", encoding="cp1252") as r:
            anno = json.loads(r.read())

    inform = anno["shapes"]
    comments = anno["text"]
    is_sentence = anno["is_sentence"]

    height, width = img.shape[:2]

    ### sort polies by area
    area_list = []
    valid_poly_list = []
    for i in inform:
        label_id = i["label"]
        points = i["points"]
        if "flag" == label_id.lower():  ## meaningless deprecated annotations
            continue

        tmp_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.polylines(tmp_mask, np.array([points], dtype=np.int32), True, 1, 1)
        cv2.fillPoly(tmp_mask, np.array([points], dtype=np.int32), 1)
        tmp_area = tmp_mask.sum()

        area_list.append(tmp_area)
        valid_poly_list.append(i)

    ### ground-truth mask
    sort_index = np.argsort(area_list)[::-1].astype(np.int32)
    sort_index = list(sort_index)
    sort_inform = []
    for s_idx in sort_index:
        sort_inform.append(valid_poly_list[s_idx])

    mask = np.zeros((height, width), dtype=np.uint8)
    for i in sort_inform:
        label_id = i["label"]
        points = i["points"]

        if "ignore" in label_id.lower():
            label_value = 255  # ignored during evaluation
        else:
            label_value = 1  # target

        cv2.polylines(mask, np.array([points], dtype=np.int32), True, label_value, 1)
        cv2.fillPoly(mask, np.array([points], dtype=np.int32), label_value)

    return mask, comments, is_sentence