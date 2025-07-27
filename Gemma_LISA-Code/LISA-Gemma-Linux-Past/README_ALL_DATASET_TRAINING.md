# LISA-Gemma3 全データセット順次学習

## 概要

このスクリプト（`train_all_datasets.py`）は、LISA（Gemma3とSAM）モデルをすべての利用可能なデータセットを使用して順次トレーニングするためのものです。デバッグモードでは少量のサンプルを使用して各データセットでの学習が正常に行えるかを確認できます。

対応するデータセットのタイプは以下の通りです:
- セマンティックセグメンテーション (sem_seg)
- 参照セグメンテーション (refer_seg)
- 視覚質問応答 (vqa)
- 理由付きセグメンテーション (reason_seg)

## 必要条件

- Python 3.8以上
- PyTorch
- transformers
- LISA-Gemma3の必要なパッケージ

## データセット構造

スクリプトは以下のデータセット構造を前提としています：

```
H:/download/LISA-dataset/dataset/
├── ade20k/
├── cocostuff/
├── mapillary/
├── refer_seg/
│   └── images/
│       ├── mscoco/
│       │   └── images/
│       ├── saiapr_tc-12/
│       └── saiapr_tc-12-sub/
├── llava_dataset/
├── reason_seg/
│   └── ReasonSeg/
│       ├── train/
│       ├── val/
│       └── test/
└── coco/
```

## 使い方

### 基本実行

すべてのデータセットを順次学習するには：

```
python train_all_datasets.py
```

### 特定のデータセットタイプのみを実行

セマンティックセグメンテーションデータセットのみを実行するには：

```
python train_all_datasets.py --sem_seg --no-refer_seg --no-vqa --no-reason_seg
```

### データセットパスのみ検証（実行なし）

データセットパスが正しく設定されているか検証するだけで、実際のトレーニングは実行しない場合：

```
python train_all_datasets.py --dry_run
```

### デバッグサンプル数の変更

```
python train_all_datasets.py --debug_samples 10
```

### 特定のデータセットから再開

```
python train_all_datasets.py --continue_from_dataset_type refer_seg --continue_from_dataset refcoco
```

### データキャッシュの有効化

```
python train_all_datasets.py --use_cached_data --save_cached_data
```

### 完全な設定例

```
python train_all_datasets.py \
  --batch_size 2 \
  --grad_accumulation_steps 8 \
  --lr 0.0003 \
  --epochs 1 \
  --steps_per_epoch 5 \
  --debug_samples 5 \
  --load_in_4bit \
  --precision bf16 \
  --vision_tower sigLIP \
  --dataset_dir "H:/download/LISA-dataset/dataset" \
  --log_base_dir "./runs" \
  --sem_seg \
  --refer_seg \
  --vqa \
  --reason_seg
```

## 出力とログ

### ログディレクトリ構造

トレーニング実行の結果は以下の構造で出力されます：

```
./runs/training_logs_YYYYMMDD-HHMMSS/
├── all_datasets_training.log   # メインログファイル
├── execution_settings.json     # 実行時の設定パラメータ
├── execution_summary.json      # 全実行結果のJSON形式サマリー
├── execution_report.txt        # 人間が読みやすい実行レポート
├── sem_seg_ade20k.log          # データセット固有のログ
├── sem_seg_ade20k_command.txt  # 実行されたコマンド
├── sem_seg_ade20k_output.log   # train_full.pyの出力ログ
└── ... (他のデータセット)
```

### ログの種類

1. **メインログ** (`all_datasets_training.log`):  
   全体の実行状況と進捗が記録されます。

2. **データセット固有のログ** (`{dataset_type}_{dataset}.log`):  
   各データセットの処理詳細が記録されます。

3. **コマンドファイル** (`{dataset_type}_{dataset}_command.txt`):  
   各データセットの学習に使用された正確なコマンドが保存されます。

4. **出力ログ** (`{dataset_type}_{dataset}_output.log`):  
   `train_full.py`の実行出力（標準出力と標準エラー出力）が保存されます。

5. **実行サマリー** (`execution_summary.json`):  
   実行結果の詳細がJSON形式で保存され、他のプログラムで解析可能です。

6. **実行レポート** (`execution_report.txt`):  
   人間が読みやすい形式で全体の実行結果がまとめられています。

### モデルチェックポイント

各データセットのトレーニングで生成されたモデルチェックポイントは、個別のディレクトリに保存されます：

```
./runs/lisa-gemma3-{dataset_type}-{dataset}-{timestamp}/
```

## 注意事項

- このスクリプトはデフォルトでデバッグモードで実行され、各データセットから少量のサンプルのみを使用します
- フルトレーニングを行う場合は、`--debug` フラグを削除し、`--epochs` や `--steps_per_epoch` などのパラメータを適切に調整してください
- すべてのデータセットで順次トレーニングを行うと、非常に時間がかかる場合があります
- 実行前にデータセットパスが正しいことを確認してください
- 存在しないデータセットはスキップされます 
- ログファイルは `--log_base_dir` で指定されたディレクトリに保存されます 