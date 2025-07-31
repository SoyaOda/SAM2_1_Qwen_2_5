#!/usr/bin/env python3
"""
第5節: 堅牢性と将来の開発に向けた事前検証
5.3. サニティチェック3：設定ファイルの読み込みと検証

多くの複雑なプロジェクトは、設定ファイル（.yamlなど）の単純なミスによって頓挫します。
ファイルパスのタイプミス、学習率の桁間違い、真偽値フラグの間違いなどは、
数週間にわたる学習を無に帰す可能性があります。

この事前チェックは、設定ファイルをソースコードの重要な一部とみなし、その内容を検証します。

論理的根拠:
- 設定ファイルの単純なミス（ファイルパスのタイプミス、学習率の桁間違い等）は数週間の学習を無に帰す
- 高コストな学習ジョブを開始する前に設定ファイル内のタイプミスや論理エラーを発見
- 実験設定全体のクリーンで包括的なサマリーを提供し、レビューを容易にする
"""

import argparse
import os
import sys
import json
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple

# プロジェクトのルートディレクトリをsys.pathに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def get_config():
    """
    config_linux.pyを必須として読み込む
    読み込めない場合はエラーで停止
    
    仕様書準拠: 他の検証スクリプト（第1-4節）と完全に一致する厳密な設定読み込み
    フォールバック機能は排除し、設定ファイルの不整合を早期発見
    """
    try:
        import config_linux as config
        print(f"設定: config_linux.py を使用")
        return config, 'config_linux'
    except ImportError as e:
        print(f"❌ ERROR: config_linux.pyが見つかりません")
        print(f"   詳細: {e}")
        print(f"   現在のディレクトリ: {os.getcwd()}")
        print(f"   ファイル存在確認: {os.path.exists('config_linux.py')}")
        raise SystemExit("config_linux.pyが必須です。ファイルが存在することを確認してください。")

def extract_config_dict(config_module) -> Dict[str, Any]:
    """設定モジュールから辞書形式で設定を抽出"""
    config_dict = {}
    
    # 設定項目のカテゴリ別に分類
    categories = {
        'model_config': ['GEMMA_MODEL_ID', 'GEMMA_MODEL_PATH', 'GEMMA_IMAGE_SIZE', 'SAM_CHECKPOINT_PATH', 'GEMMA_HIDDEN_SIZE', 'SAM_IMAGE_SIZE', 'MODEL_MAX_LENGTH', 'SEG_PROJECTION_DIM', 'SEG_TOKEN'],
        'dataset_config': ['DATASET_BASE_DIR', 'DATASET_STRUCTURE', 'SEM_SEG_DATA', 'REFER_SEG_DATA', 'VQA_DATA', 'REASON_SEG_DATA', 'VAL_DATASET', 'DATASET_SAMPLE_RATES'],
        'training_config': ['EPOCHS', 'STEPS_PER_EPOCH', 'BATCH_SIZE_PER_GPU', 'LEARNING_RATE', 'WEIGHT_DECAY', 'GRADIENT_ACCUMULATION_STEPS', 'BETA1', 'BETA2'],
        'optimization_config': ['WARMUP_STEPS', 'WARMUP_RATIO', 'SAVE_STEPS', 'LOGGING_STEPS', 'EVAL_STEPS'],
        'system_config': ['MIXED_PRECISION', 'GRADIENT_CHECKPOINTING', 'DATALOADER_NUM_WORKERS'],
        'lora_config': ['LORA_R', 'LORA_ALPHA', 'LORA_DROPOUT', 'LORA_TARGET_MODULES'],
        'loss_config': ['CE_LOSS_WEIGHT', 'DICE_LOSS_WEIGHT', 'BCE_LOSS_WEIGHT']
    }
    
    for category, attrs in categories.items():
        config_dict[category] = {}
        for attr in attrs:
            if hasattr(config_module, attr):
                config_dict[category][attr] = getattr(config_module, attr)
    
    # その他の設定項目も追加
    other_attrs = []
    for attr in dir(config_module):
        if not attr.startswith('_') and attr.isupper():
            # 既にカテゴリに含まれていない項目を追加
            found_in_category = False
            for category_attrs in categories.values():
                if attr in category_attrs:
                    found_in_category = True
                    break
            if not found_in_category:
                other_attrs.append(attr)
    
    if other_attrs:
        config_dict['other_config'] = {}
        for attr in other_attrs:
            config_dict['other_config'][attr] = getattr(config_module, attr)
    
    return config_dict

def validate_paths(config_dict: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """パスの存在確認"""
    errors = []
    warnings = []
    
    # 重要なパスをチェック
    path_checks = [
        ('DATASET_BASE_DIR', 'dataset_config'),
        ('SAM_CHECKPOINT_PATH', 'model_config'),
        ('GEMMA_MODEL_PATH', 'model_config'),
    ]
    
    for path_key, category in path_checks:
        if category in config_dict and path_key in config_dict[category]:
            path_value = config_dict[category][path_key]
            if isinstance(path_value, str):
                path_obj = Path(path_value)
                if not path_obj.exists():
                    if path_key == 'GEMMA_MODEL_PATH':
                        warnings.append(f"{path_key}: {path_value} (Hugging Face modelの場合は正常)")
                    else:
                        errors.append(f"{path_key}: {path_value} が見つかりません")
        else:
            # パスが設定に含まれていない場合の処理
            if path_key == 'GEMMA_MODEL_PATH':
                warnings.append(f"{path_key}: 未設定 (GEMMA_MODEL_IDを使用している場合は正常)")
            else:
                warnings.append(f"{path_key}: 設定項目が見つかりません")
    
    return errors, warnings

def validate_hyperparameters(config_dict: Dict[str, Any]) -> List[str]:
    """ハイパーパラメータの妥当性チェック（LISA-Gemma仕様書準拠）"""
    warnings = []
    
    # 学習率の妥当性チェック
    if 'training_config' in config_dict and 'LEARNING_RATE' in config_dict['training_config']:
        lr = config_dict['training_config']['LEARNING_RATE']
        if isinstance(lr, (int, float)):
            if lr > 1e-2:
                warnings.append(f"学習率が高すぎる可能性があります: {lr} (推奨: 1e-4～1e-5)")
            elif lr < 1e-6:
                warnings.append(f"学習率が低すぎる可能性があります: {lr} (推奨: 1e-4～1e-5)")
    
    # バッチサイズの妥当性チェック
    if 'training_config' in config_dict and 'BATCH_SIZE_PER_GPU' in config_dict['training_config']:
        batch_size = config_dict['training_config']['BATCH_SIZE_PER_GPU']
        if isinstance(batch_size, int):
            if batch_size > 8:
                warnings.append(f"GPU毎バッチサイズが大きすぎる可能性があります: {batch_size} (LISA-Gemmaでは1-4推奨)")
            elif batch_size < 1:
                warnings.append(f"GPU毎バッチサイズが無効です: {batch_size}")
    
    # 勾配蓄積ステップの妥当性チェック
    if 'training_config' in config_dict and 'GRADIENT_ACCUMULATION_STEPS' in config_dict['training_config']:
        grad_acc = config_dict['training_config']['GRADIENT_ACCUMULATION_STEPS']
        if isinstance(grad_acc, int):
            if grad_acc > 32:
                warnings.append(f"勾配蓄積ステップが大きすぎる可能性があります: {grad_acc} (推奨: 4-16)")
            elif grad_acc < 1:
                warnings.append(f"勾配蓄積ステップが無効です: {grad_acc}")
    
    # システム設定の妥当性チェック
    if 'system_config' in config_dict:
        system_config = config_dict['system_config']
        
        # データローダーワーカー数のチェック
        if 'DATALOADER_NUM_WORKERS' in system_config:
            num_workers = system_config['DATALOADER_NUM_WORKERS']
            if isinstance(num_workers, int):
                if num_workers > 16:
                    warnings.append(f"データローダーワーカー数が多すぎる可能性があります: {num_workers} (推奨: 4-8)")
                elif num_workers < 0:
                    warnings.append(f"データローダーワーカー数が無効です: {num_workers}")
    
    # LoRA設定の妥当性チェック
    if 'lora_config' in config_dict:
        lora_config = config_dict['lora_config']
        
        if 'LORA_R' in lora_config and 'LORA_ALPHA' in lora_config:
            lora_r = lora_config['LORA_R']
            lora_alpha = lora_config['LORA_ALPHA']
            if isinstance(lora_r, int) and isinstance(lora_alpha, int):
                if lora_alpha < lora_r:
                    warnings.append(f"LoRA ALPHA({lora_alpha})がR({lora_r})より小さいです (推奨: ALPHA ≥ R)")
                if lora_r > 64:
                    warnings.append(f"LoRA Rが大きすぎる可能性があります: {lora_r} (推奨: 8-32)")
        
        if 'LORA_DROPOUT' in lora_config:
            lora_dropout = lora_config['LORA_DROPOUT']
            if isinstance(lora_dropout, (int, float)):
                if lora_dropout > 0.3:
                    warnings.append(f"LoRAドロップアウトが高すぎる可能性があります: {lora_dropout} (推奨: 0.05-0.1)")
                elif lora_dropout < 0:
                    warnings.append(f"LoRAドロップアウトが無効です: {lora_dropout}")
    
    # 重みの減衰率チェック
    if 'training_config' in config_dict and 'WEIGHT_DECAY' in config_dict['training_config']:
        wd = config_dict['training_config']['WEIGHT_DECAY']
        if isinstance(wd, (int, float)):
            if wd > 0.1:
                warnings.append(f"重み減衰が大きすぎる可能性があります: {wd} (推奨: 0.01～0.05)")
    
    # モデル設定の妥当性チェック
    if 'model_config' in config_dict:
        model_config = config_dict['model_config']
        
        # 画像サイズのチェック
        if 'GEMMA_IMAGE_SIZE' in model_config:
            gemma_size = model_config['GEMMA_IMAGE_SIZE']
            if isinstance(gemma_size, int) and gemma_size != 896:
                warnings.append(f"Gemma画像サイズが標準と異なります: {gemma_size} (推奨: 896)")
        
        if 'SAM_IMAGE_SIZE' in model_config:
            sam_size = model_config['SAM_IMAGE_SIZE']
            if isinstance(sam_size, int) and sam_size != 1024:
                warnings.append(f"SAM画像サイズが標準と異なります: {sam_size} (推奨: 1024)")
        
        # モデル最大長のチェック
        if 'MODEL_MAX_LENGTH' in model_config:
            max_length = model_config['MODEL_MAX_LENGTH']
            if isinstance(max_length, int):
                if max_length < 1024:
                    warnings.append(f"モデル最大長が短すぎる可能性があります: {max_length} (推奨: 2048以上)")
                elif max_length > 8192:
                    warnings.append(f"モデル最大長が長すぎる可能性があります: {max_length} (メモリ使用量に注意)")
    
    return warnings

def format_config_summary(config_dict: Dict[str, Any], config_name: str) -> str:
    """設定の整形されたサマリーを作成"""
    summary = []
    summary.append("=" * 80)
    summary.append(f"LISA-Gemma 設定ファイル検証レポート")
    summary.append(f"設定ファイル: {config_name}")
    summary.append(f"検証日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary.append("=" * 80)
    
    for category, items in config_dict.items():
        if not items:
            continue
            
        summary.append(f"\n[{category.upper().replace('_', ' ')}]")
        summary.append("-" * 40)
        
        for key, value in items.items():
            if isinstance(value, dict):
                summary.append(f"  {key}:")
                for sub_key, sub_value in value.items():
                    summary.append(f"    {sub_key}: {sub_value}")
            elif isinstance(value, list):
                summary.append(f"  {key}: [{', '.join(map(str, value))}]")
            else:
                summary.append(f"  {key}: {value}")
    
    return "\n".join(summary)

def create_hyperparameter_table(config_dict: Dict[str, Any]) -> str:
    """重要なハイパーパラメータの表を作成"""
    table = []
    table.append("\n" + "=" * 80)
    table.append("重要なハイパーパラメータ サマリー")
    table.append("=" * 80)
    table.append(f"{'Parameter':<30} | {'Value':<25} | {'Status':<20}")
    table.append("-" * 80)
    
    # 重要なパラメータを定義（仕様書準拠 + LISA-Gemma固有設定）
    important_params = [
        ('LEARNING_RATE', 'training_config', '学習率'),
        ('BATCH_SIZE_PER_GPU', 'training_config', 'GPU毎バッチサイズ'),
        ('WEIGHT_DECAY', 'training_config', '重み減衰'),
        ('EPOCHS', 'training_config', 'エポック数'),
        ('STEPS_PER_EPOCH', 'training_config', 'エポック当たりステップ数'),
        ('GRADIENT_ACCUMULATION_STEPS', 'training_config', '勾配蓄積ステップ'),
        ('GEMMA_IMAGE_SIZE', 'model_config', 'Gemma画像サイズ'),
        ('SAM_IMAGE_SIZE', 'model_config', 'SAM画像サイズ'),
        ('MODEL_MAX_LENGTH', 'model_config', 'モデル最大長'),
        ('LORA_R', 'lora_config', 'LoRA-R値'),
        ('LORA_ALPHA', 'lora_config', 'LoRA-Alpha値'),
        ('LORA_DROPOUT', 'lora_config', 'LoRAドロップアウト'),
        ('MIXED_PRECISION', 'system_config', '混合精度'),
        ('GRADIENT_CHECKPOINTING', 'system_config', '勾配チェックポイント'),
        ('DATALOADER_NUM_WORKERS', 'system_config', 'データローダーワーカー数'),
        ('WARMUP_STEPS', 'optimization_config', 'ウォームアップステップ'),
        ('SAVE_STEPS', 'optimization_config', 'チェックポイント保存間隔'),
    ]
    
    for param_key, category, description in important_params:
        if category in config_dict and param_key in config_dict[category]:
            value = config_dict[category][param_key]
            
            # ステータスを判定（LISA-Gemma仕様書準拠）
            status = "✓ OK"
            if param_key == 'LEARNING_RATE' and isinstance(value, (int, float)):
                if value > 1e-2 or value < 1e-6:
                    status = "⚠ 要確認"
            elif param_key == 'BATCH_SIZE_PER_GPU' and isinstance(value, int):
                if value > 8 or value < 1:
                    status = "⚠ 要確認"
            elif param_key == 'GRADIENT_ACCUMULATION_STEPS' and isinstance(value, int):
                if value > 32 or value < 1:
                    status = "⚠ 要確認"
            elif param_key == 'WEIGHT_DECAY' and isinstance(value, (int, float)):
                if value > 0.1:
                    status = "⚠ 要確認"
            elif param_key == 'GEMMA_IMAGE_SIZE' and isinstance(value, int):
                if value != 896:
                    status = "⚠ 要確認"
            elif param_key == 'SAM_IMAGE_SIZE' and isinstance(value, int):
                if value != 1024:
                    status = "⚠ 要確認"
            elif param_key == 'MODEL_MAX_LENGTH' and isinstance(value, int):
                if value < 1024 or value > 8192:
                    status = "⚠ 要確認"
            elif param_key == 'LORA_R' and isinstance(value, int):
                if value < 4 or value > 64:
                    status = "⚠ 要確認"
            elif param_key == 'LORA_ALPHA' and isinstance(value, int):
                if value < 8 or value > 128:
                    status = "⚠ 要確認"
            elif param_key == 'LORA_DROPOUT' and isinstance(value, (int, float)):
                if value > 0.3 or value < 0:
                    status = "⚠ 要確認"
            elif param_key == 'DATALOADER_NUM_WORKERS' and isinstance(value, int):
                if value > 16 or value < 0:
                    status = "⚠ 要確認"
            elif param_key == 'WARMUP_STEPS' and isinstance(value, int):
                if value > 1000 or value < 0:
                    status = "⚠ 要確認"
            elif param_key == 'SAVE_STEPS' and isinstance(value, int):
                if value > 5000 or value < 10:
                    status = "⚠ 要確認"
            
            table.append(f"{description:<30} | {str(value):<25} | {status:<20}")
        else:
            table.append(f"{description:<30} | {'未設定':<25} | {'⚠ 未設定':<20}")
    
    return "\n".join(table)

def save_verification_report(summary: str, config_dict: Dict[str, Any], errors: List[str], warnings: List[str]) -> str:
    """検証レポートをファイルに保存"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("verification_output")
    output_dir.mkdir(exist_ok=True)
    
    # テキストレポート
    report_path = output_dir / f"config_verification_report_{timestamp}.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(summary)
        f.write("\n\n")
        
        if errors:
            f.write("❌ エラー:\n")
            for error in errors:
                f.write(f"  - {error}\n")
            f.write("\n")
        
        if warnings:
            f.write("⚠️ 警告:\n")
            for warning in warnings:
                f.write(f"  - {warning}\n")
            f.write("\n")
    
    # JSON形式でも保存
    json_path = output_dir / f"config_verification_data_{timestamp}.json"
    verification_data = {
        'timestamp': timestamp,
        'config_dict': config_dict,
        'errors': errors,
        'warnings': warnings,
        'validation_status': 'failed' if errors else 'passed_with_warnings' if warnings else 'passed'
    }
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(verification_data, f, indent=2, ensure_ascii=False, default=str)
    
    return str(report_path)

def parse_args():
    parser = argparse.ArgumentParser(description="Configuration file verification and validation")
    parser.add_argument("--save_report", "-s", action="store_true", help="Save verification report to file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("=" * 80)
    print("第5節: 堅牢性と将来の開発に向けた事前検証")
    print("5.3. サニティチェック3：設定ファイルの読み込みと検証")
    print("=" * 80)
    
    try:
        # 設定ファイルの読み込み
        config_module, config_name = get_config()
        print(f"✅ 設定ファイル読み込み成功: {config_name}")
        
        # 設定辞書の抽出
        config_dict = extract_config_dict(config_module)
        print(f"✅ 設定項目抽出完了: {sum(len(items) for items in config_dict.values())} 項目")
        
        # パスの検証
        errors, path_warnings = validate_paths(config_dict)
        
        # ハイパーパラメータの検証
        hyper_warnings = validate_hyperparameters(config_dict)
        
        all_warnings = path_warnings + hyper_warnings
        
        # サマリーの生成
        summary = format_config_summary(config_dict, config_name)
        hyperparameter_table = create_hyperparameter_table(config_dict)
        
        # 結果表示
        print(summary)
        print(hyperparameter_table)
        
        # エラーと警告の表示
        if errors:
            print("\n❌ エラー:")
            for error in errors:
                print(f"  - {error}")
        
        if all_warnings:
            print("\n⚠️ 警告:")
            for warning in all_warnings:
                print(f"  - {warning}")
        
        # レポート保存
        if args.save_report:
            report_path = save_verification_report(summary + hyperparameter_table, config_dict, errors, all_warnings)
            print(f"\n✅ 検証レポートを保存: {report_path}")
        
        # 最終判定
        print("\n" + "=" * 80)
        print("🎯 最終判定")
        print("=" * 80)
        
        if errors:
            print("❌ サニティチェック3: 設定ファイルの読み込みと検証 - 失敗")
            print("   重要なパスまたは設定にエラーがあります。")
            print("   学習を開始する前に上記のエラーを修正してください。")
            return False
        elif all_warnings:
            print("⚠️ サニティチェック3: 設定ファイルの読み込みと検証 - 警告あり")
            print("   設定ファイルは読み込み可能ですが、一部の設定に注意が必要です。")
            print("   警告を確認し、必要に応じて設定を調整してください。")
            return True
        else:
            print("✅ サニティチェック3: 設定ファイルの読み込みと検証 - 成功")
            print("   設定ファイルは正常に読み込まれ、全ての設定が適切です。")
            print("   学習を開始する準備が整っています。")
            return True
        
    except Exception as e:
        print(f"\n❌ 設定ファイル検証中にエラーが発生しました: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 