#!/usr/bin/env python3
"""
LISA-Gemmaプロジェクト 核心構造スクリプト出力ツール

プロジェクトの大まかな構造把握に必要な核心的なスクリプトのみを
パス付きでまとめてTXTファイルに出力します。

対象スクリプト（優先度高のみ）:
- 第4節: verify_loss_and_gradients.py
- 第5節: overfit_single_batch.py, test_inference_pipeline.py
- 核心設定: config_linux.py
"""

import os
import sys
import ast
import importlib.util
from pathlib import Path
from datetime import datetime

class CoreProjectExporter:
    def __init__(self, project_root="."):
        self.project_root = Path(project_root).resolve()
        self.output_file = f"core_project_structure_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        self.processed_files = set()
        self.related_files = []
        
        # 核心的な検証スクリプトのみ（第1-3節除外）
        self.core_scripts = [
            # 第4節: 損失計算と勾配伝播の精査（重要）
            "verify_loss_and_gradients.py",
            
            # 第5節: 堅牢性検証（重要）
            "overfit_single_batch.py",
            "test_inference_pipeline.py",
        ]
        
        # 核心的な関連ファイルのみ
        self.core_files = [
            "config_linux.py",
            
            # モデル実装の核心のみ
            "model/gemma_lisa.py", 
            "model/losses.py",
            
            # ユーティリティの核心のみ
            "utils/constants.py",
            "utils/conversation.py",
        ]
        
    def find_imports_in_file(self, file_path):
        """ファイル内のimport文を解析して依存関係を取得"""
        imports = set()
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.add(node.module)
        except Exception as e:
            print(f"警告: {file_path} の解析に失敗: {e}")
        
        return imports
    
    def resolve_import_to_file(self, import_name):
        """import名を実際のファイルパスに解決（核心ファイルのみ）"""
        possible_files = []
        
        # プロジェクト内のローカルモジュールを検索
        parts = import_name.split('.')
        
        # 核心的なモジュールのみ許可
        core_modules = {
            'config_linux', 'model.gemma_lisa', 'model.losses', 
            'utils.constants', 'utils.conversation'
        }
        
        if import_name not in core_modules and not any(import_name.startswith(m) for m in core_modules):
            return possible_files
        
        # utils.constants -> utils/constants.py
        if len(parts) >= 2 and parts[0] in ['utils', 'model']:
            potential_path = self.project_root / '/'.join(parts[:-1]) / f"{parts[-1]}.py"
            if potential_path.exists() and self.is_core_file(potential_path):
                possible_files.append(potential_path)
        
        # config_linux -> config_linux.py
        if len(parts) == 1 and parts[0] == 'config_linux':
            potential_path = self.project_root / f"{parts[0]}.py"
            if potential_path.exists():
                possible_files.append(potential_path)
        
        return possible_files
    
    def is_core_file(self, file_path):
        """ファイルが核心ファイルかどうか判定"""
        core_file_names = {
            'constants.py', 'conversation.py', 'gemma_lisa.py', 'losses.py', 'config_linux.py'
        }
        return file_path.name in core_file_names
    
    def collect_related_files(self, start_file):
        """核心ファイルのみ収集"""
        start_path = Path(start_file).resolve()
        
        if start_path in self.processed_files or not start_path.exists():
            return
        
        self.processed_files.add(start_path)
        self.related_files.append(start_path)
        
        print(f"処理中: {start_path.relative_to(self.project_root)}")
        
        # ファイル内のimportを解析（核心ファイルのみ）
        imports = self.find_imports_in_file(start_path)
        
        for import_name in imports:
            # 標準ライブラリやサードパーティライブラリをスキップ
            if self.is_standard_or_third_party(import_name):
                continue
                
            # 核心ファイルのみ検索
            possible_files = self.resolve_import_to_file(import_name)
            for file_path in possible_files:
                self.collect_related_files(file_path)
    
    def is_standard_or_third_party(self, import_name):
        """標準ライブラリまたはサードパーティライブラリかどうか判定"""
        standard_libs = {
            'os', 'sys', 'json', 'glob', 'random', 'datetime', 'pathlib',
            'collections', 'functools', 'itertools', 'typing', 'dataclasses',
            'argparse', 'logging', 'warnings', 'traceback', 'copy', 'time',
            'yaml', 'ast', 'importlib'
        }
        
        third_party_libs = {
            'torch', 'torchvision', 'torchaudio', 'transformers', 'accelerate',
            'peft', 'bitsandbytes', 'huggingface_hub', 'deepspeed',
            'pycocotools', 'numpy', 'tqdm', 'tensorboard', 'PIL', 'Pillow',
            'cv2', 'matplotlib', 'seaborn', 'pandas', 'scipy', 'sklearn'
        }
        
        base_name = import_name.split('.')[0]
        return base_name in standard_libs or base_name in third_party_libs
    
    def add_core_files(self):
        """核心ファイルを手動で追加"""
        for file_path in self.core_files:
            full_path = self.project_root / file_path
            if full_path.exists() and full_path not in self.processed_files:
                self.related_files.append(full_path)
                self.processed_files.add(full_path)
                print(f"手動追加: {file_path}")
    
    def categorize_files(self):
        """ファイルをカテゴリ別に分類"""
        categories = {
            '🔍 核心検証スクリプト (第4-5節のみ)': [],
            '⚙️ 設定ファイル': [],
            '🧮 モデル実装 (核心のみ)': [],
            '📦 ユーティリティ (核心のみ)': [],
        }
        
        for file_path in self.related_files:
            relative_path = file_path.relative_to(self.project_root)
            file_name = file_path.name
            
            if file_name in self.core_scripts:
                categories['🔍 核心検証スクリプト (第4-5節のみ)'].append(file_path)
            elif file_name == 'config_linux.py':
                categories['⚙️ 設定ファイル'].append(file_path)
            elif str(relative_path).startswith('model/'):
                categories['🧮 モデル実装 (核心のみ)'].append(file_path)
            elif str(relative_path).startswith('utils/'):
                categories['📦 ユーティリティ (核心のみ)'].append(file_path)
        
        return categories
    
    def export_to_txt(self):
        """TXTファイルに出力"""
        categories = self.categorize_files()
        
        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("LISA-Gemmaプロジェクト 核心構造スクリプト集\n")
            f.write(f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
            
            f.write("📌 概要:\n")
            f.write("プロジェクトの大まかな構造把握のため、第1-3節スクリプトと\n")
            f.write("優先順位の低いファイルを除外し、核心的なファイルのみを収録。\n\n")
            
            total_files = len(self.related_files)
            f.write(f"📊 統計: 合計 {total_files} ファイル\n\n")
            
            for category, files in categories.items():
                if not files:
                    continue
                    
                f.write(f"{category}\n")
                f.write("-" * 60 + "\n")
                
                for file_path in sorted(files):
                    relative_path = file_path.relative_to(self.project_root)
                    f.write(f"\n📁 {relative_path}\n")
                    f.write("="*60 + "\n")
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8') as file:
                            content = file.read()
                            f.write(content)
                            f.write("\n\n")
                    except Exception as e:
                        f.write(f"❌ ファイル読み込みエラー: {e}\n\n")
                
                f.write("\n")
    
    def run(self):
        """メイン処理"""
        print("🚀 LISA-Gemmaプロジェクト核心構造エクスポート開始")
        print(f"📂 プロジェクトルート: {self.project_root}")
        
        # 核心検証スクリプトから依存関係を収集
        for script in self.core_scripts:
            script_path = self.project_root / script
            if script_path.exists():
                print(f"\n🔍 {script} の依存関係を解析中...")
                self.collect_related_files(script_path)
            else:
                print(f"⚠️ {script} が見つかりません")
        
        # 核心ファイルを手動追加
        print("\n📦 核心ファイルを追加中...")
        self.add_core_files()
        
        # TXTファイルに出力
        print(f"\n📝 {self.output_file} に出力中...")
        self.export_to_txt()
        
        print(f"\n✅ 完了！ 出力ファイル: {self.output_file}")
        self.show_statistics()
    
    def show_statistics(self):
        """統計情報を表示"""
        categories = self.categorize_files()
        
        print("\n📊 ファイル統計:")
        for category, files in categories.items():
            print(f"  {category}: {len(files)}ファイル")
        
        print(f"\n📁 合計: {len(self.related_files)}ファイル")
        
        # ファイルサイズ概算
        total_size = 0
        for file_path in self.related_files:
            try:
                total_size += file_path.stat().st_size
            except:
                pass
        
        size_mb = total_size / (1024 * 1024)
        print(f"📏 概算サイズ: {size_mb:.1f}MB")

def main():
    if len(sys.argv) > 1:
        project_root = sys.argv[1]
    else:
        project_root = "."
    
    exporter = CoreProjectExporter(project_root)
    exporter.run()

if __name__ == "__main__":
    main() 