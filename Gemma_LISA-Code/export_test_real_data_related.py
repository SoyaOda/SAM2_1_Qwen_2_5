#!/usr/bin/env python3
"""
test_real_data.py関連スクリプト出力ツール

test_real_data.pyとその依存関係にあるすべてのスクリプトを
パス付きでまとめてTXTファイルに出力します。
"""

import os
import sys
import ast
import importlib.util
from pathlib import Path
from datetime import datetime

class TestRealDataExporter:
    def __init__(self, project_root="."):
        self.project_root = Path(project_root).resolve()
        self.output_file = f"test_real_data_related_scripts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        self.processed_files = set()
        self.related_files = []
        
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
        """import名を実際のファイルパスに解決"""
        possible_files = []
        
        # プロジェクト内のローカルモジュールを検索
        parts = import_name.split('.')
        
        # utils.dataset -> utils/dataset.py
        if len(parts) >= 2:
            potential_path = self.project_root / '/'.join(parts[:-1]) / f"{parts[-1]}.py"
            if potential_path.exists():
                possible_files.append(potential_path)
        
        # model.gemma_lisa -> model/gemma_lisa.py
        potential_path = self.project_root / '/'.join(parts) / "__init__.py"
        if potential_path.exists():
            possible_files.append(potential_path)
            
        potential_path = self.project_root / f"{'/'.join(parts)}.py"
        if potential_path.exists():
            possible_files.append(potential_path)
            
        # config_linux -> config_linux.py
        if len(parts) == 1:
            potential_path = self.project_root / f"{parts[0]}.py"
            if potential_path.exists():
                possible_files.append(potential_path)
        
        return possible_files
    
    def collect_related_files(self, start_file):
        """再帰的に関連ファイルを収集"""
        start_path = Path(start_file).resolve()
        
        if start_path in self.processed_files or not start_path.exists():
            return
        
        self.processed_files.add(start_path)
        self.related_files.append(start_path)
        
        print(f"処理中: {start_path}")
        
        # ファイル内のimportを解析
        imports = self.find_imports_in_file(start_path)
        
        for import_name in imports:
            # 標準ライブラリやサードパーティライブラリをスキップ
            if self.is_standard_or_third_party(import_name):
                continue
                
            # プロジェクト内のファイルを検索
            possible_files = self.resolve_import_to_file(import_name)
            for file_path in possible_files:
                self.collect_related_files(file_path)
    
    def is_standard_or_third_party(self, import_name):
        """標準ライブラリまたはサードパーティライブラリかどうか判定"""
        standard_libs = {
            'os', 'sys', 'json', 'glob', 'random', 'datetime', 'pathlib',
            'collections', 'functools', 'itertools', 'typing', 'dataclasses',
            'argparse', 'logging', 'warnings', 'traceback', 'copy'
        }
        
        third_party_libs = {
            'torch', 'torchvision', 'torchaudio', 'transformers', 'accelerate',
            'peft', 'bitsandbytes', 'huggingface_hub', 'deepspeed',
            'pycocotools', 'numpy', 'tqdm', 'tensorboard', 'PIL', 'Pillow',
            'cv2', 'matplotlib', 'seaborn', 'pandas', 'scipy', 'sklearn'
        }
        
        base_name = import_name.split('.')[0]
        return base_name in standard_libs or base_name in third_party_libs
    
    def add_manual_files(self):
        """手動で重要なファイルを追加"""
        manual_files = [
            "config_linux.py",
            "config_small_test.py",
            "ds_config.json",
            "requirements.txt",
            "utils/__init__.py",
            "model/__init__.py",
        ]
        
        for file_path in manual_files:
            full_path = self.project_root / file_path
            if full_path.exists() and full_path not in self.processed_files:
                self.related_files.append(full_path)
                self.processed_files.add(full_path)
    
    def export_to_txt(self):
        """関連ファイルをTXTファイルに出力"""
        # ファイルをパス順にソート
        self.related_files.sort(key=lambda x: str(x))
        
        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("test_real_data.py 関連スクリプト集\n")
            f.write(f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"プロジェクトルート: {self.project_root}\n")
            f.write(f"総ファイル数: {len(self.related_files)}\n")
            f.write("=" * 80 + "\n\n")
            
            # 目次
            f.write("📋 目次\n")
            f.write("-" * 40 + "\n")
            for i, file_path in enumerate(self.related_files, 1):
                relative_path = file_path.relative_to(self.project_root)
                f.write(f"{i:2d}. {relative_path}\n")
            f.write("\n")
            
            # 各ファイルの内容
            for i, file_path in enumerate(self.related_files, 1):
                relative_path = file_path.relative_to(self.project_root)
                f.write("=" * 80 + "\n")
                f.write(f"ファイル {i:2d}: {relative_path}\n")
                f.write(f"絶対パス: {file_path}\n")
                f.write(f"サイズ: {file_path.stat().st_size:,} bytes\n")
                f.write("=" * 80 + "\n")
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as source_file:
                        content = source_file.read()
                        f.write(content)
                        if not content.endswith('\n'):
                            f.write('\n')
                except Exception as e:
                    f.write(f"❌ ファイル読み込みエラー: {e}\n")
                
                f.write("\n\n")
        
        print(f"✅ 出力完了: {self.output_file}")
        print(f"📊 総ファイル数: {len(self.related_files)}")
        
        # ファイルサイズを表示
        output_size = Path(self.output_file).stat().st_size
        print(f"📄 出力ファイルサイズ: {output_size:,} bytes ({output_size/1024/1024:.2f} MB)")
    
    def run(self):
        """メイン実行関数"""
        print("🔍 test_real_data.py関連スクリプトを収集中...")
        
        # test_real_data.pyから開始
        start_file = self.project_root / "test_real_data.py"
        if not start_file.exists():
            print(f"❌ エラー: {start_file} が見つかりません")
            return
        
        # 関連ファイルを収集
        self.collect_related_files(start_file)
        
        # 手動で重要なファイルを追加
        self.add_manual_files()
        
        # TXTファイルに出力
        self.export_to_txt()
        
        # 統計情報を表示
        self.show_statistics()
    
    def show_statistics(self):
        """統計情報を表示"""
        print("\n📊 統計情報:")
        print("-" * 40)
        
        file_types = {}
        total_size = 0
        
        for file_path in self.related_files:
            ext = file_path.suffix.lower()
            if ext not in file_types:
                file_types[ext] = {'count': 0, 'size': 0}
            
            file_size = file_path.stat().st_size
            file_types[ext]['count'] += 1
            file_types[ext]['size'] += file_size
            total_size += file_size
        
        for ext, info in sorted(file_types.items()):
            print(f"  {ext or '(拡張子なし)'}: {info['count']} ファイル, {info['size']:,} bytes")
        
        print(f"\n合計: {len(self.related_files)} ファイル, {total_size:,} bytes")

def main():
    """メイン関数"""
    if len(sys.argv) > 1:
        project_root = sys.argv[1]
    else:
        project_root = "."
    
    exporter = TestRealDataExporter(project_root)
    exporter.run()

if __name__ == "__main__":
    main() 