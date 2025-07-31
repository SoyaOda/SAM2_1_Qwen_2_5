#!/usr/bin/env python3
"""
LISA-Llama4プロジェクト Utilsフォルダ核心スクリプト出力ツール

utilsフォルダ内の核心的なファイルの内容をパス付きで
まとめてTXTファイルに出力します。

対象ファイル:
- constants.py
- conversation.py
- data_processing.py
- dataset.py
- utils.py
"""

import os
import sys
from pathlib import Path
from datetime import datetime

class UtilsExporter:
    def __init__(self, project_root="."):
        self.project_root = Path(project_root).resolve()
        self.output_file = f"utils_scripts_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        # 対象のutilsファイル
        self.target_files = [
            "utils/constants.py",
            "utils/conversation.py", 
            "utils/data_processing.py",
            "utils/dataset.py",
            "utils/utils.py"
        ]
        
        self.existing_files = []
        
    def check_files_exist(self):
        """対象ファイルの存在確認"""
        print("📋 対象ファイルの存在確認...")
        
        for file_path in self.target_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                self.existing_files.append(full_path)
                file_size = full_path.stat().st_size / 1024  # KB
                print(f"✅ {file_path} ({file_size:.1f}KB)")
            else:
                print(f"❌ {file_path} (存在しません)")
        
        print(f"\n📊 発見されたファイル: {len(self.existing_files)}/{len(self.target_files)}")
        return len(self.existing_files) > 0
    
    def get_file_info(self, file_path):
        """ファイル情報を取得"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            file_size = file_path.stat().st_size
            line_count = len(lines)
            
            # ファイルの最初の数行でdocstringを探す
            description = ""
            for i, line in enumerate(lines[:20]):
                line = line.strip()
                if line.startswith('"""') or line.startswith("'''"):
                    # docstringの開始を発見
                    end_quote = '"""' if line.startswith('"""') else "'''"
                    if line.count(end_quote) >= 2:
                        # 同じ行にdocstringが完結
                        description = line.strip(end_quote).strip()
                        break
                    else:
                        # 複数行のdocstring
                        desc_lines = [line.strip(end_quote)]
                        for j in range(i+1, min(i+10, len(lines))):
                            next_line = lines[j].strip()
                            if end_quote in next_line:
                                desc_lines.append(next_line.split(end_quote)[0])
                                break
                            desc_lines.append(next_line)
                        description = " ".join(desc_lines).strip()
                        break
            
            return {
                'size_bytes': file_size,
                'size_kb': file_size / 1024,
                'line_count': line_count,
                'description': description[:200] + "..." if len(description) > 200 else description
            }
        except Exception as e:
            return {
                'size_bytes': 0,
                'size_kb': 0,
                'line_count': 0,
                'description': f"読み込みエラー: {e}"
            }
    
    def categorize_files(self):
        """ファイルをカテゴリ別に分類"""
        categories = {
            '🔧 核心ユーティリティ': [],
            '💬 会話・プロンプト処理': [],
            '📊 データ処理・変換': [],
            '📦 データセット統合': [],
            '⚙️ 設定・定数': []
        }
        
        file_categories = {
            'constants.py': '⚙️ 設定・定数',
            'conversation.py': '💬 会話・プロンプト処理',
            'data_processing.py': '📊 データ処理・変換',
            'dataset.py': '📦 データセット統合',
            'utils.py': '🔧 核心ユーティリティ'
        }
        
        for file_path in self.existing_files:
            file_name = file_path.name
            category = file_categories.get(file_name, '🔧 核心ユーティリティ')
            categories[category].append(file_path)
        
        return categories
    
    def export_to_txt(self):
        """TXTファイルに出力"""
        categories = self.categorize_files()
        
        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("LISA-Llama4プロジェクト Utilsフォルダ核心スクリプト集\n")
            f.write(f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
            
            f.write("📌 概要:\n")
            f.write("utilsフォルダ内の核心的なファイル（constants.py, conversation.py,\n")
            f.write("data_processing.py, dataset.py, utils.py）の完全なソースコードを収録。\n\n")
            
            total_files = len(self.existing_files)
            total_size = sum(file_path.stat().st_size for file_path in self.existing_files) / 1024
            total_lines = 0
            
            # 各ファイルの行数を計算
            for file_path in self.existing_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        total_lines += len(file.readlines())
                except:
                    pass
            
            f.write(f"📊 統計: 合計 {total_files} ファイル, {total_size:.1f}KB, {total_lines:,} 行\n\n")
            
            # 目次
            f.write("📑 目次:\n")
            for category, files in categories.items():
                if files:
                    f.write(f"\n{category}:\n")
                    for file_path in files:
                        relative_path = file_path.relative_to(self.project_root)
                        info = self.get_file_info(file_path)
                        f.write(f"  - {relative_path} ({info['line_count']} 行, {info['size_kb']:.1f}KB)\n")
                        if info['description']:
                            f.write(f"    {info['description']}\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("詳細なソースコード\n")
            f.write("="*80 + "\n\n")
            
            # カテゴリ別にファイル内容を出力
            for category, files in categories.items():
                if not files:
                    continue
                    
                f.write(f"\n{category}\n")
                f.write("="*60 + "\n\n")
                
                for file_path in sorted(files, key=lambda x: x.name):
                    relative_path = file_path.relative_to(self.project_root)
                    info = self.get_file_info(file_path)
                    
                    f.write(f"📄 ファイル: {relative_path}\n")
                    f.write(f"📏 サイズ: {info['size_kb']:.1f}KB ({info['line_count']} 行)\n")
                    if info['description']:
                        f.write(f"📝 説明: {info['description']}\n")
                    f.write("-" * 60 + "\n\n")
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8') as file:
                            content = file.read()
                            f.write(content)
                            if not content.endswith('\n'):
                                f.write('\n')
                    except Exception as e:
                        f.write(f"❌ ファイル読み込みエラー: {e}\n")
                    
                    f.write("\n" + "-" * 60 + "\n\n")
        
        return self.output_file
    
    def run(self):
        """メイン実行フロー"""
        print("="*60)
        print("🚀 LISA-Llama4 Utilsフォルダ スクリプト出力ツール")
        print("="*60)
        
        if not self.check_files_exist():
            print("❌ 対象ファイルが見つかりません。")
            return None
        
        print(f"\n📝 TXTファイルに出力中...")
        output_file = self.export_to_txt()
        
        print(f"✅ 出力完了: {output_file}")
        
        # 統計情報表示
        self.show_statistics(output_file)
        
        return output_file
    
    def show_statistics(self, output_file):
        """統計情報を表示"""
        print("\n📊 出力統計:")
        
        total_size = sum(file_path.stat().st_size for file_path in self.existing_files)
        output_size = Path(output_file).stat().st_size
        
        print(f"  対象ファイル数: {len(self.existing_files)}")
        print(f"  総ファイルサイズ: {total_size/1024:.1f}KB")
        print(f"  出力ファイルサイズ: {output_size/1024:.1f}KB")
        print(f"  圧縮率: {(1 - output_size/total_size)*100:.1f}%")
        
        print(f"\n📁 出力ファイル: {output_file}")
        print("💡 このファイルをテキストエディタで開いて確認してください。")

def main():
    if len(sys.argv) > 1:
        project_root = sys.argv[1]
    else:
        project_root = "."
    
    exporter = UtilsExporter(project_root)
    result = exporter.run()
    
    if result:
        print(f"\n🎉 Utils スクリプト出力が完了しました: {result}")
        return 0
    else:
        print(f"\n❌ Utils スクリプト出力に失敗しました")
        return 1

if __name__ == "__main__":
    exit(main()) 