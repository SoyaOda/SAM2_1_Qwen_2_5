#!/usr/bin/env python3
"""
LISA-Llama4プロジェクト 核心ファイル出力ツール

llama4_lisa.pyとconfig_linux.pyの内容をパス付きで
まとめてTXTファイルに出力します。

対象ファイル:
- model/llama4_lisa.py (Llama4統合モデル)
- config_linux.py (Linux環境設定)
"""

import os
import sys
from pathlib import Path
from datetime import datetime

class Llama4CoreExporter:
    def __init__(self, project_root="."):
        self.project_root = Path(project_root).resolve()
        self.output_file = f"llama4_core_scripts_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        # 対象の核心ファイル
        self.target_files = [
            "model/llama4_lisa.py",
            "config_linux.py"
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
            
            # ファイルの最初の数行でdocstringまたはコメントを探す
            description = ""
            for i, line in enumerate(lines[:30]):
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
                        for j in range(i+1, min(i+15, len(lines))):
                            next_line = lines[j].strip()
                            if end_quote in next_line:
                                desc_lines.append(next_line.split(end_quote)[0])
                                break
                            desc_lines.append(next_line)
                        description = " ".join(desc_lines).strip()
                        break
                elif line.startswith('#') and 'model/' in str(file_path):
                    # モデルファイルのコメント
                    description = line.strip('#').strip()
                    break
            
            # クラス数、関数数をカウント
            class_count = 0
            function_count = 0
            for line in lines:
                stripped = line.strip()
                if stripped.startswith('class ') and ':' in stripped:
                    class_count += 1
                elif stripped.startswith('def ') and ':' in stripped:
                    function_count += 1
            
            return {
                'size_bytes': file_size,
                'size_kb': file_size / 1024,
                'line_count': line_count,
                'class_count': class_count,
                'function_count': function_count,
                'description': description[:300] + "..." if len(description) > 300 else description
            }
        except Exception as e:
            return {
                'size_bytes': 0,
                'size_kb': 0,
                'line_count': 0,
                'class_count': 0,
                'function_count': 0,
                'description': f"読み込みエラー: {e}"
            }
    
    def categorize_files(self):
        """ファイルをカテゴリ別に分類"""
        categories = {
            '🧠 Llama4統合モデル': [],
            '⚙️ Linux環境設定': []
        }
        
        file_categories = {
            'llama4_lisa.py': '🧠 Llama4統合モデル',
            'config_linux.py': '⚙️ Linux環境設定'
        }
        
        for file_path in self.existing_files:
            file_name = file_path.name
            category = file_categories.get(file_name, '🧠 Llama4統合モデル')
            categories[category].append(file_path)
        
        return categories
    
    def export_to_txt(self):
        """TXTファイルに出力"""
        categories = self.categorize_files()
        
        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("LISA-Llama4プロジェクト 核心ファイル集\n")
            f.write(f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
            
            f.write("📌 概要:\n")
            f.write("LISA-Llama4プロジェクトの核心的なファイル（llama4_lisa.py、config_linux.py）\n")
            f.write("の完全なソースコードと設定を収録。\n\n")
            
            f.write("🎯 目的:\n")
            f.write("- Llama4とSAMの統合アーキテクチャの理解\n")
            f.write("- Lambda Cloud環境での学習設定の確認\n")
            f.write("- プロジェクト構造の把握\n\n")
            
            total_files = len(self.existing_files)
            total_size = sum(file_path.stat().st_size for file_path in self.existing_files) / 1024
            total_lines = 0
            total_classes = 0
            total_functions = 0
            
            # 各ファイルの統計を計算
            for file_path in self.existing_files:
                info = self.get_file_info(file_path)
                total_lines += info['line_count']
                total_classes += info['class_count']
                total_functions += info['function_count']
            
            f.write(f"📊 統計: 合計 {total_files} ファイル, {total_size:.1f}KB, {total_lines:,} 行\n")
            f.write(f"🏗️ 構造: {total_classes} クラス, {total_functions} 関数\n\n")
            
            # 目次
            f.write("📑 目次:\n")
            for category, files in categories.items():
                if files:
                    f.write(f"\n{category}:\n")
                    for file_path in files:
                        relative_path = file_path.relative_to(self.project_root)
                        info = self.get_file_info(file_path)
                        f.write(f"  - {relative_path}\n")
                        f.write(f"    📏 {info['line_count']} 行, {info['size_kb']:.1f}KB\n")
                        f.write(f"    🏗️ {info['class_count']} クラス, {info['function_count']} 関数\n")
                        if info['description']:
                            f.write(f"    📝 {info['description']}\n")
            
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
                    f.write(f"🏗️ 構造: {info['class_count']} クラス, {info['function_count']} 関数\n")
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
        print("🚀 LISA-Llama4 核心ファイル出力ツール")
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
        
        # 詳細統計
        total_lines = 0
        total_classes = 0
        total_functions = 0
        
        for file_path in self.existing_files:
            info = self.get_file_info(file_path)
            total_lines += info['line_count']
            total_classes += info['class_count']
            total_functions += info['function_count']
        
        print(f"  対象ファイル数: {len(self.existing_files)}")
        print(f"  総ファイルサイズ: {total_size/1024:.1f}KB")
        print(f"  総行数: {total_lines:,} 行")
        print(f"  総クラス数: {total_classes}")
        print(f"  総関数数: {total_functions}")
        print(f"  出力ファイルサイズ: {output_size/1024:.1f}KB")
        
        print(f"\n📁 出力ファイル: {output_file}")
        print("💡 このファイルをテキストエディタで開いて確認してください。")
        print("\n🔍 主要内容:")
        print("  - Llama4+SAM統合モデルの実装詳細")
        print("  - Lambda Cloud環境用の最適化設定")
        print("  - LoRA微調整の設定パラメータ")
        print("  - データセット構造と処理設定")

def main():
    if len(sys.argv) > 1:
        project_root = sys.argv[1]
    else:
        project_root = "."
    
    exporter = Llama4CoreExporter(project_root)
    result = exporter.run()
    
    if result:
        print(f"\n🎉 Llama4核心ファイル出力が完了しました: {result}")
        return 0
    else:
        print(f"\n❌ Llama4核心ファイル出力に失敗しました")
        return 1

if __name__ == "__main__":
    exit(main()) 