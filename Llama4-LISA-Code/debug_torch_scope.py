#!/usr/bin/env python3
"""
torch UnboundLocalError デバッグスクリプト
"""
import ast
import sys

def find_torch_assignments(filename):
    """ファイル内でtorchへの代入を探す"""
    with open(filename, 'r') as f:
        content = f.read()
    
    tree = ast.parse(content)
    
    class TorchAssignmentFinder(ast.NodeVisitor):
        def __init__(self):
            self.assignments = []
            self.current_function = None
            
        def visit_FunctionDef(self, node):
            old_function = self.current_function
            self.current_function = node.name
            self.generic_visit(node)
            self.current_function = old_function
            
        def visit_Assign(self, node):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == 'torch':
                    self.assignments.append({
                        'line': node.lineno,
                        'function': self.current_function,
                        'code': ast.unparse(node) if hasattr(ast, 'unparse') else str(node.lineno)
                    })
            self.generic_visit(node)
            
        def visit_Import(self, node):
            for alias in node.names:
                if alias.name == 'torch' and alias.asname:
                    self.assignments.append({
                        'line': node.lineno,
                        'function': self.current_function,
                        'code': f"import torch as {alias.asname}"
                    })
            self.generic_visit(node)
    
    finder = TorchAssignmentFinder()
    finder.visit(tree)
    
    return finder.assignments

if __name__ == "__main__":
    filename = sys.argv[1] if len(sys.argv) > 1 else "test_phase3b_integration_real.py"
    
    print(f"=== {filename} 内のtorch代入検索 ===")
    
    try:
        assignments = find_torch_assignments(filename)
        
        if assignments:
            print(f"\n⚠️ torch への代入が見つかりました:")
            for assign in assignments:
                print(f"  - 行 {assign['line']}: {assign['code']}")
                if assign['function']:
                    print(f"    関数: {assign['function']}")
        else:
            print("\n✅ torch への代入は見つかりませんでした")
            
        # 関数内でのtorch使用を確認
        print("\n🔍 setup_individual_models関数の最初の10行を確認...")
        with open(filename, 'r') as f:
            lines = f.readlines()
            
        in_function = False
        line_count = 0
        for i, line in enumerate(lines):
            if 'def setup_individual_models' in line:
                in_function = True
                continue
            if in_function:
                print(f"{i+1:4d}: {line.rstrip()}")
                line_count += 1
                if line_count >= 15:
                    break
                    
    except Exception as e:
        print(f"❌ エラー: {e}")