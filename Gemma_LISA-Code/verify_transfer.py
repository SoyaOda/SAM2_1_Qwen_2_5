#!/usr/bin/env python3
"""
Lambda Cloud データ転送検証スクリプト

Pascal PartとPACOデータセットの転送結果を検証し、
データの整合性を確認します。
"""

import os
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# --- 設定 ---
INSTANCE_IP = "132.145.192.105"
REMOTE_USER = "ubuntu"
SSH_KEY_PATH = "~/.ssh/lambda_cloud_key"
FILESYSTEM_ROOT = "/lambda/nfs/lisa-gemma-project-fs"

# ローカル設定
LOCAL_DATASET_BASE = "/mnt/h/download/LISA-dataset/dataset"
LOCAL_PASCAL_PART = f"{LOCAL_DATASET_BASE}/vlpart/pascal_part"
LOCAL_PACO = f"{LOCAL_DATASET_BASE}/vlpart/paco"

# リモート設定
REMOTE_DATASET_BASE = f"{FILESYSTEM_ROOT}/data/dataset"
REMOTE_PASCAL_PART = f"{REMOTE_DATASET_BASE}/vlpart/pascal_part"
REMOTE_PACO = f"{REMOTE_DATASET_BASE}/vlpart/paco"

def run_ssh_command(command: str) -> Tuple[int, str, str]:
    """SSH経由でリモートコマンドを実行"""
    ssh_cmd = [
        "ssh", "-i", os.path.expanduser(SSH_KEY_PATH),
        "-o", "ConnectTimeout=10",
        f"{REMOTE_USER}@{INSTANCE_IP}",
        command
    ]
    
    try:
        result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=30)
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "SSH command timed out"
    except Exception as e:
        return -1, "", str(e)

def check_ssh_connection() -> bool:
    """SSH接続確認"""
    print("🔍 SSH接続確認中...")
    returncode, stdout, stderr = run_ssh_command("echo 'SSH接続テスト成功'")
    
    if returncode == 0:
        print("✅ SSH接続成功")
        return True
    else:
        print(f"❌ SSH接続失敗: {stderr}")
        return False

def get_directory_info(path: str, is_remote: bool = False) -> Dict:
    """ディレクトリ情報を取得"""
    if is_remote:
        # リモートディレクトリ情報
        cmd = f"if [ -d '{path}' ]; then find '{path}' -type f | wc -l; du -sh '{path}' 2>/dev/null | cut -f1; ls -la '{path}' | wc -l; else echo '0'; echo '0'; echo '0'; fi"
        returncode, stdout, stderr = run_ssh_command(cmd)
        
        if returncode != 0:
            return {"exists": False, "files": 0, "size": "0", "dirs": 0}
        
        lines = stdout.strip().split('\n')
        return {
            "exists": True,
            "files": int(lines[0]) if lines[0].isdigit() else 0,
            "size": lines[1] if len(lines) > 1 else "0",
            "dirs": int(lines[2]) - 1 if len(lines) > 2 and lines[2].isdigit() else 0  # -1 for total line
        }
    else:
        # ローカルディレクトリ情報
        path_obj = Path(path)
        if not path_obj.exists():
            return {"exists": False, "files": 0, "size": "0", "dirs": 0}
        
        try:
            files = len(list(path_obj.rglob('*'))) if path_obj.is_dir() else 0
            # サイズ計算
            result = subprocess.run(['du', '-sh', str(path)], capture_output=True, text=True)
            size = result.stdout.split()[0] if result.returncode == 0 else "Unknown"
            dirs = len([d for d in path_obj.rglob('*') if d.is_dir()]) if path_obj.is_dir() else 0
            
            return {
                "exists": True,
                "files": files,
                "size": size,
                "dirs": dirs
            }
        except Exception as e:
            return {"exists": True, "files": 0, "size": f"Error: {e}", "dirs": 0}

def verify_pascal_part():
    """Pascal Part データ検証"""
    print("\n" + "="*50)
    print("📊 Pascal Part データ検証")
    print("="*50)
    
    # ローカル情報
    local_info = get_directory_info(LOCAL_PASCAL_PART, False)
    print(f"📁 ローカル ({LOCAL_PASCAL_PART}):")
    print(f"   存在: {'✅' if local_info['exists'] else '❌'}")
    if local_info['exists']:
        print(f"   ファイル数: {local_info['files']}")
        print(f"   サイズ: {local_info['size']}")
        print(f"   ディレクトリ数: {local_info['dirs']}")
    
    # リモート情報
    remote_info = get_directory_info(REMOTE_PASCAL_PART, True)
    print(f"\n☁️  リモート ({REMOTE_PASCAL_PART}):")
    print(f"   存在: {'✅' if remote_info['exists'] else '❌'}")
    if remote_info['exists']:
        print(f"   ファイル数: {remote_info['files']}")
        print(f"   サイズ: {remote_info['size']}")
        print(f"   ディレクトリ数: {remote_info['dirs']}")
    
    # 整合性チェック
    if local_info['exists'] and remote_info['exists']:
        files_match = local_info['files'] == remote_info['files']
        print(f"\n🔍 整合性チェック:")
        print(f"   ファイル数一致: {'✅' if files_match else '❌'} ({local_info['files']} vs {remote_info['files']})")
        
        # JSONファイル確認
        for json_file in ['train.json', 'val.json']:
            local_json = Path(LOCAL_PASCAL_PART) / json_file
            if local_json.exists():
                try:
                    with open(local_json, 'r') as f:
                        data = json.load(f)
                    print(f"   {json_file}: {len(data.get('images', []))} images, {len(data.get('annotations', []))} annotations")
                except Exception as e:
                    print(f"   {json_file}: エラー - {e}")
    
    return local_info['exists'] and remote_info['exists']

def verify_paco():
    """PACO データ検証"""
    print("\n" + "="*50)
    print("📊 PACO データ検証")
    print("="*50)
    
    # ローカル情報
    local_info = get_directory_info(LOCAL_PACO, False)
    print(f"📁 ローカル ({LOCAL_PACO}):")
    print(f"   存在: {'✅' if local_info['exists'] else '❌'}")
    if local_info['exists']:
        print(f"   ファイル数: {local_info['files']}")
        print(f"   サイズ: {local_info['size']}")
        print(f"   ディレクトリ数: {local_info['dirs']}")
    
    # リモート情報
    remote_info = get_directory_info(REMOTE_PACO, True)
    print(f"\n☁️  リモート ({REMOTE_PACO}):")
    print(f"   存在: {'✅' if remote_info['exists'] else '❌'}")
    if remote_info['exists']:
        print(f"   ファイル数: {remote_info['files']}")
        print(f"   サイズ: {remote_info['size']}")
        print(f"   ディレクトリ数: {remote_info['dirs']}")
    
    # 整合性チェック
    if local_info['exists'] and remote_info['exists']:
        files_match = local_info['files'] == remote_info['files']
        print(f"\n🔍 整合性チェック:")
        print(f"   ファイル数一致: {'✅' if files_match else '❌'} ({local_info['files']} vs {remote_info['files']})")
    
    return local_info['exists'] and remote_info['exists']

def main():
    """メイン実行"""
    print("🚀 Lambda Cloud データ転送検証")
    print(f"インスタンス: {INSTANCE_IP}")
    print(f"ファイルシステム: {FILESYSTEM_ROOT}")
    
    # SSH接続確認
    if not check_ssh_connection():
        print("❌ SSH接続に失敗しました。SSH鍵のパスを確認してください。")
        sys.exit(1)
    
    # データ検証
    pascal_ok = verify_pascal_part()
    paco_ok = verify_paco()
    
    # 結果サマリー
    print("\n" + "="*50)
    print("📋 検証結果サマリー")
    print("="*50)
    print(f"Pascal Part: {'✅ 正常' if pascal_ok else '❌ 問題あり'}")
    print(f"PACO: {'✅ 正常' if paco_ok else '❌ 問題あり'}")
    
    if pascal_ok and paco_ok:
        print("\n🎉 すべてのデータ転送が正常に完了しました！")
        return 0
    else:
        print("\n⚠️  一部のデータ転送に問題があります。")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 