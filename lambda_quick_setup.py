#!/usr/bin/env python3
"""
Lambda Cloud A100*8 一括セットアップスクリプト
WandB、HuggingFace認証、環境確認を一括で実行

使用方法:
python lambda_quick_setup.py --ip YOUR_LAMBDA_IP
"""

import os
import sys
import subprocess
import argparse
import time
from pathlib import Path

def run_ssh_command(ip, command, timeout=60):
    """SSH経由でコマンドを実行"""
    ssh_key = "~/.ssh/lambda_cloud_key"
    # SSH fingerprint自動承認を追加
    full_command = f'ssh -o StrictHostKeyChecking=no -i {ssh_key} ubuntu@{ip} "{command}"'
    
    try:
        result = subprocess.run(
            full_command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out"
    except Exception as e:
        return False, "", str(e)

def run_local_command(command, timeout=60):
    """ローカルでコマンドを実行"""
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out"
    except Exception as e:
        return False, "", str(e)

def check_ssh_connection(ip):
    """SSH接続確認"""
    print("🔌 SSH接続確認中...")
    success, stdout, stderr = run_ssh_command(ip, "echo 'Lambda Cloud接続OK'")
    if success:
        print("   ✅ SSH接続成功")
        return True
    else:
        print(f"   ❌ SSH接続失敗: {stderr}")
        return False

def check_environment(ip):
    """Python環境とGPU確認"""
    print("🐍 Python環境確認中...")
    
    env_command = (
        "source /lambda/nfs/lisa-gemma-project-fs/venvs/lisa_gemma_venv/bin/activate && "
        "python --version"
    )
    
    success, stdout, stderr = run_ssh_command(ip, env_command)
    if success:
        print(f"   ✅ Python環境: {stdout.strip()}")
    else:
        print(f"   ❌ Python環境確認失敗: {stderr}")
        return False
    
    print("🖥️ GPU確認中...")
    success, stdout, stderr = run_ssh_command(ip, "nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits")
    if success:
        print("   ✅ GPU情報:")
        for line in stdout.strip().split('\n'):
            if line.strip():
                print(f"      {line.strip()}")
    else:
        print(f"   ❌ GPU確認失敗: {stderr}")
        return False
    
    return True

def setup_huggingface_auth(ip):
    """HuggingFace認証設定"""
    print("🤗 HuggingFace認証設定中...")
    
    # ローカルのHuggingFaceトークンを取得
    try:
        with open(os.path.expanduser("~/.cache/huggingface/token"), "r") as f:
            local_token = f.read().strip()
    except FileNotFoundError:
        try:
            # 別の場所も確認
            with open(os.path.expanduser("~/.huggingface/token"), "r") as f:
                local_token = f.read().strip()
        except FileNotFoundError:
            print("   ❌ ローカルHuggingFaceトークンが見つかりません")
            print("   📝 まずローカルで 'huggingface-cli login' を実行してください")
            return False
    
    if not local_token:
        print("   ❌ HuggingFaceトークンが空です")
        return False
    
    # Lambda Cloud上でHuggingFace認証を設定
    auth_command = f"echo '{local_token}' | huggingface-cli login --token stdin"
    success, stdout, stderr = run_ssh_command(ip, auth_command)
    
    if success and "Token is valid" in stdout:
        print("   ✅ HuggingFace認証設定完了")
        return True
    else:
        # 別の方法を試す
        env_command = f"mkdir -p ~/.cache/huggingface && echo '{local_token}' > ~/.cache/huggingface/token"
        success2, _, _ = run_ssh_command(ip, env_command)
        
        if success2:
            print("   ✅ HuggingFace認証設定完了（環境変数方式）")
            return True
        else:
            print(f"   ❌ HuggingFace認証設定失敗: {stderr}")
            return False

def setup_wandb_auth(ip):
    """WandB認証設定（DeepSpeed分散学習対応版）"""
    print("📊 WandB認証設定中...")
    
    # ローカルのWandBトークンを取得
    wandb_token = os.environ.get('WANDB_API_KEY')
    
    if not wandb_token:
        # wandb_api_key.txtファイルから確認（優先）
        try:
            with open("wandb_api_key.txt", "r") as f:
                wandb_token = f.read().strip()
                if wandb_token:
                    print("   ✅ WandB token found: wandb_api_key.txt")
        except FileNotFoundError:
            pass
    
    if not wandb_token:
        # netrcファイルからも確認
        try:
            import netrc
            netrc_data = netrc.netrc()
            if 'api.wandb.ai' in netrc_data.hosts:
                wandb_token = netrc_data.hosts['api.wandb.ai'][2]
        except (ImportError, FileNotFoundError, Exception):
            pass
    
    if not wandb_token:
        # wandb設定ファイルからも確認
        try:
            wandb_config_path = os.path.expanduser("~/.config/wandb/settings")
            with open(wandb_config_path, 'r') as f:
                content = f.read()
                if 'api_key' in content:
                    for line in content.split('\n'):
                        if line.startswith('api_key'):
                            wandb_token = line.split('=')[1].strip()
                            break
        except:
            pass
    
    if not wandb_token:
        print("   ⚠️ ローカルWandBトークンが見つかりません")
        print("   📝 以下のいずれかの方法でトークンを設定してください:")
        print("      1. wandb_api_key.txt ファイルにトークンを記載")
        print("      2. ローカルで 'wandb login' を実行")
        print("      3. 環境変数: export WANDB_API_KEY=your_token")
        return False
    
    # Step 1: wandb_api_key.txtファイルを Lambda Cloud に転送
    print("   📤 WandB APIキーファイルをLambda Cloudに転送中...")
    
    # ローカルにwandb_api_key.txtを作成/更新
    with open("wandb_api_key.txt", "w") as f:
        f.write(wandb_token)
    
    # rsyncでファイル転送
    rsync_command = f'rsync -avz -e "ssh -i ~/.ssh/lambda_cloud_key" wandb_api_key.txt ubuntu@{ip}:/lambda/nfs/lisa-gemma-project-fs/code/LISA-Gemma-Linux/'
    success, stdout, stderr = run_local_command(rsync_command)
    
    if not success:
        print(f"   ❌ WandB APIキーファイル転送失敗: {stderr}")
        return False
    
    print("   ✅ WandB APIキーファイル転送完了")
    
    # Step 2: Lambda Cloud上でDeepSpeed対応のWandB認証実行
    print("   🔐 Lambda Cloud上でWandB認証実行中...")
    
    # 成功した手動方式と同じコマンド構成
    auth_command = (
        "cd /lambda/nfs/lisa-gemma-project-fs/code/LISA-Gemma-Linux && "
        "source ../../venvs/lisa_gemma_venv/bin/activate && "
        "wandb login $(cat wandb_api_key.txt)"
    )
    
    success, stdout, stderr = run_ssh_command(ip, auth_command, timeout=120)
    
    if success and ("Successfully logged in" in stdout or "W&B API key is configured" in stdout):
        print("   ✅ WandB認証設定完了（DeepSpeed対応）")
        
        # Step 3: 認証確認
        verify_command = (
            "cd /lambda/nfs/lisa-gemma-project-fs/code/LISA-Gemma-Linux && "
            "source ../../venvs/lisa_gemma_venv/bin/activate && "
            "wandb status"
        )
        
        success_verify, stdout_verify, _ = run_ssh_command(ip, verify_command)
        if success_verify and "logged in" in stdout_verify:
            print("   ✅ WandB認証確認完了")
            return True
        else:
            print("   ⚠️ WandB認証確認で問題あり（ただし認証は成功）")
            return True
    else:
        print(f"   ❌ WandB認証設定失敗: {stderr}")
        
        # フォールバック: 環境変数方式
        print("   🔄 環境変数方式でWandB認証を再試行...")
        env_command = (
            f"export WANDB_API_KEY='{wandb_token}' && "
            f"echo 'export WANDB_API_KEY={wandb_token}' >> ~/.bashrc && "
            f"echo 'export WANDB_API_KEY={wandb_token}' >> ~/.profile"
        )
        success2, _, _ = run_ssh_command(ip, env_command)
        
        if success2:
            print("   ✅ WandB認証設定完了（環境変数フォールバック）")
            return True
        else:
            print("   ❌ WandB認証設定完全失敗")
            return False

def setup_project_environment(ip):
    """プロジェクト環境設定"""
    print("📁 プロジェクト環境設定中...")
    
    # プロジェクトディレクトリの確認
    check_command = "ls -la /lambda/nfs/lisa-gemma-project-fs/code/LISA-Gemma-Linux/"
    success, stdout, stderr = run_ssh_command(ip, check_command)
    
    if success:
        print("   ✅ プロジェクトディレクトリ確認完了")
    else:
        print(f"   ❌ プロジェクトディレクトリ確認失敗: {stderr}")
        return False
    
    # PyTorch環境確認（最も安全な形式）
    pytorch_command = "python3 -c 'import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count())'"
    success, stdout, stderr = run_ssh_command(ip, pytorch_command)
    
    if success:
        print(f"   ✅ PyTorch環境確認完了")
        print(f"      {stdout.strip()}")
    else:
        print(f"   ❌ PyTorch環境確認失敗: {stderr}")
        return False
    
    # 仮想環境の確認
    venv_command = "source /lambda/nfs/lisa-gemma-project-fs/venvs/lisa_gemma_venv/bin/activate && python -c 'import sys; print(sys.executable)'"
    success, stdout, stderr = run_ssh_command(ip, venv_command)
    
    if success:
        print(f"   ✅ 仮想環境確認完了: {stdout.strip()}")
        return True
    else:
        print(f"   ⚠️ 仮想環境確認失敗: {stderr}")
        print("   📝 仮想環境が見つからない場合は、後で手動で設定してください")
        return True  # 仮想環境は必須ではないのでTrueを返す

def create_setup_aliases(ip):
    """開発用エイリアス作成"""
    print("⚡ 開発用エイリアス設定中...")
    
    aliases = f'''
# LISA-Gemma Lambda Cloud エイリアス（チェックポイント保護版）
export LAMBDA_IP="{ip}"

# プロジェクト全体転送（チェックポイント保護）
alias lc-sync-all='rsync -avz --progress --delete --exclude=".git" --exclude="__pycache__" --exclude="*.pyc" --exclude="lambda_results" --exclude="verification_output" --exclude="vis_output" --exclude="preprocessed_checkpoints" --exclude="weights" --exclude="checkpoints" --exclude=".gitignore" -e "ssh -i ~/.ssh/lambda_cloud_key" ./ ubuntu@$LAMBDA_IP:/lambda/nfs/lisa-gemma-project-fs/code/LISA-Gemma-Linux/'

# 個別ファイル転送
alias lc-sync='rsync -avz --exclude=".git" --exclude="__pycache__" --exclude="*.pyc" --exclude=".gitignore" -e "ssh -i ~/.ssh/lambda_cloud_key"'

# Python実行
alias lc-run='ssh -i ~/.ssh/lambda_cloud_key ubuntu@{ip} "cd /lambda/nfs/lisa-gemma-project-fs/code/LISA-Gemma-Linux && source ../../venvs/lisa_gemma_venv/bin/activate && export TF_CPP_MIN_LOG_LEVEL=3 && export TF_ENABLE_ONEDNN_OPTS=0 && python"'

# DeepSpeed実行
alias lc-deepspeed='ssh -i ~/.ssh/lambda_cloud_key ubuntu@{ip} "cd /lambda/nfs/lisa-gemma-project-fs/code/LISA-Gemma-Linux && source ../../venvs/lisa_gemma_venv/bin/activate && export TF_CPP_MIN_LOG_LEVEL=3 && export TF_ENABLE_ONEDNN_OPTS=0 && deepspeed"'

# システム管理
alias lc-tmux='ssh -i ~/.ssh/lambda_cloud_key ubuntu@{ip} -t "tmux attach-session -t training || tmux new-session -s training"'
alias lc-gpu='ssh -i ~/.ssh/lambda_cloud_key ubuntu@{ip} "nvidia-smi"'
alias lc-connect='ssh -i ~/.ssh/lambda_cloud_key ubuntu@{ip}'

# 使用例:
# lc-sync-all  # プロジェクト全体転送（チェックポイント保護）
# lc-sync your_file.py ubuntu@{ip}:/lambda/nfs/lisa-gemma-project-fs/code/LISA-Gemma-Linux/
# lc-run your_file.py
# lc-deepspeed train_script.py
# lc-gpu
'''
    
    alias_file = Path.home() / ".lisa_lambda_aliases"
    try:
        with open(alias_file, 'w') as f:
            f.write(aliases)
        print(f"   ✅ エイリアス設定完了: {alias_file}")
        print("   📝 以下をbashrcに追加してください:")
        print(f"      source {alias_file}")
        return True
    except Exception as e:
        print(f"   ❌ エイリアス設定失敗: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Lambda Cloud A100*8 一括セットアップ")
    parser.add_argument("--ip", required=True, help="Lambda CloudインスタンスのIPアドレス")
    parser.add_argument("--skip-hf", action="store_true", help="HuggingFace認証をスキップ")
    parser.add_argument("--skip-wandb", action="store_true", help="WandB認証をスキップ")
    args = parser.parse_args()
    
    ip = args.ip
    
    print("🚀 Lambda Cloud A100*8 一括セットアップ開始")
    print(f"📍 対象IP: {ip}")
    print("=" * 60)
    
    # セットアップステップ
    steps = [
        ("SSH接続確認", lambda: check_ssh_connection(ip)),
        ("環境確認", lambda: check_environment(ip)),
    ]
    
    if not args.skip_hf:
        steps.append(("HuggingFace認証", lambda: setup_huggingface_auth(ip)))
    
    if not args.skip_wandb:
        steps.append(("WandB認証", lambda: setup_wandb_auth(ip)))
    
    steps.extend([
        ("プロジェクト環境設定", lambda: setup_project_environment(ip)),
        ("開発用エイリアス作成", lambda: create_setup_aliases(ip)),
    ])
    
    # セットアップ実行
    failed_steps = []
    
    for step_name, step_func in steps:
        print(f"\n{step_name}...")
        try:
            if step_func():
                print(f"✅ {step_name}: 成功")
            else:
                print(f"❌ {step_name}: 失敗")
                failed_steps.append(step_name)
        except Exception as e:
            print(f"❌ {step_name}: エラー - {e}")
            failed_steps.append(step_name)
        
        time.sleep(1)  # 少し待機
    
    # 結果レポート
    print("============================================================")
    print("🎯 セットアップ完了レポート")
    print("============================================================")
    
    if failed_steps:
        print(f"⚠️ {len(failed_steps)}個のセットアップが失敗しました:")
        for step in failed_steps:
            print(f"   - {step}")
        
        print("\n📝 手動で以下を確認してください:")
        if "HuggingFace認証" in failed_steps:
            print("   🤗 HuggingFace:")
            print("      1. ローカル: huggingface-cli login")
            print("      2. Lambda: huggingface-cli login（SSH接続後）")
        
        if "WandB認証" in failed_steps:
            print("   📊 WandB:")
            print("      1. ローカル: wandb login")
            print("      2. Lambda: wandb login（SSH接続後）")
            print("      3. または: export WANDB_API_KEY=your_key")
        
        if "プロジェクト環境設定" in failed_steps:
            print("   🐍 環境:")
            print("      1. 仮想環境確認: source /lambda/nfs/lisa-gemma-project-fs/venvs/lisa_gemma_venv/bin/activate")
            print("      2. PyTorch確認: python -c 'import torch; print(torch.__version__)'")
    else:
        print("🎉 全てのセットアップが成功しました！")
    
    print("\n🚀 次のステップ:")
    print("   1. エイリアス読み込み: source ~/.lisa_lambda_aliases")
    print("   2. ファイル転送テスト: lc-sync test.py")
    print("   3. コマンド実行テスト: lc-run 'echo Hello Lambda!'")
    print("   4. GPU確認: lc-gpu")
    
    print("\n🎯 今後の開発フロー:")
    print("   lc-sync your_file.py    # ファイル転送")
    print("   lc-run python your_file.py    # Python実行")
    print("   lc-connect    # SSH接続")
    print("   lc-tmux      # tmux管理")
    
    return len(failed_steps) == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 