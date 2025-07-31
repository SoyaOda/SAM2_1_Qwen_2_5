#!/usr/bin/env python3
"""
Lambda Cloud Development Utilities
Lambda Cloud環境での開発・学習実行を支援するユーティリティスクリプト
"""

import subprocess
import sys
import os
import time
from datetime import datetime
import json

# SSH設定
SSH_KEY = "~/.ssh/lambda_cloud_key"
LAMBDA_USER = "ubuntu"
CODE_PATH = "/lambda/nfs/lisa-gemma-project-fs/code/LISA-Gemma-Linux"
VENV_PATH = "/lambda/nfs/lisa-gemma-project-fs/venvs/lisa_gemma_venv"

# グローバル変数でIPアドレスをキャッシュ
_cached_lambda_ip = None

def get_lambda_ip():
    """Lambda Cloud IPアドレスを取得（キャッシュ機能付き）"""
    global _cached_lambda_ip
    
    # 既にキャッシュされている場合はそれを返す
    if _cached_lambda_ip:
        return _cached_lambda_ip
    
    # 環境変数から取得を試行
    if 'LAMBDA_IP' in os.environ:
        ip = os.environ['LAMBDA_IP']
        print(f"🔍 環境変数からIP取得: {ip}")
        _cached_lambda_ip = ip
        return ip
    
    # 対話式で入力
    print("🌐 Lambda Cloud IPアドレスを入力してください")
    print("   例: 150.136.114.187 (A100)")
    print("   例: 150.136.47.58 (A10)")
    
    while True:
        ip = input("Lambda Cloud IP: ").strip()
        if not ip:
            print("❌ IPアドレスを入力してください")
            continue
        
        # 簡易的なIPアドレス形式チェック
        parts = ip.split('.')
        if len(parts) != 4:
            print("❌ 正しいIPアドレス形式で入力してください (例: 150.136.114.187)")
            continue
        
        try:
            for part in parts:
                int(part)
            print(f"✅ IP設定: {ip}")
            _cached_lambda_ip = ip
            return ip
        except ValueError:
            print("❌ 正しいIPアドレス形式で入力してください (例: 150.136.114.187)")
            continue

def set_lambda_ip(ip):
    """Lambda Cloud IPアドレスを設定"""
    global _cached_lambda_ip
    _cached_lambda_ip = ip

# Hugging Face Token設定
HF_TOKEN_FILE = "hf_token.txt"

# WandB API Key設定
WANDB_API_KEY_FILE = "wandb_api_key.txt"

def run_ssh_command(command, use_tmux=False, session_name=None, detach=False, timeout=None, lambda_ip=None):
    """SSH経由でコマンドを実行"""
    # IPアドレスの取得
    if lambda_ip is None:
        lambda_ip = get_lambda_ip()
    
    # TensorFlow初期化を無効化する環境変数を追加
    tf_disable_env = "export TF_CPP_MIN_LOG_LEVEL=3 && export TF_ENABLE_ONEDNN_OPTS=0 && "
    ssh_base = f"ssh -i {SSH_KEY} {LAMBDA_USER}@{lambda_ip}"
    
    if use_tmux:
        session_name = session_name or "lisa_dev"
        if detach:
            # バックグラウンドで tmux セッションを作成し、コマンドを実行
            tmux_cmd = f"tmux new-session -d -s {session_name} 'cd {CODE_PATH} && source {VENV_PATH}/bin/activate && {tf_disable_env}{command}'"
        else:
            # tmux セッションにアタッチ
            tmux_cmd = f"tmux attach-session -t {session_name} || tmux new-session -s {session_name} 'cd {CODE_PATH} && source {VENV_PATH}/bin/activate && {tf_disable_env}{command}'"
        
        full_command = f"{ssh_base} \"{tmux_cmd}\""
    else:
        full_command = f"{ssh_base} \"cd {CODE_PATH} && source {VENV_PATH}/bin/activate && {tf_disable_env}{command}\""
    
    print(f"🚀 実行中: {full_command}")
    
    if detach:
        # バックグラウンド実行
        process = subprocess.Popen(full_command, shell=True)
        print(f"✅ バックグラウンドで実行開始 (PID: {process.pid})")
        return process
    else:
        # 前景実行
        try:
            if timeout:
                result = subprocess.run(full_command, shell=True, capture_output=True, text=True, timeout=timeout)
            else:
                result = subprocess.run(full_command, shell=True, capture_output=True, text=True)
            return result.returncode == 0, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return False, "", "Command timed out"
        except Exception as e:
            return False, "", str(e)

def sync_code(lambda_ip=None):
    """ローカルコードをLambda Cloudに同期"""
    if lambda_ip is None:
        lambda_ip = get_lambda_ip()
    
    print("📤 コードをLambda Cloudに同期中...")
    
    rsync_cmd = f"""rsync -avz --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
        --exclude='lambda_results' --exclude='runs' \
        ./ {LAMBDA_USER}@{lambda_ip}:{CODE_PATH}/"""
    
    result = subprocess.run(rsync_cmd, shell=True)
    if result.returncode == 0:
        print("✅ コード同期完了")
    else:
        print("❌ コード同期失敗")
    return result.returncode == 0

def sync_results(lambda_ip=None):
    """Lambda Cloudから結果をローカルに同期"""
    if lambda_ip is None:
        lambda_ip = get_lambda_ip()
    
    print("📥 結果をLambda Cloudから取得中...")
    
    # ローカルの結果ディレクトリを作成
    os.makedirs("lambda_results", exist_ok=True)
    
    rsync_cmd = f"""rsync -avz {LAMBDA_USER}@{lambda_ip}:/lambda/nfs/lisa-gemma-project-fs/artifacts/ \
        ./lambda_results/"""
    
    result = subprocess.run(rsync_cmd, shell=True)
    if result.returncode == 0:
        print("✅ 結果取得完了")
    else:
        print("❌ 結果取得失敗")
    return result.returncode == 0

def list_tmux_sessions():
    """tmuxセッション一覧を表示"""
    print("📋 Lambda Cloud上のtmuxセッション:")
    run_ssh_command("tmux list-sessions", use_tmux=False)

def attach_tmux_session(session_name="lisa_dev"):
    """tmuxセッションにアタッチ"""
    print(f"🔗 tmuxセッション '{session_name}' にアタッチ中...")
    run_ssh_command("", use_tmux=True, session_name=session_name)

def kill_tmux_session(session_name="lisa_dev"):
    """tmuxセッションを終了"""
    print(f"🛑 tmuxセッション '{session_name}' を終了中...")
    run_ssh_command(f"tmux kill-session -t {session_name}", use_tmux=False)

def check_gpu_status():
    """GPU使用状況を確認"""
    print("🖥️  GPU使用状況:")
    run_ssh_command("nvidia-smi", use_tmux=False)

def emergency_stop():
    """緊急停止: 全てのPythonプロセスを停止"""
    print("🚨 緊急停止: 全Pythonプロセスを終了中...")
    
    # 確認
    response = input("⚠️  本当に全てのPythonプロセスを停止しますか？ (y/N): ")
    if response.lower() != 'y':
        print("❌ キャンセルされました")
        return
    
    # 全Pythonプロセスを終了
    run_ssh_command("pkill -f python", use_tmux=False)
    print("✅ 緊急停止完了")

def start_training(script_name, experiment_name=None, use_tmux=True):
    """学習開始"""
    if experiment_name is None:
        experiment_name = f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print(f"🏋️  学習開始: {script_name} (実験名: {experiment_name})")
    
    # まずコードを同期
    if not sync_code():
        print("❌ コード同期に失敗しました")
        return False
    
    # 学習コマンドを構築
    training_cmd = f"python {script_name} --experiment_name {experiment_name}"
    
    if use_tmux:
        print("📺 tmuxセッションで学習を開始します...")
        print("   SSH接続が切れても学習は継続されます")
        print(f"   再接続するには: python {__file__} attach")
        
        # tmuxでバックグラウンド実行
        process = run_ssh_command(training_cmd, use_tmux=True, 
                                session_name=f"training_{experiment_name}", 
                                detach=True)
        
        time.sleep(2)  # tmuxセッション開始を待つ
        print(f"✅ 学習開始完了 (tmuxセッション: training_{experiment_name})")
        return True
    else:
        # 前景実行（非推奨）
        print("⚠️  前景実行での学習開始 (SSH切断で停止します)")
        success, stdout, stderr = run_ssh_command(training_cmd, use_tmux=False)
        return success

def monitor_training():
    """学習状況をモニタリング"""
    print("📊 学習状況モニタリング:")
    print("=" * 50)
    
    # GPU使用状況
    check_gpu_status()
    print()
    
    # tmuxセッション一覧
    list_tmux_sessions()
    print()
    
    # 最新のログファイルを表示
    print("📝 最新ログ:")
    run_ssh_command("find /lambda/nfs/lisa-gemma-project-fs/artifacts/logs -name '*.log' -type f -exec ls -lt {} + | head -5", use_tmux=False)

def setup_instance():
    """新しいLambda Cloudインスタンスをセットアップ"""
    print("🏗️  Lambda Cloudインスタンスをセットアップ中...")
    
    setup_commands = [
        # システム更新と基本ツールインストール
        "sudo apt-get update",
        "sudo apt-get install -y tmux htop tree",
        
        # シンボリックリンク作成
        "ln -sfn /lambda/nfs/lisa-gemma-project-fs ~/persistent_storage",
        "ln -sfn /lambda/nfs/lisa-gemma-project-fs/code/LISA-Gemma-Linux ~/project",
        
        # 環境確認
        f"cd {CODE_PATH} && source {VENV_PATH}/bin/activate && python config_lambda_cloud.py"
    ]
    
    print("📋 実行するセットアップ手順:")
    for i, cmd in enumerate(setup_commands, 1):
        print(f"  {i}. {cmd}")
    
    response = input("\n⚠️  セットアップを実行しますか？ (y/N): ")
    if response.lower() != 'y':
        print("❌ セットアップをキャンセルしました")
        return
    
    # セットアップ実行
    for i, cmd in enumerate(setup_commands, 1):
        print(f"\n🔧 ステップ {i}/{len(setup_commands)}: {cmd}")
        success, stdout, stderr = run_ssh_command(cmd, use_tmux=False)
        if not success:
            print(f"❌ ステップ {i} でエラーが発生しました")
            return False
    
    print("\n✅ セットアップ完了！")
    print("📂 利用可能なショートカット:")
    print("  ~/persistent_storage → Persistent Filesystem")
    print("  ~/project → プロジェクトルート")
    
    return True

def check_cloud():
    """Lambda Cloud環境をチェック"""
    print("🔍 Lambda Cloud環境をチェック中...")
    
    checks = {}
    
    # SSH接続をチェック
    print("📋 SSH接続をチェック中...")
    success, stdout, stderr = run_ssh_command("echo 'SSH OK'", use_tmux=False)
    checks["SSH接続"] = success
    print(f"   {'✅' if checks['SSH接続'] else '❌'} SSH接続")
    
    # Filesystemをチェック
    print("📋 Filesystem存在をチェック中...")
    success, stdout, stderr = run_ssh_command("ls -la /lambda/nfs/lisa-gemma-project-fs", use_tmux=False)
    checks["Filesystem存在"] = success
    print(f"   {'✅' if checks['Filesystem存在'] else '❌'} Filesystem存在")
    
    # 仮想環境をチェック
    print("📋 仮想環境をチェック中...")
    success, stdout, stderr = run_ssh_command(f"source {VENV_PATH}/bin/activate && python --version", use_tmux=False)
    checks["仮想環境"] = success
    print(f"   {'✅' if checks['仮想環境'] else '❌'} 仮想環境")
    
    # GPU認識をチェック
    print("📋 GPU認識をチェック中...")
    success, stdout, stderr = run_ssh_command("nvidia-smi --query-gpu=name --format=csv,noheader", use_tmux=False)
    checks["GPU認識"] = success
    print(f"   {'✅' if checks['GPU認識'] else '❌'} GPU認識")
    if success and stdout:
        print(f"       GPU: {stdout.strip()}")
    
    # プロジェクト設定をチェック
    print("📋 プロジェクト設定をチェック中...")
    success, stdout, stderr = run_ssh_command(f"cd {CODE_PATH} && python config_linux.py", use_tmux=False)
    checks["プロジェクト設定"] = success
    print(f"   {'✅' if checks['プロジェクト設定'] else '❌'} プロジェクト設定")
    
    # パッケージをチェック
    print("📋 必要パッケージをチェック中...")
    success, stdout, stderr = run_ssh_command(
        f"source {VENV_PATH}/bin/activate && pip list | grep -E 'torch|transformers|deepspeed'",
        use_tmux=False
    )
    # 出力から必要なパッケージが見つかるかチェック
    checks["必要パッケージ"] = success
    print(f"   {'✅' if checks['必要パッケージ'] else '❌'} 必要パッケージ")
    
    # Hugging Face認証をチェック
    print("📋 Hugging Face認証をチェック中...")
    success, stdout, stderr = run_ssh_command(
        f"source {VENV_PATH}/bin/activate && huggingface-cli whoami",
        use_tmux=False
    )
    checks["HF認証"] = success
    print(f"   {'✅' if checks['HF認証'] else '❌'} Hugging Face認証")
    
    # WandB認証をチェック
    print("📋 WandB認証をチェック中...")
    success, stdout, stderr = run_ssh_command(
        f"source {VENV_PATH}/bin/activate && wandb status",
        use_tmux=False
    )
    checks["WandB認証"] = success
    print(f"   {'✅' if checks['WandB認証'] else '❌'} WandB認証")
    
    print("\n📊 環境チェック結果:")
    print("=" * 30)
    for check_name, status in checks.items():
        print(f"{'✅' if status else '❌'} {check_name}")
    
    all_passed = all(checks.values())
    if all_passed:
        print("\n🎉 全てのチェックが成功しました！")
    else:
        print("\n⚠️  一部のチェックが失敗しています。修正が必要です。")
        if not checks.get("HF認証", False):
            print("💡 Hugging Face認証が必要です: python lambda_dev_utils.py setup_hf <your_token>")
        if not checks.get("WandB認証", False):
            print("💡 WandB認証が必要です: python lambda_dev_utils.py setup_wandb <your_api_key>")
        
    return all_passed

def install_missing_packages():
    """不足パッケージの追加インストール"""
    print("📦 追加パッケージをインストール中...")
    
    additional_packages = [
        "mlflow",
        "boto3", 
        "tensorboard",
        "wandb",
        "matplotlib",
        "seaborn"
    ]
    
    install_cmd = f"source {VENV_PATH}/bin/activate && pip install " + " ".join(additional_packages)
    
    print(f"インストール対象: {', '.join(additional_packages)}")
    response = input("インストールを実行しますか？ (y/N): ")
    
    if response.lower() == 'y':
        success, stdout, stderr = run_ssh_command(install_cmd, use_tmux=False)
        if success:
            print("✅ パッケージインストール完了")
        else:
            print("❌ パッケージインストール失敗")
        return success
    else:
        print("❌ インストールをキャンセルしました")
        return False

def run_validation_test():
    """エンドツーエンド検証テスト"""
    print("🧪 エンドツーエンド検証テストを実行中...")
    
    # まずコードを同期
    if not sync_code():
        print("❌ コード同期に失敗しました")
        return False
    
    # 検証スクリプトを実行
    test_commands = [
        # GPU確認
        "nvidia-smi",
        
        # 基本的なPythonテスト
        f"cd {CODE_PATH} && source {VENV_PATH}/bin/activate && python -c \"import torch; print(f'PyTorch: {{torch.__version__}}'); print(f'CUDA available: {{torch.cuda.is_available()}}'); print(f'GPU count: {{torch.cuda.device_count()}}')\"",
        
        # 設定ファイルテスト
        f"cd {CODE_PATH} && source {VENV_PATH}/bin/activate && python config_lambda_cloud.py",
        
        # 簡単なモデルテスト（利用可能な場合）
        f"cd {CODE_PATH} && source {VENV_PATH}/bin/activate && python -c \"from transformers import AutoTokenizer; t = AutoTokenizer.from_pretrained('google/gemma-2-9b-it'); print('Tokenizer loaded successfully')\"" if os.path.exists("test_basic_model.py") else "echo 'Model test skipped'"
    ]
    
    print("🧪 実行する検証テスト:")
    for i, cmd in enumerate(test_commands, 1):
        print(f"  {i}. {cmd[:60]}...")
    
    success_count = 0
    for i, cmd in enumerate(test_commands, 1):
        print(f"\n🧪 テスト {i}/{len(test_commands)}")
        success, stdout, stderr = run_ssh_command(cmd, use_tmux=False)
        if success:
            print(f"✅ テスト {i} 成功")
            success_count += 1
        else:
            print(f"❌ テスト {i} 失敗")
    
    print(f"\n📊 検証結果: {success_count}/{len(test_commands)} テスト成功")
    
    if success_count == len(test_commands):
        print("🎉 全ての検証テストが成功しました！")
        print("🚀 Lambda Cloud環境は使用準備完了です")
        return True
    else:
        print("⚠️  一部のテストが失敗しています。環境の確認が必要です。")
        return False

def setup_hf_token(token=None):
    """Hugging Face Tokenを安全に設定"""
    if token:
        # ローカルにトークンファイルを作成（.gitignoreで除外済み）
        with open(HF_TOKEN_FILE, 'w') as f:
            f.write(token.strip())
        print(f"✅ Hugging Face Tokenをローカルに保存しました: {HF_TOKEN_FILE}")
        
        # Lambda CloudでHugging Face CLIに直接ログイン
        print("🔐 Lambda CloudでHugging Face CLIにログイン中...")
        success, stdout, stderr = run_ssh_command(
            f"source {VENV_PATH}/bin/activate && huggingface-cli login --token {token.strip()}",
            use_tmux=False
        )
        if success:
            print("✅ Hugging Face CLIログイン成功")
            return True
        else:
            print("❌ Hugging Face CLIログイン失敗")
            print(f"エラー出力: {stderr}")
            print(f"標準出力: {stdout}")
            return False
    else:
        # 既存のトークンファイルから読み込み
        if os.path.exists(HF_TOKEN_FILE):
            with open(HF_TOKEN_FILE, 'r') as f:
                token = f.read().strip()
            return setup_hf_token(token)
        else:
            print(f"❌ Tokenファイルが見つかりません: {HF_TOKEN_FILE}")
            print("使用方法: python lambda_dev_utils.py setup_hf <your_token>")
            return False

def check_hf_auth():
    """Hugging Face認証状態をチェック"""
    print("🔐 Hugging Face認証状態をチェック中...")
    success, stdout, stderr = run_ssh_command(
        f"source {VENV_PATH}/bin/activate && huggingface-cli whoami",
        use_tmux=False
    )
    if success:
        print("✅ Hugging Face認証済み")
        return True
    else:
        print("❌ Hugging Face未認証")
        return False

def sync_hf_token(lambda_ip=None):
    """hf_token.txtファイルをLambda Cloudに転送"""
    if lambda_ip is None:
        lambda_ip = get_lambda_ip()
    
    if not os.path.exists(HF_TOKEN_FILE):
        print(f"❌ Tokenファイルが見つかりません: {HF_TOKEN_FILE}")
        print("💡 まず、Hugging Face Tokenをhf_token.txtファイルに保存してください")
        return False
    
    print(f"📤 {HF_TOKEN_FILE}をLambda Cloudに転送中...")
    
    rsync_cmd = f"""rsync -avz -e "ssh -i {SSH_KEY}" {HF_TOKEN_FILE} {LAMBDA_USER}@{lambda_ip}:{CODE_PATH}/"""
    
    result = subprocess.run(rsync_cmd, shell=True)
    if result.returncode == 0:
        print("✅ Tokenファイル転送完了")
        print("💡 次に以下のコマンドでHugging Face認証を設定してください：")
        print(f"   python lambda_dev_utils.py setup_hf --ip {lambda_ip}")
        return True
    else:
        print("❌ Tokenファイル転送失敗")
        return False

def setup_wandb_key(api_key=None):
    """WandB API Keyを安全に設定"""
    if api_key:
        # ローカルにAPIキーファイルを作成（.gitignoreで除外済み）
        with open(WANDB_API_KEY_FILE, 'w') as f:
            f.write(api_key.strip())
        print(f"✅ WandB API Keyをローカルに保存しました: {WANDB_API_KEY_FILE}")
        
        # Lambda CloudでWandBにログイン
        print("🔐 Lambda CloudでWandBにログイン中...")
        success, stdout, stderr = run_ssh_command(
            f"source {VENV_PATH}/bin/activate && wandb login {api_key.strip()}",
            use_tmux=False
        )
        if success:
            print("✅ WandBログイン成功")
            return True
        else:
            print("❌ WandBログイン失敗")
            print(f"エラー出力: {stderr}")
            print(f"標準出力: {stdout}")
            return False
    else:
        # 既存のAPIキーファイルから読み込み
        if os.path.exists(WANDB_API_KEY_FILE):
            with open(WANDB_API_KEY_FILE, 'r') as f:
                api_key = f.read().strip()
            return setup_wandb_key(api_key)
        else:
            print(f"❌ API Keyファイルが見つかりません: {WANDB_API_KEY_FILE}")
            print("使用方法: python lambda_dev_utils.py setup_wandb <your_api_key>")
            return False

def check_wandb_auth():
    """WandB認証状態をチェック"""
    print("🔐 WandB認証状態をチェック中...")
    success, stdout, stderr = run_ssh_command(
        f"source {VENV_PATH}/bin/activate && wandb status",
        use_tmux=False
    )
    if success:
        print("✅ WandB認証済み")
        return True
    else:
        print("❌ WandB未認証")
        return False

def sync_wandb_token(lambda_ip=None):
    """wandb_api_key.txtファイルをLambda Cloudに転送"""
    if lambda_ip is None:
        lambda_ip = get_lambda_ip()
    
    if not os.path.exists(WANDB_API_KEY_FILE):
        print(f"❌ API Keyファイルが見つかりません: {WANDB_API_KEY_FILE}")
        print("💡 まず、WandB API Keyをwandb_api_key.txtファイルに保存してください")
        return False
    
    print(f"📤 {WANDB_API_KEY_FILE}をLambda Cloudに転送中...")
    
    rsync_cmd = f"""rsync -avz -e "ssh -i {SSH_KEY}" {WANDB_API_KEY_FILE} {LAMBDA_USER}@{lambda_ip}:{CODE_PATH}/"""
    
    result = subprocess.run(rsync_cmd, shell=True)
    if result.returncode == 0:
        print("✅ API Keyファイル転送完了")
        print("💡 次に以下のコマンドでWandB認証を設定してください：")
        print(f"   python lambda_dev_utils.py setup_wandb --ip {lambda_ip}")
        return True
    else:
        print("❌ API Keyファイル転送失敗")
        return False

def main():
    """メイン関数"""
    if len(sys.argv) < 2:
        print("🚀 Lambda Cloud開発ユーティリティ")
        print("=" * 50)
        print("使用可能なコマンド:")
        print("  sync        - ローカルコードをLambda Cloudに同期")
        print("  train       - 学習スクリプトを実行 (tmux)")
        print("  monitor     - GPU使用状況とトレーニング状況を監視")
        print("  results     - 学習結果を取得")
        print("  setup       - 新しいインスタンスのセットアップ")
        print("  check       - 環境の健全性チェック")
        print("  install     - 追加パッケージのインストール")
        print("  validate    - エンドツーエンド検証テスト")
        print("  emergency   - 緊急停止（全tmuxセッション終了）")
        print("  setup_hf    - Hugging Face Tokenを設定")
        print("  sync_token  - hf_token.txtをLambda Cloudに転送")
        print("  setup_wandb - WandB API Keyを設定")
        print("  sync_wandb  - wandb_api_key.txtをLambda Cloudに転送")
        print("")
        print("オプション:")
        print("  --ip IP_ADDRESS  - Lambda Cloud IPアドレスを指定")
        print("")
        print("例:")
        print("  python lambda_dev_utils.py sync")
        print("  python lambda_dev_utils.py check --ip 150.136.114.187")
        print("  python lambda_dev_utils.py train train_ds.py --ip 150.136.47.58")
        print("  python lambda_dev_utils.py setup_hf hf_xxxxxxx")
        print("  python lambda_dev_utils.py setup_hf")
        print("  python lambda_dev_utils.py sync_token --ip 129.213.22.187")
        print("  python lambda_dev_utils.py setup_wandb a389f0xxxxxxx")
        print("  python lambda_dev_utils.py setup_wandb")
        print("  python lambda_dev_utils.py sync_wandb --ip 150.136.36.116")
        return
    
    # IPアドレスの処理
    lambda_ip = None
    args = sys.argv[1:]
    
    # --ipオプションを探す
    if '--ip' in args:
        ip_index = args.index('--ip')
        if ip_index + 1 < len(args):
            lambda_ip = args[ip_index + 1]
            set_lambda_ip(lambda_ip)
            print(f"🔍 コマンドライン引数からIP取得: {lambda_ip}")
            # --ipとIPアドレスを引数リストから削除
            args = args[:ip_index] + args[ip_index + 2:]
        else:
            print("❌ --ipオプションにIPアドレスが指定されていません")
            return
    
    if not args:
        print("❌ コマンドが指定されていません")
        return
    
    command = args[0]
    
    if command == "sync":
        sync_code()
    
    elif command == "results":
        sync_results()
    
    elif command == "train":
        if len(args) < 2:
            print("❌ 学習スクリプト名を指定してください")
            return
        script_name = args[1]
        experiment_name = args[2] if len(args) > 2 else None
        start_training(script_name, experiment_name)
    
    elif command == "monitor":
        monitor_training()
    
    elif command == "gpu":
        check_gpu_status()
    
    elif command == "setup":
        setup_instance()
    
    elif command == "check":
        check_cloud()
    
    elif command == "install":
        install_missing_packages()
    
    elif command == "validate":
        run_validation_test()
    
    elif command == "tmux-list":
        list_tmux_sessions()
    
    elif command == "attach":
        session_name = args[1] if len(args) > 1 else "lisa_dev"
        attach_tmux_session(session_name)
    
    elif command == "kill":
        session_name = args[1] if len(args) > 1 else "lisa_dev"
        kill_tmux_session(session_name)
    
    elif command == "emergency":
        emergency_stop()
    
    elif command == "setup_hf":
        if len(args) >= 2:
            # コマンドライン引数からToken指定
            token = args[1]
            setup_hf_token(token)
        else:
            # hf_token.txtファイルから自動読取り
            print("📄 hf_token.txtファイルから自動読取り中...")
            if setup_hf_token():
                print("✅ Hugging Face認証設定完了")
            else:
                print("❌ 認証設定に失敗しました")
                print("💡 使用方法: python lambda_dev_utils.py setup_hf <your_token>")
    
    elif command == "sync_token":
        sync_hf_token()
    
    elif command == "setup_wandb":
        if len(args) >= 2:
            # コマンドライン引数からAPI Key指定
            api_key = args[1]
            setup_wandb_key(api_key)
        else:
            # wandb_api_key.txtファイルから自動読取り
            print("📄 wandb_api_key.txtファイルから自動読取り中...")
            if setup_wandb_key():
                print("✅ WandB認証設定完了")
            else:
                print("❌ 認証設定に失敗しました")
                print("💡 使用方法: python lambda_dev_utils.py setup_wandb <your_api_key>")
    
    elif command == "sync_wandb":
        sync_wandb_token()
    
    else:
        print(f"❌ 不明なコマンド: {command}")

if __name__ == "__main__":
    main() 