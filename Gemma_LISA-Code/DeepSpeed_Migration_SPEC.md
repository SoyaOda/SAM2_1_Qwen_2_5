LISA-GemmaモデルのLambda Cloudにおけるスケーラブルなファインチューニング実装仕様書序文：戦略的概要とアーキテクチャ設計本仕様書は、ローカル環境で開発されたLISA-Gemmaプロジェクトを、Lambda CloudのGPUインフラストラクチャへ移行し、スケーラブルなファインチューニングから本番学習までを完遂するための段階的実装計画を定義するものです。本計画は、コスト効率、再現性、および拡張性を最大化するモダンなMLOps（Machine Learning Operations）のベストプラクティスに基づいています。1.1. 用語の明確化：Lambda CloudとAWS Lambda本仕様書全体を通じて、「Lambda」という用語は、機械学習ワークロードに特化したIaaS（Infrastructure as a Service）プロバイダーであるLambda Labs GPU Cloudを指します 1。同様に、「Lambdaデータベース」という用語は、Lambda Cloudが提供する永続ストレージサービスであるPersistent Filesystemを指すものと解釈します 3。これは、サーバーレスコンピューティングサービスであるAWS Lambdaとは異なるサービスです。AWS Lambdaもファイルシステム（Amazon EFS）との連携が可能ですが 4、本プロジェクトのような大規模なモデルの長期的なファインチューニングや、H100のような高性能GPUを複数使用するユースケースには、Lambda Cloudのような特化型GPUクラウドがより適しています。この区別は、以降のアーキテクチャ設計の根幹をなすため、極めて重要です。1.2. ターゲットアーキテクチャの全体像本プロジェクトで構築する最終的なアーキテクチャは、ストレージとコンピュートを完全に分離する設計思想に基づいています。これにより、高価なGPUインスタンスを必要な時にのみ起動し、コストを最適化しつつ、データやモデル、環境の永続性を担保します。データと制御のフローは以下の通りです：ローカル開発環境: 初期コードとデータの供給源。GitHubリポジトリ: 全てのコードの単一の信頼できる情報源（Single Source of Truth）。Lambda Cloud Persistent Filesystem: データセット、ソースコード、Python仮想環境、生成されたモデルアーティファクト（チェックポイント、ログ、最終モデル）を永続的に保管する中央ストレージ。Lambda Cloud GPUインスタンス: 起動時にPersistent Filesystemをマウントする、エフェメラル（一時的）な計算リソース。デバッグ用の単一GPUから本番学習用の8x H100まで、フェーズに応じて使い分ける。Cursor IDE: SSH経由でGPUインスタンスに接続し、AIエージェント（Claude-4-sonnet）に対話的に指示を与えるためのインターフェース。このアーキテクチャの最大の利点は、GPUインスタンスを終了しても、全ての重要な資産がPersistent Filesystem上に安全に保持される点です 1。これにより、計算コストを時間単位で厳密に管理できます。1.3. コア技術スタックの選定理由本プロジェクトで採用する主要な技術とその選定理由は以下の通りです。rsync: ローカル環境からリモートサーバーへの大規模な初回データ転送において、その堅牢性、SSH経由での利便性、および実績から最適なツールとして選定します 7。また、Lambda Cloudが提供するFilesystem S3 Adapterが利用可能な場合、より高速な並列転送を実現するs5cmdを代替手段として検討します 3。Lambda Stack: 全てのLambda Cloudインスタンスにプリインストールされており、CUDA、PyTorch、cuDNNなど、テスト済みの機械学習関連ソフトウェア群を提供します。これにより、環境構築の手間が大幅に削減されます 9。DeepSpeed: 複数GPUにまたがる大規模なモデル学習を効率化するためのライブラリです。特に、メモリ使用量を最適化するZeRO (Zero Redundancy Optimizer) を活用します 10。Hugging Face peft & trl: ユーザーの既存リポジトリでも採用されている、LoRA (Low-Rank Adaptation) などのパラメータ効率の良いファインチューニング（PEFT）における標準的なライブラリです。CSV Logger / MLflow: 本番学習前の「パラメータや学習戦略の試行錯誤」フェーズにおいて、ハイパーパラメータ最適化（HPO）の実験結果を体系的に追跡・管理するために導入します。単純なCSVロガーから始め、必要に応じてMLflowのような高機能なツールへ拡張することも可能です 12。表1：フェーズ別インスタンス割り当てとコスト概算以下の表は、プロジェクトの各フェーズで使用を推奨するLambda Cloudインスタンス、その目的、およびオンデマンド料金に基づいたコストの概算を示します。これにより、技術的なロードマップと予算計画を明確に連携させます。フェーズ目的推奨インスタンスGPU構成VRAM/GPUオンデマンド料金 (USD/hr)選定理由1データ移行1x NVIDIA A101x A1024 GB$0.75データ転送用の安価な一時的エンドポイントとして利用 2。2開発・デバッグ1x NVIDIA A100 SXM1x A10040 GB$1.29全体のパイプラインを検証するための、性能とコストのバランスが良い単一GPU環境 2。3スケールアップ検証4x NVIDIA H100 SXM4x H10080 GB$12.36 (4 * $3.09)8x H100への移行前段階として、マルチGPU設定とDeepSpeedの動作を検証 2。4 & 5HPO・本番学習8x NVIDIA H100 SXM8x H10080 GB$23.92 (8 * $2.99)プロジェクトの最終目標である、最大規模でのハイパーパラメータ探索および本番学習を実行 14。注：料金は2024年時点のオンデマンド価格に基づく概算であり、変動する可能性があります 2。フェーズ1：基盤構築 — データとコードの移行目的: プロジェクトの全てのデータとコードを、ローカル環境から完全に分離し、Lambda Cloudエコシステム内に単一の永続的かつ信頼できる情報源を確立する。2.1. Lambda Cloud Persistent FilesystemのプロビジョニングLambda Cloudのダッシュボードにログインします。左側のメニューから「Filesystems」を選択し、「Create a filesystem」をクリックします。以下の情報を設定します。Name: プロジェクト固有の識別可能な名前（例: lisa-gemma-project-fs）を設定します。Region: GPUインスタンスを起動する予定のリージョンと同一のリージョンを選択します。リージョンが異なるとインスタンスにアタッチできないため、これは必須要件です 3。「Create filesystem」をクリックしてプロビジョニングを開始します。このファイルシステムは、アタッチされているインスタンスがなくても、存在する限りGBあたりの月額料金が発生します 3。これは、長期的なデータ保管庫としての役割を裏付けています。2.2. ローカルからPersistent Filesystemへのデータ移行このステップでは、ローカルのデータセットをクラウド上のPersistent Filesystemに転送します。rsyncコマンドを使用し、堅牢なデータ転送を実現します。一時的な転送用インスタンスの起動:Lambda Cloudダッシュボードから、最も安価なGPUインスタンス（例: 1x A10）を1台起動します。起動設定時に、必ずステップ2.1で作成したlisa-gemma-project-fsを「Attach filesystems」で選択します。インスタンスが「Running」状態になったら、IPアドレスを控えます。データ転送スクリプトの実行:ローカルマシンで以下の内容のシェルスクリプト migrate_data.sh を作成します。Bash#!/bin/bash
set -e # エラーが発生したら即座に終了

# --- ユーザー設定項目 ---
LOCAL_DATA_PATH="/path/to/your/local/database" # ローカルのデータセットディレクトリ
LAMBDA_SSH_KEY="/path/to/your/lambda_ssh_key.pem" # Lambda CloudのSSH秘密鍵
INSTANCE_IP="<YOUR_INSTANCE_IP>" # 手順1で控えたIPアドレス
FILESYSTEM_NAME="lisa-gemma-project-fs"
# --- 設定項目終了 ---

REMOTE_USER="ubuntu"
REMOTE_TARGET_PATH="/lambda/nfs/${FILESYSTEM_NAME}/data"

echo "ローカルパス: ${LOCAL_DATA_PATH}"
echo "転送先: ${REMOTE_USER}@${INSTANCE_IP}:${REMOTE_TARGET_PATH}"
echo "データ転送を開始します..."

# rsyncコマンドの実行
# -a: アーカイブモード（パーミッション、タイムスタンプ等を保持）
# -v: 詳細な情報を表示
# -z: 転送中にデータを圧縮
# --info=progress2: 全体の進捗を表示
# -e: 使用するSSHコマンドを指定
rsync -avz --info=progress2 -e "ssh -i ${LAMBDA_SSH_KEY}" "${LOCAL_DATA_PATH}/" "${REMOTE_USER}@${INSTANCE_IP}:${REMOTE_TARGET_PATH}/"

echo "データ転送が完了しました。"
ローカルマシンのターミナルで bash migrate_data.sh を実行し、データ転送を開始します。2.3. Persistent Filesystem上のプロジェクト構造の確立データ転送後、転送用インスタンス上でプロジェクト全体のディレクトリ構造を整備します。転送用インスタンスにSSHで接続します。以下のコマンドを実行して、必要なディレクトリを作成し、GitHubからソースコードをクローンします。Bash# スクリプト化して実行することを推奨
# setup_filesystem.sh

# 変数の定義
FILESYSTEM_ROOT="/lambda/nfs/lisa-gemma-project-fs"
PROJECT_REPO="https://github.com/SoyaOda/LISA-Gemma-Linux_2.git"
PROJECT_DIR_NAME="LISA-Gemma-Linux_2"
TARGET_BRANCH="deepspeed_migration"

# 1. プロジェクト用のディレクトリを作成
mkdir -p "${FILESYSTEM_ROOT}/code"
mkdir -p "${FILESYSTEM_ROOT}/artifacts/checkpoints"
mkdir -p "${FILESYSTEM_ROOT}/artifacts/logs"
mkdir -p "${FILESYSTEM_ROOT}/artifacts/final_models"
mkdir -p "${FILESYSTEM_ROOT}/venvs" # Python仮想環境用

# 2. 'code'ディレクトリにリポジトリをクローン
cd "${FILESYSTEM_ROOT}/code"
if; then
    git clone "${PROJECT_REPO}"
fi

# 3. 指定ブランチにチェックアウト
cd "${PROJECT_DIR_NAME}"
git checkout "${TARGET_BRANCH}"
git pull # 最新の状態に更新

echo "ファイルシステムのセットアップが完了しました。"
2.4. テストと検証転送用インスタンス上で、ls -lR ${FILESYSTEM_ROOT} を実行し、意図した通りのディレクトリ構造が作成されていることを確認します。du -sh ${FILESYSTEM_ROOT}/data を実行し、転送されたデータのおおよそのサイズがローカルのソースと一致することを確認します。cd ${FILESYSTEM_ROOT}/code/${PROJECT_DIR_NAME} し、git status を実行して、正しいブランチにいること、およびリポジトリがクリーンな状態であることを確認します。全ての確認が完了したら、転送用インスタンスは不要なコストを避けるために**速やかに終了（Terminate）**します。フェーズ2：単一GPUでの開発とデバッグ目的: 低コストの単一GPUインスタンス上で、データパス、依存関係、学習スクリプトを含むエンドツーエンドの学習パイプラインがLambda Cloud環境で正しく動作することを検証する。3.1. 開発用インスタンスの起動Lambda Cloudダッシュボードから、On-demand 1x NVIDIA A100 SXM (40GB) インスタンスを起動します 2。選定理由: このインスタンスは、Gemma-7Bモデルのテスト実行に十分な40GBのVRAMを持ちながら、時間あたり$1.29と比較的手頃な価格で利用できます。これにより、コストを抑えつつ、パイプライン全体のデバッグを効率的に行うことができます。起動設定時、必ずフェーズ1でセットアップしたPersistent Filesystem (lisa-gemma-project-fs) をアタッチします。これは、プロジェクトの状態をインスタンス間で引き継ぐための最も重要なステップです 6。3.2. インスタンスのセットアップと環境設定新しいインスタンスに初めてSSH接続した後、以下のセットアップスクリプトを実行します。このスクリプトは、計算インスタンス自体をステートレス（状態を持たない）に保ち、再現性を確保するためのものです。プロジェクトの状態（コード、データ、仮想環境）は全てPersistent Filesystem上に存在するため、このスクリプトを実行するだけで、どのインスタンスでも即座に開発環境を復元できます。setup_dev_instance.sh:Bash#!/bin/bash
set -e # エラーが発生したら即座に終了

# --- プロジェクト設定 ---
PROJECT_NAME="LISA-Gemma-Linux_2"
FILESYSTEM_NAME="lisa-gemma-project-fs"
VENV_NAME="lisa_gemma_venv"
# --- 設定終了 ---

# 1. パスの定義
FS_ROOT="/lambda/nfs/${FILESYSTEM_NAME}"
PROJECT_ROOT="${FS_ROOT}/code/${PROJECT_NAME}"
VENV_PATH="${FS_ROOT}/venvs/${VENV_NAME}"

# 2. システムの更新と基本ツールのインストール
sudo apt-get update
sudo apt-get install -y tmux htop

# 3. Persistent Filesystem上にPython仮想環境を作成・有効化
# Lambda Stackのパッケージを継承するために --system-site-packages を使用 [9]
if; then
    echo "仮想環境を ${VENV_PATH} に作成します..."
    python3 -m venv --system-site-packages "${VENV_PATH}"
else
    echo "既存の仮想環境を使用します。"
fi
source "${VENV_PATH}/bin/activate"
echo "仮想環境が有効化されました。"

# 4. プロジェクト固有の依存関係をインストール
echo "requirements.txt からPythonの依存関係をインストールします..."
pip install --upgrade pip
pip install -r "${PROJECT_ROOT}/requirements.txt"
# 後続フェーズで必要となるライブラリも明示的にインストール
pip install mlflow boto3

# 5. Hugging Face認証の設定
# ユーザーは事前に 'write' 権限を持つトークンを作成し、環境変数として設定する必要がある [15, 16]
if; then
    echo "警告: 環境変数 HF_TOKEN が設定されていません。"
    echo "Hugging Face Hubへのプッシュ機能などを使用するには、'export HF_TOKEN=your_token_here' を実行してください。"
else
    huggingface-cli login --token $HF_TOKEN
    echo "Hugging Faceトークンが設定されました。"
fi

# 6. ホームディレクトリからのアクセスを容易にするためのシンボリックリンクを作成
ln -sfn "${FS_ROOT}" ~/persistent_storage
ln -sfn "${PROJECT_ROOT}" ~/project
echo "シンボリックリンクがホームディレクトリに作成されました: ~/persistent_storage, ~/project"

echo "セットアップ完了。 'cd ~/project' で作業を開始できます。"
このスクリプトをインスタンス上で実行します。3.3. 検証用の学習実行パイプライン全体が機能することを確認するため、データセットのごく一部を使い、数ステップだけの短い学習ジョブを実行します。cd ~/project でプロジェクトディレクトリに移動します。source../../venvs/lisa_gemma_venv/bin/activate で仮想環境を有効化します。以下のコマンドを実行します。既存の学習スクリプト (train.py等) が、これらの引数を受け取れるように、必要に応じて修正してください。Bashpython train.py \
  --model_name_or_path "google/gemma-7b" \
  --data_path "${FS_ROOT}/data/your_subset_for_validation.json" \
  --output_dir "${FS_ROOT}/artifacts/checkpoints/validation_run_$(date +%Y%m%d-%H%M%S)" \
  --num_train_epochs 1 \
  --max_steps 10 \
  --per_device_train_batch_size 1 \
  --save_strategy "steps" \
  --save_steps 5 \
  --logging_steps 1
3.4. テストと検証学習ジョブの実行中に、依存関係やパス関連のエラーが発生しないことを監視します。ジョブが正常に完了した後、output_dirで指定したパスにチェックポイントディレクトリが作成されていることを ls -l コマンドで確認します。チェックポイントディレクトリ内に、adapter_model.bin (または .safetensors) と adapter_config.json が存在することを確認します。インスタンスのローカルストレージ（例: /home/ubuntu）に重要なファイルが書き込まれていないことを確認し、ステートレスなインスタンス運用が維持されていることを保証します。フェーズ3：マルチGPUインスタンス（8x H100）でのスケールアップ学習目的: ターゲットプラットフォームである 8x NVIDIA H100 SXM 上で、DeepSpeedを用いて学習プロセスを最適化し、最大のパフォーマンスと効率を引き出すための設定を確立する。4.1. 本番用インスタンスの起動Lambda Cloudダッシュボードから、On-demand 8x NVIDIA H100 SXM インスタンスを起動します 2。極めて重要: 起動設定時に、これまでのフェーズで使用してきたものと同一のPersistent Filesystem (lisa-gemma-project-fs) を必ずアタッチします。インスタンスにSSH接続後、フェーズ2で使用した setup_dev_instance.sh スクリプトを再度実行します。このスクリプトは汎用的に設計されているため、新しい高性能インスタンス上でも環境を正しく構築します。4.2. 8x H100 LoRAファインチューニング向けDeepSpeed設定マルチGPUでの性能を最大化する鍵は、DeepSpeedの設定ファイルにあります。ここでは、本プロジェクトの特性に最適化された設定を提供します。Gemma-7Bモデル（BF16で約14GB）は、単一のH100 GPU（80GB VRAM）に余裕を持って収まります。この条件下でのLoRAファインチューニングでは、モデルパラメータ全体をシャード化するZeRO-3は過剰な通信オーバーヘッドを生む可能性があります。なぜなら、LoRAではベースモデルのパラメータは凍結されており、更新されないため、これらを毎回のフォワード/バックワードパスで全GPUに集める（all-gather）必要がないからです。代わりに、勾配とオプティマイザの状態のみをシャード化するZeRO-2が最適です 17。LoRAアダプタのパラメータ数は非常に小さいため、ZeRO-2における通信コストは最小限に抑えられ、ZeRO-3で発生するベースモデルの通信オーバーヘッドを回避できます。これにより、学習スループットが大幅に向上し、コスト効率が改善されます。表2：8x H100 LoRAファインチューニング向けDeepSpeed ZeRO-2推奨設定以下の内容で deepspeed_config_zero2.json ファイルをプロジェクトルート (~/project) に作成します。JSON{
  "train_batch_size": 128,
  "train_micro_batch_size_per_gpu": 4,
  "gradient_accumulation_steps": 4,
  "steps_per_print": 10,

  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "contiguous_gradients": true,
    "overlap_comm": true,
    "reduce_bucket_size": 5e8
  },

  "bf16": {
    "enabled": true
  },

  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": "auto",
      "betas": [0.9, 0.999],
      "eps": 1e-8,
      "weight_decay": 0.01
    }
  },

  "scheduler": {
    "type": "WarmupLR",
    "params": {
      "warmup_min_lr": "auto",
      "warmup_max_lr": "auto",
      "warmup_num_steps": "auto"
    }
  },

  "wall_clock_breakdown": false
}
train_batch_size: 8GPU全体での実効的なバッチサイズ。train_micro_batch_size_per_gpu: 1GPUあたりのマイクロバッチサイズ。VRAM使用量を見ながら調整します。gradient_accumulation_steps: train_batch_size / (num_gpus * micro_batch_size) となるように設定します。zero_optimization: ZeRO-2を有効化し、通信と計算のオーバーラップなど各種最適化を適用します 10。bf16: H100 GPUの性能を最大限に引き出すために、bfloat16混合精度学習を有効化します 19。4.3. accelerate launch を用いた学習スクリプトの実行Hugging Face Trainer を使用する既存のスクリプトは、accelerate を用いて容易に分散学習に対応できます。accelerate の設定:インスタンス上で accelerate config コマンドを実行し、対話的に設定を行います。DeepSpeedを使用する設定を選択し、上記で作成した deepspeed_config_zero2.json のパスを指定します。TrainingArguments の修正:学習スクリプト (train.py) 内で、TrainingArguments の初期化時に deepspeed パラメータを追加します。Pythonfrom transformers import TrainingArguments

training_args = TrainingArguments(
    #... 他の引数...
    deepspeed="deepspeed_config_zero2.json",
)
学習の開始:以下のコマンドで分散学習を開始します。Bashcd ~/project
source../../venvs/lisa_gemma_venv/bin/activate

# accelerate launch は8プロセス（8GPU）で train.py を実行する
accelerate launch train.py \
  --model_name_or_path "google/gemma-7b" \
  --data_path "${FS_ROOT}/data/full_dataset.json" \
  --output_dir "${FS_ROOT}/artifacts/checkpoints/scaled_run_1" \
  #... その他の学習引数...
4.4. テストと検証上記のコマンドで、まずは短い学習（例: 50ステップ）を実行します。学習中に新しいターミナルセッションを開き、nvtop または watch -n 1 nvidia-smi を実行します。検証項目:8つ全てのGPUが高い使用率（理想的には90%以上）を示していること。各GPUにメモリが割り当てられていること。Trainer のログに出力されるスループット（samples/secなど）を確認し、ベースライン性能を記録します。フェーズ4：体系的なハイパーパラメータ最適化（HPO）目的: 最適なモデル構成を見つけるための「試行錯誤」プロセスを、構造化され、自動化され、追跡可能なワークフローとして実装する。5.1. 実験追跡の統合各学習のハイパーパラメータと結果メトリクスを記録するため、シンプルなCSVロガーを導入します。これにより、どのパラメータの組み合わせが最良の結果をもたらしたかをデータに基づいて判断できます 12。プロジェクト内に log_utils.py のようなユーティリティファイルを作成します。Python# log_utils.py
import csv
import os
from datetime import datetime

LOG_FILE = '/lambda/nfs/lisa-gemma-project-fs/artifacts/logs/hpo_log.csv'
HEADERS = [
    'timestamp', 'run_id', 'status',
    'lora_r', 'lora_alpha', 'learning_rate', 'num_train_epochs', 'per_device_train_batch_size',
    'final_train_loss', 'final_eval_loss', 'eval_accuracy' # 必要に応じてメトリクスを追加
]

def initialize_log_file():
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(HEADERS)

def log_run(run_data):
    initialize_log_file()
    run_data['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(LOG_FILE, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=HEADERS)
        writer.writerow(run_data)
メインの学習スクリプト (train.py) の最後に、Trainer の train() メソッドの返り値や evaluate() の結果を用いてこのロギング関数を呼び出すコードを追加します。5.2. HPOスイープスクリプトの作成異なるハイパーパラメータの組み合わせで複数の学習を自動実行するための、マスターbashスクリプトを作成します。LoRAのファインチューニングにおいて特に影響が大きい lora_r と learning_rate をスイープするのが効果的です 21。一般的に lora_alpha は r の2倍に設定するヒューリスティックが有効です。run_hpo_sweep.sh:Bash#!/bin/bash
set -e

# --- HPO設定 ---
LORA_RANKS=(16 32 64)
LEARNING_RATES=(5e-5 1e-4 2e-4)
EPOCHS=3
BATCH_SIZE=4
# --- 設定終了 ---

# 環境の有効化
source /lambda/nfs/lisa-gemma-project-fs/venvs/lisa_gemma_venv/bin/activate
cd ~/project

# HPOループ
for r in "${LORA_RANKS[@]}"; do
  for lr in "${LEARNING_RATES[@]}"; do
    RUN_ID="r${r}_lr${lr}_$(date +%s)"
    OUTPUT_DIR="/lambda/nfs/lisa-gemma-project-fs/artifacts/checkpoints/hpo/${RUN_ID}"
    
    echo "--- HPO実行開始: ${RUN_ID} ---"
    echo "Lora Rank: ${r}, Learning Rate: ${lr}"

    accelerate launch train.py \
      --output_dir "${OUTPUT_DIR}" \
      --lora_r ${r} \
      --lora_alpha $((${r} * 2)) \
      --learning_rate ${lr} \
      --num_train_epochs ${EPOCHS} \
      --per_device_train_batch_size ${BATCH_SIZE} \
      #... その他の固定パラメータ...

    echo "--- HPO実行完了: ${RUN_ID} ---"
  done
done
このスクリプトを実行する前に、train.py が --lora_r や --learning_rate などの引数をコマンドラインから受け取れるように修正する必要があります。5.3. 評価プロトコルと最良モデルの選定HPOスイープ完了後、hpo_log.csv ファイルを分析して最良のモデルを選定します。主要な評価指標を final_eval_loss（検証損失）とします。この値が最も低いモデルが、未知のデータに対して最も汎化性能が高いと期待されます。hpo_log.csv をスプレッドシートソフトで開くか、Pythonのpandasライブラリなどを用いて読み込み、final_eval_loss 列で昇順にソートします。最も損失が低い行に対応する run_id とハイパーパラメータの組み合わせが、本番学習で採用すべき構成となります。表3：ハイパーパラメータ最適化追跡ログ（テンプレート）hpo_log.csv ファイルは以下のような形式で記録されます。timestamprun_idstatuslora_rlora_alphalearning_rate...final_train_lossfinal_eval_loss2024-09-15 10:30:00r16_lr5e-5_...COMPLETED16325e-5...0.850.922024-09-15 12:45:00r32_lr1e-4_...COMPLETED32641e-4...0.780.85...........................5.4. テストと検証run_hpo_sweep.sh スクリプトを、ごく小さな探索空間（例: LORA_RANKS=(8), LEARNING_RATES=(1e-4)）と短い学習ステップ（--max_steps=10 を train.py に渡す）で実行します。スクリプトがエラーなく完了することを確認します。hpo_log.csv ファイルに、実行した設定に対応する新しい行が追加されていることを確認します。指定されたHPO用のチェックポイントディレクトリに、新しいrun_idのフォルダが作成されていることを確認します。フェーズ5：本番学習とモデルの最終化目的: HPOで特定した最適なハイパーパラメータを用いて完全なデータセットで本番学習を実行し、推論・デプロイに適した形式の最終的なモデルアーティファクトを作成する。6.1. 本番学習ジョブの開始フェーズ4で特定した最良のハイパーパラメータを用いて、最終的な学習ジョブを開始します。run_production_training.sh:Bash#!/bin/bash
set -e

# --- 最適化された本番設定 ---
BEST_LORA_RANK=32
BEST_LR=1e-4
BEST_ALPHA=$((${BEST_LORA_RANK} * 2))
NUM_EPOCHS=5 # 本番用のエポック数
BATCH_SIZE=4
RUN_VERSION="v1.0"
# --- 設定終了 ---

# 環境の有効化と移動
source /lambda/nfs/lisa-gemma-project-fs/venvs/lisa_gemma_venv/bin/activate
cd ~/project

# 出力ディレクトリの定義
OUTPUT_DIR="/lambda/nfs/lisa-gemma-project-fs/artifacts/final_models/production_${RUN_VERSION}"

echo "--- 本番学習開始: Version ${RUN_VERSION} ---"
echo "Lora Rank: ${BEST_LORA_RANK}, Learning Rate: ${BEST_LR}"

accelerate launch train.py \
  --output_dir "${OUTPUT_DIR}" \
  --lora_r ${BEST_LORA_RANK} \
  --lora_alpha ${BEST_ALPHA} \
  --learning_rate ${BEST_LR} \
  --num_train_epochs ${NUM_EPOCHS} \
  --per_device_train_batch_size ${BATCH_SIZE} \
  --data_path "/lambda/nfs/lisa-gemma-project-fs/data/full_dataset.json" \
  #... その他の固定パラメータ...

echo "--- 本番学習完了 ---"
6.2. モデルの最終化：アダプタのマージと保存ファインチューニングで得られるLoRAアダプタは、それ単体ではデプロイできません。推論時の効率を最大化し、デプロイメントを簡素化するために、学習したアダプタの重みをベースモデルの重みにマージ（統合）する必要があります 22。この処理により、推論時に追加の計算（W=W0​+BA）が不要になり、標準的なHugging Faceモデルとして扱えるようになります。このマージ処理を行うための専用スクリプト finalize_model.py を作成します。Python# finalize_model.py
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse

def main(args):
    print(f"ベースモデルをロード中: {args.base_model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu" # マージ処理はCPUで行う
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)

    print(f"LoRAアダプタをロード中: {args.adapter_path}")
    model = PeftModel.from_pretrained(model, args.adapter_path)

    print("アダプタをベースモデルにマージしています...")
    model = model.merge_and_unload()
    print("マージが正常に完了しました。")

    print(f"マージ済みモデルを保存中: {args.merged_model_save_path}")
    model.save_pretrained(args.merged_model_save_path)
    tokenizer.save_pretrained(args.merged_model_save_path)
    print("保存が完了しました。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str, default="google/gemma-7b")
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to the trained LoRA adapter directory.")
    parser.add_argument("--merged_model_save_path", type=str, required=True, help="Path to save the final merged model.")
    args = parser.parse_args()
    main(args)
実行コマンド:Bashpython finalize_model.py \
  --adapter_path "/lambda/nfs/lisa-gemma-project-fs/artifacts/final_models/production_v1.0/checkpoint-XXXX" \
  --merged_model_save_path "/lambda/nfs/lisa-gemma-project-fs/artifacts/final_models/lisa-gemma-7b-merged-v1.0"
6.3. （任意）Hugging Face Hubへのアーティファクトのプッシュ学習成果を共有・バックアップするために、LoRAアダプタとマージ済みモデルの両方をHugging Face Hubにプッシュすることを推奨します。Python# スクリプトの一部として
from huggingface_hub import HfApi
api = HfApi()

# LoRAアダプタのプッシュ (軽量で再利用性が高い)
model.push_to_hub("your-username/lisa-gemma-7b-lora-adapter-v1.0", private=True)
tokenizer.push_to_hub("your-username/lisa-gemma-7b-lora-adapter-v1.0", private=True)

# マージ済みモデルのプッシュ (推論に即時利用可能)
merged_model.push_to_hub("your-username/lisa-gemma-7b-merged-v1.0", private=True)
tokenizer.push_to_hub("your-username/lisa-gemma-7b-merged-v1.0", private=True)
push_to_hub メソッドは Trainer やモデルオブジェクトから直接呼び出すことも可能です 24。6.4. テストと検証最終的なマージ済みモデルが正しく機能するかを確認します。以下の内容で test_inference.py を作成します。Python# test_inference.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "/lambda/nfs/lisa-gemma-project-fs/artifacts/final_models/lisa-gemma-7b-merged-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto")

prompt = "Describe the LISA model and its capabilities:" # テスト用のプロンプト
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs, skip_special_tokens=True))
スクリプトを実行し、意味のある、破綻していないテキストが出力されることを確認します。これにより、マージ処理が成功し、最終モデルが有効であることが検証されます。付録：Cursor Agentを用いた実装ワークフローの管理目的: AIエージェント（CursorのClaude-4-sonnet）に本仕様書の複雑な実装を段階的に、かつ確実に実行させるための、堅牢でステートフルなプロトコルを提供する。AIエージェントは対話ごとにコンテキストを失う可能性があるため、プロジェクトの進捗状況を外部の永続的なファイルに記録することが不可欠です。これにより、人間とAIの両方が常に「次に何をすべきか」を正確に把握できます。A.1. PROGRESS.md — 状態追跡ファイルプロジェクトの進捗を管理するために、Gitリポジトリのルートに以下の PROGRESS.md ファイルを作成します。これは、バージョン管理されるチェックリストとして機能し、完了したタスクと未完了のタスクを明確にします。PROGRESS.md テンプレート:LISA-Gemma on Lambda Cloud: Implementation ProgressPhase 1: Foundational Setup[ ] 1.1 Provision Lambda Cloud Persistent Filesystem.[ ] 1.2 Execute migrate_data.sh to transfer data from local.[ ] 1.3 Execute commands to establish project structure on the filesystem.[ ] 1.4 Perform verification checks on transferred data and structure.[ ] 1.5 Terminate the temporary data transfer instance.Phase 2: Development & Debugging[ ] 2.1 Launch 1x A100 development instance with filesystem attached.[ ] 2.2 Execute setup_dev_instance.sh on the new instance.[ ] 2.3 Run the validation training job for 10 steps.[ ] 2.4 Verify checkpoint creation on the persistent filesystem.[ ] 2.5 Terminate the development instance.Phase 3: Scaled Fine-Tuning[ ] 3.1 Launch 8x H100 production instance with filesystem attached.[ ] 3.2 Execute setup_dev_instance.sh on the production instance.[ ] 3.3 Create deepspeed_config_zero2.json file.[ ] 3.4 Run a short scaled training job to verify multi-GPU utilization.[ ] 3.5 Verify all 8 GPUs are active during the test run.Phase 4: Hyperparameter Optimization[ ] 4.1 Implement log_utils.py for experiment tracking.[ ] 4.2 Modify train.py to call the logging function.[ ] 4.3 Create run_hpo_sweep.sh for automated HPO.[ ] 4.4 Run a minimal HPO test and verify logging.[ ] 4.5 Execute the full HPO sweep.[ ] 4.6 Analyze hpo_log.csv and select the best hyperparameters.Phase 5: Production Training and Finalization[ ] 5.1 Create run_production_training.sh with the best hyperparameters.[ ] 5.2 Execute the full production training job.[ ] 5.3 Create finalize_model.py to merge the adapter.[ ] 5.4 Run the finalization script to create the merged model.[ ] 5.5 Create and run test_inference.py to verify the merged model.[ ] 5.6 (Optional) Push final artifacts to Hugging Face Hub.[ ] 5.7 Terminate the production instance.A.2. Cursor Agentへのプロンプトプロトコル以下のプロンプトテンプレートを用いて、Cursorエージェントにタスクを指示します。初期セットアッププロンプト私は新しいプロジェクトを開始します。完全な実装仕様書はにあります。あなたの最初のタスクは、現在のGitリポジトリのルートに PROGRESS.md という名前のファイルを作成し、仕様書の付録A.1の内容をコピー＆ペーストすることです。その後、「feat: Initialize project progress tracker」というコミットメッセージでこのファイルをコミットしてください。標準タスク実行プロンプトPROGRESS.md ファイルを参照し、次に実行すべき未チェックのタスクを特定してください。そのタスクに関する詳細な指示を、実装仕様書から読み取ってください。指示に従って、必要なスクリプトの実装やコードの修正を行ってください。実装後、仕様書に記載されているテストと検証の手順を実行してください。テストが成功した場合、PROGRESS.md ファイルを更新し、完了したタスクの [ ] を [x] に変更してください。最後に、全ての変更を、タスク内容を説明する適切なコミットメッセージ（例: feat(phase2): Implement validation training run）と共にコミットしてください。いずれかのステップで失敗した場合は、コミットせずにエラー内容を報告してください。