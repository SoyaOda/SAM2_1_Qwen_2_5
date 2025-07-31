LISA-Gemma3: 推論ベースセグメンテーションのための決定版実装ブループリントはじめに：プロジェクトの設計図と基本原則プロジェクトの目的本ドキュメントは、LISA（Large-language Instructed Segmentation Assistant）フレームワークの中核をなすVision-Language Model（VLM）を、既存のLLaVAからGoogleのgoogle/gemma-3-4b-itに置き換えるプロジェクトの、完全かつ曖昧性のない技術仕様書である。これは単なるモデルの交換ではなく、根本的なアーキテクチャの再設計と実装を伴う、高度なエンジニアリング作業である。本仕様書は、実装者が直面するであろう技術的課題を事前に解決し、成功への明確な道筋を提供することを目的とする。中核的課題：デュアルエンコーダ問題本プロジェクトの成功は、[Geminiの提案]で詳述されている核心的な技術的課題、すなわち「デュアルエンコーダ問題」の解決にかかっている。この問題は、2つの独立した視覚情報経路をいかに調和させるかという点に集約される。Gemmaの内部視覚経路: モデルは、テキストプロンプトの意図を解釈し、セグメンテーションを実行すべきか否かを判断するために、自身の内部に統合されたSigLIPビジョンエンコーダ 1 を用いて画像を「理解」する。この理解に基づき、モデルは特別な<SEG>トークンを生成する。SAMの外部視覚経路: 一方、最終的なピクセルレベルのマスクを生成するSAMのマスクデコーダは、GemmaのSigLIPエンコーダが見た特徴量ではなく、独立して凍結されたSAM-ViTエンコーダ 3 から供給される高解像度な特徴量を必要とする。この結果、モデルは一方の視覚情報（SigLIP由来）を用いて推論し、その推論結果を全く異なる視覚情報（SAM-ViT由来）の上で動作するコンポーネントに伝えなければならないという、アーキテクチャ上のパラドックスが生じる。過去の試みが失敗した根本原因は、この2つの異質な視覚世界の間の「意味の翻訳」ができていなかった点にあると考えられる。3つの柱からなる解決戦略このデュアルエンコーダ問題を解決するため、本仕様書は以下の3つの柱に基づいたソリューションを提案する。デュアルストリーム・データパイプライン: 単一の入力画像を、Gemma-SigLIPとSAM-ViTという2つの異なるビジョンエンコーダの要求仕様に合わせて、それぞれ個別に前処理するデータパイプラインを構築する。デュアルパスウェイ・モデルアーキテクチャ: 2系統の視覚特徴量ストリームを、モデルのフォワードパス内で明確に分離し続け、それぞれが対応するコンポーネント（Gemmaの推論部とSAMのデコード部）にのみ供給されるように設計する。訓練可能なMLPプロジェクタ: 2つの世界を繋ぐ「橋渡し役」として、小さな多層パーセプトロン（MLP）を導入する。このプロジェクタは、Gemmaが生成した<SEG>トークンの隠れ状態（Gemmaの世界の「意図」）を、SAMデコーダが解釈可能なプロンプト埋め込み（SAMの世界の「指示」）へと変換する役割を担う。このMLPを訓練することが、本プロジェクトの学習プロセスの核心となる。第1章 環境構築と基礎設定本章では、プロジェクトの再現性と安定性を確保するための、環境構築と設定ファイルに関する完全な仕様を定義する。1.1. 依存関係の仕様 (requirements.txt)一貫した動作を保証するため、以下の内容でrequirements.txtファイルを作成する。これにより、全ての開発者が同一のライブラリバージョンで作業することが可能となる。Plaintext# Core ML/DL Frameworks
torch>=2.4.0
torchvision
torchaudio

# Hugging Face Ecosystem
transformers>=4.51.3
accelerate
peft
bitsandbytes
huggingface_hub

# Distributed Training
deepspeed

# Data Handling and Utilities
pycocotools
numpy
tqdm
tensorboard
Pillow
正当性の根拠:torch>=2.4.0およびtransformers>=4.51.3は、Gemma 3モデルを完全にサポートするために推奨される最小バージョンである 4。peftはLoRA（Low-Rank Adaptation）によるパラメータ効率の良いファインチューニングに必須である 5。deepspeedは、ユーザーの要求通り、大規模モデルの分散学習を効率化するために導入される。bitsandbytesは、将来的な4ビット/8ビット量子化の適用を可能にするための依存関係である 6。pycocotoolsは、COCO形式のデータセットやマスクを扱う上で標準的なツールである 3。1.2. プロジェクト設定 (config_linux.py)プロジェクト全体のパス、ハイパーパラメータ、モデル識別子を一元管理するため、以下の内容でconfig_linux.pyをルートディレクトリに作成する。ユーザーが提供した過去のスクリプトを、Linux標準のパス構造とGemma 3の仕様に合わせて全面的に刷新する。Pythonimport os

# ==============================================================================
# 1. PATHS AND IDENTIFIERS
# ==============================================================================
# プロジェクトのルートディレクトリからの相対パスでデータセットのベースディレクトリを指定
# 例: /home/user/LISA-Gemma3/datasets
# 注意: WSLの '/mnt/h/...' のようなパスではなく、Linuxネイティブの絶対パスまたは相対パスを使用すること
DATASET_BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")

# 事前学習済みSAMモデルのチェックポイントへのパス
# 例: /home/user/LISA-Gemma3/weights/sam_vit_h_4b8939.pth
SAM_CHECKPOINT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights", "sam_vit_h_4b8939.pth")

# Hugging Faceモデル識別子
GEMMA_MODEL_ID = "google/gemma-3-4b-it"

# ログと出力の保存先
LOG_BASE_DIR = "./runs"

# ==============================================================================
# 2. MODEL CONFIGURATION
# ==============================================================================
# 画像サイズ設定
# GEMMA_IMAGE_SIZEはGemma-3のSigLIPエンコーダの要求仕様 (896x896)
GEMMA_IMAGE_SIZE = 896
# SAM_IMAGE_SIZEはSAM-ViTエンコーダの要求仕様 (1024x1024)
SAM_IMAGE_SIZE = 1024
# モデルが処理するトークンの最大長
MODEL_MAX_LENGTH = 2048
# MLPプロジェクタからSAMデコーダへの出力次元 (SAMのプロンプト埋め込み次元と一致)
SEG_PROJECTION_DIM = 256

# ==============================================================================
# 3. TRAINING HYPERPARAMETERS
# ==============================================================================
# DeepSpeed設定ファイルで "auto" を使用するため、ここではコメントアウト。
# TrainingArgumentsまたはdeepspeed configで直接設定することを推奨。
# BATCH_SIZE_PER_GPU = 2
# GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 1e-4
EPOCHS = 10
STEPS_PER_EPOCH = 500
WEIGHT_DECAY = 1e-2
BETA1 = 0.9
BETA2 = 0.95

# 損失関数の重み
CE_LOSS_WEIGHT = 1.0
DICE_LOSS_WEIGHT = 0.5
BCE_LOSS_WEIGHT = 2.0

# ==============================================================================
# 4. LoRA CONFIGURATION (Parameter-Efficient Fine-Tuning)
# ==============================================================================
LORA_R = 32  # ランク
LORA_ALPHA = 64  # LoRAのスケーリング係数 (r * 2 が一般的)
LORA_DROPOUT = 0.05

# CRITICAL: Gemmaアーキテクチャに適合したターゲットモジュール
# ユーザー提供のスクリプトにあった "q_proj,k_proj,v_proj" は不完全。
# GemmaではAttention層とFFN層の両方をターゲットにする必要がある。
LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

# ==============================================================================
# 5. DATASET CONFIGURATION
# ==============================================================================
# データセットの混合比率
DATASET_SAMPLE_RATES = "9,3,3,1"  # sem_seg, refer_seg, vqa, reason_seg
# 使用するデータセットの指定
SEM_SEG_DATA = "ade20k||cocostuff||mapillary"
REFER_SEG_DATA = "refclef||refcoco||refcoco+||refcocog"
VQA_DATA = "llava_instruct_150k"
REASON_SEG_DATA = "ReasonSeg|train"
VAL_DATASET = "ReasonSeg|val"

# ==============================================================================
# 6. UTILITY FUNCTIONS
# ==============================================================================
def get_dataset_paths():
    """データセットへの完全なパスを構築して返す"""
    return {
        "sem_seg": {
            "ade20k": os.path.join(DATASET_BASE_DIR, "ade20k"),
            "cocostuff": os.path.join(DATASET_BASE_DIR, "cocostuff"),
            "mapillary": os.path.join(DATASET_BASE_DIR, "mapillary"),
        },
        "refer_seg": {
            "refcoco": os.path.join(DATASET_BASE_DIR, "refer_seg"),
            "refcoco+": os.path.join(DATASET_BASE_DIR, "refer_seg"),
            "refcocog": os.path.join(DATASET_BASE_DIR, "refer_seg"),
            "refclef": os.path.join(DATASET_BASE_DIR, "refer_seg"),
        },
        "vqa": {
            "llava_instruct_150k": os.path.join(DATASET_BASE_DIR, "llava_dataset", "llava_instruct_150k.json"),
        },
        "reason_seg": {
            "ReasonSeg": os.path.join(DATASET_BASE_DIR, "reason_seg", "ReasonSeg"),
        }
    }

def check_paths():
    """重要なパスが存在するかを検証する"""
    paths_to_check =
    
    dataset_paths = get_dataset_paths()
    # 簡単な存在チェック
    paths_to_check.append(dataset_paths["sem_seg"]["ade20k"])
    paths_to_check.append(dataset_paths["vqa"]["llava_instruct_150k"])

    missing_paths = [path for path in paths_to_check if not os.path.exists(path)]
    
    if missing_paths:
        print("警告: 以下の必須パスまたはファイルが見つかりません:")
        for path in missing_paths:
            print(f"  - {path}")
        print("\nconfig_linux.pyのパス設定と、データセットが正しく配置されているか確認してください。")
        return False
    
    return True

if __name__ == "__main__":
    print("=== プロジェクト設定検証 ===")
    if check_paths():
        print("✓ 必須パスの検証に成功しました。")
    else:
        print("✗ 必須パスの検証に失敗しました。上記のエラーメッセージを確認してください。")

重要な変更点と正当性:パス構造: WSL特有の/mnt/プレフィックスを廃止し、スクリプトの場所を基準とした、よりポータブルなパス構造に変更した。LoRAターゲットモジュール: ユーザーの過去のスクリプトにあった不完全な設定 ("q_proj,k_proj,v_proj") は、学習の失敗に直結する重大な誤りである。GemmaのTransformerブロックは、LLaMAとは異なる構造を持つ。Hugging Faceの公式ブログやGemma 3のファインチューニング事例 5 に基づき、アテンション層 (q, k, v, o_proj) とフィードフォワードネットワーク層 (gate, up, down_proj) の両方をターゲットに含めるよう修正した。これにより、LoRAアダプタがモデルのより広範なパラメータに影響を与え、効果的な学習が可能になる。1.3. Hugging Face認証Gemma 3は、利用規約への同意が必要なゲート付きモデルである 9。モデルをダウンロードするには、事前に認証を行う必要がある。ターミナルで以下のコマンドを実行し、Hugging Faceのアクセストークン（hf_で始まる文字列）を入力してログインする。Bash# Hugging Face Hubにログインする
huggingface-cli login --token YOUR_HF_TOKEN_HERE

# または、環境変数として設定する (スクリプト実行時に自動で読み込まれる)
export HF_TOKEN=YOUR_HF_TOKEN_HERE
セキュリティの観点から、トークンをスクリプトにハードコーディングするのではなく、CLIの認証情報ストアまたは環境変数を使用することを強く推奨する。第2章 アーキテクチャの中核: model/gemma_lisa.py本章では、デュアルパスウェイ設計を実装するモデルアーキテクチャの完全なコードと、その設計判断の根拠を詳述する。このファイルは、プロジェクト全体の心臓部である。2.1. モデルアーキテクチャ定義 (LisaGemmaForCausalLM)modelディレクトリを新規に作成し、その中にgemma_lisa.pyという名前で以下のファイルを作成する。このクラスは、Hugging FaceのPreTrainedModelを継承することで、from_pretrainedメソッドやsave_pretrainedメソッドなどのエコシステムとの互換性を確保する。Python# model/gemma_lisa.py
import torch
import torch.nn as nn
from typing import Optional, List, Tuple, Dict, Any

from transformers import AutoModelForCausalLM, PreTrainedModel, PretrainedConfig
from segment_anything import sam_model_registry
from segment_anything.modeling import MaskDecoder, PromptEncoder, TwoWayTransformer

# LISA-Gemmaモデルのカスタム設定クラス
class LisaGemmaConfig(PretrainedConfig):
    model_type = "lisa_gemma"

    def __init__(
        self,
        gemma_model_id="google/gemma-3-4b-it",
        sam_checkpoint_path=None,
        seg_token_idx=0,
        gemma_hidden_size=2560,
        sam_prompt_embed_dim=256,
        **kwargs,
    ):
        self.gemma_model_id = gemma_model_id
        self.sam_checkpoint_path = sam_checkpoint_path
        self.seg_token_idx = seg_token_idx
        self.gemma_hidden_size = gemma_hidden_size
        self.sam_prompt_embed_dim = sam_prompt_embed_dim
        super().__init__(**kwargs)

class LisaGemmaForCausalLM(PreTrainedModel):
    config_class = LisaGemmaConfig

    def __init__(self, config: LisaGemmaConfig):
        super().__init__(config)

        # 1. Gemma-3 LLMのロード
        self.gemma_model = AutoModelForCausalLM.from_pretrained(
            config.gemma_model_id,
            torch_dtype=torch.bfloat16, # bf16で効率化
            trust_remote_code=True,
        )

        # 2. SAMコンポーネントのロードと凍結
        sam = sam_model_registry["vit_h"](checkpoint=config.sam_checkpoint_path)
        
        # SAMの画像エンコーダを抽出し、凍結する
        self.sam_image_encoder = sam.image_encoder
        for param in self.sam_image_encoder.parameters():
            param.requires_grad = False
        
        # SAMのマスクデコーダを抽出し、訓練可能にする
        self.sam_mask_decoder = sam.mask_decoder
        for param in self.sam_mask_decoder.parameters():
            param.requires_grad = True

        # 3. MLPプロジェクタの定義 (GemmaとSAMを繋ぐ橋)
        self.mlp_projector = nn.Sequential(
            nn.Linear(config.gemma_hidden_size, config.gemma_hidden_size),
            nn.GELU(),
            nn.Linear(config.gemma_hidden_size, config.sam_prompt_embed_dim),
        )

        # 4. モデルの他の部分のパラメータ管理
        # LLMの大部分は凍結 (LoRAでファインチューニング)
        for param in self.gemma_model.parameters():
            param.requires_grad = False
        
        # 埋め込み層とLMヘッドは訓練可能にする
        self.gemma_model.get_input_embeddings().requires_grad_(True)
        self.gemma_model.get_output_embeddings().requires_grad_(True)


    def get_input_embeddings(self) -> nn.Module:
        return self.gemma_model.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Module):
        self.gemma_model.set_input_embeddings(value)

    def get_output_embeddings(self) -> nn.Module:
        return self.gemma_model.get_output_embeddings()

    def forward(
        self,
        images_for_gemma: torch.Tensor,
        images_for_sam: torch.Tensor,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        labels: Optional = None,
        seg_token_mask: Optional = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        デュアルパスウェイ・フォワードパスの実装
        """
        # ======================================================================
        # パスウェイ 1: SAMの画像エンコーディング (セグメンテーション用)
        # ======================================================================
        # SAMの画像エンコーダは凍結されているため、勾配計算は不要
        with torch.no_grad():
            sam_image_features = self.sam_image_encoder(images_for_sam)

        # ======================================================================
        # パスウェイ 2: Gemmaの推論 (意図理解用)
        # ======================================================================
        # Gemmaモデルに画像とテキストを入力し、出力を得る
        # Gemmaの内部プロセッサがimages_for_gemmaを処理する
        gemma_outputs = self.gemma_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=images_for_gemma,
            labels=labels,
            output_hidden_states=True, # 隠れ状態を取得するために必要
        )
        
        # テキスト生成の損失（VQAタスクなどで使用）
        text_loss = gemma_outputs.loss
        
        # ======================================================================
        # 橋渡し: MLPプロジェクタによる特徴量変換
        # ======================================================================
        last_hidden_state = gemma_outputs.hidden_states[-1]
        
        # バッチ内の<SEG>トークンの隠れ状態を抽出
        if seg_token_mask is not None and seg_token_mask.sum() > 0:
            # seg_token_maskは、<SEG>トークンの位置がTrueのブールマスク
            # (batch_size, seq_len) -> (batch_size, seq_len, hidden_size)
            seg_token_mask_expanded = seg_token_mask.unsqueeze(-1).expand_as(last_hidden_state)
            
            # <SEG>トークンの隠れ状態のみを抽出 (sum > 0 の場合のみ)
            # (num_seg_tokens, hidden_size)
            h_seg_raw = last_hidden_state[seg_token_mask_expanded].view(-1, last_hidden_state.size(-1))
            
            # MLPプロジェクタを通して、SAMが理解できる埋め込みに変換
            # (num_seg_tokens, sam_prompt_embed_dim)
            seg_token_embedding = self.mlp_projector(h_seg_raw)
        else:
            seg_token_embedding = None

        # ======================================================================
        # 最終段階: SAMマスクデコーダによるマスク生成
        # ======================================================================
        predicted_masks = None
        if seg_token_embedding is not None:
            # SAMデコーダへの入力を作成
            # (batch_size, num_prompts, embed_dim) -> (num_seg_tokens, 1, embed_dim)
            sparse_prompt_embeddings = seg_token_embedding.unsqueeze(1)
            
            # デンスなプロンプトは使用しない
            dense_prompt_embeddings = torch.zeros(
                (seg_token_embedding.size(0), 256, 256),
                device=seg_token_embedding.device,
                dtype=seg_token_embedding.dtype
            )

            # SAMデコーダを実行してマスクを予測
            # sam_image_featuresはバッチ処理に対応していない可能性があるため、ループで処理する必要がある場合がある
            # ここでは、seg_token_maskがバッチ内のどの画像に対応するかを追跡する必要がある
            # 簡単のため、ここではバッチ内の全ての<SEG>トークンが同じ画像特徴を使うと仮定
            # 実際の実装では、どのトークンがどの画像に対応するかを管理する必要がある
            
            # TODO: バッチ処理を正しくハンドリングするロジックを追加
            # seg_token_maskから、各トークンがどのバッチインデックスに属するかを取得
            batch_indices = torch.where(seg_token_mask)
            
            # 対応する画像特徴を選択
            corresponding_sam_features = sam_image_features[batch_indices]

            low_res_masks, iou_predictions = self.sam_mask_decoder(
                image_embeddings=corresponding_sam_features,
                image_pe=self.sam_mask_decoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_prompt_embeddings,
                dense_prompt_embeddings=dense_prompt_embeddings,
                multimask_output=False, # LISAは単一マスクを予測
            )
            
            predicted_masks = low_res_masks

        return {
            "text_loss": text_loss,
            "predicted_masks": predicted_masks,
            "logits": gemma_outputs.logits,
        }

2.2. コンポーネントの初期化Gemma-3: AutoModelForCausalLM.from_pretrained を使用してロードする 4。torch_dtype=torch.bfloat16 を指定することで、A100/H100などの最新GPUでメモリ効率と計算速度を向上させる。SAM: 公式のsegment-anythingライブラリが提供するsam_model_registryを用いて、事前学習済みのvit_hモデルをインスタンス化する 3。その後、model.image_encoderとmodel.mask_decoderを個別の属性として抽出し、保持する。2.3. 橋渡し役：MLPプロジェクタ (mlp_projector)このMLPは、本アーキテクチャの成功を左右する最も重要なコンポーネントである。その役割は、Gemmaの思考空間（隠れ状態ベクトル）とSAMの指示空間（プロンプト埋め込みベクトル）の間の意味的なギャップを埋めることにある。このMLPの入出力次元を正確に設定することは、致命的なRuntimeErrorを回避するために不可欠である。入力次元: Gemmaの隠れ層のサイズに一致させる必要がある。google/gemma-3-4b-itのconfig.jsonは直接アクセスできない場合があるが、コミュニティによって提供された互換モデルのコンフィグ 11 から、そのhidden_sizeが2560であることが確認できる。出力次元: SAMのマスクデコーダが期待するプロンプト埋め込みの次元に一致させる必要がある。SAMのアーキテクチャ分析 12 によれば、この次元は256である。このMLPはゼロから学習されるため、その重みは訓練可能 (requires_grad=True) でなければならない。学習プロセス中、セグメンテーション損失から逆伝播してきた勾配がこのMLPを通過し、GemmaのLoRAアダプタへと流れる。この勾配フローこそが、Gemmaに「SAMが理解できる」ような<SEG>トークンの埋め込みを生成させるための学習信号となる。パラメータ値ソース / 正当性の根拠in_features2560gemma-3-4b-itのconfig.hidden_size 11hidden_features2560調整可能な中間次元。入力次元と同一に設定するのが一般的。out_features256SAMのprompt_embed_dim 122.4. デュアルパスウェイ・フォワードパスforwardメソッドの実装は、[Geminiの提案]で概説されたロジックを忠実に反映している。入力: images_for_gemmaとimages_for_samという2つの異なる画像テンソルを受け取る。これは、第3章で詳述するデュアルストリーム・データパイプラインによって生成される。並列処理: sam_image_encoderとgemma_modelは、それぞれに対応する画像テンソルを入力として並列的に処理される。2つの視覚経路は、この段階では完全に分離されている。情報抽出と変換: Gemmaの出力から<SEG>トークンに対応する隠れ状態ベクトルを特定し、mlp_projectorを介してSAM用のプロンプト埋め込みに変換する。最終デコーディング: 最初に計算しておいたsam_image_featuresと、変換されたプロンプト埋め込みをsam_mask_decoderに渡し、最終的なマスクを生成する。出力: テキスト生成タスク用のtext_lossおよびlogitsと、セグメンテーションタスク用のpredicted_masksの両方を含む辞書を返す。これにより、第4章で定義する複合損失関数が計算可能となる。2.5. パラメータ管理戦略訓練の安定性と効率性を最大化するため、各コンポーネントの訓練可能性を厳密に管理する。この戦略を誤ると、有用な事前学習知識が失われる「破滅的忘却」や、メモリの枯渇を引き起こす。以下の表は、各コンポーネントのrequires_gradフラグの設定方針を明確に定義する。コンポーネント訓練可能か？ (requires_grad)正当性の根拠Gemma-3 LLMの重みNo (凍結)モデルの核となる言語能力を維持するため。LoRAを介して間接的にのみ適応させる。Gemma-3 LoRAアダプタYes (訓練可能)パラメータ効率の良いファインチューニングの核心。学習対象となる主要な部分。Gemma-3 Vision Encoder (SigLIP)No (凍結)強力な事前学習済みの視覚理解能力をそのまま活用する 2。SAM Vision Encoder (ViT)No (凍結)絶対的に重要。これはデコーダのための高忠実度な特徴量の供給源である。これを訓練すると、その汎化能力が破壊され、プロジェクトの前提が崩れる ([Geminiの提案])。SAM Mask DecoderYes (訓練可能)MLPプロジェクタから来る新しい形式のプロンプト埋め込みを解釈する方法を学習する必要がある ([Geminiの提案])。MLP Projector (γ)Yes (訓練可能)このアーキテクチャのために構築される「橋」そのものであり、ゼロから学習されなければならない ([Geminiの提案])。LLM Token EmbeddingsYes (訓練可能)新しく追加された<SEG>トークンに意味のあるベクトル表現を学習させるために必要 ([Geminiの提案])。LLM Head (lm_head)Yes (訓練可能)最終出力層であり、新しいタスクの分布に適応させる必要がある ([Geminiの提案])。第3章 デュアルストリーム・データパイプライン: utils/dataset.py本章では、デュアルエンコーダアーキテクチャを支えるデータ処理パイプラインを定義する。モデルが正しくても、入力データが不適切であれば学習は成功しない。__getitem__メソッドとcollate_fnの実装が、この複雑なモデルの成否を分ける。3.1. デュアル前処理の論理的根拠モデルはGemma-SigLIPとSAM-ViTという2つの異なる視覚エンコーダを搭載しているため、単一の入力画像に対して、それぞれのエンコーダが要求する仕様に合わせた2種類の前処理を施す必要がある。これは本プロジェクトの根幹をなす要件である。Gemma用前処理: Gemma 3は、内部のSigLIPエンコーダが期待する形式に画像を変換する必要がある。これには、896x896ピクセルへのリサイズ、特定の正規化、そしてテンソル化が含まれる 1。この処理は、Hugging FaceのAutoProcessor.from_pretrained("google/gemma-3-4b-it")を使用するのが最も確実かつ簡単である。SAM用前処理: SAM-ViTは、独自の前処理パイプラインを要求する。公式実装 3 によれば、通常、画像の最も長い辺を1024ピクセルにリサイズし、短い辺をパディングして1024x1024の正方形にした後、特定の平均と標準偏差で正規化する。この2つのパイプラインの違いを以下の表にまとめる。HybridDatasetクラスは、この2つの異なるテンソルを同時に生成する責任を負う。前処理ステップGemma-3用 (AutoProcessor経由)SAM-ViT用 (カスタム変換)ターゲット解像度896 x 8961024 x 1024リサイズ戦略リサイズしてクロップ 13最長辺を1024にリサイズし、パディング 14正規化の平均値[0.5, 0.5, 0.5] 15[123.675, 116.28, 103.53] (ピクセル値)正規化の標準偏差[0.5, 0.5, 0.5] 15[58.395, 57.12, 57.375] (ピクセル値)3.2. HybridDatasetの実装utilsディレクトリを新規作成し、その中にdataset.pyを配置する。オリジナルのLISAリポジトリのHybridDatasetを参考にしつつ、Gemma 3の仕様に合わせて全面的に書き換える。Python# utils/dataset.py
import torch
from torch.utils.data import Dataset
import json
from PIL import Image
import numpy as np
import random
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from transformers import AutoProcessor

from.utils import preprocess, preprocess_mask # LISAの元コードから持ってくるヘルパー関数

class HybridDataset(Dataset):
    def __init__(self, config, tokenizer, samples_per_epoch):
        self.config = config
        self.tokenizer = tokenizer
        self.samples_per_epoch = samples_per_epoch
        
        # 1. Gemma用のプロセッサをロード
        self.gemma_processor = AutoProcessor.from_pretrained(config.GEMMA_MODEL_ID)

        # 2. SAM用の画像変換を定義
        self.sam_transform = Compose(, std=[0.229, 0.224, 0.225]),
        ])

        # LISAの元コードと同様に、各種データセットのメタ情報をロード
        self.dataset_list = config.DATASET.split("||")
        self.sample_rates =
        
        self.sem_seg_data = config.SEM_SEG_DATA.split("||")
        #... (以下、LISAの元コードと同様に各データセットのパスやアノテーションをロードする処理)
        # self.refer_seg_data =...
        # self.vqa_data =...
        # self.reason_seg_data =...
        
        # この例では、簡単のため、データロードのロジックは省略し、
        # __getitem__での処理に焦点を当てる。
        # 実際には、LISAの元のdataset.pyのロジックをここに移植する必要がある。
        print("データセットの初期化が完了しました。")


    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        # LISAのロジックに従い、サンプリングレートに基づいてデータセットを選択
        # この例では、reason_segからデータを取得するケースを想定
        
        # 1. データのロード (画像、テキスト、マスク)
        # この部分は、実際のデータセット構造に合わせて実装する必要がある
        # 例:
        # image_path =...
        # text_prompt = "運転方向を制御するために操作する部分をセグメント化して"
        # mask_path =...
        
        # ダミーデータで処理を例示
        image = Image.new('RGB', (1200, 800), color = 'red')
        text_prompt = "Show me the red area. <SEG>"
        gt_mask = np.ones((800, 1200), dtype=np.uint8)

        # 2. デュアル画像前処理
        # パスウェイA: Gemma用
        # AutoProcessorはリサイズ、正規化、テンソル化を全て行う
        # textもここで一緒に処理できるが、LISAの構造に合わせて別々に処理する
        gemma_processed = self.gemma_processor(images=image, return_tensors="pt")
        image_for_gemma = gemma_processed['pixel_values'].squeeze(0) # (3, 896, 896)

        # パスウェイB: SAM用
        # 元のLISAのpreprocess関数を参考にSAM用の前処理を行う
        # SAMの入力は1024x1024の正方形である必要がある
        image_for_sam, _ = preprocess(image, self.config.SAM_IMAGE_SIZE) # (3, 1024, 1024)
        
        # 3. テキストとマスクの前処理
        # LISAの元コードと同様に会話形式に変換
        conversations =
        
        # トークン化
        # この部分はLISAの元のコードのロジックを流用する
        # input_ids, labels =...
        
        # ダミーのトークン化
        tokenized_output = self.tokenizer(
            text_prompt,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.config.MODEL_MAX_LENGTH,
        )
        input_ids = tokenized_output.input_ids.squeeze(0)
        labels = input_ids.clone() # 簡単のため
        
        # <SEG>トークンの位置を特定するマスクを作成
        seg_token_idx = self.tokenizer.convert_tokens_to_ids("")
        seg_token_mask = (input_ids == seg_token_idx)

        # マスクの前処理
        processed_mask = preprocess_mask(gt_mask) # (1, 1024, 1024)

        return {
            "images_for_gemma": image_for_gemma,
            "images_for_sam": image_for_sam,
            "input_ids": input_ids,
            "labels": labels,
            "seg_token_mask": seg_token_mask,
            "ground_truth_mask": processed_mask,
            "has_mask": True, # マスクデータの有無を示すフラグ
        }

def collate_fn(batch: List) -> Dict:
    """
    バッチ内のデータを結合するカスタムcollate関数
    """
    # 各キーごとにデータをスタックする
    images_for_gemma = torch.stack([item["images_for_gemma"] for item in batch])
    images_for_sam = torch.stack([item["images_for_sam"] for item in batch])
    input_ids = torch.stack([item["input_ids"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    seg_token_mask = torch.stack([item["seg_token_mask"] for item in batch])
    
    # マスクが存在するサンプルのみをスタック
    masks = [item["ground_truth_mask"] for item in batch if item["has_mask"]]
    if masks:
        ground_truth_mask = torch.stack(masks)
    else:
        ground_truth_mask = None

    return {
        "images_for_gemma": images_for_gemma,
        "images_for_sam": images_for_sam,
        "input_ids": input_ids,
        "attention_mask": (input_ids!= 0), # パディングトークンは0と仮定
        "labels": labels,
        "seg_token_mask": seg_token_mask,
        "ground_truth_mask": ground_truth_mask,
    }

3.3. バッチの結合 (collate_fn)collate_fnは、データローダが__getitem__から受け取ったサンプルのリストを、モデルが一度に処理できる単一のバッチにまとめる役割を担う。特に、images_for_gemmaとimages_for_samという2つの画像テンソルを正しくスタックし、テキストシーケンスの長さをバッチ内で最長のものに合わせるパディング処理（tokenizerが自動で行う）をハンドリングする。また、VQAデータのようにマスクが存在しないサンプルを考慮し、マスクが存在するサンプルのみを結合するロジックを含めることが重要である。第4章 学習のオーケストレーション: train_deepspeed.py本章では、モデルの訓練プロセス全体を管理するメインスクリプトtrain_deepspeed.pyの仕様を定義する。これには、引数の解析、LoRAの設定、モデルとオプティマイザの初期化、複合損失関数の計算、そしてDeepSpeedによる訓練ループの実行が含まれる。4.1. 引数解析と設定argparseを用いて、訓練プロセスを柔軟に制御するためのコマンドライン引数を定義する。これにより、config_linux.pyの値を上書きしたり、実験ごとに設定を容易に変更したりすることが可能になる。4.2. Gemma-3のためのLoRA設定peftライブラリのLoraConfigを用いて、パラメータ効率の良いファインチューニングを設定する。ここで最も重要なのはtarget_modulesの指定である。前述の通り、Gemmaアーキテクチャに適したモジュール名を指定する必要がある。パラメータ推奨値正当性の根拠 / ソースr (ランク)16 or 32学習可能なパラメータ数と性能のトレードオフ。まずは16から始めるのが良い 7。lora_alphar * 2 (例: 32 or 64)学習された重みをスケーリングするための一般的な慣習 ([Geminiの提案])。lora_dropout0.05アダプタの重みの過学習を防ぐための標準的な正則化 7。target_modules["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]最重要: GemmaのTransformerブロック内の線形層の正しい名称。LLaMA用の設定を流用すると学習が失敗する 5。task_type"CAUSAL_LM"peftライブラリに対し、モデルを自己回帰型の言語生成タスク用にラップするよう指示する 5。4.3. モデルとオプティマイザの初期化訓練スクリプト内での初期化シーケンスは以下の通りである。LisaGemmaConfigとLisaGemmaForCausalLMを用いて、第2章で定義したカスタムモデルをインスタンス化する。tokenizer.add_tokens("", special_tokens=True)を呼び出して、新しい<SEG>トークンを語彙に追加する。model.resize_token_embeddings(len(tokenizer))を呼び出して、モデルの埋め込み層のサイズを新しい語彙サイズに合わせて拡張する。上記で定義したLoraConfigを作成し、get_peft_model(model, lora_config)を呼び出してモデルにLoRAアダプタを適用する。model.print_trainable_parameters()を実行し、訓練可能なパラメータの数と割合が意図通り（全パラメータの1%未満など、非常に小さい値）であることを確認する。これは重要な健全性チェックである。4.4. 複合損失関数訓練ループ内で、モデルのフォワードパスから返された出力を用いて複合損失を計算する。Python# 訓練ループ内での損失計算のロジック
outputs = model(**batch)
text_loss = outputs.get("text_loss")
predicted_masks = outputs.get("predicted_masks")
ground_truth_mask = batch.get("ground_truth_mask")

# マスク損失の計算 (マスクデータが存在する場合のみ)
mask_loss = torch.tensor(0.0, device=model.device)
if predicted_masks is not None and ground_truth_mask is not None:
    # DICE損失とBCE損失を計算するヘルパー関数を想定
    dice_loss = calculate_dice_loss(predicted_masks, ground_truth_mask)
    bce_loss = calculate_bce_loss(predicted_masks, ground_truth_mask)
    mask_loss = (config.DICE_LOSS_WEIGHT * dice_loss) + (config.BCE_LOSS_WEIGHT * bce_loss)

# 複合損失
# text_lossがNoneでないことを確認 (labelsが渡された場合のみ計算される)
total_loss = mask_loss
if text_loss is not None:
    total_loss += config.CE_LOSS_WEIGHT * text_loss
このロジックにより、セグメンテーションデータ（テキスト＋マスク）とVQAデータ（テキストのみ）が混在するバッチを正しく処理できる。VQAサンプルではmask_lossが0となり、テキスト生成能力の維持に貢献する。4.5. DeepSpeed訓練ループdeepspeed.initializeを用いてモデル、オプティマイザ、データローダをラップし、model_engineを作成する。標準的な訓練ループは以下のようになる。Python# DeepSpeedで初期化されたモデルエンジンを想定
model_engine, optimizer, train_loader, _ = deepspeed.initialize(...)

for epoch in range(config.EPOCHS):
    model_engine.train()
    for step, batch in enumerate(train_loader):
        # バッチをGPUに転送
        batch = {k: v.to(model_engine.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        # フォワードパス
        outputs = model_engine(**batch)
        
        # 損失計算 (前節のロジック)
        total_loss = calculate_composite_loss(outputs, batch, config)
        
        # バックワードパス
        model_engine.backward(total_loss)
        
        # オプティマイザステップ
        model_engine.step()
第5章 分散学習設定: ds_config.json本章では、DeepSpeedを用いた効率的な分散学習を実現するための設定ファイルds_config.jsonを定義する。特に、本プロジェクトの特性（LoRAファインチューニング）に最適なZeRO（Zero Redundancy Optimizer）戦略を選択することが、訓練スループットを最大化する鍵となる。5.1. 最適戦略の選択：ZeRO Stage 2の採用DeepSpeed ZeROは、複数のGPUにまたがるメモリの冗長性を削減するための最適化技術である。Stage 3: モデルのパラメータ、勾配、オプティマイザ状態の3つ全てを全GPUに分割する最も強力なステージ。これは、モデルのパラメータ自体が単一GPUのメモリに収まらない巨大なモデル（例：70Bパラメータモデル）を訓練する際には必須である 16。Stage 2: 勾配とオプティマイザ状態のみを分割し、各GPUはモデルパラメータの完全なコピーを保持する 17。本プロジェクトではgemma-3-4b-it（4Bパラメータ、bf16で約8GB）をベースモデルとしており、これは現代的なGPUのメモリに十分に収まる。さらに重要なのは、LoRAによるファインチューニングを行なっている点である。LoRAでは、モデルパラメータの99%以上が凍結されており、訓練対象となるのはごく一部のアダプタのみである。この状況下でStage 3を使用すると、フォワードパスとバックワードパスの度に、更新されない大部分の凍結パラメータを含む全モデルパラメータをGPU間で集約する必要が生じ、これが深刻な通信オーバーヘッドとなる。一方、Stage 2では、各GPUがパラメータのコピーを保持しているため、この通信は不要である。LoRAで更新される勾配とオプティマイザ状態は非常に小さいため、それらを分割する際の通信コストは無視できるレベルである。この分析は、文献によっても裏付けられている。「モデル全体が単一GPUに収まる場合はZeRO-2（またはそれ以下）を使用すべきである... ZeRO-2は、オプティマイザ状態と勾配に関する通信コストが低いため、特にLoRAに有用である」16。したがって、本プロジェクトにおいてZeRO Stage 2を選択することは、Stage 3と比較して訓練速度を大幅に向上させる、極めて重要な最適化である。5.2. ZeRO Stage 2のための完全なds_config.json以下の内容でds_config.jsonファイルをプロジェクトのルートディレクトリに作成する。JSON{
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_accumulation_steps": "auto",
  "steps_per_print": 10,

  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": "auto",
      "betas": [0.9, 0.95],
      "eps": 1e-8,
      "weight_decay": 1e-2
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

  "fp16": {
    "enabled": false
  },
  "bf16": {
    "enabled": true
  },

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

  "gradient_clipping": 1.0,
  "wall_clock_breakdown": false
}
設定の解説:"stage": 2: ZeRO Stage 2を有効にする。"bf16": { "enabled": true }: Gemma 3で推奨されるbfloat16混合精度学習を有効にする 4。"offload_optimizer": オプティマイザの状態をCPUメモリにオフロードすることで、GPUメモリをさらに節約する。"contiguous_gradients": メモリの断片化を減らし、通信効率を向上させる。"overlap_comm": 計算と通信をオーバーラップさせ、訓練のボトルネックを削減する。"auto"フィールド: DeepSpeedがtrain_micro_batch_size_per_gpuやgradient_accumulation_stepsなどの値を、train_batch_sizeとGPU数から自動的に推論することを可能にする。これにより、設定の柔軟性が向上する。第6章 実行および検証プロトコル本章では、訓練を確実に開始し、モデルが正しく学習していることを体系的に検証するための、必須の手順を定義する。6.1. 起動コマンドdeepspeedランチャーを使用して訓練スクリプトを実行する。以下は、4つのGPU（localhost:0,1,2,3）を使用する場合のコマンド例である。Bashdeepspeed --include localhost:0,1,2,3 train_deepspeed.py \
    --deepspeed_config ds_config.json \
    --exp_name "lisa-gemma3-run1" \
    --batch_size 16 \
    --grad_accumulation_steps 2 \
    --lr 1e-4 \
    --epochs 10
--batch_sizeはグローバルバッチサイズ（全GPUの合計）を指定し、--grad_accumulation_stepsと組み合わせて実効バッチサイズを調整する。DeepSpeedはこれらの値から、各GPUのマイクロバッチサイズを自動的に計算する。6.2. 段階的検証ガイド本格的な訓練を開始する前に、以下の検証ステップを順に実行することは、問題の早期発見とデバッグ時間の短縮に不可欠である。データ健全性チェック: データローダから1バッチ分のデータを抽出し、その内容を可視化する。image_for_gemmaとimage_for_samのテンソル形状がそれぞれ(B, 3, 896, 896)と(B, 3, 1024, 1024)になっているか確認する。テキストプロンプトをデコードして、意図通りにフォーマットされているか確認する。ground_truth_maskを画像に重ねて表示し、アノテーションが正しいか視覚的に確認する。このステップで問題が見つかった場合、utils/dataset.pyの__getitem__メソッドと前処理パイプラインにバグがある可能性が高い。フォワードパス・テスト: 訓練ループを実行せずに、モデルのforwardメソッドに1バッチ分のデータを渡してみる。RuntimeError（特に次元不一致エラー）が発生しないことを確認する。これはmlp_projectorの入出力次元が正しいかのテストになる。出力されるlogitsとpredicted_masksの形状が期待通りであることを確認する。このステップで失敗する場合、問題はmodel/gemma_lisa.pyのforwardメソッドのロジックまたはコンポーネントの初期化にある。単一バッチ過学習: これは最も重要な健全性テストである。データセットから1つのバッチだけを取り出し、そのバッチを繰り返し（例：100〜200ステップ）モデルに与えて訓練する。成功基準: 複合損失（total_loss）が着実に減少し、ほぼゼロに収束すること。失敗した場合: 損失が減少しない、または発散する場合、学習率が不適切、勾配フローがどこかで断絶している（例：requires_gradの設定ミス）、損失関数の実装が誤っている、オプティマイザの設定に問題がある、などの根本的な問題が存在することを示す。勾配フロー検査: 単一バッチ過学習で1ステップのbackward()を実行した後、訓練可能なパラメータの勾配を確認する。model.named_parameters()をループし、p.gradがNoneでないことを確認するべきパラメータ（LoRAアダプタ、MLPプロジェクタ、SAMデコーダ、埋め込み層、LMヘッド）と、Noneであるべきパラメータ（Gemmaの本体、SAM-ViTエンコーダ）を調べる。意図しない場所に勾配が流れていたり、流れるべき場所に流れていなかったりする場合、第2章のパラメータ管理戦略が正しく実装されていないことを意味する。6.3. 結果の解釈と定性的評価本格的な訓練が始まったら、TensorBoardなどのロギングツールで以下のメトリクスを監視する。損失: 訓練および検証データに対するtotal_loss、text_loss、mask_lossの推移。検証損失が訓練損失から大きく乖離し始めたら、過学習の兆候である。セグメンテーションIoU: 検証セット（例：ReasonSegのvalセット）に対するIntersection over Union（IoU）スコア。これが訓練の進行と共に向上することが、セグメンテーション能力が学習されていることの定量的証拠となる。さらに、定性的な評価として、訓練中に固定されたいくつかの検証サンプルに対して予測されたマスクを定期的に保存し、視覚的に確認することを強く推奨する。最初はノイズのようだったマスクが、エポックを重ねるごとに意味のあるオブジェクトの形状に近づいていく様子を観察できれば、モデルが正しく学習していることの強力な証拠となる。結論：動作するモデルへの道筋と今後の展望重要な成功要因の要約本仕様書は、LISAフレームワークにgemma-3-4b-itを統合するという野心的なプロジェクトを成功させるための、包括的な技術的ブループリントを提示した。実装者が直ちに検証し、実行すべき最も重要な成功要因は以下の通りである。デュアルストリーム・データパイプラインの厳密な実装: utils/dataset.pyにおいて、Gemma用（896x896）とSAM用（1024x1024）の2つの異なる前処理パイプラインを正しく実装すること。これは本アーキテクチャの根幹をなす。MLPプロジェクタの正確な次元設定: model/gemma_lisa.pyにおいて、プロジェクタの入力次元をGemmaの隠れ層サイズ（2560）に、出力次元をSAMのプロンプト埋め込み次元（256）に正確に一致させること。Gemma-3に最適化されたLoRA設定の適用: train_deepspeed.pyにおいて、Gemmaアーキテクチャに適したtarget_modules（q,k,v,o_projおよびgate,up,down_proj）を使用すること。パラメータ凍結戦略の徹底: 安定性と効率性を確保するため、意図したコンポーネント（LoRA、MLP、SAMデコーダ等）のみが訓練可能であり、特にSAM-ViTエンコーダが確実に凍結されていることを二重に確認すること。DeepSpeed ZeRO Stage 2の選択: LoRAファインチューニングの通信オーバーヘッドを最小化し、訓練スループットを最大化するために、ds_config.jsonでStage 2を選択すること。体系的な検証プロトコルの遵守: 特に、単一バッチの過学習テストは、訓練パイプライン全体の健全性を確認するための必須のマイルストーンである。将来的な機能強化と研究の方向性本仕様書で定義されたベースラインモデルの訓練が成功した暁には、以下のようなエキサイティングな研究の方向性を探求することが可能となる。段階的な凍結解除: ベースラインが安定した後、より多くのレイヤーの凍結を解除する実験が考えられる。例えば、GemmaのSigLIPエンコーダの最上位層や、Gemma LLM自体の凍結を部分的に解除（Full-tuningに近づける）することで、計算コストの増加と引き換えに性能が向上する可能性がある。高度なフュージョン戦略の探求: 本仕様書で提案したアーキテクチャは、MLPを介したシンプルな「レイトフュージョン」である。より高度な手法として、SAM-ViTの特徴量をGemmaの内部トークンと連結（concatenate）し、LLMに直接入力するような、より密結合なアーキテクチャを探求することも可能である。これは実装の複雑性を増すが、より強力な性能を発揮する可能性がある。LISA++およびVideoLISAコンセプトの適用: オリジナルのLISAプロジェクトは、インスタンスセグメンテーション（LISA++）やビデオセグメンテーション（VideoLISA）へと進化している [Geminiの提案]。Gemmaベースの統合が確固たるものになれば、異なるインスタンスに対して複数の<SEG>トークンを生成したり、モデルを時間的なダイナミクスを扱えるように拡張したりするなど、これらのより高度な機能の適応を探求することができる。これにより、本プロジェクトは単なるモデルの置き換えに留まらず、コンピュータビジョン研究の最前線に位置づけられるだろう。