from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (AutoProcessor, AutoTokenizer, 
                         AutoModelForCausalLM, AutoConfig,
                         Gemma3ForConditionalGeneration)

from model.segment_anything import build_sam_vit_h
from ..mm_utils import get_gemma_processor, GemmaImageProcessor
from ..utils import build_gemma_inputs, init_gemma_tokenizer


def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    scale=1000,  # 100000.0,
    eps=1e-6,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1, 2)
    targets = targets.flatten(1, 2)
    numerator = 2 * (inputs / scale * targets).sum(-1)
    denominator = (inputs / scale).sum(-1) + (targets / scale).sum(-1)
    loss = 1 - (numerator + eps) / (denominator + eps)
    loss = loss.sum() / (num_masks + 1e-8)
    return loss


def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = loss.flatten(1, 2).mean(1).sum() / (num_masks + 1e-8)
    return loss


class GemmaLISAMetaModel(nn.Module):
    """Gemmaモデルとサムを統合したメタモデルクラス"""
    
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super().__init__()

        self.config = config
        if not hasattr(self.config, "train_mask_decoder"):
            self.config.train_mask_decoder = kwargs["train_mask_decoder"]
            self.config.out_dim = kwargs["out_dim"]
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
        else:
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
            self.initialize_lisa_modules(self.config)

    def initialize_lisa_modules(self, config):
        # SAM
        self.visual_model = build_sam_vit_h(self.vision_pretrained)
        for param in self.visual_model.parameters():
            param.requires_grad = False
        if config.train_mask_decoder:
            self.visual_model.mask_decoder.train()
            for param in self.visual_model.mask_decoder.parameters():
                param.requires_grad = True

        # Projection layer - Gemma3の隠れ状態からSAMのプロンプト次元へ
        if hasattr(config, 'hidden_size'):
            in_dim = config.hidden_size
        elif hasattr(config, 'text_config'):
            in_dim = config.text_config.hidden_size
        else:
            # デフォルト値（Gemma3-4bの場合）
            in_dim = 2048
        
        out_dim = config.out_dim
        text_fc = [
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
            nn.Dropout(0.0),
        ]
        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_fc)])
        self.text_hidden_fcs.train()
        for param in self.text_hidden_fcs.parameters():
            param.requires_grad = True


class GemmaLISAModel(GemmaLISAMetaModel):
    """GemmaとSAMを統合したモデルクラス"""
    
    def __init__(
        self,
        config,
        **kwargs,
    ):
        # 親クラスの初期化
        super(GemmaLISAModel, self).__init__(config, **kwargs)
        
        # Configure model parameters
        self.config.use_cache = False
        
        # SAM関連の初期化
        self.sam_checkpoint = kwargs.get("sam_checkpoint", None)
        self.seg_token_idx = kwargs.get("seg_token_idx", None)
        
        # Gemma3モデル
        self.gemma_model = None
        
        if self.sam_checkpoint is not None:
            self.initialize_lisa_modules(self.config)
    
    def forward(self, *args, **kwargs):
        """
        フォワードメソッド - Gemma3モデルに処理を委譲
        """
        if self.gemma_model is not None:
            return self.gemma_model(*args, **kwargs)
        else:
            raise ValueError("gemma_modelが初期化されていません")

    def encode_images(self, pixel_values):
        """
        Gemma3のビジョンエンコーダで画像をエンコードする
        """
        if self.gemma_model is not None and hasattr(self.gemma_model, "vision_model"):
            # Gemma3モデルの内部エンコーダを使用
            vision_outputs = self.gemma_model.vision_model(pixel_values)
            return vision_outputs.last_hidden_state
        else:
            raise ValueError("Gemma3モデルが初期化されていないか、vision_modelが見つかりません")
        
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, sam_checkpoint=None, **kwargs):
        """
        GemmaLISAModelのプリトレーニング済みモデルをロードするメソッド
        sam_checkpointパラメータを受け取り、SAMをロードします
        """
        # LISAに特化したパラメータを抽出
        seg_token_idx = kwargs.pop("seg_token_idx", None)
        train_mask_decoder = kwargs.pop("train_mask_decoder", True)
        out_dim = kwargs.pop("out_dim", 256)
        
        # Gemma3ForConditionalGenerationの設定を取得
        print(f"Gemma3モデルをロード中: {pretrained_model_name_or_path}")
        config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=True
        )
        
        # text_configからhidden_sizeなどの属性をトップレベルにコピー
        if hasattr(config, 'text_config'):
            config.hidden_size = getattr(config.text_config, 'hidden_size', 2048)
            config.vocab_size = getattr(config.text_config, 'vocab_size', 262144)
            config.num_hidden_layers = getattr(config.text_config, 'num_hidden_layers', 18)
            config.num_attention_heads = getattr(config.text_config, 'num_attention_heads', 8)
            print(f"text_configからhidden_size={config.hidden_size}を設定しました")
        
        # LISA用のカスタム設定を追加
        config.seg_token_idx = seg_token_idx
        config.train_mask_decoder = train_mask_decoder
        config.out_dim = out_dim
        
        # GemmaLISAModelインスタンスを作成するためのパラメータ
        lisa_model_args = {
            "sam_checkpoint": sam_checkpoint,
            "seg_token_idx": seg_token_idx,
            "train_mask_decoder": train_mask_decoder,
            "out_dim": out_dim,
        }
        
        # LISAモデルのインスタンス化
        lisa_model = cls(config, **lisa_model_args)
        
        # Gemma3モデルをロード
        # SAM関連のパラメータをkwargsから削除してGemma3モデルに渡さないようにする
        gemma_model = Gemma3ForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path,
            config=config,
            trust_remote_code=True,
            **kwargs  # SAM関連パラメータは既に取り除いたkwargsのみを渡す
        )
        
        # Gemma3モデルのattributesをコピー
        lisa_model.gemma_model = gemma_model
        
        return lisa_model


class GemmaLISAForCausalLM(nn.Module):
    """
    Gemmaをベースとしたモデル
    SAMを統合してセグメンテーション機能を持たせる
    """
    
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super().__init__()
        
        # 引数からSAM関連の設定を取得
        if not hasattr(config, "train_mask_decoder"):
            self.ce_loss_weight = kwargs.pop("ce_loss_weight", 1.0)
            self.dice_loss_weight = kwargs.pop("dice_loss_weight", 0.5)
            self.bce_loss_weight = kwargs.pop("bce_loss_weight", 2.0)
            
        # セグメンテーショントークンのインデックスを取得
        self.seg_token_idx = kwargs.pop("seg_token_idx", None)
        
        # SAMのチェックポイントパス
        self.sam_checkpoint = kwargs.pop("sam_checkpoint", None)
        
        # モデルの初期化
        self.model = GemmaLISAModel(config, **kwargs)
        
        # プロセッサの初期化
        self.processor = None
        
        # SAMモジュールの初期化
        self.initialize_lisa_modules(config, **kwargs)
    
    def initialize_lisa_modules(self, config, **kwargs):
        """
        LISAモジュール（SAM統合）を初期化する
        """
        # SAM
        self.visual_model = build_sam_vit_h(self.sam_checkpoint)
        for param in self.visual_model.parameters():
            param.requires_grad = False
            
        # マスクデコーダを学習する場合
        train_mask_decoder = kwargs.get("train_mask_decoder", True)
        if train_mask_decoder:
            self.visual_model.mask_decoder.train()
            for param in self.visual_model.mask_decoder.parameters():
                param.requires_grad = True

        # テキスト特徴からセグメンテーション用埋め込みへの変換層
        if hasattr(config, 'hidden_size'):
            in_dim = config.hidden_size
        elif hasattr(config, 'text_config'):
            in_dim = config.text_config.hidden_size
        else:
            # デフォルト値（Gemma3-4bの場合）
            in_dim = 2048
        
        out_dim = kwargs.get("out_dim", 256)
        
        # 変換層の定義 - 勾配が伝播するように実装
        text_fc = [
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
            nn.Dropout(0.0),
        ]
        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_fc)])
        
        # 明示的にtrainモードに設定
        self.text_hidden_fcs.train()
        for param in self.text_hidden_fcs.parameters():
            param.requires_grad = True
            
    def load_processor(self, model_name):
        """
        モデル名からプロセッサをロードする
        
        Args:
            model_name: Gemmaモデル名
        """
        self.processor = get_gemma_processor(model_name)
        # トークナイザの初期化
        if hasattr(self.processor, "tokenizer"):
            init_gemma_tokenizer(self.processor.tokenizer)
        
        # 画像プロセッサの設定
        self.image_processor = GemmaImageProcessor(self.processor)
    
    def get_visual_embs(self, pixel_values: torch.FloatTensor):
        """
        画像からSAMの視覚エンコーダで特徴量を抽出する
        
        Args:
            pixel_values: 画像ピクセル値 (B, C, H, W)
            
        Returns:
            torch.Tensor: SAMの視覚特徴量
        """
        with torch.no_grad():
            image_embeddings_list = []
            for i in range(pixel_values.shape[0]):
                torch.cuda.empty_cache()
                image_embeddings = self.visual_model.image_encoder(
                    pixel_values[i].unsqueeze(0)
                )
                image_embeddings_list.append(image_embeddings)
            torch.cuda.empty_cache()
            image_embeddings = torch.cat(image_embeddings_list, 0)
        return image_embeddings
    
    def forward(
        self,
        input_ids=None,
        pixel_values=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        images=None,
        image_sizes=None,
        return_dict=None,
        **kwargs,
    ):
        """
        フォワードメソッド: Gemma3モデルの入力と画像入力を処理し、SAMを統合する
        
        Args:
            input_ids: 入力テキストのトークンID
            pixel_values: 入力画像のピクセル値
            attention_mask: 注意マスク
            position_ids: 位置ID
            past_key_values: キャッシュ用の過去のKV値
            inputs_embeds: 埋め込み入力
            labels: 教師ラベル
            use_cache: キャッシュを使用するかどうか
            output_attentions: 注意スコアを出力するかどうか
            output_hidden_states: 隠れ状態を出力するかどうか
            images: 画像入力
            image_sizes: 画像サイズ
            return_dict: 辞書形式で返すかどうか
        """
        return_dict = return_dict if return_dict is not None else True
        output_attentions = output_attentions if output_attentions is not None else False
        output_hidden_states = output_hidden_states if output_hidden_states is not None else False
        
        # SAMモデルの入力としての画像を処理
        # imagesが存在する場合はそれを使用
        if images is not None:
            # SAM用の画像入力を準備
            images_tensor = images
            
            # 画像入力が提供されている場合のみ処理
            if images_tensor is not None:
                device = images_tensor.device
                
                # mask_decoder以外のSAMコンポーネントはgrad不要
                with torch.no_grad():
                    # 画像エンコーダで特徴を抽出
                    image_embeddings = self.visual_model.image_encoder(images_tensor)
                    
                    # プロンプトエンコーダでポジショナル・エンコーディングを取得
                    pe = self.visual_model.prompt_encoder.get_dense_pe()
                    
                    # 画像サイズ情報が指定されていない場合はデフォルト値を使用
                    if image_sizes is None:
                        image_sizes = [(images_tensor.shape[2], images_tensor.shape[3])] * images_tensor.shape[0]
                    
                    # 画像サイズ情報をテンソルに変換
                    if not isinstance(image_sizes, torch.Tensor):
                        image_sizes = torch.tensor(image_sizes, device=device)
                
                # LLMの出力から、セグメンテーショントークンの位置を特定
                if self.seg_token_idx is not None:
                    # 勾配が伝播するために、入力テンソルにrequires_gradを設定
                    if input_ids is not None and input_ids.requires_grad == False:
                        input_ids.requires_grad = True
                    if inputs_embeds is not None and inputs_embeds.requires_grad == False:
                        inputs_embeds.requires_grad = True
                    
                    # Gemma3モデルの出力を取得
                    gemma_output = self.model.gemma_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                        inputs_embeds=inputs_embeds,
                        use_cache=use_cache,
                        output_attentions=output_attentions,
                        output_hidden_states=True,  # 隠れ状態が必要
                        return_dict=True,
                    )
                    
                    # 隠れ状態を取得
                    hidden_states = gemma_output.hidden_states[-1]
                    
                    # セグメンテーショントークンの位置を特定
                    seg_token_pos = torch.nonzero(input_ids == self.seg_token_idx, as_tuple=True)
                    batch_indices, seg_indices = seg_token_pos
                    
                    # サンプル数を取得
                    num_samples = batch_indices.shape[0]
                    
                    # セグメンテーショントークンの特徴を抽出
                    seg_hidden_states = hidden_states[batch_indices, seg_indices]
                    
                    # SAMのマスクデコーダに入力するために特徴を変換
                    # 明示的に勾配計算を有効化
                    seg_hidden_states.requires_grad = True
                    sparse_embeddings = []
                    for i, fc in enumerate(self.text_hidden_fcs):
                        sparse_embeddings.append(fc(seg_hidden_states))
                    
                    sparse_embeddings = torch.stack(sparse_embeddings, dim=1).to(device)
                    
                    # SAMのマスクデコーダにより、マスクを予測
                    # 勾配計算が可能なように処理
                    masks, iou_pred = self.visual_model.mask_decoder(
                        image_embeddings=image_embeddings,
                        image_pe=pe,
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=torch.zeros_like(image_embeddings),
                        multimask_output=False,
                    )
                    
                    # 勾配が伝播するように明示的に設定
                    masks.requires_grad = True
                    iou_pred.requires_grad = True
                    
                    # マスクを元の画像サイズにリサイズ
                    masks = masks.to(device)
                    
                    # 結果を辞書にまとめる
                    output_dict = {
                        **gemma_output,
                        "masks": masks,
                        "iou_scores": iou_pred,
                        "seg_token_hidden_states": seg_hidden_states,
                    }
                    
                    return output_dict
        
        # 画像が提供されていない場合やセグメンテーショントークンが無い場合は通常のLLM出力を返す
        outputs = self.model.gemma_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        return outputs
    
    def generate(self, *args, **kwargs):
        """
        Gemma3モデルのgenerate関数を呼び出すラッパー
        """
        return self.model.gemma_model.generate(*args, **kwargs)
    
    def evaluate(
        self,
        pixel_values,
        images,
        input_ids,
        resize_list,
        original_size_list,
        max_new_tokens=32,
        tokenizer=None,
    ):
        """
        モデルによる評価処理（推論）
        
        Args:
            pixel_values: Gemma3用の画像テンソル
            images: SAM用の高解像度画像テンソル
            input_ids: 入力テキストのトークンID
            resize_list: リサイズ情報のリスト
            original_size_list: 元画像サイズのリスト
            max_new_tokens: 生成する最大トークン数
            tokenizer: トークナイザ
            
        Returns:
            dict: 生成テキストとセグメンテーションマスク
        """
        # 入力準備
        attention_mask = torch.ones_like(input_ids)
        offset = torch.Tensor([0, 1]).long().to(input_ids.device)
        label_dummy = None
        masks_dummy = [None]
        label_list_dummy = [None]
        
        # 自己回帰生成
        with torch.no_grad():
            outputs = self.generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
            )
        
        # 生成されたトークンをデコード
        outputs = outputs[:, input_ids.shape[1]:] # 入力を除いた生成部分だけを抽出
        response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        
        # セグメンテーション実行
        with torch.no_grad():
            # セグメンテーション予測
            seg_results = self.forward(
                images=images,
                pixel_values=pixel_values,
                input_ids=outputs,
                labels=label_dummy,
                attention_masks=torch.ones_like(outputs),
                position_ids=None,
                past_key_values=None,
                inputs_embeds=None,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=True,
                inference=True,
            )
        
        # セグメンテーション結果の後処理
        if "masks" in seg_results:
            mask_pred = seg_results["masks"]
            # マスクのリサイズ、後処理など
            masks = []
            for i, (pred, (h, w), original_size) in enumerate(zip(mask_pred, resize_list, original_size_list)):
                pred = pred.unsqueeze(0)
                pred = F.interpolate(
                    pred, (h, w), mode="bilinear", align_corners=False
                )
                
                # マスクを元のサイズにリサイズ
                pred = F.interpolate(
                    pred, original_size, mode="bilinear", align_corners=False
                )
                pred = pred.sigmoid().gt(0.5)
                masks.append(pred.squeeze())
        else:
            masks = None
        
        # 結果を返す
        return {"response": response, "masks": masks} 