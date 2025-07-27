import os
import gc
import math
import torch
import torch.nn as nn
import transformers
import logging
from transformers import Trainer
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers.trainer_pt_utils import get_parameter_names
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Union, Any, Tuple

from utils.utils import AverageMeter, dict_to_cuda


# ロガーの初期化
logger = logging.getLogger(__name__)


class GemmaLISATrainer(Trainer):
    """
    Gemma3とSAMを統合したLISAモデルのカスタムTrainer
    
    言語モデルの損失とセグメンテーション損失の両方を考慮して学習を行います。
    """
    
    def __init__(self, **kwargs):
        # 追加ハイパーパラメータを先に取り出す
        self.ce_loss_weight = kwargs.pop("ce_loss_weight", 1.0)
        self.bce_loss_weight = kwargs.pop("bce_loss_weight", 2.0)
        self.dice_loss_weight = kwargs.pop("dice_loss_weight", 0.5)
        
        # processing_classを使用するように変更（tokenizer非推奨の対応）
        if "tokenizer" in kwargs and "processing_class" not in kwargs:
            kwargs["processing_class"] = kwargs.pop("tokenizer")
        
        # 親クラスの初期化
        super().__init__(**kwargs)
        
        # use_amp属性を追加（混合精度トレーニングを使用するかどうか）
        self.use_amp = self.args.fp16 or self.args.bf16
        logger.info(f"混合精度トレーニング(AMP): {'有効' if self.use_amp else '無効'}")
        
        # PyTorchバージョン情報のログ
        logger.info(f"PyTorchバージョン: {torch.__version__}")
        
        # 混合精度トレーニング用のスケーラーを初期化
        try:
            self.scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() and self.use_amp else None
            logger.info("GradScaler: torch.cuda.amp.GradScalerを使用")
        except AttributeError:
            logger.warning("torch.cuda.ampにGradScalerが見つかりません。代替パスを試みます...")
            try:
                self.scaler = torch.amp.GradScaler() if torch.cuda.is_available() and self.use_amp else None
                logger.info("GradScaler: torch.amp.GradScalerを使用")
            except AttributeError:
                logger.error("GradScalerが見つかりません。混合精度トレーニングは無効化されます。")
                self.scaler = None
                self.use_amp = False
        
        # GPUデバイス情報のログ
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            logger.info(f"GemmaLISATrainer: {device_count}台のGPUを使用してトレーニングを開始します")
            for i in range(device_count):
                logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            logger.info("GemmaLISATrainer: CPU環境でトレーニングを開始します（警告: GPUを使用すると大幅に高速化されます）")
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        言語モデル損失とセグメンテーション損失を計算する
        
        Args:
            model: 学習するモデル
            inputs: 入力データ (ディクショナリ)
            return_outputs: 出力も返すかどうか
            
        Returns:
            損失値または (損失値, 出力) のタプル
        """
        # inputs内のテンソルをCUDAに転送
        inputs = dict_to_cuda(inputs)
        
        # モデルの出力を取得
        outputs = model(**inputs)
        
        # 損失へのアクセス - 辞書と属性の両方に対応
        try:
            # 1. 属性としてアクセスを試みる（Gemma3などのモデルで期待される方法）
            if hasattr(outputs, "loss"):
                loss = outputs.loss
            # 2. 辞書としてアクセスを試みる
            elif isinstance(outputs, dict) and "loss" in outputs:
                loss = outputs["loss"]
            else:
                # どちらの方法でもlossが見つからない場合、例外を発生
                logger.error(f"損失が見つかりません。outputs型: {type(outputs)}")
                if isinstance(outputs, dict):
                    logger.error(f"出力キー: {list(outputs.keys())}")
                else:
                    logger.error(f"利用可能な属性: {dir(outputs)}")
                raise ValueError("モデル出力から損失を取得できません。モデルの出力形式を確認してください。")
            
            # 非テンソル型をテンソルに変換
            if not isinstance(loss, torch.Tensor):
                logger.info(f"compute_loss: 損失が非テンソル型です（{type(loss)}）: {loss}")
                
                # 辞書の場合
                if isinstance(loss, dict):
                    # 数値のみを含む辞書の場合
                    try:
                        # 辞書内の数値を抽出して平均を計算
                        numeric_values = []
                        for k, v in loss.items():
                            if isinstance(v, (int, float)):
                                numeric_values.append(float(v))
                            elif isinstance(v, torch.Tensor):
                                numeric_values.append(v.item())
                        
                        if numeric_values:
                            loss = torch.tensor(sum(numeric_values) / len(numeric_values), device=model.device)
                            logger.info(f"辞書から数値を抽出して損失を計算しました: {loss}")
                        else:
                            # 数値がない場合はゼロの損失を返す
                            logger.warning(f"辞書から数値を抽出できませんでした。ゼロの損失を使用します。")
                            loss = torch.tensor(0.0, device=model.device)
                    except Exception as e:
                        logger.error(f"辞書からの損失計算中にエラーが発生しました: {e}")
                        # エラーが発生した場合はゼロの損失を返す
                        loss = torch.tensor(0.0, device=model.device)
                # mapオブジェクトの場合
                elif isinstance(loss, map):
                    loss_list = list(loss)
                    # 文字列を含む可能性があるのでフィルタリング
                    numeric_loss = [float(x) for x in loss_list if isinstance(x, (int, float)) or (isinstance(x, str) and x.replace('.', '', 1).isdigit())]
                    if numeric_loss:
                        loss = torch.tensor(numeric_loss, device=model.device).mean()
                    else:
                        logger.warning(f"mapオブジェクトから数値を抽出できませんでした。ゼロの損失を使用します。")
                        loss = torch.tensor(0.0, device=model.device)
                # イテラブルな場合
                elif hasattr(loss, "__iter__"):
                    # 文字列を含む可能性があるのでフィルタリング
                    try:
                        numeric_loss = []
                        for x in loss:
                            if isinstance(x, (int, float)):
                                numeric_loss.append(float(x))
                            elif isinstance(x, str) and x.replace('.', '', 1).isdigit():
                                numeric_loss.append(float(x))
                            elif isinstance(x, torch.Tensor):
                                numeric_loss.append(x.item())
                        
                        if numeric_loss:
                            loss = torch.tensor(numeric_loss, device=model.device).mean()
                        else:
                            logger.warning(f"イテラブルから数値を抽出できませんでした。ゼロの損失を使用します。")
                            loss = torch.tensor(0.0, device=model.device)
                    except Exception as e:
                        logger.error(f"イテラブルからの損失計算中にエラーが発生しました: {e}")
                        loss = torch.tensor(0.0, device=model.device)
                # 単一の値の場合
                else:
                    try:
                        # 文字列の場合は数値に変換を試みる
                        if isinstance(loss, str):
                            if loss.replace('.', '', 1).isdigit():
                                loss = torch.tensor(float(loss), device=model.device)
                            else:
                                logger.warning(f"文字列を数値に変換できませんでした: {loss}. ゼロの損失を使用します。")
                                loss = torch.tensor(0.0, device=model.device)
                        else:
                            loss = torch.tensor(loss, device=model.device)
                    except Exception as e:
                        logger.error(f"損失値の変換中にエラーが発生しました: {e}")
                        loss = torch.tensor(0.0, device=model.device)
                
                logger.info(f"compute_loss: 損失をテンソルに変換しました: {loss}")
        except Exception as e:
            logger.error(f"compute_loss: 損失へのアクセス中にエラーが発生しました: {e}")
            logger.error(f"outputs型: {type(outputs)}")
            raise e
        
        # 損失の内訳を記録（属性アクセスと辞書アクセスの両方をサポート）
        # lm_loss
        if hasattr(outputs, "lm_loss") and outputs.lm_loss is not None:
            lm_loss = outputs.lm_loss
            # 非テンソル型の場合は変換
            if not isinstance(lm_loss, torch.Tensor):
                try:
                    if isinstance(lm_loss, dict):
                        # 辞書内の数値を抽出して平均を計算
                        numeric_values = []
                        for k, v in lm_loss.items():
                            if isinstance(v, (int, float)):
                                numeric_values.append(float(v))
                            elif isinstance(v, torch.Tensor):
                                numeric_values.append(v.item())
                        
                        if numeric_values:
                            lm_loss = torch.tensor(sum(numeric_values) / len(numeric_values), device=model.device)
                        else:
                            lm_loss = torch.tensor(0.0, device=model.device)
                    elif isinstance(lm_loss, map):
                        loss_list = list(lm_loss)
                        numeric_loss = [float(x) for x in loss_list if isinstance(x, (int, float)) or (isinstance(x, str) and x.replace('.', '', 1).isdigit())]
                        if numeric_loss:
                            lm_loss = torch.tensor(numeric_loss, device=model.device).mean()
                        else:
                            lm_loss = torch.tensor(0.0, device=model.device)
                    elif hasattr(lm_loss, "__iter__"):
                        numeric_loss = []
                        for x in lm_loss:
                            if isinstance(x, (int, float)):
                                numeric_loss.append(float(x))
                            elif isinstance(x, str) and x.replace('.', '', 1).isdigit():
                                numeric_loss.append(float(x))
                            elif isinstance(x, torch.Tensor):
                                numeric_loss.append(x.item())
                        
                        if numeric_loss:
                            lm_loss = torch.tensor(numeric_loss, device=model.device).mean()
                        else:
                            lm_loss = torch.tensor(0.0, device=model.device)
                    else:
                        if isinstance(lm_loss, str):
                            if lm_loss.replace('.', '', 1).isdigit():
                                lm_loss = torch.tensor(float(lm_loss), device=model.device)
                            else:
                                lm_loss = torch.tensor(0.0, device=model.device)
                        else:
                            lm_loss = torch.tensor(lm_loss, device=model.device)
                except Exception as e:
                    logger.error(f"lm_loss値の変換中にエラーが発生しました: {e}")
                    lm_loss = torch.tensor(0.0, device=model.device)
            self.log({"lm_loss": lm_loss.detach().cpu().item()})
            
        elif isinstance(outputs, dict) and "lm_loss" in outputs and outputs["lm_loss"] is not None:
            lm_loss = outputs["lm_loss"]
            # 非テンソル型の場合は変換
            if not isinstance(lm_loss, torch.Tensor):
                try:
                    if isinstance(lm_loss, dict):
                        # 辞書内の数値を抽出して平均を計算
                        numeric_values = []
                        for k, v in lm_loss.items():
                            if isinstance(v, (int, float)):
                                numeric_values.append(float(v))
                            elif isinstance(v, torch.Tensor):
                                numeric_values.append(v.item())
                        
                        if numeric_values:
                            lm_loss = torch.tensor(sum(numeric_values) / len(numeric_values), device=model.device)
                        else:
                            lm_loss = torch.tensor(0.0, device=model.device)
                    elif isinstance(lm_loss, map):
                        loss_list = list(lm_loss)
                        numeric_loss = [float(x) for x in loss_list if isinstance(x, (int, float)) or (isinstance(x, str) and x.replace('.', '', 1).isdigit())]
                        if numeric_loss:
                            lm_loss = torch.tensor(numeric_loss, device=model.device).mean()
                        else:
                            lm_loss = torch.tensor(0.0, device=model.device)
                    elif hasattr(lm_loss, "__iter__"):
                        numeric_loss = []
                        for x in lm_loss:
                            if isinstance(x, (int, float)):
                                numeric_loss.append(float(x))
                            elif isinstance(x, str) and x.replace('.', '', 1).isdigit():
                                numeric_loss.append(float(x))
                            elif isinstance(x, torch.Tensor):
                                numeric_loss.append(x.item())
                        
                        if numeric_loss:
                            lm_loss = torch.tensor(numeric_loss, device=model.device).mean()
                        else:
                            lm_loss = torch.tensor(0.0, device=model.device)
                    else:
                        if isinstance(lm_loss, str):
                            if lm_loss.replace('.', '', 1).isdigit():
                                lm_loss = torch.tensor(float(lm_loss), device=model.device)
                            else:
                                lm_loss = torch.tensor(0.0, device=model.device)
                        else:
                            lm_loss = torch.tensor(lm_loss, device=model.device)
                except Exception as e:
                    logger.error(f"lm_loss値の変換中にエラーが発生しました: {e}")
                    lm_loss = torch.tensor(0.0, device=model.device)
            self.log({"lm_loss": lm_loss.detach().cpu().item()})
        
        # mask_loss
        if hasattr(outputs, "mask_loss") and outputs.mask_loss is not None:
            mask_loss = outputs.mask_loss
            # 非テンソル型の場合は変換
            if not isinstance(mask_loss, torch.Tensor):
                try:
                    if isinstance(mask_loss, dict):
                        # 辞書内の数値を抽出して平均を計算
                        numeric_values = []
                        for k, v in mask_loss.items():
                            if isinstance(v, (int, float)):
                                numeric_values.append(float(v))
                            elif isinstance(v, torch.Tensor):
                                numeric_values.append(v.item())
                        
                        if numeric_values:
                            mask_loss = torch.tensor(sum(numeric_values) / len(numeric_values), device=model.device)
                        else:
                            mask_loss = torch.tensor(0.0, device=model.device)
                    elif isinstance(mask_loss, map):
                        loss_list = list(mask_loss)
                        numeric_loss = [float(x) for x in loss_list if isinstance(x, (int, float)) or (isinstance(x, str) and x.replace('.', '', 1).isdigit())]
                        if numeric_loss:
                            mask_loss = torch.tensor(numeric_loss, device=model.device).mean()
                        else:
                            mask_loss = torch.tensor(0.0, device=model.device)
                    elif hasattr(mask_loss, "__iter__"):
                        numeric_loss = []
                        for x in mask_loss:
                            if isinstance(x, (int, float)):
                                numeric_loss.append(float(x))
                            elif isinstance(x, str) and x.replace('.', '', 1).isdigit():
                                numeric_loss.append(float(x))
                            elif isinstance(x, torch.Tensor):
                                numeric_loss.append(x.item())
                        
                        if numeric_loss:
                            mask_loss = torch.tensor(numeric_loss, device=model.device).mean()
                        else:
                            mask_loss = torch.tensor(0.0, device=model.device)
                    else:
                        if isinstance(mask_loss, str):
                            if mask_loss.replace('.', '', 1).isdigit():
                                mask_loss = torch.tensor(float(mask_loss), device=model.device)
                            else:
                                mask_loss = torch.tensor(0.0, device=model.device)
                        else:
                            mask_loss = torch.tensor(mask_loss, device=model.device)
                except Exception as e:
                    logger.error(f"mask_loss値の変換中にエラーが発生しました: {e}")
                    mask_loss = torch.tensor(0.0, device=model.device)
            self.log({"mask_loss": mask_loss.detach().cpu().item()})
            
        elif isinstance(outputs, dict) and "mask_loss" in outputs and outputs["mask_loss"] is not None:
            mask_loss = outputs["mask_loss"]
            # 非テンソル型の場合は変換
            if not isinstance(mask_loss, torch.Tensor):
                try:
                    if isinstance(mask_loss, dict):
                        # 辞書内の数値を抽出して平均を計算
                        numeric_values = []
                        for k, v in mask_loss.items():
                            if isinstance(v, (int, float)):
                                numeric_values.append(float(v))
                            elif isinstance(v, torch.Tensor):
                                numeric_values.append(v.item())
                        
                        if numeric_values:
                            mask_loss = torch.tensor(sum(numeric_values) / len(numeric_values), device=model.device)
                        else:
                            mask_loss = torch.tensor(0.0, device=model.device)
                    elif isinstance(mask_loss, map):
                        loss_list = list(mask_loss)
                        numeric_loss = [float(x) for x in loss_list if isinstance(x, (int, float)) or (isinstance(x, str) and x.replace('.', '', 1).isdigit())]
                        if numeric_loss:
                            mask_loss = torch.tensor(numeric_loss, device=model.device).mean()
                        else:
                            mask_loss = torch.tensor(0.0, device=model.device)
                    elif hasattr(mask_loss, "__iter__"):
                        numeric_loss = []
                        for x in mask_loss:
                            if isinstance(x, (int, float)):
                                numeric_loss.append(float(x))
                            elif isinstance(x, str) and x.replace('.', '', 1).isdigit():
                                numeric_loss.append(float(x))
                            elif isinstance(x, torch.Tensor):
                                numeric_loss.append(x.item())
                        
                        if numeric_loss:
                            mask_loss = torch.tensor(numeric_loss, device=model.device).mean()
                        else:
                            mask_loss = torch.tensor(0.0, device=model.device)
                    else:
                        if isinstance(mask_loss, str):
                            if mask_loss.replace('.', '', 1).isdigit():
                                mask_loss = torch.tensor(float(mask_loss), device=model.device)
                            else:
                                mask_loss = torch.tensor(0.0, device=model.device)
                        else:
                            mask_loss = torch.tensor(mask_loss, device=model.device)
                except Exception as e:
                    logger.error(f"mask_loss値の変換中にエラーが発生しました: {e}")
                    mask_loss = torch.tensor(0.0, device=model.device)
            self.log({"mask_loss": mask_loss.detach().cpu().item()})
        
        if return_outputs:
            return loss, outputs
        return loss
    
    def _save_checkpoint(self, model, trial):
        """
        チェックポイントを保存する
        
        Args:
            model: 保存するモデル
            trial: Trialオブジェクト (HPO用)
        
        Returns:
            保存先のパス
        """
        # Trainer標準の保存処理を行う
        output_dir = super()._save_checkpoint(model, trial)
        
        # ここでSAM関連の追加コンポーネントのセーブやその他のカスタム処理を行うことも可能
        # 例: text_hidden_fcs など追加モジュールの保存
        
        return output_dir
    
    def _save(self, output_dir, state_dict=None, save_model=True, safe_serialization=True):
        """
        モデルを保存する
        
        Args:
            output_dir: 保存先ディレクトリ
            state_dict: 保存する状態辞書
            save_model: モデルを保存するかどうか
            safe_serialization: safetensorsを使用するかどうか
        """
        # safetensorsの保存エラーを回避するため、safe_serializationを無効化
        safe_serialization = False
        
        # ロギングレベルを一時的に変更して冗長な警告を抑制
        import logging
        import warnings
        
        # 元のロギングレベルを保存
        original_level = logging.getLogger().level
        original_transformers_level = logging.getLogger("transformers").level
        
        try:
            # ロギングレベルをERRORに設定して警告を抑制
            if safe_serialization:
                logging.getLogger().setLevel(logging.ERROR)
                logging.getLogger("transformers").setLevel(logging.ERROR)
                
                # safetensorsの警告を抑制
                warnings.filterwarnings("ignore", message="Some tensors share memory")
            
            logger.info(f"モデルを保存します: {output_dir} (safe_serialization={safe_serialization})")
            
            # Transformersの新しいバージョンでは_saveメソッドの引数が異なる
            if hasattr(super(), "_save") and callable(getattr(super(), "_save")):
                # 親クラスの_saveメソッドのシグネチャを確認
                import inspect
                signature = inspect.signature(super()._save)
                params = list(signature.parameters.keys())
                
                if len(params) >= 3 and "safe_serialization" in params:
                    # 新しいバージョン: 4引数
                    super()._save(output_dir, state_dict, save_model, safe_serialization)
                elif len(params) >= 3 and "save_model" in params:
                    # 中間バージョン: 3引数
                    super()._save(output_dir, state_dict, save_model)
                else:
                    # 古いバージョン: 2引数
                    super()._save(output_dir, state_dict)
            else:
                # 直接モデルを保存
                if state_dict is None:
                    state_dict = self.model.state_dict()
                
                if save_model:
                    torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))
                    self.model.config.save_pretrained(output_dir)
        except RuntimeError as e:
            # safetensorsエラーの場合、PyTorchの標準保存方法を使用（ログは最小限に）
            if "Some tensors share memory" in str(e):
                logger.info("safetensors形式での保存に失敗したため、PyTorch形式で保存します")
                
                if state_dict is None:
                    state_dict = self.model.state_dict()
                
                torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))
                self.model.config.save_pretrained(output_dir)
            else:
                # その他のエラーは再発生
                raise e
        finally:
            # 元のロギングレベルを復元
            logging.getLogger().setLevel(original_level)
            logging.getLogger("transformers").setLevel(original_transformers_level)
            warnings.resetwarnings()
    
    def create_optimizer(self):
        """
        最適化アルゴリズムを作成する
        
        Returns:
            optimizer: 最適化アルゴリズム
        """
        if self.optimizer is None:
            # 重み減衰を適用するパラメータ名の集合を取得
            # LayerNormやバイアスには重み減衰を適用しない
            decay_parameters = get_parameter_names(self.model, [nn.LayerNorm])
            decay_parameters = {name for name in decay_parameters if "bias" not in name}
            
            # パラメータをグループ化
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if n in decay_parameters],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if n not in decay_parameters],
                    "weight_decay": 0.0,
                },
            ]
            
            # オプティマイザの初期化
            optimizer_cls = (
                torch.optim.AdamW
                if self.args.optim == "adamw_torch"
                else transformers.optimization.AdamW
            )
            
            # オプティマイザを作成
            self.optimizer = optimizer_cls(
                optimizer_grouped_parameters,
                lr=self.args.learning_rate,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
            )
            
            # モデルのパラメータ名からパラメータ本体への辞書を作成（デバッグ・ログ用）
            param_name_map = {param: name for name, param in self.model.named_parameters()}
            
            # パラメータグループの情報を出力
            print("最適化アルゴリズムのパラメータグループ:")
            for i, pg in enumerate(self.optimizer.param_groups):
                print(f"  パラメータグループ {i}:")
                print(f"    学習率: {pg['lr']}")
                print(f"    Weight Decay: {pg.get('weight_decay', 0.0)}")
                print(f"    パラメータ数: {len(pg['params'])}")
                
                # 詳細なパラメータ名をログに残す場合はコメントを外す
                # param_names = [param_name_map.get(p, "unknown") for p in pg['params']]
                # print(f"    パラメータ: {param_names[:5]}... (合計 {len(param_names)} 個)")
        
        return self.optimizer
    
    def get_train_dataloader(self) -> DataLoader:
        """
        学習用のDataLoaderを取得する
        
        Returns:
            DataLoader: 学習用のDataLoader
        """
        # collate_fnが指定されていない場合のみDefaultDataCollatorを使用
        if self.data_collator is None:
            from utils.utils import collate_fn
            self.data_collator = collate_fn
        
        return super().get_train_dataloader()
    
    def log(self, logs: Dict[str, float], start_time=None) -> None:
        """
        ログを記録する
        
        Args:
            logs: ログデータ (辞書形式)
            start_time: 開始時間（オプション）、親クラスとの互換性のため
        """
        # GCを実行してメモリリークを防止
        if self.args.local_rank <= 0 and self.state.global_step % 100 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # 親クラスのログ処理
        super().log(logs, start_time)
    
    def _get_checkpoint_path(self):
        """
        最新のチェックポイントパスを取得する
        
        Returns:
            str: 最新のチェックポイントパス、存在しない場合はNone
        """
        output_dir = self.args.output_dir
        if not os.path.exists(output_dir):
            return None
            
        # チェックポイントディレクトリが存在するか確認
        checkpoint_dirs = [
            d for d in os.listdir(output_dir)
            if os.path.isdir(os.path.join(output_dir, d)) and d.startswith(PREFIX_CHECKPOINT_DIR)
        ]
        
        if len(checkpoint_dirs) == 0:
            return None
            
        # チェックポイントディレクトリを番号順にソート
        checkpoint_dirs = sorted(
            checkpoint_dirs,
            key=lambda x: int(x.replace(PREFIX_CHECKPOINT_DIR + "-", ""))
        )
        
        # 最新のチェックポイントディレクトリを返す
        return os.path.join(output_dir, checkpoint_dirs[-1])
    
    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        学習ステップを実行する関数
        入力からモデルの出力を計算し、損失を逆伝播させる

        Args:
            model: モデル
            inputs: 入力データ
            num_items_in_batch: バッチ内のアイテム数（オプション）

        Returns:
            損失値
        """
        # パラメータのrequires_gradを確認して設定
        for name, param in model.named_parameters():
            if param.requires_grad and not hasattr(param, 'grad_fn'):
                if 'mask_decoder' in name or 'text_hidden_fcs' in name:
                    # マスクデコーダとテキスト変換層は勾配が必要
                    param.requires_grad = True
                    
        # 入力テンソルについても確認
        if 'images' in inputs and inputs['images'] is not None:
            # 画像入力があることを確認
            images = inputs['images']
            # 画像テンソルを浮動小数点型に変換（必要な場合）
            if images.dtype in [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]:
                images = images.float()
                inputs['images'] = images
            if not images.requires_grad:
                images.requires_grad = True
            
        # 勾配を0にリセット
        model.zero_grad()
        self.optimizer.zero_grad()
        
        # モデル出力を計算
        with self.compute_loss_context_manager():
            outputs = self.compute_loss(model, inputs, return_outputs=True)
            
        if isinstance(outputs, tuple):
            loss, outputs = outputs
        else:
            loss = outputs
            outputs = None
        
        # MixedPrecisionで学習している場合
        if self.use_amp and self.scaler is not None:
            # テンソルチェック
            if not isinstance(loss, torch.Tensor):
                logger.warning(f"backward前の損失が非テンソル型です（{type(loss)}）。テンソルに変換します。")
                # 辞書から損失を抽出
                if isinstance(loss, dict):
                    try:
                        if "loss" in loss:
                            loss = loss["loss"]
                        else:
                            # 辞書内の数値を抽出して合計
                            loss_value = 0.0
                            for k, v in loss.items():
                                if isinstance(v, torch.Tensor):
                                    # 勾配計算のために必要な設定
                                    if not v.requires_grad:
                                        v.requires_grad = True
                                    loss_value += v.item()
                                elif isinstance(v, (int, float)):
                                    loss_value += float(v)
                            loss = torch.tensor(loss_value, device=model.device, requires_grad=True)
                    except Exception as e:
                        logger.error(f"辞書からの損失計算中にエラーが発生しました: {e}")
                        loss = torch.tensor(0.0, device=model.device, requires_grad=True)
                else:
                    try:
                        if isinstance(loss, str):
                            if loss.replace('.', '', 1).isdigit():
                                loss = torch.tensor(float(loss), device=model.device, requires_grad=True)
                            else:
                                loss = torch.tensor(0.0, device=model.device, requires_grad=True)
                        else:
                            loss = torch.tensor(loss, device=model.device, requires_grad=True)
                    except:
                        loss = torch.tensor(0.0, device=model.device, requires_grad=True)
            
            # 損失が勾配計算可能か確認
            if not loss.requires_grad:
                loss.requires_grad = True
                
            logger.info(f"バックワード前の損失: {loss.item()}, requires_grad={loss.requires_grad}")
            
            # GradScalerを使用した勾配計算
            self.scaler.scale(loss).backward()
            
            # 勾配クリッピング（CUDAが利用可能な場合のみ）
            if self.args.max_grad_norm is not None and self.args.max_grad_norm > 0 and torch.cuda.is_available():
                # クリッピング前にスケールを戻す
                self.scaler.unscale_(self.optimizer)
                self.accelerator.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)
            
            # オプティマイザステップとスケーラー更新
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # 通常の学習
            # テンソルチェック（念のため）
            if not isinstance(loss, torch.Tensor):
                logger.warning(f"backward前の損失が非テンソル型です（{type(loss)}）。テンソルに変換します。")
                # 再度変換を試みる
                if isinstance(loss, dict):
                    # 辞書内の数値を抽出して平均を計算
                    try:
                        if "loss" in loss:
                            loss = loss["loss"]
                            if not loss.requires_grad:
                                loss.requires_grad = True
                        else:
                            numeric_values = []
                            for k, v in loss.items():
                                if isinstance(v, (int, float)):
                                    numeric_values.append(float(v))
                                elif isinstance(v, torch.Tensor):
                                    if not v.requires_grad:
                                        v.requires_grad = True
                                    numeric_values.append(v.item())
                            
                            if numeric_values:
                                loss = torch.tensor(sum(numeric_values) / len(numeric_values), 
                                                   device=model.device, requires_grad=True)
                            else:
                                loss = torch.tensor(0.0, device=model.device, requires_grad=True)
                    except Exception as e:
                        logger.error(f"辞書からの損失計算中にエラーが発生しました: {e}")
                        loss = torch.tensor(0.0, device=model.device, requires_grad=True)
                elif isinstance(loss, map):
                    loss_list = list(loss)
                    numeric_loss = [float(x) for x in loss_list if isinstance(x, (int, float)) or (isinstance(x, str) and x.replace('.', '', 1).isdigit())]
                    if numeric_loss:
                        loss = torch.tensor(numeric_loss, device=model.device, requires_grad=True).mean()
                    else:
                        loss = torch.tensor(0.0, device=model.device, requires_grad=True)
                elif hasattr(loss, "__iter__"):
                    try:
                        numeric_loss = []
                        for x in loss:
                            if isinstance(x, (int, float)):
                                numeric_loss.append(float(x))
                            elif isinstance(x, str) and x.replace('.', '', 1).isdigit():
                                numeric_loss.append(float(x))
                            elif isinstance(x, torch.Tensor):
                                if not x.requires_grad:
                                    x.requires_grad = True
                                numeric_loss.append(x.item())
                        
                        if numeric_loss:
                            loss = torch.tensor(numeric_loss, device=model.device, requires_grad=True).mean()
                        else:
                            loss = torch.tensor(0.0, device=model.device, requires_grad=True)
                    except:
                        loss = torch.tensor(0.0, device=model.device, requires_grad=True)
                else:
                    try:
                        if isinstance(loss, str):
                            if loss.replace('.', '', 1).isdigit():
                                loss = torch.tensor(float(loss), device=model.device, requires_grad=True)
                            else:
                                loss = torch.tensor(0.0, device=model.device, requires_grad=True)
                        else:
                            loss = torch.tensor(loss, device=model.device, requires_grad=True)
                    except:
                        loss = torch.tensor(0.0, device=model.device, requires_grad=True)
            
            # 損失が勾配計算可能か確認
            if not loss.requires_grad:
                loss.requires_grad = True
                
            logger.info(f"バックワード前の損失: {loss.item()}, requires_grad={loss.requires_grad}")
            loss.backward()
            
            # 勾配クリッピング（CUDAが利用可能な場合のみ）
            if self.args.max_grad_norm is not None and self.args.max_grad_norm > 0 and torch.cuda.is_available():
                self.accelerator.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)
                
            self.optimizer.step()
        
        # スケジューラのステップ
        self.lr_scheduler.step()
        
        # ログに損失を記録
        logs = {"loss": loss.detach().cpu().item()}
        
        # 個別の損失をログに記録 - 辞書と属性の両方をチェック
        # lm_loss
        if hasattr(outputs, "lm_loss") and outputs.lm_loss is not None:
            logs["lm_loss"] = outputs.lm_loss.detach().cpu().item()
        elif isinstance(outputs, dict) and "lm_loss" in outputs and outputs["lm_loss"] is not None:
            logs["lm_loss"] = outputs["lm_loss"].detach().cpu().item()
            
        # mask_loss
        if hasattr(outputs, "mask_loss") and outputs.mask_loss is not None:
            logs["mask_loss"] = outputs.mask_loss.detach().cpu().item()
        elif isinstance(outputs, dict) and "mask_loss" in outputs and outputs["mask_loss"] is not None:
            logs["mask_loss"] = outputs["mask_loss"].detach().cpu().item()
        
        # ログをTrainerクラスのログ機構に追加
        self.log(logs)
        
        return loss.detach() 