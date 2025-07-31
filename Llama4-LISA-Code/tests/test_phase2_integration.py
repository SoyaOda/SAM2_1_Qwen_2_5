# tests/test_phase2_integration.py
"""
Phase 2: 部分統一トークン空間の統合テスト
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import unittest
from typing import Dict, Any

# Phase 2コンポーネント
from model.adaptive_token_compressor import AdaptiveTokenCompressor, create_adaptive_compressor
from model.cross_modal_unifier import CrossModalUnifier, create_cross_modal_unifier
from model.moe_integration_adapter import MoEIntegrationAdapter, create_moe_integration_adapter
import config_linux


class MockConfig:
    """テスト用のモック設定"""
    def __init__(self):
        # Q-Former設定
        self.qformer_config = {
            'hidden_size': 768,
            'num_queries': 32,
            'sam_prompt_dim': 256,
        }
        # Llama設定
        self.llama_hidden_size = 5120
        
        # Phase 2設定
        self.use_partial_unified_space = True
        self.use_adaptive_compression = True
        self.use_moe_integration = True
        
        # その他の設定
        self.use_seg_token = True
        self.use_multi_frame_seg = False
        self.seg_token_return_attention = False
        

class TestAdaptiveTokenCompressor(unittest.TestCase):
    """適応的トークン圧縮器のテスト"""
    
    def setUp(self):
        """テスト前の初期化"""
        self.config = MockConfig()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 2
        self.compressor = create_adaptive_compressor(self.config).to(self.device)
    
    def test_initialization(self):
        """初期化テスト"""
        self.assertIsInstance(self.compressor, AdaptiveTokenCompressor)
        self.assertEqual(self.compressor.qformer_dim, 768)
        self.assertEqual(self.compressor.llama_dim, 5120)
        self.assertEqual(self.compressor.max_compressed_tokens, 8)
        self.assertEqual(self.compressor.min_compressed_tokens, 1)
    
    def test_token_prediction(self):
        """トークン数予測テスト"""
        query_embeds = torch.randn(self.batch_size, 32, 768, device=self.device)
        num_tokens = self.compressor.predict_num_tokens(query_embeds)
        
        # 予測されたトークン数が範囲内
        self.assertGreaterEqual(num_tokens, 1)
        self.assertLessEqual(num_tokens, 8)
    
    def test_compression(self):
        """トークン圧縮テスト"""
        query_embeds = torch.randn(self.batch_size, 32, 768, device=self.device)
        qformer_outputs = {'query_embeds': query_embeds}
        
        # 圧縮実行
        outputs = self.compressor(qformer_outputs, return_details=True)
        
        # 出力検証
        self.assertIn('compressed_tokens', outputs)
        self.assertIn('num_tokens', outputs)
        self.assertIn('compression_ratio', outputs)
        
        # 形状検証
        compressed_tokens = outputs['compressed_tokens']
        self.assertEqual(compressed_tokens.shape[0], self.batch_size)
        self.assertEqual(compressed_tokens.shape[1], outputs['num_tokens'])
        self.assertEqual(compressed_tokens.shape[2], 5120)  # Llama次元
        
        # 圧縮率検証
        self.assertGreater(outputs['compression_ratio'], 1.0)
    
    def test_forced_compression(self):
        """強制トークン数での圧縮テスト"""
        query_embeds = torch.randn(self.batch_size, 32, 768, device=self.device)
        qformer_outputs = {'query_embeds': query_embeds}
        
        # 特定のトークン数で圧縮
        for num_tokens in [1, 4, 8]:
            outputs = self.compressor(
                qformer_outputs, 
                force_num_tokens=num_tokens,
                return_details=True
            )
            
            self.assertEqual(outputs['num_tokens'], num_tokens)
            self.assertEqual(outputs['compressed_tokens'].shape[1], num_tokens)


class TestCrossModalUnifier(unittest.TestCase):
    """クロスモーダル統一層のテスト"""
    
    def setUp(self):
        """テスト前の初期化"""
        self.config = MockConfig()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 2
        self.unifier = create_cross_modal_unifier(self.config).to(self.device)
    
    def test_initialization(self):
        """初期化テスト"""
        self.assertIsInstance(self.unifier, CrossModalUnifier)
        self.assertEqual(self.unifier.llama_dim, 5120)
        self.assertEqual(len(self.unifier.fusion_layers), 2)
        self.assertEqual(len(self.unifier.modality_embeddings.weight), 3)
    
    def test_single_modality(self):
        """単一モダリティ処理テスト"""
        # ビジョンのみ
        vision_tokens = torch.randn(self.batch_size, 8, 5120, device=self.device)
        outputs = self.unifier(vision_tokens=vision_tokens)
        
        self.assertIn('unified_tokens', outputs)
        self.assertIn('modality_masks', outputs)
        self.assertEqual(outputs['unified_tokens'].shape, (self.batch_size, 8, 5120))
    
    def test_multi_modality(self):
        """複数モダリティ処理テスト"""
        vision_tokens = torch.randn(self.batch_size, 8, 5120, device=self.device)
        text_tokens = torch.randn(self.batch_size, 10, 5120, device=self.device)
        seg_tokens = torch.randn(self.batch_size, 1, 5120, device=self.device)
        
        outputs = self.unifier(
            vision_tokens=vision_tokens,
            text_tokens=text_tokens,
            seg_tokens=seg_tokens,
            return_separated=True
        )
        
        # 統一トークンの検証
        self.assertIn('unified_tokens', outputs)
        self.assertEqual(outputs['unified_tokens'].shape, (self.batch_size, 19, 5120))  # 8 + 10 + 1
        
        # モダリティマスクの検証
        self.assertIn('modality_masks', outputs)
        self.assertIn('vision', outputs['modality_masks'])
        self.assertIn('text', outputs['modality_masks'])
        self.assertIn('seg', outputs['modality_masks'])
        
        # 分離されたトークンの検証
        self.assertIn('separated_tokens', outputs)
        self.assertEqual(outputs['separated_tokens']['vision'].shape, (self.batch_size, 8, 5120))
        self.assertEqual(outputs['separated_tokens']['text'].shape, (self.batch_size, 10, 5120))
        self.assertEqual(outputs['separated_tokens']['seg'].shape, (self.batch_size, 1, 5120))


class TestMoEIntegrationAdapter(unittest.TestCase):
    """MoE統合アダプターのテスト"""
    
    def setUp(self):
        """テスト前の初期化"""
        self.config = MockConfig()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 2
        self.adapter = create_moe_integration_adapter(self.config).to(self.device)
    
    def test_initialization(self):
        """初期化テスト"""
        self.assertIsInstance(self.adapter, MoEIntegrationAdapter)
        self.assertEqual(self.adapter.num_experts, 16)
        self.assertEqual(self.adapter.num_active_experts, 2)
        
        # エキスパート専門性の検証
        expert_spec = self.adapter.expert_specialization
        self.assertEqual(expert_spec.shape, (16, 3))
        # Softmax正規化されている
        self.assertTrue(torch.allclose(
            expert_spec.sum(dim=1), 
            torch.ones(16, device=expert_spec.device)
        ))
    
    def test_routing(self):
        """ルーティング機能テスト"""
        unified_tokens = torch.randn(self.batch_size, 10, 5120, device=self.device)
        
        # モダリティマスクの作成
        modality_masks = {
            'vision': torch.zeros(self.batch_size, 10, dtype=torch.bool, device=self.device),
            'text': torch.zeros(self.batch_size, 10, dtype=torch.bool, device=self.device)
        }
        modality_masks['vision'][:, :5] = True
        modality_masks['text'][:, 5:] = True
        
        outputs = self.adapter(
            unified_tokens=unified_tokens,
            modality_masks=modality_masks,
            return_routing_info=True
        )
        
        # 出力検証
        self.assertIn('adapted_tokens', outputs)
        self.assertIn('load_balance_loss', outputs)
        self.assertIn('routing_weights', outputs)
        self.assertIn('expert_indices', outputs)
        
        # 形状検証
        self.assertEqual(outputs['adapted_tokens'].shape, unified_tokens.shape)
        self.assertGreaterEqual(outputs['load_balance_loss'].item(), 0)
        
        # ルーティング情報の検証
        routing_weights = outputs['routing_weights']
        self.assertEqual(routing_weights.shape, (self.batch_size, 10, 16))
        # 各トークンのルーティング重みの合計が1
        self.assertTrue(torch.allclose(
            routing_weights.sum(dim=-1), 
            torch.ones(self.batch_size, 10, device=self.device),
            atol=1e-4
        ))


class TestPhase2Integration(unittest.TestCase):
    """Phase 2統合テスト"""
    
    def setUp(self):
        """テスト前の初期化"""
        self.config = MockConfig()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 2
        
        # 全コンポーネントの初期化
        self.compressor = create_adaptive_compressor(self.config).to(self.device)
        self.unifier = create_cross_modal_unifier(self.config).to(self.device)
        self.adapter = create_moe_integration_adapter(self.config).to(self.device)
    
    def test_full_pipeline(self):
        """完全なパイプラインテスト"""
        # 入力の準備
        qformer_outputs = {
            'query_embeds': torch.randn(self.batch_size, 32, 768, device=self.device)
        }
        text_tokens = torch.randn(self.batch_size, 10, 5120, device=self.device)
        seg_tokens = torch.randn(self.batch_size, 1, 5120, device=self.device)
        
        # 1. 適応的圧縮
        compressed = self.compressor(qformer_outputs, return_details=True)
        print(f"圧縮完了: 32 → {compressed['num_tokens']} (圧縮率: {compressed['compression_ratio']:.1f}x)")
        
        # 2. クロスモーダル統一
        unified = self.unifier(
            vision_tokens=compressed['compressed_tokens'],
            text_tokens=text_tokens,
            seg_tokens=seg_tokens,
            return_separated=True
        )
        print(f"統一完了: 総トークン数={unified['total_seq_len']}")
        
        # 3. MoE統合
        adapted = self.adapter(
            unified_tokens=unified['unified_tokens'],
            modality_masks=unified['modality_masks'],
            return_routing_info=True
        )
        print(f"MoE適用完了: ロードバランス損失={adapted['load_balance_loss'].item():.4f}")
        
        # 最終出力の検証
        self.assertEqual(adapted['adapted_tokens'].shape[0], self.batch_size)
        self.assertEqual(adapted['adapted_tokens'].shape[-1], 5120)
        
        # 分離されたビジョントークンの取得
        vision_tokens_final = unified['separated_tokens']['vision']
        self.assertEqual(vision_tokens_final.shape[0], self.batch_size)
        self.assertEqual(vision_tokens_final.shape[1], compressed['num_tokens'])


def run_tests():
    """テスト実行関数"""
    unittest.main(argv=[''], exit=False)


if __name__ == '__main__':
    print("=== Phase 2: 部分統一トークン空間 統合テスト ===")
    run_tests()