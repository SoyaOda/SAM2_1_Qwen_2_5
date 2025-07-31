# tests/test_seg_token_generator.py
"""
Sa2VA風[SEG]トークン生成器のユニットテスト
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import unittest
from typing import Dict

from model.seg_token_generator import (
    LightweightSEGTokenGenerator,
    MultiFrameSEGTokenGenerator,
    create_seg_token_generator
)
import config_linux


class MockConfig:
    """テスト用のモック設定"""
    def __init__(self):
        self.qformer_config = {
            'hidden_size': 768,
            'num_queries': 32,
            'sam_prompt_dim': 256,
        }
        self.llama_hidden_size = 5120
        self.use_seg_token = True
        self.use_multi_frame_seg = False
        self.seg_token_return_attention = True


class TestSEGTokenGenerator(unittest.TestCase):
    """[SEG]トークン生成器のテストクラス"""
    
    def setUp(self):
        """テスト前の初期化"""
        self.config = MockConfig()
        self.batch_size = 2
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def test_lightweight_generator_init(self):
        """軽量版生成器の初期化テスト"""
        generator = LightweightSEGTokenGenerator(self.config)
        
        # 各コンポーネントの存在確認
        self.assertIsNotNone(generator.seg_token_projector)
        self.assertIsNotNone(generator.to_sam_prompt)
        self.assertIsNotNone(generator.seg_token_embedding)
        
        # 次元確認
        self.assertEqual(generator.qformer_dim, 768)
        self.assertEqual(generator.llama_dim, 5120)
        self.assertEqual(generator.sam_prompt_dim, 256)
        
        # 学習可能パラメータの形状確認
        self.assertEqual(generator.seg_token_embedding.shape, (1, 1, 768))
        
    def test_lightweight_generator_forward(self):
        """軽量版生成器のフォワードパステスト"""
        generator = LightweightSEGTokenGenerator(self.config).to(self.device)
        
        # テスト入力作成
        qformer_outputs = {
            'query_embeds': torch.randn(
                self.batch_size, 32, 768, 
                device=self.device
            )
        }
        llama_hidden_states = torch.randn(
            self.batch_size, 100, 5120,
            device=self.device
        )
        
        # フォワードパス実行
        outputs = generator(
            qformer_outputs=qformer_outputs,
            llama_hidden_states=llama_hidden_states,
            return_attention=True
        )
        
        # 出力検証
        self.assertIn('seg_token', outputs)
        self.assertIn('sam_prompt', outputs)
        self.assertIn('attention_weights', outputs)
        
        # 形状検証
        self.assertEqual(outputs['seg_token'].shape, (self.batch_size, 5120))
        self.assertEqual(outputs['sam_prompt'].shape, (self.batch_size, 256))
        self.assertEqual(outputs['attention_weights'].shape, (self.batch_size, 32))
        
        # 注意重みの合計が1になることを確認
        attention_sum = outputs['attention_weights'].sum(dim=-1)
        torch.testing.assert_close(
            attention_sum, 
            torch.ones_like(attention_sum),
            rtol=1e-4, 
            atol=1e-4
        )
        
    def test_multi_frame_generator_init(self):
        """動画対応版生成器の初期化テスト"""
        generator = MultiFrameSEGTokenGenerator(self.config)
        
        # 追加コンポーネントの存在確認
        self.assertIsNotNone(generator.temporal_fusion)
        self.assertIsNotNone(generator.frame_predictor)
        
        # LSTM設定確認
        self.assertEqual(generator.temporal_fusion.input_size, 5120)
        self.assertEqual(generator.temporal_fusion.hidden_size, 2560)  # 5120 // 2
        self.assertTrue(generator.temporal_fusion.bidirectional)
        
    def test_multi_frame_generator_forward(self):
        """動画対応版生成器のフォワードパステスト"""
        generator = MultiFrameSEGTokenGenerator(self.config).to(self.device)
        
        # テスト入力作成
        qformer_outputs = {
            'query_embeds': torch.randn(
                self.batch_size, 32, 768,
                device=self.device
            )
        }
        
        # 固定フレーム数でテスト
        num_frames = 4
        outputs = generator(
            qformer_outputs=qformer_outputs,
            num_frames=num_frames,
            return_attention=False
        )
        
        # 基本出力検証
        self.assertIn('seg_token', outputs)
        self.assertIn('sam_prompt', outputs)
        
        # 時間的出力検証
        self.assertIn('seg_tokens_temporal', outputs)
        self.assertIn('sam_prompts_temporal', outputs)
        self.assertIn('num_frames', outputs)
        
        # 形状検証
        self.assertEqual(
            outputs['seg_tokens_temporal'].shape, 
            (self.batch_size, num_frames, 5120)
        )
        self.assertEqual(
            outputs['sam_prompts_temporal'].shape,
            (self.batch_size, num_frames, 256)
        )
        self.assertEqual(outputs['num_frames'], num_frames)
        
    def test_factory_function(self):
        """ファクトリ関数のテスト"""
        # 軽量版
        generator1 = create_seg_token_generator(self.config, multi_frame=False)
        self.assertIsInstance(generator1, LightweightSEGTokenGenerator)
        
        # 動画対応版
        generator2 = create_seg_token_generator(self.config, multi_frame=True)
        self.assertIsInstance(generator2, MultiFrameSEGTokenGenerator)
        
    def test_gradient_flow(self):
        """勾配フローのテスト"""
        generator = LightweightSEGTokenGenerator(self.config).to(self.device)
        
        # テスト入力（requires_grad=True）
        qformer_outputs = {
            'query_embeds': torch.randn(
                self.batch_size, 32, 768,
                device=self.device,
                requires_grad=True
            )
        }
        
        # フォワードパス
        outputs = generator(qformer_outputs=qformer_outputs)
        
        # ダミー損失でバックプロパゲーション
        loss = outputs['seg_token'].sum() + outputs['sam_prompt'].sum()
        loss.backward()
        
        # 勾配が流れていることを確認
        self.assertIsNotNone(qformer_outputs['query_embeds'].grad)
        self.assertIsNotNone(generator.seg_token_embedding.grad)
        
        # 勾配が0でないことを確認
        self.assertTrue(torch.any(qformer_outputs['query_embeds'].grad != 0))
        self.assertTrue(torch.any(generator.seg_token_embedding.grad != 0))
        
    def test_device_compatibility(self):
        """異なるデバイスでの互換性テスト"""
        generator = LightweightSEGTokenGenerator(self.config)
        
        # CPU上でのテスト
        cpu_device = torch.device('cpu')
        generator = generator.to(cpu_device)
        
        qformer_outputs = {
            'query_embeds': torch.randn(1, 32, 768, device=cpu_device)
        }
        
        outputs = generator(qformer_outputs=qformer_outputs)
        
        # 出力がCPU上にあることを確認
        self.assertEqual(outputs['seg_token'].device.type, 'cpu')
        self.assertEqual(outputs['sam_prompt'].device.type, 'cpu')
        
    def test_error_handling(self):
        """エラーハンドリングのテスト"""
        generator = LightweightSEGTokenGenerator(self.config)
        
        # query_embedsがない場合
        with self.assertRaises(ValueError) as context:
            generator(qformer_outputs={})
        self.assertIn("must contain 'query_embeds'", str(context.exception))
            
        # 空のテンソルの場合（0次元）
        with self.assertRaises(ValueError) as context:
            generator(qformer_outputs={'query_embeds': torch.tensor([])})
        self.assertIn("empty tensor", str(context.exception))
        
        # 空のテンソルの場合（3次元だがサイズ0）
        with self.assertRaises(ValueError) as context:
            generator(qformer_outputs={'query_embeds': torch.zeros(0, 0, 0)})
        self.assertIn("empty tensor", str(context.exception))
        
        # 次元が不足している場合（2次元）
        with self.assertRaises(ValueError) as context:
            generator(qformer_outputs={'query_embeds': torch.randn(2, 768)})
        self.assertIn("must be at least 3D", str(context.exception))


class TestIntegrationWithConfig(unittest.TestCase):
    """config_linuxとの統合テスト"""
    
    def test_config_integration(self):
        """実際のconfig_linuxとの統合テスト"""
        # 実際の設定を使用
        seg_config = config_linux.get_seg_token_config()
        
        # 設定値の確認
        self.assertTrue(seg_config['use_seg_token'])
        self.assertFalse(seg_config['use_multi_frame'])
        self.assertEqual(seg_config['llama_hidden_size'], 5120)
        self.assertEqual(seg_config['sam_prompt_dim'], 256)
        
        # 期待性能向上値の確認
        self.assertEqual(seg_config['expected_improvement_phase1'], 5.0)
        self.assertEqual(seg_config['expected_improvement_phase2'], 10.0)
        self.assertEqual(seg_config['expected_improvement_phase3'], 15.0)


def run_tests():
    """テスト実行関数"""
    unittest.main(argv=[''], exit=False)


if __name__ == '__main__':
    print("=== Sa2VA風[SEG]トークン生成器ユニットテスト ===")
    run_tests()