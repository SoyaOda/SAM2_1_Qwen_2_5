#!/usr/bin/env python3
"""
QFormerSegmentationBridgeのforward メソッドシグネチャを調査するデバッグスクリプト
"""

import os
import sys
import inspect

# プロジェクトルートをパスに追加
sys.path.append('.')

# 環境設定
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
os.environ['PYTHONUNBUFFERED'] = '1'

def debug_qformer_bridge():
    """QFormerSegmentationBridgeの構造を詳細に調査"""
    print("=== QFormerSegmentationBridge デバッグ調査 ===")
    
    try:
        # 必要なモジュールのインポート
        from model.llama4_qformer_sam2 import QFormerSegmentationBridge, LlamaQFormerSAM2Config
        print("✓ QFormerSegmentationBridge インポート成功")
        
        # 1. クラスの基本情報
        print("\n--- クラス情報 ---")
        print(f"クラス名: {QFormerSegmentationBridge.__name__}")
        print(f"モジュール: {QFormerSegmentationBridge.__module__}")
        print(f"継承元: {QFormerSegmentationBridge.__bases__}")
        
        # 2. forward メソッドのシグネチャ
        print("\n--- forward メソッドシグネチャ ---")
        forward_method = QFormerSegmentationBridge.forward
        signature = inspect.signature(forward_method)
        print(f"シグネチャ: {signature}")
        
        # パラメータの詳細
        print("\n--- forward メソッドパラメータ詳細 ---")
        for param_name, param in signature.parameters.items():
            print(f"  {param_name}:")
            print(f"    - デフォルト値: {param.default if param.default != inspect.Parameter.empty else 'なし'}")
            print(f"    - アノテーション: {param.annotation if param.annotation != inspect.Parameter.empty else 'なし'}")
            print(f"    - 種類: {param.kind}")
        
        # 3. __init__ メソッドのシグネチャ
        print("\n--- __init__ メソッドシグネチャ ---")
        init_method = QFormerSegmentationBridge.__init__
        init_signature = inspect.signature(init_method)
        print(f"シグネチャ: {init_signature}")
        
        # 4. forward メソッドのソースコード（最初の20行）
        print("\n--- forward メソッドソースコード（抜粋） ---")
        try:
            source_lines = inspect.getsource(forward_method).split('\n')
            for i, line in enumerate(source_lines[:30]):  # 最初の30行
                print(f"{i+1:3d}: {line}")
        except Exception as e:
            print(f"ソースコード取得エラー: {e}")
        
        # 5. クラスの全メソッド一覧
        print("\n--- QFormerSegmentationBridge の全メソッド ---")
        methods = [method for method in dir(QFormerSegmentationBridge) 
                  if callable(getattr(QFormerSegmentationBridge, method)) 
                  and not method.startswith('_')]
        for method in sorted(methods):
            print(f"  - {method}")
        
        # 6. 設定クラスの確認
        print("\n--- LlamaQFormerSAM2Config の属性 ---")
        config = LlamaQFormerSAM2Config()
        config_attrs = [attr for attr in dir(config) if not attr.startswith('_')]
        for attr in sorted(config_attrs):
            value = getattr(config, attr)
            if not callable(value):
                print(f"  - {attr}: {value}")
        
        # 7. デュアルエンコーダー関連の属性を探す
        print("\n--- デュアルエンコーダー関連の検索 ---")
        for attr in dir(config):
            if 'dual' in attr.lower() or 'sam' in attr.lower() or 'encoder' in attr.lower():
                if not attr.startswith('_'):
                    value = getattr(config, attr)
                    print(f"  - {attr}: {value}")
        
        # 8. 実際にインスタンスを作成してみる
        print("\n--- インスタンス作成テスト ---")
        try:
            # 簡易設定でインスタンス作成
            test_config = LlamaQFormerSAM2Config()
            
            # デュアルエンコーダー設定を試す
            if hasattr(test_config, 'use_dual_encoder'):
                test_config.use_dual_encoder = True
                print("  - use_dual_encoder 設定: True")
            
            # 注意: 実際のモデルロードは重いのでスキップ
            print("  ✓ 設定オブジェクト作成成功")
            
        except Exception as e:
            print(f"  ❌ インスタンス作成エラー: {e}")
        
        print("\n=== デバッグ調査完了 ===")
        
    except Exception as e:
        print(f"\n❌ デバッグエラー: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    debug_qformer_bridge()