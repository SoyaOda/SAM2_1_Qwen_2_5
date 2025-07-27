"""
GPU Memory Monitoring and Management Utilities for 4x H100 80GB
Webリサーチ結果に基づく2025年最新のメモリ最適化実装
"""

import torch
import gc
import psutil
import time
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from contextlib import contextmanager

logger = logging.getLogger(__name__)

@dataclass
class MemorySnapshot:
    """メモリ使用量スナップショット"""
    timestamp: float
    gpu_allocated: Dict[int, float]  # GB
    gpu_reserved: Dict[int, float]   # GB
    gpu_total: Dict[int, float]      # GB
    gpu_free: Dict[int, float]       # GB
    cpu_memory: float                # GB
    cpu_percent: float               # %
    
    def gpu_usage_percent(self, gpu_id: int) -> float:
        """GPU使用率計算"""
        if gpu_id in self.gpu_total and self.gpu_total[gpu_id] > 0:
            return (self.gpu_allocated[gpu_id] / self.gpu_total[gpu_id]) * 100
        return 0.0
    
    def is_memory_critical(self, gpu_id: int, threshold: float = 85.0) -> bool:
        """メモリ使用量が危険レベルかチェック"""
        return self.gpu_usage_percent(gpu_id) > threshold

class GPUMemoryMonitor:
    """
    GPU メモリモニタリングクラス（Webリサーチベストプラクティス準拠）
    
    H100 80GB最適化:
    - expandable_segments対応
    - フラグメンテーション検出
    - 自動メモリクリーンアップ
    """
    
    def __init__(self, enable_detailed_logging: bool = True):
        self.enable_detailed_logging = enable_detailed_logging
        self.snapshots: List[MemorySnapshot] = []
        self.device_count = torch.cuda.device_count()
        
        # Webリサーチ推奨設定適用
        self._apply_memory_optimizations()
        
        logger.info(f"🔍 GPU Memory Monitor初期化: {self.device_count} GPUs")
        self._log_initial_state()
    
    def _apply_memory_optimizations(self):
        """Webリサーチ推奨のメモリ最適化設定を適用"""
        try:
            import os
            
            # 1. Expandable Segments（フラグメンテーション対策）
            current_alloc_conf = os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '')
            if 'expandable_segments' not in current_alloc_conf:
                if current_alloc_conf:
                    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = f"{current_alloc_conf},expandable_segments:True"
                else:
                    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "expandable_segments:True"
                logger.info("✅ expandable_segments:True 設定適用")
            
            # 2. PyTorch 2.x最適化設定
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            logger.info("✅ TF32最適化有効化")
            
            # 3. メモリプール設定（H100特化、train script最適化）
            for i in range(self.device_count):
                torch.cuda.set_per_process_memory_fraction(0.95, device=i)  # 95%まで使用許可
            logger.info("✅ GPU RAM使用率95%設定（test_phase3b成功パターン）")
            
        except Exception as e:
            logger.warning(f"⚠️ メモリ最適化設定エラー: {e}")
    
    def _log_initial_state(self):
        """初期状態をログ出力"""
        logger.info("📊 GPU初期状態:")
        for i in range(self.device_count):
            props = torch.cuda.get_device_properties(i)
            total_gb = props.total_memory / (1024**3)
            logger.info(f"  GPU {i}: {props.name} ({total_gb:.1f}GB)")
    
    def capture_snapshot(self) -> MemorySnapshot:
        """現在のメモリ使用状況をキャプチャ"""
        timestamp = time.time()
        gpu_allocated = {}
        gpu_reserved = {}
        gpu_total = {}
        gpu_free = {}
        
        for i in range(self.device_count):
            allocated = torch.cuda.memory_allocated(i) / (1024**3)
            reserved = torch.cuda.memory_reserved(i) / (1024**3)
            props = torch.cuda.get_device_properties(i)
            total = props.total_memory / (1024**3)
            free = total - allocated
            
            gpu_allocated[i] = allocated
            gpu_reserved[i] = reserved
            gpu_total[i] = total
            gpu_free[i] = free
        
        # CPU メモリ情報
        cpu_memory = psutil.virtual_memory()
        cpu_memory_gb = cpu_memory.used / (1024**3)
        cpu_percent = cpu_memory.percent
        
        snapshot = MemorySnapshot(
            timestamp=timestamp,
            gpu_allocated=gpu_allocated,
            gpu_reserved=gpu_reserved,
            gpu_total=gpu_total,
            gpu_free=gpu_free,
            cpu_memory=cpu_memory_gb,
            cpu_percent=cpu_percent
        )
        
        self.snapshots.append(snapshot)
        return snapshot
    
    def log_memory_status(self, prefix: str = ""):
        """メモリ状況をログ出力"""
        snapshot = self.capture_snapshot()
        
        if prefix:
            logger.info(f"📊 {prefix} - メモリ状況:")
        else:
            logger.info("📊 メモリ状況:")
        
        for i in range(self.device_count):
            usage_percent = snapshot.gpu_usage_percent(i)
            allocated = snapshot.gpu_allocated[i]
            reserved = snapshot.gpu_reserved[i]
            total = snapshot.gpu_total[i]
            free = snapshot.gpu_free[i]
            
            status_icon = "🔴" if snapshot.is_memory_critical(i) else "🟢"
            logger.info(
                f"  {status_icon} GPU {i}: {allocated:.1f}GB使用 / "
                f"{reserved:.1f}GB予約 / {total:.1f}GB総容量 "
                f"({usage_percent:.1f}% 使用, {free:.1f}GB空き)"
            )
        
        logger.info(f"  💻 CPU: {snapshot.cpu_memory:.1f}GB使用 ({snapshot.cpu_percent:.1f}%)")
        
        # メモリ危険レベル警告
        for i in range(self.device_count):
            if snapshot.is_memory_critical(i):
                logger.warning(f"⚠️ GPU {i}: メモリ使用量が危険レベル ({snapshot.gpu_usage_percent(i):.1f}%)")
    
    def detect_fragmentation(self) -> Dict[int, bool]:
        """メモリフラグメンテーション検出"""
        fragmentation_detected = {}
        
        for i in range(self.device_count):
            allocated = torch.cuda.memory_allocated(i)
            reserved = torch.cuda.memory_reserved(i)
            
            # 予約済みメモリが割当済みメモリより大幅に多い場合はフラグメンテーション
            if reserved > 0:
                fragmentation_ratio = (reserved - allocated) / reserved
                fragmentation_detected[i] = fragmentation_ratio > 0.3  # 30%以上の差があれば問題
                
                if fragmentation_detected[i] and self.enable_detailed_logging:
                    logger.warning(
                        f"🔴 GPU {i}: メモリフラグメンテーション検出 "
                        f"(予約: {reserved/(1024**3):.1f}GB, 使用: {allocated/(1024**3):.1f}GB, "
                        f"断片化率: {fragmentation_ratio*100:.1f}%)"
                    )
            else:
                fragmentation_detected[i] = False
        
        return fragmentation_detected
    
    def emergency_cleanup(self) -> bool:
        """緊急メモリクリーンアップ（Webリサーチベストプラクティス）"""
        logger.info("🧹 緊急メモリクリーンアップ実行...")
        
        cleanup_success = False
        
        try:
            # 1. Python ガベージコレクション
            collected = gc.collect()
            logger.info(f"  ✅ Python GC: {collected} オブジェクト解放")
            
            # 2. PyTorch キャッシュクリア
            torch.cuda.empty_cache()
            logger.info("  ✅ PyTorch CUDA キャッシュクリア")
            
            # 3. 各GPUで同期実行
            for i in range(self.device_count):
                torch.cuda.synchronize(i)
            logger.info("  ✅ 全GPU同期完了")
            
            # 4. クリーンアップ後の状況確認
            time.sleep(0.1)  # 短時間待機
            self.log_memory_status("クリーンアップ後")
            
            cleanup_success = True
            
        except Exception as e:
            logger.error(f"❌ メモリクリーンアップエラー: {e}")
        
        return cleanup_success
    
    @contextmanager
    def monitor_section(self, section_name: str):
        """コードセクションのメモリ使用量を監視"""
        logger.info(f"🔍 {section_name} - 開始")
        self.log_memory_status(f"{section_name} 開始前")
        
        start_snapshot = self.capture_snapshot()
        
        try:
            yield self
        except torch.OutOfMemoryError as e:
            logger.error(f"❌ {section_name} - CUDA OOM発生: {e}")
            self.log_memory_status(f"{section_name} OOM時")
            
            # 緊急クリーンアップ試行
            if self.emergency_cleanup():
                logger.info(f"🔄 {section_name} - クリーンアップ後に再試行してください")
            
            raise
        finally:
            end_snapshot = self.capture_snapshot()
            self.log_memory_status(f"{section_name} 完了後")
            
            # メモリ増加量を計算
            for i in range(self.device_count):
                memory_diff = end_snapshot.gpu_allocated[i] - start_snapshot.gpu_allocated[i]
                if abs(memory_diff) > 0.1:  # 100MB以上の変化
                    logger.info(
                        f"  📈 GPU {i}: {section_name}で{memory_diff:+.1f}GB変化"
                    )
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """メモリ使用量サマリを取得"""
        if not self.snapshots:
            return {}
        
        latest = self.snapshots[-1]
        summary = {
            "timestamp": latest.timestamp,
            "total_gpus": self.device_count,
            "gpu_usage": {},
            "cpu_usage": {
                "memory_gb": latest.cpu_memory,
                "percent": latest.cpu_percent
            },
            "alerts": []
        }
        
        for i in range(self.device_count):
            summary["gpu_usage"][i] = {
                "allocated_gb": latest.gpu_allocated[i],
                "reserved_gb": latest.gpu_reserved[i],
                "total_gb": latest.gpu_total[i],
                "usage_percent": latest.gpu_usage_percent(i),
                "is_critical": latest.is_memory_critical(i)
            }
            
            if latest.is_memory_critical(i):
                summary["alerts"].append(f"GPU {i} memory critical: {latest.gpu_usage_percent(i):.1f}%")
        
        # フラグメンテーション検出
        fragmentation = self.detect_fragmentation()
        for gpu_id, is_fragmented in fragmentation.items():
            if is_fragmented:
                summary["alerts"].append(f"GPU {gpu_id} memory fragmentation detected")
        
        return summary


# グローバルモニターインスタンス
_global_monitor: Optional[GPUMemoryMonitor] = None

def get_memory_monitor() -> GPUMemoryMonitor:
    """グローバルメモリモニターを取得"""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = GPUMemoryMonitor()
    return _global_monitor

def log_memory_status(prefix: str = ""):
    """メモリ状況をログ出力（簡易関数）"""
    monitor = get_memory_monitor()
    monitor.log_memory_status(prefix)

def emergency_cleanup() -> bool:
    """緊急メモリクリーンアップ（簡易関数）"""
    monitor = get_memory_monitor()
    return monitor.emergency_cleanup()

@contextmanager
def memory_monitor_section(section_name: str):
    """メモリ監視コンテキストマネージャー（簡易関数）"""
    monitor = get_memory_monitor()
    with monitor.monitor_section(section_name):
        yield