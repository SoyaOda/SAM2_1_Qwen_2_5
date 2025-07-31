"""
Configuration for SAM2.1 + Qwen2.5-VL Integration
統合モデルの設定ファイル
"""
import torch
from typing import Dict, Any, Optional, List


class SAMQwenConfig:
    """Configuration class for SAM-Qwen integrated model"""
    
    def __init__(self):
        # Model identifiers
        self.QWEN_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
        self.SAM_MODEL_ID = "facebook/sam2.1-hiera-large"
        
        # Model settings
        self.TORCH_DTYPE = torch.float16
        self.DEVICE_MAP = "auto"
        self.TRUST_REMOTE_CODE = True
        self.LOW_CPU_MEM_USAGE = True
        
        # Visual projector settings
        self.USE_QFORMER = True
        self.NUM_QUERIES = 32
        self.ENABLE_POSITION_ENCODING = True
        self.ENHANCED_PROJECTOR = True
        
        # Training settings
        self.LEARNING_RATE = 2e-5
        self.BATCH_SIZE = 1  # Adjust based on GPU memory
        self.MAX_LENGTH = 2048
        self.MAX_NEW_TOKENS = 128
        
        # Generation settings
        self.GENERATION_CONFIG = {
            'max_new_tokens': 128,
            'do_sample': False,
            'temperature': 0.7,
            'top_p': 0.9,
            'pad_token_id': None  # Will be set from tokenizer
        }
        
        # SAM2.1 settings
        self.SAM_CONFIG = {
            'multimask_output': True,
            'return_logits': True,
            'use_stability_score': True,
            'stability_score_thresh': 0.9
        }
        
        # Image processing settings
        self.IMAGE_SIZE = 1024  # SAM2.1 default
        self.VISION_TOKEN_COUNT = 32  # Should match num_queries for Q-Former
        
        # Training optimization settings
        self.OPTIMIZER_CONFIG = {
            'type': 'AdamW',
            'lr': 2e-5,
            'weight_decay': 0.01,
            'betas': (0.9, 0.999),
            'eps': 1e-8
        }
        
        # Loss function settings
        self.LOSS_CONFIG = {
            'segmentation_loss_weight': 1.0,
            'text_loss_weight': 1.0,
            'use_focal_loss': True,
            'focal_alpha': 0.25,
            'focal_gamma': 2.0
        }
        
        # Memory optimization settings
        self.MEMORY_CONFIG = {
            'gradient_checkpointing': True,
            'mixed_precision': True,
            'max_memory_gb': 24  # For 24GB GPU
        }
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration dictionary"""
        return {
            'model_name': self.QWEN_MODEL_ID,
            'sam_model_name': self.SAM_MODEL_ID,
            'use_qformer': self.USE_QFORMER,
            'num_queries': self.NUM_QUERIES,
            'torch_dtype': self.TORCH_DTYPE,
            'device_map': self.DEVICE_MAP,
            'trust_remote_code': self.TRUST_REMOTE_CODE
        }
    
    def get_visual_projector_config(self) -> Dict[str, Any]:
        """Get visual projector configuration"""
        return {
            'image_dim': 256,  # SAM2.1 default embedding dimension
            'hidden_dim': 2048,  # Qwen2.5-VL-3B hidden dimension (approximate)
            'num_queries': self.NUM_QUERIES,
            'use_qformer': self.USE_QFORMER,
            'enhanced': self.ENHANCED_PROJECTOR,
            'enable_position_encoding': self.ENABLE_POSITION_ENCODING
        }
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration"""
        return {
            'learning_rate': self.LEARNING_RATE,
            'batch_size': self.BATCH_SIZE,
            'max_length': self.MAX_LENGTH,
            'optimizer_config': self.OPTIMIZER_CONFIG,
            'loss_config': self.LOSS_CONFIG,
            'memory_config': self.MEMORY_CONFIG
        }
    
    def update_generation_config(self, tokenizer):
        """Update generation config with tokenizer-specific settings"""
        self.GENERATION_CONFIG['pad_token_id'] = tokenizer.eos_token_id
        return self.GENERATION_CONFIG


# Default configuration instance
config = SAMQwenConfig()

# Quick access to commonly used settings
QWEN_MODEL_ID = config.QWEN_MODEL_ID
SAM_MODEL_ID = config.SAM_MODEL_ID
TORCH_DTYPE = config.TORCH_DTYPE
DEVICE_MAP = config.DEVICE_MAP
USE_QFORMER = config.USE_QFORMER
NUM_QUERIES = config.NUM_QUERIES
BATCH_SIZE = config.BATCH_SIZE
LEARNING_RATE = config.LEARNING_RATE

# Environment-specific configurations
class DevelopmentConfig(SAMQwenConfig):
    """Development environment configuration"""
    
    def __init__(self):
        super().__init__()
        self.BATCH_SIZE = 1
        self.MAX_NEW_TOKENS = 64
        self.MEMORY_CONFIG['max_memory_gb'] = 12  # For development GPUs
        self.NUM_QUERIES = 16  # Reduce for faster development


class ProductionConfig(SAMQwenConfig):
    """Production environment configuration"""
    
    def __init__(self):
        super().__init__()
        self.BATCH_SIZE = 4
        self.MAX_NEW_TOKENS = 256
        self.MEMORY_CONFIG['max_memory_gb'] = 80  # For high-end GPUs
        self.NUM_QUERIES = 64  # More queries for better performance


class TestConfig(SAMQwenConfig):
    """Test environment configuration"""
    
    def __init__(self):
        super().__init__()
        self.BATCH_SIZE = 1
        self.MAX_NEW_TOKENS = 32
        self.NUM_QUERIES = 8  # Minimal for testing
        self.USE_QFORMER = False  # Use simpler linear projection for testing
        self.ENHANCED_PROJECTOR = False


# Factory function to get config by environment
def get_config(env: str = 'development') -> SAMQwenConfig:
    """
    Get configuration based on environment.
    
    Args:
        env: Environment name ('development', 'production', 'test')
        
    Returns:
        Configuration instance
    """
    configs = {
        'development': DevelopmentConfig,
        'production': ProductionConfig,
        'test': TestConfig,
        'default': SAMQwenConfig
    }
    
    config_class = configs.get(env, SAMQwenConfig)
    return config_class()


# Hyperparameter search spaces (for optimization)
HYPERPARAMETER_SEARCH_SPACE = {
    'learning_rate': [1e-5, 2e-5, 5e-5, 1e-4],
    'num_queries': [16, 32, 64, 128],
    'batch_size': [1, 2, 4, 8],
    'segmentation_loss_weight': [0.5, 1.0, 2.0],
    'text_loss_weight': [0.5, 1.0, 2.0],
    'focal_alpha': [0.25, 0.5, 0.75],
    'focal_gamma': [1.0, 2.0, 3.0]
}

# Model size configurations
MODEL_SIZE_CONFIGS = {
    'small': {
        'qwen_model': "Qwen/Qwen2.5-VL-3B-Instruct",
        'sam_model': "facebook/sam2.1-hiera-base-plus",
        'num_queries': 16,
        'hidden_dim': 2048
    },
    'medium': {
        'qwen_model': "Qwen/Qwen2.5-VL-7B-Instruct",
        'sam_model': "facebook/sam2.1-hiera-large",
        'num_queries': 32,
        'hidden_dim': 3072
    },
    'large': {
        'qwen_model': "Qwen/Qwen2.5-VL-72B-Instruct",
        'sam_model': "facebook/sam2.1-hiera-large",
        'num_queries': 64,
        'hidden_dim': 8192
    }
}


def get_model_size_config(size: str) -> Dict[str, Any]:
    """
    Get model configuration based on size.
    
    Args:
        size: Model size ('small', 'medium', 'large')
        
    Returns:
        Model size configuration
    """
    return MODEL_SIZE_CONFIGS.get(size, MODEL_SIZE_CONFIGS['small'])


# Validation functions
def validate_config(config: SAMQwenConfig) -> bool:
    """
    Validate configuration settings.
    
    Args:
        config: Configuration to validate
        
    Returns:
        True if valid, raises ValueError if invalid
    """
    # Check required attributes
    required_attrs = [
        'QWEN_MODEL_ID', 'SAM_MODEL_ID', 'TORCH_DTYPE',
        'USE_QFORMER', 'NUM_QUERIES', 'BATCH_SIZE'
    ]
    
    for attr in required_attrs:
        if not hasattr(config, attr):
            raise ValueError(f"Missing required config attribute: {attr}")
    
    # Validate ranges
    if config.NUM_QUERIES <= 0:
        raise ValueError("NUM_QUERIES must be positive")
    
    if config.BATCH_SIZE <= 0:
        raise ValueError("BATCH_SIZE must be positive")
    
    if config.LEARNING_RATE <= 0:
        raise ValueError("LEARNING_RATE must be positive")
    
    return True


# Export commonly used configurations
__all__ = [
    'SAMQwenConfig',
    'DevelopmentConfig',
    'ProductionConfig', 
    'TestConfig',
    'config',
    'get_config',
    'get_model_size_config',
    'validate_config',
    'QWEN_MODEL_ID',
    'SAM_MODEL_ID',
    'TORCH_DTYPE',
    'USE_QFORMER',
    'NUM_QUERIES',
    'BATCH_SIZE',
    'LEARNING_RATE'
]