"""
Visual Feature Projector for SAM2.1 + Qwen2.5-VL Integration
o3_spec.mdの仕様に基づいたQ-FormerベースおよびLinearベースの視覚特徴プロジェクター
"""
import torch
import torch.nn as nn
from typing import Optional


class VisualProjector(nn.Module):
    """
    Projects image features to the LLM hidden dimension.
    Supports two modes:
    - Q-Former mode: Uses learnable query embeddings and cross-attention to distill image features
    - Linear mode: Uses simple linear projection with pooling to produce fixed visual tokens
    """
    
    def __init__(self, image_dim: int, hidden_dim: int, num_queries: int = 32, use_qformer: bool = True):
        """
        Args:
            image_dim: SAM2.1 image embedding dimension (typically 256)
            hidden_dim: Qwen2.5-VL hidden dimension (e.g., 2048 for 3B model)
            num_queries: Number of query tokens for Q-Former mode
            use_qformer: Whether to use Q-Former (True) or Linear projection (False)
        """
        super().__init__()
        self.use_qformer = use_qformer
        self.num_queries = num_queries if use_qformer else 4  # 4 tokens for linear mode
        self.hidden_dim = hidden_dim
        self.image_dim = image_dim
        
        if use_qformer:
            # Q-Former implementation
            self._init_qformer_components()
        else:
            # Linear projection implementation
            self._init_linear_components()
    
    def _init_qformer_components(self):
        """Initialize Q-Former components"""
        # Learnable queries for cross-attention
        self.query_embed = nn.Parameter(torch.randn(self.num_queries, self.hidden_dim))
        
        # Project image feature dimension to hidden_dim if needed
        self.feat_proj = nn.Linear(self.image_dim, self.hidden_dim) if self.image_dim != self.hidden_dim else nn.Identity()
        
        # Cross-attention layer (multi-head attention) and feed-forward network
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_dim, 
            num_heads=8, 
            batch_first=True,
            dropout=0.1
        )
        self.attn_norm = nn.LayerNorm(self.hidden_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, 4 * self.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(4 * self.hidden_dim, self.hidden_dim)
        )
        self.ffn_norm = nn.LayerNorm(self.hidden_dim)
        
        # Initialize parameters
        self._init_weights()
    
    def _init_linear_components(self):
        """Initialize Linear projection components"""
        # Linear projection mode: use average pooling to one vector, then expand to a few tokens
        self.feat_proj = nn.Linear(self.image_dim, self.hidden_dim) if self.image_dim != self.hidden_dim else nn.Identity()
        self.num_queries = 4  # Fixed to 4 tokens for linear mode
        
        # Additional projection layer for better representation
        self.linear_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        # Initialize parameters
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        
        # Special initialization for query embeddings in Q-Former mode
        if self.use_qformer:
            nn.init.normal_(self.query_embed, std=0.02)
    
    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the visual projector.
        
        Args:
            image_features: Tensor of shape [B, N, image_dim] 
                          (N = number of image feature patches/tokens from SAM2.1)
        
        Returns:
            visual_embeds: Tensor of shape [B, num_queries, hidden_dim] 
                          Visual token embeddings for LLM input
        """
        B, N, D = image_features.shape
        
        if self.use_qformer:
            return self._forward_qformer(image_features, B, N, D)
        else:
            return self._forward_linear(image_features, B, N, D)
    
    def _forward_qformer(self, image_features: torch.Tensor, B: int, N: int, D: int) -> torch.Tensor:
        """Q-Former forward pass"""
        # Project features to hidden dimension
        feats = self.feat_proj(image_features)  # [B, N, hidden_dim]
        
        # Initialize query tokens (expand to batch size)
        queries = self.query_embed.unsqueeze(0).expand(B, -1, -1)  # [B, num_queries, hidden_dim]
        
        # Cross-attention: queries attend to image features
        attn_output, _ = self.cross_attn(query=queries, key=feats, value=feats)  # [B, num_queries, hidden_dim]
        
        # Add & normalize (residual connection)
        queries = self.attn_norm(queries + attn_output)
        
        # Feed-forward network on queries
        ffn_output = self.ffn(queries)  # [B, num_queries, hidden_dim]
        queries = self.ffn_norm(queries + ffn_output)
        
        return queries  # [B, num_queries, hidden_dim]
    
    def _forward_linear(self, image_features: torch.Tensor, B: int, N: int, D: int) -> torch.Tensor:
        """Linear projection forward pass"""
        # Pool across the N dimension (features) -> shape [B, D]
        pooled = image_features.mean(dim=1)  # [B, image_dim]
        
        # Project to hidden dimension
        proj = self.feat_proj(pooled)  # [B, hidden_dim]
        
        # Apply additional projection
        proj = self.linear_proj(proj)  # [B, hidden_dim]
        
        # Repeat the projected vector to form a small set of visual tokens
        visual_tokens = proj.unsqueeze(1).expand(B, self.num_queries, -1)  # [B, num_queries, hidden_dim]
        
        return visual_tokens


class EnhancedVisualProjector(VisualProjector):
    """
    Enhanced version with additional features for better performance
    """
    
    def __init__(self, image_dim: int, hidden_dim: int, num_queries: int = 32, 
                 use_qformer: bool = True, enable_position_encoding: bool = True):
        """
        Args:
            enable_position_encoding: Whether to add positional encoding to image features
        """
        super().__init__(image_dim, hidden_dim, num_queries, use_qformer)
        self.enable_position_encoding = enable_position_encoding
        
        if enable_position_encoding:
            # Learnable positional encoding for image patches
            max_patches = 1024  # Maximum number of patches (32x32 for typical SAM2.1 output)
            self.pos_embedding = nn.Parameter(torch.randn(1, max_patches, self.hidden_dim if use_qformer else image_dim))
    
    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        """Enhanced forward with optional positional encoding"""
        B, N, D = image_features.shape
        
        # Add positional encoding if enabled
        if self.enable_position_encoding:
            if hasattr(self, 'pos_embedding'):
                pos_embed = self.pos_embedding[:, :N, :]  # [1, N, dim]
                if pos_embed.size(-1) != D:
                    # Project positional embedding to match feature dimension
                    if not hasattr(self, 'pos_proj'):
                        self.pos_proj = nn.Linear(pos_embed.size(-1), D).to(pos_embed.device)
                    pos_embed = self.pos_proj(pos_embed)
                image_features = image_features + pos_embed  # [B, N, D]
        
        # Call parent forward method
        return super().forward(image_features)


def create_visual_projector(config: dict) -> VisualProjector:
    """
    Factory function to create visual projector based on configuration
    
    Args:
        config: Configuration dictionary containing:
            - image_dim: SAM2.1 image embedding dimension
            - hidden_dim: Qwen2.5-VL hidden dimension
            - num_queries: Number of query tokens
            - use_qformer: Whether to use Q-Former
            - enhanced: Whether to use enhanced version
            - enable_position_encoding: Whether to use positional encoding (enhanced only)
    
    Returns:
        VisualProjector instance
    """
    if config.get('enhanced', False):
        return EnhancedVisualProjector(
            image_dim=config['image_dim'],
            hidden_dim=config['hidden_dim'],
            num_queries=config.get('num_queries', 32),
            use_qformer=config.get('use_qformer', True),
            enable_position_encoding=config.get('enable_position_encoding', True)
        )
    else:
        return VisualProjector(
            image_dim=config['image_dim'],
            hidden_dim=config['hidden_dim'],
            num_queries=config.get('num_queries', 32),
            use_qformer=config.get('use_qformer', True)
        )