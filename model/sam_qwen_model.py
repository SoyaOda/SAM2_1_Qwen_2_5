"""
SAM2.1 + Qwen2.5-VL Integrated Multi-Modal Model
o3_spec.mdの仕様に基づいた統合モデル実装
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Optional, Dict, Any, List, Tuple
import numpy as np
from PIL import Image
import warnings

# Transformers imports
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor

# SAM2.1 imports  
try:
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    SAM2_AVAILABLE = True
except ImportError:
    warnings.warn("SAM2.1 not available. Please install sam2 package.")
    SAM2_AVAILABLE = False

# Local imports
from .visual_projector import VisualProjector, create_visual_projector

# Type aliases
ImageType = Union[torch.Tensor, np.ndarray, Image.Image]


class SAMQwenModel(nn.Module):
    """
    Integrated model combining Qwen2.5-VL 3B with SAM2.1 for multi-modal segmentation and QA.
    
    Key features:
    - Uses SAM2.1 as visual feature extractor
    - Integrates with Qwen2.5-VL via visual projector (Q-Former or Linear)
    - Monkey-patches Qwen's visual encoder
    - Supports both text generation and segmentation in single forward pass
    """
    
    def __init__(self, 
                 model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
                 sam_model_name: str = "facebook/sam2.1-hiera-large",
                 use_qformer: bool = True,
                 num_queries: int = 32,
                 torch_dtype: torch.dtype = torch.float16,
                 device_map: str = "auto",
                 trust_remote_code: bool = True):
        """
        Initialize the integrated SAM-Qwen model.
        
        Args:
            model_name: Qwen2.5-VL model name/path
            sam_model_name: SAM2.1 model name/path  
            use_qformer: Whether to use Q-Former for visual projection
            num_queries: Number of query tokens (if Q-Former is used)
            torch_dtype: Model dtype
            device_map: Device mapping strategy
            trust_remote_code: Whether to trust remote code
        """
        super().__init__()
        
        if not SAM2_AVAILABLE:
            raise ImportError("SAM2.1 is required but not available. Please install sam2 package.")
        
        self.model_name = model_name
        self.sam_model_name = sam_model_name
        self.use_qformer = use_qformer
        self.num_queries = num_queries
        self.torch_dtype = torch_dtype
        
        # Initialize components
        self._init_qwen_components(device_map, trust_remote_code)
        self._init_sam_components()
        self._init_visual_projector()
        self._setup_monkey_patch()
        
        print(f"✅ SAMQwenModel initialized:")
        print(f"  - Qwen model: {model_name}")
        print(f"  - SAM model: {sam_model_name}")
        print(f"  - Visual projector: {'Q-Former' if use_qformer else 'Linear'} ({num_queries} queries)")
    
    def _init_qwen_components(self, device_map: str, trust_remote_code: bool):
        """Initialize Qwen2.5-VL components"""
        print(f"🧠 Loading Qwen2.5-VL model: {self.model_name}")
        
        # Load Qwen2.5-VL model
        self.llm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=self.torch_dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            low_cpu_mem_usage=True
        )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=trust_remote_code
        )
        
        # Configure processor to control visual token count
        vis_tokens = self.num_queries if self.use_qformer else 4
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            min_pixels=vis_tokens * 28 * 28,
            max_pixels=vis_tokens * 28 * 28,
            trust_remote_code=trust_remote_code
        )
        
        print(f"✅ Qwen2.5-VL loaded with {vis_tokens} visual tokens")
    
    def _init_sam_components(self):
        """Initialize SAM2.1 components"""
        print(f"🎯 Loading SAM2.1 model: {self.sam_model_name}")
        
        # Load SAM2.1 predictor
        self.sam_predictor = SAM2ImagePredictor.from_pretrained(self.sam_model_name)
        
        # Move SAM model to same device as LLM
        device = next(self.llm.parameters()).device
        self.sam_predictor.model = self.sam_predictor.model.to(device)
        
        print(f"✅ SAM2.1 loaded and moved to device: {device}")
    
    def _init_visual_projector(self):
        """Initialize visual feature projector"""
        print(f"🔗 Initializing visual projector...")
        
        # Get dimensions
        hidden_dim = self.llm.config.hidden_size  # Qwen LLM hidden dimension
        image_dim = 256  # SAM2.1 default embedding dimension
        
        # Configure projector
        projector_config = {
            'image_dim': image_dim,
            'hidden_dim': hidden_dim,
            'num_queries': self.num_queries,
            'use_qformer': self.use_qformer,
            'enhanced': True,  # Use enhanced version with positional encoding
            'enable_position_encoding': True
        }
        
        self.visual_projector = create_visual_projector(projector_config)
        
        # Move projector to same device as LLM
        device = next(self.llm.parameters()).device
        self.visual_projector = self.visual_projector.to(device)
        
        print(f"✅ Visual projector initialized: {image_dim} -> {hidden_dim}")
        print(f"  - Mode: {'Q-Former' if self.use_qformer else 'Linear'}")
        print(f"  - Queries: {self.num_queries}")
    
    def _setup_monkey_patch(self):
        """Setup monkey-patch for Qwen's visual encoder"""
        print(f"🐒 Setting up monkey-patch for Qwen visual encoder...")
        
        # Replace Qwen's vision module with this instance
        if hasattr(self.llm, "model") and hasattr(self.llm.model, "visual"):
            self.original_visual_encoder = self.llm.model.visual
            self.llm.model.visual = self
            print("✅ Monkey-patch applied to Qwen visual encoder")
        else:
            print("⚠️ Could not find Qwen visual encoder to monkey-patch")
    
    def __call__(self, pixel_values: Optional[torch.Tensor] = None, **kwargs):
        """
        Monkey-patched forward for Qwen's visual encoder.
        This is invoked inside Qwen's forward to get visual embeddings.
        
        Args:
            pixel_values: Input pixel values tensor
            **kwargs: Additional arguments
            
        Returns:
            visual_embeds: Visual embeddings for LLM input
        """
        if pixel_values is None:
            return None
        
        # Ensure pixel_values is on the same device as SAM predictor
        device = next(self.sam_predictor.model.parameters()).device
        pixel_values = pixel_values.to(device)
        
        # Check if image has been set in SAM predictor
        if not hasattr(self.sam_predictor, "_features") or self.sam_predictor._features is None:
            # Convert pixel_values tensor to numpy image for predictor
            # Assuming pixel_values is [B, C, H, W] and normalized to [0, 1]
            img_np = (pixel_values[0].cpu().permute(1, 2, 0).numpy() * 255).astype("uint8")
            self.sam_predictor.set_image(img_np)
        
        # Get image embedding from SAM predictor
        image_embed = self.sam_predictor.get_image_embedding()  # [1, C, H, W]
        
        # Flatten spatial dimensions to sequence of patches
        B, C, H, W = image_embed.shape
        image_tokens = image_embed.view(B, C, H * W).permute(0, 2, 1)  # [B, N_patches, C]
        
        # Project image tokens to LLM hidden dimension
        visual_embeds = self.visual_projector(image_tokens)  # [B, num_queries, hidden_dim]
        
        return visual_embeds
    
    def forward(self, 
                image: ImageType,
                question: str,
                point_coords: Optional[torch.Tensor] = None,
                point_labels: Optional[torch.Tensor] = None,
                box: Optional[torch.Tensor] = None,
                return_text_only: bool = False,
                return_mask_only: bool = False,
                max_new_tokens: int = 128) -> Dict[str, Any]:
        """
        Unified forward pass returning both segmentation mask and generated text.
        
        Args:
            image: Input image (PIL, NumPy array, or Torch tensor)
            question: Text question/query about the image
            point_coords: Optional point prompts for segmentation [N, 2]
            point_labels: Optional point labels [N] (1=foreground, 0=background)
            box: Optional box prompt [4] in XYXY format
            return_text_only: If True, only return text generation
            return_mask_only: If True, only return segmentation mask
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Dictionary containing:
                - mask: Segmentation mask [H, W] 
                - generated_text: Generated text response
                - iou_scores: IoU scores from SAM2.1
                - visual_features: Visual features from SAM2.1
                - text_logits: Text generation logits (optional)
        """
        results = {}
        
        # Prepare image input
        img_tensor, img_np = self._prepare_image_input(image)
        
        # Generate segmentation mask (if requested)
        if not return_text_only:
            mask_results = self._generate_mask(img_np, point_coords, point_labels, box)
            results.update(mask_results)
        
        # Generate text response (if requested)
        if not return_mask_only:
            text_results = self._generate_text(img_np, question, max_new_tokens)
            results.update(text_results)
        
        # Get visual features for analysis
        if hasattr(self.sam_predictor, "_features") and self.sam_predictor._features is not None:
            results['visual_features'] = self.sam_predictor.get_image_embedding()
        
        return results
    
    def _prepare_image_input(self, image: ImageType) -> Tuple[torch.Tensor, np.ndarray]:
        """Prepare image input in both tensor and numpy formats"""
        device = next(self.sam_predictor.model.parameters()).device
        
        if isinstance(image, torch.Tensor):
            img_tensor = image.clone()
            if img_tensor.dim() == 3:
                img_tensor = img_tensor.unsqueeze(0)  # Add batch dim
            # Convert to numpy for SAM
            img_np = (img_tensor[0].permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
        elif isinstance(image, Image.Image):
            img_np = np.array(image.convert("RGB"))
            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        else:
            # Assume numpy array
            img_np = image.astype("uint8")
            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        
        img_tensor = img_tensor.to(device)
        return img_tensor, img_np
    
    def _generate_mask(self, 
                      img_np: np.ndarray,
                      point_coords: Optional[torch.Tensor] = None,
                      point_labels: Optional[torch.Tensor] = None,
                      box: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """Generate segmentation mask using SAM2.1"""
        # Set image in SAM predictor
        self.sam_predictor.set_image(img_np)
        
        # Prepare prompts
        point_coords_np = point_coords.cpu().numpy() if point_coords is not None else None
        point_labels_np = point_labels.cpu().numpy() if point_labels is not None else None
        box_np = box.cpu().numpy() if box is not None else None
        
        # Predict masks
        masks, scores, _ = self.sam_predictor.predict(
            point_coords=point_coords_np[None] if point_coords_np is not None else None,
            point_labels=point_labels_np[None] if point_labels_np is not None else None,
            box=box_np[None] if box_np is not None else None,
            multimask_output=True  # Get multiple masks with confidence scores
        )
        
        # Return best mask (highest IoU score)
        best_mask_idx = np.argmax(scores)
        best_mask = masks[best_mask_idx]
        
        return {
            'mask': torch.from_numpy(best_mask).bool(),
            'all_masks': torch.from_numpy(masks),
            'iou_scores': torch.from_numpy(scores),
            'best_mask_idx': best_mask_idx
        }
    
    def _generate_text(self, img_np: np.ndarray, question: str, max_new_tokens: int) -> Dict[str, Any]:
        """Generate text response using Qwen2.5-VL"""
        # Prepare input for Qwen
        inputs = self.processor(
            text=[question], 
            images=[img_np], 
            return_tensors="pt"
        )
        
        device = next(self.llm.parameters()).device
        inputs = inputs.to(device)
        
        # Generate text response
        with torch.inference_mode():
            generated_ids = self.llm.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode generated text (skip the prompt part)
        input_len = inputs["input_ids"].shape[1]
        generated_text = self.tokenizer.decode(generated_ids[0][input_len:], skip_special_tokens=True)
        
        return {
            'generated_text': generated_text,
            'generated_ids': generated_ids,
            'input_length': input_len
        }
    
    @property
    def device(self) -> torch.device:
        """Get model device"""
        return next(self.llm.parameters()).device
    
    def eval(self):
        """Set model to evaluation mode"""
        super().eval()
        self.llm.eval()
        self.sam_predictor.model.eval()
        return self
    
    def train(self, mode: bool = True):
        """Set model to training mode"""
        super().train(mode)
        if hasattr(self, 'llm'):
            self.llm.train(mode)
        if hasattr(self, 'sam_predictor'):
            self.sam_predictor.model.train(mode)
        return self


def create_sam_qwen_model(config: Dict[str, Any]) -> SAMQwenModel:
    """
    Factory function to create SAMQwenModel with configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        SAMQwenModel instance
    """
    return SAMQwenModel(
        model_name=config.get('model_name', "Qwen/Qwen2.5-VL-3B-Instruct"),
        sam_model_name=config.get('sam_model_name', "facebook/sam2.1-hiera-large"),
        use_qformer=config.get('use_qformer', True),
        num_queries=config.get('num_queries', 32),
        torch_dtype=getattr(torch, config.get('torch_dtype', 'float16')),
        device_map=config.get('device_map', "auto"),
        trust_remote_code=config.get('trust_remote_code', True)
    )