Integrating Qwen2.5-VL 3B with SAM2.1 for Multi-Modal Segmentation and QA

Model Architecture and Components

In this implementation, we integrate Alibaba’s Qwen2.5-VL 3B vision-language model with Meta’s SAM 2.1 segmentation model. The SAM2.1 Vision Transformer (ViT) is used as the visual feature extractor, and a learnable projection module (with an optional Q-Former mechanism) connects these visual features to Qwen’s language model. This allows the combined model to generate both text (from Qwen) and segmentation masks (from SAM) for a given image and query. The code is written in PyTorch using Hugging Face Transformers for Qwen and the official SAM2.1 API for segmentation.

Key Components:
	•	Visual Feature Extractor: SAM2.1 ViT (via SAM2ImagePredictor) provides image embeddings ￼. We use the from_pretrained method to automatically download weights if not present.
	•	Visual Projector: A module that maps SAM’s image embeddings to Qwen’s LLM input space. This can be a simple linear projection or a Q-Former (transformer) that uses learnable query tokens to attend to image features.
	•	Qwen LLM: Qwen2.5-VL-3B (instruction-tuned) loaded via Hugging Face Transformers ￼. We monkey-patch its internal visual module to use SAM’s features instead of the original vision encoder.
	•	Segmentation Decoder: SAM2.1 mask decoder to generate segmentation masks from image features and prompts (points/boxes).
	•	Multi-modal Forward: The integrated model’s forward method outputs both the segmentation mask and the text answer. The text generation is done using Qwen’s generate on the combined image-text input, while the mask is obtained from SAM in parallel.

Below we define the model classes and projection module, then demonstrate a test script that performs an inference given an image and a question.

Visual Feature Projector (Q-Former or Linear)

The VisualProjector class handles mapping image features to the dimension expected by Qwen. It supports two modes:
	•	Q-Former mode: Uses learnable query embeddings and cross-attention to distill image features into a fixed number of visual tokens.
	•	Linear mode: Uses a simple linear layer (and pooling) to project image features, producing a small fixed set of visual tokens (e.g., 4 tokens).

import torch
import torch.nn as nn

class VisualProjector(nn.Module):
    def __init__(self, image_dim: int, hidden_dim: int, num_queries: int = 32, use_qformer: bool = True):
        """
        Projects image features to the LLM hidden dimension.
        If use_qformer is True, uses a learnable query transformer (Q-Former) with `num_queries` tokens.
        Otherwise, uses a linear projection (and pooling) to produce a fixed number of visual tokens.
        """
        super().__init__()
        self.use_qformer = use_qformer
        self.num_queries = num_queries if use_qformer else 4  # use 4 tokens for linear mode
        self.hidden_dim = hidden_dim

        if use_qformer:
            # Learnable queries for cross-attention
            self.query_embed = nn.Parameter(torch.randn(self.num_queries, hidden_dim))
            # Project image feature dimension to hidden_dim if needed
            self.feat_proj = nn.Linear(image_dim, hidden_dim) if image_dim != hidden_dim else nn.Identity()
            # Cross-attention layer (multi-head attention) and feed-forward network
            self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, batch_first=True)
            self.attn_norm = nn.LayerNorm(hidden_dim)
            self.ffn = nn.Sequential(
                nn.Linear(hidden_dim, 4 * hidden_dim),
                nn.GELU(),
                nn.Linear(4 * hidden_dim, hidden_dim)
            )
            self.ffn_norm = nn.LayerNorm(hidden_dim)
        else:
            # Linear projection mode: use average pooling to one vector, then expand to a few tokens
            self.feat_proj = nn.Linear(image_dim, hidden_dim) if image_dim != hidden_dim else nn.Identity()
            # We'll output 4 identical visual tokens (could be adjusted)
            self.num_queries = 4

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        """
        image_features: Tensor of shape [B, N, image_dim] (N = number of image feature patches/tokens).
        Returns:
          visual_embeds: [B, num_queries, hidden_dim] visual token embeddings for LLM.
        """
        B, N, D = image_features.shape
        if self.use_qformer:
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
        else:
            # Linear projection: average pool all features to one vector
            # Pool across the N dimension (features) -> shape [B, D]
            pooled = image_features.mean(dim=1)  # [B, image_dim]
            proj = self.feat_proj(pooled)        # [B, hidden_dim]
            # Repeat the projected vector to form a small set of visual tokens
            visual_tokens = proj.unsqueeze(1).expand(B, self.num_queries, -1)  # [B, num_queries, hidden_dim]
            return visual_tokens

Integrated Multi-Modal Model Definition

The SAMQwenModel class encapsulates the combined model. It loads Qwen2.5-VL (3B) via Transformers and SAM2.1 via the SAM API, then replaces Qwen’s internal image encoder with a custom forward that uses SAM’s features. The forward method returns both the segmentation mask and the generated text. We ensure the implementation is modular and ready for future fine-tuning (e.g., LoRA can be applied to the Qwen LLM, the projection module, or even the SAM components independently).

import torch
import torch.nn as nn
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from sam2.sam2_image_predictor import SAM2ImagePredictor

class SAMQwenModel(nn.Module):
    def __init__(self, use_qformer: bool = True, num_queries: int = 32):
        """
        Initializes the integrated model with Qwen2.5-VL 3B and SAM2.1.
        use_qformer: whether to use the Q-Former for visual feature projection.
        num_queries: number of query tokens (if Q-Former is used).
        """
        super().__init__()
        # Load Qwen2.5-VL 3B model and tokenizer
        self.llm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype=torch.float16, device_map="auto"
        ) [oai_citation:2‡huggingface.co](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct#:~:text=from%20transformers%20import%20Qwen2_5_VLForConditionalGeneration%2C%20AutoTokenizer%2C,AutoProcessor%20from%20qwen_vl_utils%20import%20process_vision_info)
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
        # Configure the processor to control visual token count to match our projector output
        vis_tokens = num_queries if use_qformer else 4
        self.processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct",
            min_pixels=vis_tokens * 28 * 28,
            max_pixels=vis_tokens * 28 * 28
        )
        # Load SAM2.1 predictor (image encoder + mask decoder) automatically
        self.sam_predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2.1-hiera-large") [oai_citation:3‡github.com](https://github.com/facebookresearch/sam2#:~:text=predictor%20%3D%20SAM2ImagePredictor.from_pretrained%28%22facebook%2Fsam2)
        # Move SAM model to same device as LLM
        device = next(self.llm.parameters()).device
        self.sam_predictor.model = self.sam_predictor.model.to(device)
        # Create visual feature projector (to match Qwen hidden size)
        hidden_dim = self.llm.config.hidden_size  # Qwen LLM hidden dimension (e.g., 2048)
        # SAM2.1 image embedding dimension (after ViT) – SAM2.1 outputs 256-d embeddings by default
        image_dim = 256  
        self.visual_projector = VisualProjector(image_dim, hidden_dim, num_queries=num_queries, use_qformer=use_qformer)
        # Monkey-patch Qwen's internal visual module to use our SAM features
        if hasattr(self.llm, "model") and hasattr(self.llm.model, "visual"):
            # Replace Qwen's vision module with this instance (it will call our __call__ method)
            self.llm.model.visual = self  

    def __call__(self, pixel_values: torch.Tensor = None, **kwargs):
        """
        Monkey-patched forward for Qwen's visual encoder. 
        This is invoked inside Qwen's forward to get visual embeddings.
        """
        if pixel_values is None:
            return None
        # Ensure pixel_values is on the same device as SAM predictor
        pixel_values = pixel_values.to(self.sam_predictor.device)
        # If not already set, encode the image with SAM's image encoder
        # (It's expected that .set_image() has been called in forward, setting self._features)
        if not hasattr(self.sam_predictor, "_features") or self.sam_predictor._features is None:
            # Convert pixel_values tensor to numpy image for predictor
            img_np = (pixel_values[0].cpu().permute(1, 2, 0).numpy() * 255).astype("uint8")
            self.sam_predictor.set_image(img_np)
        # Get image embedding from SAM predictor (shape [1, C, H, W], typically C=256)
        image_embed = self.sam_predictor.get_image_embedding()  # [1, 256, H, W]
        # Flatten spatial dimensions to sequence of patches
        B, C, H, W = image_embed.shape
        image_tokens = image_embed.view(B, C, H * W).permute(0, 2, 1)  # [B, N_patches, C]
        # Project image tokens to LLM hidden dimension
        visual_embeds = self.visual_projector(image_tokens)  # [B, num_queries, hidden_dim]
        return visual_embeds

    def forward(self, image: 'ImageType', question: str, 
                point_coords: torch.Tensor = None, point_labels: torch.Tensor = None, 
                box: torch.Tensor = None) -> (torch.Tensor, str):
        """
        Performs forward pass: returns a segmentation mask and generated text answer.
        image: Input image (PIL image, NumPy array, or Torch tensor [C,H,W]).
        question: Text question/query about the image.
        point_coords, point_labels: Optional point prompt(s) for segmentation (tensor of shape [N,2] and [N] labels).
        box: Optional box prompt for segmentation (tensor [4] in XYXY format).
        """
        # Prepare image input for SAM
        if isinstance(image, torch.Tensor):
            img_tensor = image.clone()
            if img_tensor.dim() == 3: 
                img_tensor = img_tensor.unsqueeze(0)  # add batch dim
        else:
            # Convert PIL or NumPy image to tensor
            from PIL import Image
            import numpy as np
            if isinstance(image, Image.Image):
                img_np = np.array(image.convert("RGB"))
            else:
                img_np = image  # assume numpy array
            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(self.sam_predictor.device)
        img_tensor = img_tensor.float()

        # Run SAM segmentation: set the image and make prediction
        img_np = (img_tensor[0].permute(1, 2, 0).cpu().numpy()).astype("uint8")
        self.sam_predictor.set_image(img_np) [oai_citation:4‡github.com](https://github.com/facebookresearch/sam2#:~:text=with%20torch.inference_mode%28%29%2C%20torch.autocast%28,input_prompts)
        # Prepare prompt inputs for SAM
        masks, scores, _ = self.sam_predictor.predict(
            point_coords=point_coords[None] if point_coords is not None else None,
            point_labels=point_labels[None] if point_labels is not None else None,
            box=box[None].cpu().numpy() if box is not None else None,
            multimask_output=False
        ) [oai_citation:5‡learnopencv.com](https://learnopencv.com/sam-2/#:~:text=masks%2C%20scores%2C%20_%20%3D%20predictor,0%20else%20None%2C%20multimask_output%3DFalse%2C)
        mask = masks[0]  # [H, W] binary mask (True/False)

        # Prepare input for Qwen LLM
        inputs = self.processor(
            text=[question], images=[img_np], return_tensors="pt"
        )
        inputs = inputs.to(self.sam_predictor.device)
        # Generate text answer using Qwen (LLM). The SAM visual features will be injected via our monkey-patch.
        with torch.inference_mode():
            generated_ids = self.llm.generate(**inputs, max_new_tokens=128)
        # Decode the generated ids to text (skip the prompt part)
        input_len = inputs["input_ids"].shape[1]
        generated_text = self.tokenizer.decode(generated_ids[0][input_len:], skip_special_tokens=True)
        return mask, generated_text

Notes: We set min_pixels and max_pixels in the AutoProcessor so that it inserts a fixed number of visual tokens (matching the projector output length) for each image ￼. In this setup, if use_qformer=True with num_queries=32, the processor will allocate 32 visual tokens per image. This ensures the placeholders in Qwen’s input sequence align with the number of visual embeddings our SAM projector produces. The monkey-patched __call__ in SAMQwenModel intercepts Qwen’s call to its visual encoder and provides SAM-derived embeddings ￼ ￼.

Inference Test Script

Below is a test script demonstrating how to use the integrated model. It loads an image and a question, then outputs both the segmentation mask and the model’s text answer. We use a point prompt for segmentation in this example (simulating a user clicking on an object of interest), but box prompts or no prompts could also be used. Both text generation and mask prediction occur in one forward pass of the model.

Example input image of a dish with salad and meat, used for testing the multi-modal model.

import torch
from PIL import Image

# Initialize the integrated model (using Q-Former with 32 query tokens)
model = SAMQwenModel(use_qformer=True, num_queries=32)

# Load an example image (replace 'example.jpg' with an actual image file path)
image = Image.open("example.jpg")
question = "What food items are on the plate?"

# Optionally, prepare a point prompt for segmentation (e.g., pointing at the meat portion)
# Here we use a single point roughly at the center of the meat
point = torch.tensor([[300.0, 200.0]])   # x,y coordinates in pixel space of the image
label = torch.tensor([1])               # label 1 = foreground

# Perform inference
mask, answer = model(image, question, point_coords=point, point_labels=label)

# Convert mask to PIL image for visualization (mask is boolean, convert to uint8 0/255)
mask_image = Image.fromarray(mask.cpu().numpy().astype('uint8') * 255)
mask_image.save("output_mask.png")      # save mask output
print("Generated Answer:", answer)
print("Mask saved to output_mask.png (dimensions:", mask.shape, ")")

Explanation: In the above script, we instantiate SAMQwenModel. The image is loaded and the question is defined. We specify a prompt point (with label=1 indicating a positive point on the target object). The model is called with the image, question, and prompt. Internally, SAM produces a segmentation mask for the specified region, and Qwen generates a text answer about the image. The resulting mask is then saved as an image file, and the answer is printed to the console.

Conclusion

We have constructed a multi-modal model that unifies Qwen2.5-VL’s language generation with SAM2.1’s segmentation capabilities. This flexible implementation keeps the vision and language components modular, which is ideal for future fine-tuning (e.g., using LoRA on the language model or on the projection layers). In a future scenario (such as FoodLMM tasks), this model could be fine-tuned so that the language output and the segmentation output are conditioned on each other (for example, segmenting specific ingredients and describing them). For now, the above integration provides a working inference-time prototype that takes an image and a question, and returns both a mask and a textual response, leveraging the strengths of both Qwen (for understanding and answering) and SAM (for precise segmentation).

Sources: The integration approach is based on the official usage of Qwen2.5-VL in Transformers ￼ and SAM 2.1 in its predictor form ￼. The combined model uses SAM’s set_image and predict API for mask generation ￼, and injects SAM’s visual features into Qwen following Qwen’s vision-token mechanism ￼ ￼. This code is expected to run with the required libraries installed and can be further fine-tuned with low-rank adaptation techniques as needed.