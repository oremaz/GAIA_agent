from typing import Optional, List, Any
from pydantic import Field, PrivateAttr
from llama_index.core.llms import CustomLLM, CompletionResponse, CompletionResponseGen, LLMMetadata
from llama_index.core.llms.callbacks import llm_completion_callback
from transformers import Qwen2_5_VLForConditionalGeneration
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from typing import Any, List, Optional
from llama_index.core.embeddings import BaseEmbedding
from transformers import AutoModel, BitsAndBytesConfig

class QwenVLCustomLLM(CustomLLM):
    model_name: str = Field(default="Qwen/Qwen2.5-VL-32B-Instruct-AWQ")
    context_window: int = Field(default=32768)
    num_output: int = Field(default=256)
    _model = PrivateAttr()
    _processor = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name,
            device_map="auto",
            trust_remote_code=True,
        )
        self._processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=self.model_name,
        )

    @llm_completion_callback()
    def complete(
        self,
        prompt: str,
        image_paths: Optional[List[str]] = None,
        **kwargs: Any
    ) -> CompletionResponse:
        # Prepare multimodal input
        messages = [{"role": "user", "content": []}]
        if image_paths:
            for path in image_paths:
                messages[0]["content"].append({"type": "image", "image": path})
        messages[0]["content"].append({"type": "text", "text": prompt})

        # Tokenize and process
        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self._model.device)

        # Generate output
        generated_ids = self._model.generate(**inputs, max_new_tokens=self.num_output)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = self._processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return CompletionResponse(text=output_text)

    @llm_completion_callback()
    def stream_complete(
        self,
        prompt: str,
        image_paths: Optional[List[str]] = None,
        **kwargs: Any
    ) -> CompletionResponseGen:
        response = self.complete(prompt, image_paths)
        for token in response.text:
            yield CompletionResponse(text=token, delta=token)


class GPTOSSInternVLRouterLLM(CustomLLM):
    """
    Routes text-only prompts to GPT-OSS-20B and multimodal (image+text) prompts to InternVL3-8B.
    GPT-OSS uses its native MXFP4 precision (no BitsAndBytes), InternVL loads in 4-bit.
    """
    gpt_model_name: str = Field(default="openai/gpt-oss-20b")
    vlm_model_name: str = Field(default="Qwen/Qwen2.5-VL-7B-Instruct-AWQ")
    context_window: int = Field(default=32768)
    num_output: int = Field(default=512)

    _gpt_model = PrivateAttr()
    _gpt_tokenizer = PrivateAttr()
    _vlm = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        from transformers import AutoModelForCausalLM, AutoTokenizer
        # Load GPT-OSS-20B (text LLM) with native MXFP4 (no BitsAndBytes)
        self._gpt_model = AutoModelForCausalLM.from_pretrained(
            self.gpt_model_name,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
        self._gpt_tokenizer = AutoTokenizer.from_pretrained(
            self.gpt_model_name, use_fast=True, trust_remote_code=True
        )
        # Load Qwen2.5-VL-7B-Instruct-AWQ as VLM
        self._vlm = QwenVLCustomLLM(model_name=self.vlm_model_name)

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=f"router:{self.gpt_model_name}|{self.vlm_model_name}",
        )

    def _build_vlm_pixel_values(self, image_paths: List[str]):
        """Minimal image preprocessing to 448x448 with ImageNet normalization.
        Returns a torch.Tensor shaped [N, 3, 448, 448].
        """
        import torch
        from PIL import Image
        try:
            import torchvision.transforms as T
            from torchvision.transforms.functional import InterpolationMode
        except Exception as e:
            raise RuntimeError("torchvision is required for InternVL3 image preprocessing") from e

        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)
        transform = T.Compose([
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        images = []
        for p in image_paths:
            img = Image.open(p)
            images.append(transform(img))
        pixel_values = torch.stack(images)  # [N, 3, 448, 448]
        # InternVL expects bf16 where possible
        if torch.cuda.is_available():
            pixel_values = pixel_values.to(torch.bfloat16).cuda()
        return pixel_values

    def _gpt_respond(self, prompt: str) -> str:
        import torch
        tok = self._gpt_tokenizer
        # Use chat template if available (Harmony format)
        try:
            messages = [{"role": "user", "content": prompt}]
            text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            text = prompt

        inputs = tok(text, return_tensors="pt").to(self._gpt_model.device)
        with torch.no_grad():
            out = self._gpt_model.generate(
                **inputs,
                max_new_tokens=self.num_output,
                do_sample=False,
                temperature=0.0,
                pad_token_id=tok.eos_token_id,
                eos_token_id=tok.eos_token_id,
            )
        gen = out[0, inputs["input_ids"].shape[1]:]
        return tok.decode(gen, skip_special_tokens=True)

    def _vlm_respond(self, prompt: str, image_paths: List[str]) -> str:
        # Use QwenVLCustomLLM for multimodal completion
        return self._vlm.complete(prompt, image_paths=image_paths).text

    @llm_completion_callback()
    def complete(
        self,
        prompt: str,
        image_paths: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> CompletionResponse:
        image_paths = image_paths or []
        if len(image_paths) > 0:
            text = self._vlm_respond(prompt, image_paths)
        else:
            text = self._gpt_respond(prompt)
        return CompletionResponse(text=text)

    @llm_completion_callback()
    def stream_complete(
        self,
        prompt: str,
        image_paths: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> CompletionResponseGen:
        resp = self.complete(prompt, image_paths, **kwargs)
        for ch in resp.text:
            yield CompletionResponse(text=ch, delta=ch)

    

# If BaseEmbedding is from LlamaIndex or your own base, import it accordingly.
# from llama_index.core.embeddings.base import BaseEmbedding

class JinaEmbeddingsV4(BaseEmbedding):
    """
    Multimodal embedding wrapper for jinaai/jina-embeddings-v4.
    Supports:
      - Text-only: model.encode_text(..., task=..., prompt_name=...)
      - Image-only: model.encode_image(..., task=...)
      - Text+Image: model.encode(text=..., image=..., task=...)
    Pass task/prompt_name at encode time per the official docs.
    """
    model_name: str = Field(default="jinaai/jina-embeddings-v4")

    # Keep a handle to the HF model (initialized in __init__)
    _model = PrivateAttr(default=None)

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # Load the model once. Prefer float16 for stability first.
        # To try 4-bit later, see the commented block below.
        import torch
        from transformers import AutoModel

        if self._model is None:
            #self._model = AutoModel.from_pretrained(
                #self.model_name,
                #trust_remote_code=True,
                #torch_dtype=torch.float16,   # stable default
                #device_map="auto",
            #).eval()

            # If you prefer explicit device placement (instead of device_map="auto"):
            # self._model.to("cuda" if torch.cuda.is_available() else "cpu")

            # To enable 4-bit (only after ensuring peft/transformers/bitsandbytes/CUDA versions are aligned):
            self._bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

            device_map = {"": 0} if torch.cuda.is_available() else "cpu"

            self._model = AutoModel.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                quantization_config=self._bnb_config,
                device_map=device_map,
            ).eval()

    @classmethod
    def class_name(cls) -> str:
        return "jina_v4"

    def _embed(
        self,
        texts: List[str],
        image_paths: Optional[List[Optional[str]]] = None,
        task: str = "retrieval",
        prompt_name: Optional[str] = "passage",  # "query" for queries, "passage" for docs; not used for images
        return_multivector: bool = False,
        truncate_dim: Optional[int] = None,      # e.g., 128, 256, 512, 1024, 2048
        max_length: Optional[int] = None,        # for texts
        max_pixels: Optional[int] = None,        # for images
        batch_size: Optional[int] = None,
    ) -> List[List[float]]:
        """
        Encodes a batch of inputs; each item can be text-only, image-only, or text+image.
        - task: "retrieval" | "text-matching" | "code"
        - prompt_name: for retrieval/code text inputs ("query" or "passage"); not used for image-only
        - return_multivector: if True, returns multi-vector outputs (here flattened per item)
        """
        import torch
        from PIL import Image

        model = self._model
        assert model is not None, "Model is not initialized."

        # Normalize image_paths to align with texts
        image_paths = image_paths or [None] * len(texts)
        if len(image_paths) == 1 and len(texts) > 1:
            image_paths = image_paths * len(texts)

        results: List[List[float]] = []

        # Helper to coerce model outputs to a single vector (first item if batched)
        def _to_list_vector(arr):
            # Robust conversion to a plain Python list[float].
            try:
                import torch
            except Exception:
                torch = None
            try:
                import numpy as np
            except Exception:
                np = None

            # If it's a torch Tensor, convert to numpy first
            if torch is not None and isinstance(arr, torch.Tensor):
                arr = arr.detach().cpu().numpy()

            # If it's a numpy array, reduce batch dims and return list
            if np is not None and isinstance(arr, np.ndarray):
                # Handle possible shapes: (N, D), (N, M, D), (D,)
                if arr.ndim == 3:
                    # average across the middle dimension (num_vecs) then take first batch
                    arr = arr.mean(axis=1)
                if arr.ndim == 2:
                    arr = arr[0]
                return [float(x) for x in arr.tolist()]

            # If it's a list, try to coerce the first item's vector
            if isinstance(arr, list):
                if len(arr) == 0:
                    return []
                first = arr[0]
                # If first element is tensor/ndarray/list, recurse on it
                if (torch is not None and isinstance(first, torch.Tensor)) or (
                    np is not None and isinstance(first, np.ndarray)
                ) or isinstance(first, (list, tuple)):
                    return _to_list_vector(first)
                # Otherwise assume it's already a flat list of numbers
                return [float(x) for x in arr]

            # Scalars and other types -> coerce to float
            try:
                return [float(arr)]
            except Exception:
                # Last resort: stringify then cast
                return [float(str(arr))]

        with torch.no_grad():
            for text, img_path in zip(texts, image_paths):
                # Determine path: multimodal, image-only, or text-only
                if img_path is not None and text:
                    # Multimodal
                    img = Image.open(img_path).convert("RGB")
                    emb = model.encode(
                        text=text,
                        image=img,
                        task=task,
                        return_multivector=return_multivector,
                        truncate_dim=truncate_dim,
                        max_length=max_length,
                        max_pixels=max_pixels,
                        batch_size=batch_size,
                    )
                    # For multimodal, remote code returns either single or multivector embedding
                    # Flatten to a single vector if not returning multivector explicitly
                    if return_multivector:
                        # If multivector, you may want to keep the nested structure.
                        # Here we average to return a single vector for compatibility.
                        if hasattr(emb, "detach"):
                            emb = emb.detach().cpu().numpy()
                        elif hasattr(emb, "cpu"):
                            emb = emb.cpu().numpy()
                        if emb.ndim == 3:
                            # shape (1, num_vecs, dim) -> avg across num_vecs
                            emb = emb.mean(axis=1)[0]
                        else:
                            emb = emb[0] if emb.ndim == 2 else emb
                        results.append(emb.tolist())
                    else:
                        results.append(_to_list_vector(emb))

                elif img_path is not None:
                    # Image-only
                    img = Image.open(img_path).convert("RGB")
                    emb = model.encode_image(
                        images=[img],
                        task=task,
                        return_multivector=return_multivector,
                        truncate_dim=truncate_dim,
                        max_pixels=max_pixels,
                        batch_size=batch_size,
                    )
                    results.append(_to_list_vector(emb))

                else:
                    # Text-only
                    kwargs = {
                        "texts": [text],
                        "task": task,
                        "return_multivector": return_multivector,
                        "truncate_dim": truncate_dim,
                        "max_length": max_length,
                        "batch_size": batch_size,
                    }
                    # Only text tasks use prompt_name (retrieval/code); text-matching is symmetric
                    if prompt_name:
                        kwargs["prompt_name"] = prompt_name

                    emb = model.encode_text(**kwargs)
                    results.append(_to_list_vector(emb))

        return results

    def _get_query_embedding(self, query: str, image_path: Optional[str] = None) -> List[float]:
        # Retrieval query uses prompt_name="query"
        return self._embed(
            [query],
            [image_path] if image_path else None,
            task="retrieval",
            prompt_name="query" if image_path is None else None,
        )[0]

    def _get_text_embedding(self, text: str, image_path: Optional[str] = None) -> List[float]:
        # Retrieval passage (for indexing) uses prompt_name="passage"
        return self._embed(
            [text],
            [image_path] if image_path else None,
            task="retrieval",
            prompt_name="passage" if image_path is None else None,
        )[0]

    def _get_text_embeddings(
        self,
        texts: List[str],
        image_paths: Optional[List[Optional[str]]] = None,
    ) -> List[List[float]]:
        # Default to retrieval passages for document embeddings
        return self._embed(
            texts,
            image_paths,
            task="retrieval",
            prompt_name="passage",
        )

    async def _aget_query_embedding(self, query: str, image_path: Optional[str] = None) -> List[float]:
        return self._get_query_embedding(query, image_path)

    async def _aget_text_embedding(self, text: str, image_path: Optional[str] = None) -> List[float]:
        return self._get_text_embedding(text, image_path)


class JinaMultimodalReranker:
    """
    Custom Jina multimodal reranker using jinaai/jina-reranker-m0.
    Supports text-to-text, text-to-image, image-to-text, and image-to-image reranking.
    """
    
    def __init__(self, model_name: str = "jinaai/jina-reranker-m0", top_n: int = 5, device: str = "auto"):
        self.model_name = model_name
        self.top_n = top_n
        self.device = device
        self._model = None
        self._load_model()
    
    def _load_model(self):
        """Load the Jina reranker model"""
        try:
            from transformers import AutoModel
            import torch
            
            # Load with flash attention if available and compatible GPU
            try:
                self._model = AutoModel.from_pretrained(
                    self.model_name,
                    torch_dtype="auto",
                    trust_remote_code=True,
                    attn_implementation="flash_attention_2"
                )
            except Exception:
                # Fallback without flash attention
                self._model = AutoModel.from_pretrained(
                    self.model_name,
                    torch_dtype="auto",
                    trust_remote_code=True
                )
            
            # Move to device
            if self.device == "auto":
                if torch.cuda.is_available():
                    self._model.to('cuda')
                else:
                    self._model.to('cpu')
            else:
                self._model.to(self.device)
            
            self._model.eval()
            print(f"Loaded Jina reranker model: {self.model_name}")
            
        except Exception as e:
            print(f"Error loading Jina reranker model: {e}")
            raise
    
    def postprocess_nodes(self, nodes, query_bundle):
        """
        Rerank nodes using Jina multimodal reranker.
        Automatically detects content types and uses appropriate reranking method.
        """
        if not nodes:
            return []
        
        query = query_bundle.query_str
        
        # Prepare query-document pairs for reranking
        text_pairs = []
        image_pairs = []
        node_indices = []
        
        for i, node in enumerate(nodes):
            # Check if node contains image content
            has_image = self._node_has_image(node)
            
            if has_image:
                # Get image path/URL from node
                image_path = self._extract_image_path(node)
                if image_path:
                    image_pairs.append([query, image_path])
                    node_indices.append(('image', i))
            else:
                # Text content
                text_content = node.get_content()
                text_pairs.append([query, text_content])
                node_indices.append(('text', i))
        
        # Compute scores
        all_scores = []
        
        try:
            # Score text pairs
            if text_pairs:
                text_scores = self._model.compute_score(
                    text_pairs, 
                    max_length=1024, 
                    doc_type="text"
                )
                all_scores.extend([(score, 'text', idx) for score, (_, idx) in zip(text_scores, [x for x in node_indices if x[0] == 'text'])])
            
            # Score image pairs  
            if image_pairs:
                image_scores = self._model.compute_score(
                    image_pairs, 
                    max_length=2048, 
                    doc_type="image"
                )
                all_scores.extend([(score, 'image', idx) for score, (_, idx) in zip(image_scores, [x for x in node_indices if x[0] == 'image'])])
        
        except Exception as e:
            print(f"Error during reranking: {e}")
            # Return original nodes if reranking fails
            return nodes[:self.top_n]
        
        # Sort by score (descending)
        all_scores.sort(key=lambda x: x[0], reverse=True)
        
        # Return top N reranked nodes
        reranked_nodes = []
        for score, content_type, node_idx in all_scores[:self.top_n]:
            if node_idx < len(nodes):
                node = nodes[node_idx]
                # Update node score
                if hasattr(node, 'score'):
                    node.score = score
                reranked_nodes.append(node)
        
        return reranked_nodes
    
    def _node_has_image(self, node) -> bool:
        """Check if a node contains image content"""
        # Check metadata for image indicators
        metadata = getattr(node, 'metadata', {})
        
        # Check file type
        file_type = metadata.get('file_type', '').lower()
        if file_type in ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp', 'pdf']:
            return True
        
        # Check content type
        content_type = metadata.get('type', '').lower()
        if content_type in ['image', 'web_image']:
            return True
        
        # Check if node has image_path attribute
        if hasattr(node, 'image_path') and node.image_path:
            return True
        
        # Check if metadata contains image_data
        if 'image_data' in metadata:
            return True
        
        # Check source path for image extensions
        source = metadata.get('source', '').lower()
        if any(ext in source for ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']):
            return True
        
        return False
    
    def _extract_image_path(self, node) -> Optional[str]:
        """Extract image path from node"""
        # Try image_path attribute first
        if hasattr(node, 'image_path') and node.image_path:
            return node.image_path
        
        # Try metadata
        metadata = getattr(node, 'metadata', {})
        
        # Check for direct path
        if 'path' in metadata:
            return metadata['path']
        
        # Check source if it's an image
        source = metadata.get('source', '')
        if source and any(ext in source.lower() for ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']):
            return source
        
        # For PDF or other documents with images, we might need special handling
        # For now, return None if we can't find a direct image path
        return None
