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
        # Pre-load housekeeping to reduce memory spikes and fragmentation
        import gc, os
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

        load_kwargs = {"device_map": "auto", "trust_remote_code": True, "low_cpu_mem_usage": True}

        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name,
            **load_kwargs,
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


# Removed GPTOSSInternVLRouterLLM routing class: text-only pipeline will directly use GPT-OSS via HuggingFaceLLM


class Qwen3GGUFEmbedding(BaseEmbedding):
    """Wrapper to load a Qwen3 GGUF embedding model via llama.cpp or gguf loader.

    This class exposes the same interface used in the repo: _get_query_embedding/_get_text_embedding
    and a generic _embed() method that returns list[list[float]]. It prefers llama_cpp Llama.embed when
    available and falls back to a lightweight subprocess-based llama.cpp call if needed.
    """
    model_name: str = Field(default="Qwen/Qwen3-Embedding-0.6B-GGUF")
    _llama = PrivateAttr(default=None)

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # Prefer llama_cpp Llama if available
        try:
            from llama_cpp import Llama
            # For GGUF local file usage one would pass model_path, but HF hub download is left to user
            # We'll initialize lazily in _embed to keep constructor cheap
            self._llama = None
        except Exception:
            self._llama = None

    @classmethod
    def class_name(cls) -> str:
        return "qwen3_gguf"

    def _ensure_llama(self):
        if self._llama is None:
            try:
                from huggingface_hub import hf_hub_download
                # Attempt to download the GGUF file from HF repo if present
                repo_id = self.model_name
                # try common gguf filename
                gguf_file = None
                from huggingface_hub import list_repo_files
                files = list_repo_files(repo_id)
                for f in files:
                    if f.lower().endswith('.gguf') or f.lower().endswith('.bin'):
                        gguf_file = f
                        break
                if gguf_file:
                    path = hf_hub_download(repo_id=repo_id, filename=gguf_file)
                    from llama_cpp import Llama
                    self._llama = Llama(model_path=path)
            except Exception:
                self._llama = None

    def _embed(self, texts: List[str], image_paths: Optional[List[Optional[str]]] = None, **kwargs) -> List[List[float]]:
        self._ensure_llama()
        if self._llama is not None:
            # llama_cpp exposes embed
            flat = []
            for t in texts:
                try:
                    res = self._llama.embed(t)
                    if isinstance(res, dict) and 'data' in res:
                        flat.append(list(res['data'][0].get('embedding', [])))
                    elif isinstance(res, list):
                        flat.append(list(res[0]))
                    else:
                        flat.append([0.0])
                except Exception:
                    flat.append([0.0])
            return flat
        else:
            # Last resort: return zero vectors matching a small dimension
            return [[0.0] * 384 for _ in texts]

    def _get_query_embedding(self, query: str, image_path: Optional[str] = None) -> List[float]:
        return self._embed([query])[0]

    def _get_text_embedding(self, text: str, image_path: Optional[str] = None) -> List[float]:
        return self._embed([text])[0]


class Gemma3CustomLLM(CustomLLM):
    """Dedicated Gemma-3 (27B IT) multimodal LLM loader using 4-bit BitsAndBytes quantization.

    Implements the same multimodal chat-style interface as QwenVLCustomLLM but follows
    the Gemma 3 example: use AutoProcessor.apply_chat_template and Gemma3ForConditionalGeneration.
    Loads weights with BitsAndBytesConfig (4-bit) when CUDA is available and device_map="auto".
    """
    model_name: str = Field(default="google/gemma-3-27b-it")
    context_window: int = Field(default=131072)
    num_output: int = Field(default=512)
    _model = PrivateAttr()
    _processor = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Prepare a BitsAndBytesConfig for 4-bit quantized load when possible
        try:
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        except Exception:
            bnb_cfg = None

        try:
            # Pre-load housekeeping: collect GC and free CUDA cache to reduce OOM risk
            import gc, os
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

            from transformers import Gemma3ForConditionalGeneration

            load_kwargs = {"device_map": "auto", "trust_remote_code": True}
            if bnb_cfg is not None and torch.cuda.is_available():
                load_kwargs["quantization_config"] = bnb_cfg

            # Load model (prefer 4-bit quantized when possible)
            self._model = Gemma3ForConditionalGeneration.from_pretrained(
                self.model_name,
                **load_kwargs,
                low_cpu_mem_usage=True,
            ).eval()

            # Processor for multimodal inputs
            self._processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)

        except Exception as e:
            # Surface the error for debugging in environments without HF weights
            print(f"Failed to initialize Gemma3CustomLLM: {e}")
            raise

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=self.model_name,
        )

    @llm_completion_callback()
    def complete(self, prompt: str, image_paths: Optional[List[str]] = None, **kwargs: Any) -> CompletionResponse:
        # Build chat-style messages: system + user content (images then text)
        messages = [{"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]}]
        user_content = []
        if image_paths:
            for path in image_paths:
                user_content.append({"type": "image", "image": path})
        user_content.append({"type": "text", "text": prompt})
        messages.append({"role": "user", "content": user_content})

        # Use processor.apply_chat_template per Gemma doc
        inputs = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._model.device, dtype=torch.bfloat16)

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = self._model.generate(**inputs, max_new_tokens=self.num_output, do_sample=False)
            # remove the input prefix
            generation = generation[0][input_len:]

        decoded = self._processor.decode(generation, skip_special_tokens=True)
        return CompletionResponse(text=decoded)

    @llm_completion_callback()
    def stream_complete(self, prompt: str, image_paths: Optional[List[str]] = None, **kwargs: Any):
        resp = self.complete(prompt, image_paths=image_paths, **kwargs)
        # Yield tokens/characters as delta to match previous streaming behavior
        for ch in resp.text:
            yield CompletionResponse(text=ch, delta=ch)


# If BaseEmbedding is from LlamaIndex or your own base, import it accordingly.
# from llama_index.core.embeddings.base import BaseEmbedding
 
# Lightweight GGUF / llama.cpp wrapper for Qwen2.5 Coder 3B (GGUF)
class QwenCoderGGUFLLM(CustomLLM):
    """Wrapper that loads a Qwen2.5-Coder-3B Instruct GGUF via llama_cpp.

    It downloads the HF model snapshot, finds the first .gguf file, and instantiates
    llama_cpp.Llama. This avoids a large torch-based HF load and runs via the
    local GGUF runtime (llama.cpp or llama_cpp bindings).
    """
    model_name: str = Field(default="Qwen/Qwen2.5-Coder-3B-Instruct-GGUF")
    context_window: int = Field(default=8192)
    num_output: int = Field(default=256)
    _llm = PrivateAttr(default=None)

    def __init__(self, model_name: Optional[str] = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if model_name:
            self.model_name = model_name

        try:
            # Local imports to avoid import-time dependency on llama_cpp when not used.
            import gc
            import glob
            from huggingface_hub import snapshot_download

            # Keep housekeeping to reduce fragmentation
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

            # Download model repo snapshot and locate GGUF file
            repo_dir = snapshot_download(repo_id=self.model_name, repo_type="model")
            gguf_candidates = glob.glob(f"{repo_dir}/*.gguf")
            if not gguf_candidates:
                gguf_candidates = glob.glob(f"{repo_dir}/**/*.gguf", recursive=True)
            if not gguf_candidates:
                raise RuntimeError(f"No .gguf file found for {self.model_name} in {repo_dir}")

            model_path = gguf_candidates[0]

            # Instantiate llama_cpp Llama
            from llama_cpp import Llama
            self._llm = Llama(model_path=model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize QwenCoderGGUFLLM: {e}")

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=self.model_name,
        )

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        if self._llm is None:
            raise RuntimeError("llama_cpp model not initialized")
        try:
            resp = self._llm.create(prompt=prompt, max_tokens=self.num_output)
            text = ""
            if isinstance(resp, dict) and resp.get("choices"):
                choice = resp["choices"][0]
                text = choice.get("text") or choice.get("message", {}).get("content", "")
            else:
                # fallback attribute access
                choices = getattr(resp, "choices", None)
                if choices and isinstance(choices, list):
                    text = choices[0].get("text", "") if isinstance(choices[0], dict) else str(choices[0])
            return CompletionResponse(text=text or "")
        except Exception as e:
            raise

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        resp = self.complete(prompt, **kwargs)
        for token in resp.text:
            yield CompletionResponse(text=token, delta=token)

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
            # Pre-load housekeeping to reduce CPU/GPU memory fragmentation
            import gc, os
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

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
                low_cpu_mem_usage=True,
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
            from transformers import AutoModel, BitsAndBytesConfig
            import torch

            # Determine target device. Prefer cuda:1 when multiple GPUs exist,
            # otherwise fall back to cuda:0. If no CUDA, use CPU.
            if self.device == "auto":
                if torch.cuda.is_available():
                    try:
                        dev_count = torch.cuda.device_count()
                    except Exception:
                        dev_count = 1
                    if dev_count > 1:
                        target_dev = "cuda:1"
                    else:
                        target_dev = "cuda:0"
                else:
                    target_dev = "cpu"
            else:
                target_dev = self.device

            device_map = {"": target_dev} if target_dev != "cpu" else "cpu"

            # Only attempt BitsAndBytes 4-bit quantized load when using CUDA device(s).
            if target_dev != "cpu" and torch.cuda.is_available():
                # Prefer 4-bit quantized load to reduce GPU memory usage (align with embeddings)
                try:
                    bnb_cfg = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_compute_dtype=torch.bfloat16,
                    )
                    self._model = AutoModel.from_pretrained(
                        self.model_name,
                        trust_remote_code=True,
                        quantization_config=bnb_cfg,
                        device_map=device_map,
                    )
                except Exception as e:
                    # If 4-bit loading fails, fall back to regular load on the same device_map.
                    print(f"4-bit load failed for reranker: {e}. Falling back to non-4bit load.")
                    self._model = AutoModel.from_pretrained(
                        self.model_name,
                        torch_dtype="auto",
                        trust_remote_code=True,
                        device_map=device_map,
                    )
            else:
                # For CPU targets, prefer a GGUF (llama.cpp) runtime if available
                # so we avoid heavy PyTorch CPU model loading. Try HF GGUF repo then llama_cpp.
                from huggingface_hub import list_repo_files, hf_hub_download
                # Try to find a gguf file in the companion repo (common naming convention)
                repo_id = f"{self.model_name}-GGUF" if not self.model_name.endswith("-GGUF") else self.model_name
                files = list_repo_files(repo_id)
                gguf_candidates = [f for f in files if f.lower().endswith('.gguf') or f.lower().endswith('.bin')]
                if gguf_candidates:
                    gguf_file = gguf_candidates[0]
                    gguf_path = hf_hub_download(repo_id=repo_id, filename=gguf_file)
                    try:
                        from llama_cpp import Llama

                        llm = Llama(model_path=gguf_path)

                        class _GGUFFallbackWrapper:
                            """Robust wrapper exposing compute_score(pairs, ...) using embeddings from llama_cpp.

                            Features:
                            - Attempts a single batched embedding call where supported.
                            - Normalizes multiple possible embedding return shapes.
                            - Validates embedding dimensions and returns 0.0 for invalid pairs.
                            """
                            def __init__(self, llm_obj):
                                self._llm = llm_obj
                                # Mirror a minimal PyTorch model surface so callers can .to()/.eval()
                                self.device = 'cpu'

                            def to(self, device):
                                """No-op device move for GGUF wrapper. Records device string."""
                                try:
                                    # normalize device string
                                    if isinstance(device, str):
                                        self.device = device
                                    else:
                                        self.device = str(device)
                                except Exception:
                                    self.device = 'cpu'
                                return self

                            def eval(self):
                                """No-op eval (keeps parity with torch models)."""
                                return self

                            def _call_embed(self, texts: List[str]) -> List[List[float]]:
                                """Return a list of embedding vectors for the input texts.

                                Tries a batched API first, falls back to per-text calls.
                                Normalizes possible return formats.
                                """
                                embeddings: List[List[float]] = []

                                # Try batched call first
                                try:
                                    # llama_cpp Llama.embed accepts a single string or a list
                                    res = self._llm.embed(texts)
                                    # Normalize response shapes
                                    # Possible forms: {'data': [{'embedding': [...]}, ...]} or list of embeddings
                                    if isinstance(res, dict) and 'data' in res:
                                        for item in res['data']:
                                            if isinstance(item, dict) and 'embedding' in item:
                                                embeddings.append(list(item['embedding']))
                                            else:
                                                # Unexpected item -> try to coerce
                                                try:
                                                    embeddings.append(list(item))
                                                except Exception:
                                                    embeddings.append([0.0])
                                    elif isinstance(res, list):
                                        # Assume list of embeddings or list of dicts
                                        for item in res:
                                            if isinstance(item, dict) and 'embedding' in item:
                                                embeddings.append(list(item['embedding']))
                                            else:
                                                try:
                                                    embeddings.append(list(item))
                                                except Exception:
                                                    embeddings.append([0.0])
                                    else:
                                        # Single embedding returned for whole batch
                                        try:
                                            embeddings = [list(res)] * len(texts)
                                        except Exception:
                                            embeddings = [[0.0] for _ in texts]
                                    # If result length mismatches, fall back to per-text
                                    if len(embeddings) != len(texts):
                                        raise RuntimeError("Batch embed length mismatch")
                                    return embeddings
                                except Exception:
                                    # Fall back to per-text embedding calls
                                    embeddings = []
                                    for t in texts:
                                        try:
                                            r = self._llm.embed(t)
                                            if isinstance(r, dict) and 'data' in r:
                                                emb = r['data'][0].get('embedding')
                                                embeddings.append(list(emb) if emb is not None else [0.0])
                                            elif isinstance(r, list):
                                                embeddings.append(list(r[0]) if r and isinstance(r[0], (list, tuple)) else list(r))
                                            else:
                                                embeddings.append(list(r) if hasattr(r, '__iter__') else [0.0])
                                        except Exception:
                                            embeddings.append([0.0])
                                    return embeddings

                            def compute_score(self, pairs, max_length=1024, doc_type="text"):
                                # pairs: list of [query, doc_text]
                                if not pairs:
                                    return []

                                # Build flattened list for embeddings requests: [q0, d0, q1, d1, ...]
                                texts: List[str] = []
                                for q, d in pairs:
                                    texts.append(q if isinstance(q, str) else str(q))
                                    texts.append(d if isinstance(d, str) else str(d))

                                embeddings = self._call_embed(texts)

                                # Compute cosine similarity for each pair with validation
                                scores: List[float] = []
                                import math
                                for i in range(0, len(embeddings), 2):
                                    try:
                                        qv = embeddings[i]
                                        dv = embeddings[i + 1]
                                    except Exception:
                                        scores.append(0.0)
                                        continue

                                    # Validate numeric vectors
                                    if not qv or not dv or len(qv) != len(dv):
                                        scores.append(0.0)
                                        continue

                                    # Compute dot and norms robustly
                                    try:
                                        dot = 0.0
                                        nq = 0.0
                                        nd = 0.0
                                        for a, b in zip(qv, dv):
                                            fa = float(a)
                                            fb = float(b)
                                            dot += fa * fb
                                            nq += fa * fa
                                            nd += fb * fb
                                        if nq <= 0 or nd <= 0:
                                            scores.append(0.0)
                                            continue
                                        scores.append(dot / (math.sqrt(nq) * math.sqrt(nd) + 1e-12))
                                    except Exception:
                                        scores.append(0.0)

                                return scores
                        self._model = _GGUFFallbackWrapper(llm)
                        print(f"Loaded GGUF reranker via llama_cpp from {repo_id} using file {gguf_file}")
                    except Exception as e:
                        print(f"llama_cpp GGUF init failed: {e}. Falling back to PyTorch CPU model.")
                        # Fall through to PyTorch CPU load below
                        raise
                else:
                    # No gguf file found -- raise to trigger the PyTorch fallback
                    raise RuntimeError("No GGUF candidate found in HF repo")
            # If device_map is 'cpu', ensure model is on CPU. Otherwise device_map already placed weights.
            if device_map == 'cpu':
                self._model.to('cpu')

            self._model.eval()
            print(f"Loaded Jina reranker model: {self.model_name} on {target_dev}")
            
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
