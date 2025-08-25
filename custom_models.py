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
import threading
import logging

# Lock to prevent races when creating cached instances concurrently
_CACHE_LOCK = threading.Lock()
_logger = logging.getLogger(__name__)

# Module-level caches to avoid creating multiple heavyweight model instances
_EMBEDDER_CACHE = {}
_RERANKER_CACHE = {}
_LLM_CACHE = {}

def _truncate_on_stop(text: str, stop: Optional[List[str]]) -> str:
    if not stop:
        return text
    idxs = [text.find(s) for s in stop if s and s in text]
    if not idxs:
        return text
    cut = min(i for i in idxs if i >= 0)
    return text[:cut]

def get_or_create_jina_embedder(model_name: Optional[str] = None, device: Optional[str] = None):
    """Return a cached JinaEmbeddingsV4 instance or create one.

    Keyed by (model_name, device). If model_name is None the default from
    the class will be used.
    """
    key = (model_name or "jinaai/jina-embeddings-v4", device or "auto")
    # Fast path
    inst = _EMBEDDER_CACHE.get(key)
    if inst is not None:
        return inst

    # Ensure only one creator runs at a time
    with _CACHE_LOCK:
        inst = _EMBEDDER_CACHE.get(key)
        if inst is not None:
            return inst

        _logger.info("Creating Jina embedder for key=%s", key)
        try:
            before_alloc = None
            try:
                if torch.cuda.is_available():
                    before_alloc = torch.cuda.memory_allocated()
            except Exception:
                before_alloc = None

            inst = JinaEmbeddingsV4(model_name=key[0])

            after_alloc = None
            try:
                if torch.cuda.is_available():
                    after_alloc = torch.cuda.memory_allocated()
            except Exception:
                after_alloc = None

            _logger.info("Jina embedder created for key=%s (mem_before=%s, mem_after=%s)", key, before_alloc, after_alloc)
        except Exception:
            _logger.exception("Failed to create Jina embedder for key=%s", key)
            raise

        _EMBEDDER_CACHE[key] = inst
        return inst


def get_or_create_jina_reranker(model_name: Optional[str] = None, top_n: int = 5, device: str = "cpu"):
    """Return a cached JinaMultimodalReranker instance or create one.

    Keyed by (model_name, top_n, device).
    """
    key = (model_name or "jinaai/jina-reranker-m0", top_n, device)
    inst = _RERANKER_CACHE.get(key)
    if inst is not None:
        return inst

    with _CACHE_LOCK:
        inst = _RERANKER_CACHE.get(key)
        if inst is not None:
            return inst

        _logger.info("Creating Jina reranker for key=%s", key)
        try:
            before_alloc = None
            try:
                if torch.cuda.is_available():
                    before_alloc = torch.cuda.memory_allocated()
            except Exception:
                before_alloc = None

            inst = JinaMultimodalReranker(model_name=key[0], top_n=key[1], device=key[2])

            after_alloc = None
            try:
                if torch.cuda.is_available():
                    after_alloc = torch.cuda.memory_allocated()
            except Exception:
                after_alloc = None

            _logger.info("Jina reranker created for key=%s (mem_before=%s, mem_after=%s)", key, before_alloc, after_alloc)
        except Exception:
            _logger.exception("Failed to create Jina reranker for key=%s", key)
            raise

        _RERANKER_CACHE[key] = inst
        return inst


def get_or_create_qwen_vl_llm(model_name: Optional[str] = None, device: Optional[str] = None):
    """Return cached Qwen25VLMultiModal or create one.

    Qwen25VLMultiModal is the new multimodal wrapper replacing the deprecated
    QwenVLCustomLLM. We keep the factory name for backward compatibility with
    existing import sites (e.g. agent initialization) but change the created
    class. Cached by (model_name, device).
    """
    # Avoid accessing Pydantic-model attributes at import time (AttributeError).
    # Use the same default literal as defined in the class to form the cache key.
    key = (model_name or "Qwen/Qwen2.5-VL-32B-Instruct-AWQ", device or "auto")
    inst = _LLM_CACHE.get(key)
    if inst is not None:
        return inst

    with _CACHE_LOCK:
        inst = _LLM_CACHE.get(key)
        if inst is not None:
            return inst

    _logger.info("Creating Qwen25VLMultiModal for key=%s", key)
    try:
        before_alloc = None
        try:
            if torch.cuda.is_available():
                before_alloc = torch.cuda.memory_allocated()
        except Exception:
            before_alloc = None

        # Instantiate the multimodal model wrapper (loads weights immediately)
        from_path_model_name = key[0]
        inst = Qwen25VLMultiModal(model_id=from_path_model_name, device_map=device or "auto")
        # record after-instantiation allocation (model still lazy until first call)
        after_alloc = None
        try:
            if torch.cuda.is_available():
                after_alloc = torch.cuda.memory_allocated()
        except Exception:
            after_alloc = None

        _logger.info("Qwen25VLMultiModal created for key=%s (mem_before=%s, mem_after=%s)", key, before_alloc, after_alloc)
    except Exception:
        _logger.exception("Failed to create Qwen25VLMultiModal for key=%s", key)
        raise

    _LLM_CACHE[key] = inst
    return inst


def get_or_create_qwen_coder_gguf_llm(model_name: Optional[str] = None, device: str = "cpu"):
    """Return cached QwenCoderGGUFLLM or create one."""
    # Use literal default to avoid Pydantic class attribute access at import
    key = (model_name or "Qwen/Qwen2.5-Coder-3B-Instruct-GGUF", device)
    inst = _LLM_CACHE.get(key)
    if inst is not None:
        return inst

    with _CACHE_LOCK:
        inst = _LLM_CACHE.get(key)
        if inst is not None:
            return inst

        _logger.info("Creating QwenCoderGGUFLLM for key=%s", key)
        try:
            before_alloc = None
            try:
                if torch.cuda.is_available():
                    before_alloc = torch.cuda.memory_allocated()
            except Exception:
                before_alloc = None

            inst = QwenCoderGGUFLLM(model_name=key[0])

            after_alloc = None
            try:
                if torch.cuda.is_available():
                    after_alloc = torch.cuda.memory_allocated()
            except Exception:
                after_alloc = None

            _logger.info("QwenCoderGGUFLLM created for key=%s (mem_before=%s, mem_after=%s)", key, before_alloc, after_alloc)
        except Exception:
            _logger.exception("Failed to create QwenCoderGGUFLLM for key=%s", key)
            raise

        _LLM_CACHE[key] = inst
        return inst


def get_or_create_qwen3_gguf_embedding(model_name: Optional[str] = None):
    """Return cached Qwen3GGUFEmbedding or create one."""
    # Literal default to avoid accessing Pydantic attributes at import time
    key = (model_name or "Qwen/Qwen3-Embedding-0.6B-GGUF",)
    inst = _EMBEDDER_CACHE.get(key)
    if inst is not None:
        return inst

    with _CACHE_LOCK:
        inst = _EMBEDDER_CACHE.get(key)
        if inst is not None:
            return inst

        _logger.info("Creating Qwen3GGUFEmbedding for key=%s", key)
        try:
            inst = Qwen3GGUFEmbedding(model_name=key[0])
        except Exception:
            _logger.exception("Failed to create Qwen3GGUFEmbedding for key=%s", key)
            raise

        _EMBEDDER_CACHE[key] = inst
        return inst


# Removed GPTOSSInternVLRouterLLM routing class: text-only pipeline will directly use GPT-OSS via HuggingFaceLLM

# qwen25_llamaindex_mm_agent.py

import os
from typing import List, Optional, Any, Sequence

import torch
from transformers import AutoProcessor
# Use the new model class for Qwen2.5-VL (requires latest transformers)
from transformers import Qwen2_5_VLForConditionalGeneration  # [1]

# Qwen's lightweight helper for packing images/videos into the processor inputs
from qwen_vl_utils import process_vision_info  # [17]

# LlamaIndex imports (v0.10+ API)
from llama_index.core.multi_modal_llms import MultiModalLLM  # base interface [11]
from llama_index.core.schema import ImageDocument  # pass images to agent [10]
from llama_index.core.agent.react_multimodal.step import MultimodalReActAgentWorker  # [10]

# Optional: LlamaIndex tools; here we keep it empty, but you can add QueryEngineTool etc. [10]
from llama_index.core.agent import AgentRunner  # if you prefer AgentRunner wiring [10]

MODEL_ID = "Qwen/Qwen2.5-VL-32B-Instruct-AWQ"  # AWQ quantized repository [1]

class Qwen25VLMultiModal(MultiModalLLM):
    """
    Minimal Hugging Face Qwen2.5-VL wrapper implementing the MultiModalLLM interface expected by
    LlamaIndex's multi-modal ReAct agent. [11]
    """

    def __init__(
        self,
        model_id: str = MODEL_ID,
        torch_dtype: Optional[torch.dtype] = None,
        device_map: str = "auto",
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        top_p: float = 0.95,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        use_flash_attn2: bool = False,
    ):
        """
        device_map='auto' will shard across the two Kaggle T4 GPUs automatically (Accelerate). [1]
        You can control visual token budget via min_pixels/max_pixels to save memory. [1]
        """
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

        # Build processor (optionally constrain visual token budget)
        if min_pixels is not None or max_pixels is not None:
            self.processor = AutoProcessor.from_pretrained(
                model_id,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
        else:
            self.processor = AutoProcessor.from_pretrained(model_id)  # [1]

        # Model init
        model_kwargs = {
            "torch_dtype": torch_dtype or "auto",
            "device_map": device_map,  # shards across multi-GPU automatically [1]
        }
        if use_flash_attn2:
            # Only enable on GPUs supporting FA2 (T4 generally does not). [1]
            model_kwargs["attn_implementation"] = "flash_attention_2"

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, **model_kwargs
        )  # [1]

    # Synchronous "complete" is the core used by LlamaIndex MM flows.
    def complete(
        self,
        prompt: str,
        image_documents: Optional[Sequence[ImageDocument]] = None,
        **kwargs: Any,
    ):
        """
        Execute a single-turn multimodal generation with text prompt + image(s). [11]
        """
        messages = [{"role": "user", "content": []}]

        # Pack images first (can be multiple)
        if image_documents:
            for img_doc in image_documents:
                # LlamaIndex ImageDocument exposes image_path; Qwen utils accept file:// URIs. [10][17]
                messages["content"].append(
                    {"type": "image", "image": f"file://{img_doc.image_path}"}
                )

        # Then add the text instruction
        messages["content"].append({"type": "text", "text": prompt})  # [1]

        # Apply Qwen chat template and convert images/videos into model inputs
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )  # [1]
        image_inputs, video_inputs = process_vision_info(messages)  # [17]

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )  # [1][17]

        # Put tensors on CUDA (Accelerate will map per device_map automatically)
        inputs = {k: v.to("cuda") if hasattr(v, "to") else v for k, v in inputs.items()}  # [1]

        # Generate
        gen_kwargs = dict(
            max_new_tokens=kwargs.get("max_new_tokens", self.max_new_tokens),
            do_sample=(self.temperature is not None and self.temperature > 0),
            temperature=kwargs.get("temperature", self.temperature),
            top_p=kwargs.get("top_p", self.top_p),
        )  # [1]
        generated_ids = self.model.generate(**inputs, **gen_kwargs)  # [1]

        # Trim prompt tokens and decode
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]  # [1]
        output_texts = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )  # [1]

        # Return plain string (LlamaIndex will wrap as needed in agent steps)
        return output_texts  # [10]

    # Optional: single-turn chat alias to satisfy some agent paths expecting .chat
    def chat(self, messages: List[dict], image_documents: Optional[Sequence[ImageDocument]] = None, **kwargs: Any):
        """
        Accept a list of messages in the same {role, content} format, append images if passed. [11]
        """
        # Merge any passed images into the last user turn (or create one)
        if image_documents:
            # Find last user turn
            user_idx = None
            for i in range(len(messages) - 1, -1, -1):
                if messages[i].get("role") == "user":
                    user_idx = i
                    break
            if user_idx is None:
                messages.append({"role": "user", "content": []})
                user_idx = len(messages) - 1
            if isinstance(messages[user_idx]["content"], str):
                messages[user_idx]["content"] = [{"type": "text", "text": messages[user_idx]["content"]}]
            for img_doc in image_documents:
                messages[user_idx]["content"].insert(0, {"type": "image", "image": f"file://{img_doc.image_path}"})

        # Use the same pipeline as .complete
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)  # [1]
        image_inputs, video_inputs = process_vision_info(messages)  # [17]
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
        )  # [1][17]
        inputs = {k: v.to("cuda") if hasattr(v, "to") else v for k, v in inputs.items()}  # [1]
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=(self.temperature is not None and self.temperature > 0),
            temperature=self.temperature,
            top_p=self.top_p,
        )  # [1]
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]  # [1]
        output_texts = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )  # [1]
        return output_texts  # [10]


    def _prepare_inputs_from_messages(self, messages):
        # Shared helper for streaming and non-streaming
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to("cuda") if hasattr(v, "to") else v for k, v in inputs.items()}
        return inputs

    def stream_complete(
        self,
        prompt: str,
        image_documents: Optional[Sequence[ImageDocument]] = None,
        **kwargs: Any,
    ):
        # Build Qwen-style messages
        messages = [{"role": "user", "content": []}]
        if image_documents:
            for img_doc in image_documents:
                messages["content"].append(
                    {"type": "image", "image": f"file://{img_doc.image_path}"}
                )
        messages["content"].append({"type": "text", "text": prompt})

        inputs = self._prepare_inputs_from_messages(messages)

        gen_kwargs = dict(
            max_new_tokens=kwargs.get("max_new_tokens", self.max_new_tokens),
            do_sample=(kwargs.get("temperature", self.temperature) or 0) > 0,
            temperature=kwargs.get("temperature", self.temperature),
            top_p=kwargs.get("top_p", self.top_p),
        )

        streamer = TextIteratorStreamer(
            self.processor.tokenizer,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        t = threading.Thread(
            target=self.model.generate,
            kwargs={**inputs, **gen_kwargs, "streamer": streamer},
            daemon=True,
        )
        t.start()

        for new_text in streamer:
            # Yield incremental text chunks
            yield new_text

    def stream_chat(
        self,
        messages: List[dict],
        image_documents: Optional[Sequence[ImageDocument]] = None,
        **kwargs: Any,
    ):
        # Optionally support streaming for a messages list
        # Attach images to the last user message (or create one)
        if image_documents:
            user_idx = None
            for i in range(len(messages) - 1, -1, -1):
                if messages[i].get("role") == "user":
                    user_idx = i
                    break
            if user_idx is None:
                messages.append({"role": "user", "content": []})
                user_idx = len(messages) - 1
            if isinstance(messages[user_idx]["content"], str):
                messages[user_idx]["content"] = [{"type": "text", "text": messages[user_idx]["content"]}]
            for img_doc in image_documents:
                messages[user_idx]["content"].insert(
                    0, {"type": "image", "image": f"file://{img_doc.image_path}"}
                )

        inputs = self._prepare_inputs_from_messages(messages)

        gen_kwargs = dict(
            max_new_tokens=self.max_new_tokens,
            do_sample=(self.temperature or 0) > 0,
            temperature=self.temperature,
            top_p=self.top_p,
        )

        streamer = TextIteratorStreamer(
            self.processor.tokenizer,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        t = threading.Thread(
            target=self.model.generate,
            kwargs={**inputs, **gen_kwargs, "streamer": streamer},
            daemon=True,
        )
        t.start()

        for new_text in streamer:
            yield new_text


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
                # Collect candidate gguf/bin files and prefer q4_k_m (then q4_k, q4_0, q4)
                candidates = [f for f in files if f.lower().endswith('.gguf') or f.lower().endswith('.bin')]
                if candidates:
                    pref_order = ["q4_k_m", "q4_k", "q4_0", "q4"]
                    chosen = None
                    for p in pref_order:
                        for f in candidates:
                            if p in f.lower():
                                chosen = f
                                break
                        if chosen:
                            break
                    if not chosen:
                        chosen = candidates[0]
                    path = hf_hub_download(repo_id=repo_id, filename=chosen)
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


# If BaseEmbedding is from LlamaIndex or your own base, import it accordingly.
# from llama_index.core.embeddings.base import BaseEmbedding
 
# Lightweight GGUF / llama.cpp wrapper for Qwen2.5 Coder 3B (GGUF)
class QwenCoderGGUFLLM(CustomLLM):
    model_name: str = Field(default="Qwen/Qwen2.5-Coder-3B-Instruct-GGUF")
    context_window: int = Field(default=8192)
    num_output: int = Field(default=256)

    _llm = PrivateAttr(default=None)

    def __init__(self, model_name: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        if model_name:
            self.model_name = model_name
        self._gguf_path = None

    def _ensure_llm(self):
        if self._llm is not None:
            return
        from huggingface_hub import snapshot_download, list_repo_files, hf_hub_download
        import glob, gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        files = list_repo_files(self.model_name)
        candidates = [f for f in files if f.lower().endswith(('.gguf', '.bin'))]
        fav = next((f for p in ["q4_k_m", "q4_k", "q4_0", "q4"] for f in candidates if p in f.lower()), None)
        model_path = hf_hub_download(repo_id=self.model_name, filename=fav) if fav else None

        if not model_path:
            repo_dir = snapshot_download(repo_id=self.model_name, repo_type="model")
            gguf_candidates = glob.glob(f"{repo_dir}/**/*.gguf", recursive=True)
            if not gguf_candidates:
                raise RuntimeError(f"No .gguf file found for {self.model_name}")
            model_path = gguf_candidates[0]

        from llama_cpp import Llama
        self._llm = Llama(model_path=model_path)
        self._gguf_path = model_path

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(context_window=self.context_window, num_output=self.num_output, model_name=self.model_name)

    @llm_completion_callback()
    def complete(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> CompletionResponse:
        if self._llm is None:
            self._ensure_llm()
        resp = self._llm.create(prompt=prompt, max_tokens=self.num_output)
        text = ""
        if isinstance(resp, dict) and resp.get("choices"):
            choice = resp["choices"][0]
            text = choice.get("text") or choice.get("message", {}).get("content", "")
        elif hasattr(resp, "choices"):
            choices = getattr(resp, "choices", None)
            if choices and isinstance(choices, list):
                text = str(choices[0])
        text = _truncate_on_stop(text or "", stop)
        return CompletionResponse(text=text)

    @llm_completion_callback()
    def stream_complete(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> CompletionResponseGen:
        resp = self.complete(prompt, stop=stop, **kwargs)
        for ch in resp.text:
            yield CompletionResponse(text=ch, delta=ch)


class JinaEmbeddingsV4(BaseEmbedding):
    """
    Memory-constrained wrapper for jinaai/jina-embeddings-v4.
    - 4-bit NF4 + FP16 compute with automatic sharding/offload
    - Caps sequence length and batch size to limit activation/KV memory spikes
    - Processes items sequentially (batch_size=1) to avoid multi-GiB peaks
    """
    model_name: str = Field(default="jinaai/jina-embeddings-v4")

    # Memory guardrails
    max_length_query: int = Field(default=512)     # tighten if needed (256–512)
    max_length_passage: int = Field(default=768)   # tighten if needed (512–1024)
    batch_size: int = Field(default=1)             # keep 1 for long texts
    truncate_dim: Optional[int] = Field(default=None)  # e.g., 1024 to reduce head memory
    offload_folder: str = Field(default="./offload_jina_v4")

    # Private
    _model = PrivateAttr(default=None)
    _processor = PrivateAttr(default=None)
    _loaded = PrivateAttr(default=False)
    _model_lock = PrivateAttr(default=None)

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        try:
            self._model_lock = threading.Lock()
        except Exception:
            self._model_lock = None
        self._loaded = False

    def _ensure_model(self):
        if getattr(self, "_loaded", False):
            return
        lock = getattr(self, "_model_lock", None)
        if lock is not None:
            lock.acquire()
        try:
            if getattr(self, "_loaded", False):
                return

            os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
            os.makedirs(self.offload_folder, exist_ok=True)

            from transformers import AutoModel, AutoProcessor
            # BitsAndBytesConfig is available via transformers when bitsandbytes is installed
            try:
                from transformers import BitsAndBytesConfig
            except Exception:
                from bitsandbytes import BitsAndBytesConfig  # fallback import

            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,  # choose bfloat16 if instability observed
            )

            # Let HF place layers and offload automatically (prevents pinning on GPU:0)
            self._model = AutoModel.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                device_map="auto",
                offload_folder=self.offload_folder,
                torch_dtype=torch.float16,
                quantization_config=bnb_cfg,
            ).eval()

            # Optional processor
            try:
                self._processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
            except Exception:
                self._processor = None

            self._loaded = True
        finally:
            if lock is not None:
                try:
                    lock.release()
                except Exception:
                    pass

    @classmethod
    def class_name(cls) -> str:
        return "jina_v4_memory_guarded"

    def _embed(
        self,
        texts: List[str],
        image_paths: Optional[List[Optional[str]]] = None,
        task: str = "retrieval",
        prompt_name: Optional[str] = "passage",
        return_multivector: bool = False,
        truncate_dim: Optional[int] = None,
        max_length: Optional[int] = None,
        max_pixels: Optional[int] = None,
        batch_size: Optional[int] = None,
        is_query: bool = False,
    ) -> List[List[float]]:
        """
        Strict memory controls:
        - batch_size forced to 1 by default
        - max_length capped (queries vs passages)
        - optional truncate_dim to reduce output dimensionality
        """
        from PIL import Image

        try:
            self._ensure_model()
        except Exception:
            # Safe fallback: return zeros
            dim = truncate_dim or self.truncate_dim or 2048
            return [[0.0] * dim for _ in texts]

        model = self._model

        # Effective limits
        eff_batch = 1 if batch_size is None else max(1, int(batch_size))
        default_max_len = self.max_length_query if is_query else self.max_length_passage
        eff_max_len = int(max_length or default_max_len)
        eff_truncate_dim = truncate_dim or self.truncate_dim

        # Normalize image paths
        image_paths = image_paths or [None] * len(texts)
        if len(image_paths) == 1 and len(texts) > 1:
            image_paths = image_paths * len(texts)

        def _to_list_vector(arr):
            if isinstance(arr, torch.Tensor):
                arr = arr.detach().cpu().numpy()
            try:
                import numpy as np
            except Exception:
                np = None
            if np is not None and isinstance(arr, np.ndarray):
                # possible shapes: (N, D), (N, M, D), (D,)
                if arr.ndim == 3:
                    arr = arr.mean(axis=1)
                if arr.ndim == 2:
                    arr = arr[0]
                return [float(x) for x in arr.tolist()]
            if isinstance(arr, list):
                if not arr:
                    return []
                first = arr[0]
                if isinstance(first, (list, tuple, torch.Tensor)):
                    return _to_list_vector(first)
                return [float(x) for x in arr]
            try:
                return [float(arr)]
            except Exception:
                return [float(str(arr))]

        results: List[List[float]] = []

        # One-by-one to avoid overlapping activation peaks
        with torch.inference_mode():
            for text, img_path in zip(texts, image_paths):
                kwargs = dict(
                    task=task,
                    return_multivector=return_multivector,
                    truncate_dim=eff_truncate_dim,
                    max_length=eff_max_len,
                    max_pixels=max_pixels,
                    batch_size=eff_batch,
                )

                if img_path is not None and text:
                    img = Image.open(img_path).convert("RGB")
                    emb = model.encode(text=text, image=img, **kwargs)
                    results.append(_to_list_vector(emb))
                elif img_path is not None:
                    img = Image.open(img_path).convert("RGB")
                    emb = model.encode_image(images=[img], **kwargs)
                    results.append(_to_list_vector(emb))
                else:
                    ekw = {"texts": [text], **kwargs}
                    if prompt_name:
                        ekw["prompt_name"] = prompt_name
                    emb = model.encode_text(**ekw)
                    results.append(_to_list_vector(emb))

                # Encourage allocator reuse between items
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()

        return results

    def _get_query_embedding(self, query: str, image_path: Optional[str] = None) -> List[float]:
        # prompt_name="query" for text-only retrieval queries
        return self._embed(
            [query],
            [image_path] if image_path else None,
            task="retrieval",
            prompt_name=None if image_path is not None else "query",
            is_query=True,
        )[0]

    def _get_text_embedding(self, text: str, image_path: Optional[str] = None) -> List[float]:
        # prompt_name="passage" for indexing
        return self._embed(
            [text],
            [image_path] if image_path else None,
            task="retrieval",
            prompt_name=None if image_path is not None else "passage",
            is_query=False,
        )[0]

    def _get_text_embeddings(
        self,
        texts: List[str],
        image_paths: Optional[List[Optional[str]]] = None,
    ) -> List[List[float]]:
        return self._embed(
            texts,
            image_paths,
            task="retrieval",
            prompt_name="passage",
            is_query=False,
        )

    # Async variants for compatibility with your current call sites
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
        self._loaded = False
    
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

        # Lazy-load reranker model on first use
        if not getattr(self, "_loaded", False):
            try:
                self._load_model()
                self._loaded = True
            except Exception as e:
                print(f"Reranker lazy-load failed: {e}")
                # If loading fails, return the original nodes limited to top_n
                return nodes[:self.top_n]
        
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
