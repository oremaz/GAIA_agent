from typing import Optional, List, Any, Dict, Sequence, Union
import os
import warnings
import base64
import mimetypes
from pydantic import Field, PrivateAttr
from pydantic.warnings import UnsupportedFieldAttributeWarning
# Silence LlamaIndex's validate_default Field warning during import.
warnings.filterwarnings("ignore", category=UnsupportedFieldAttributeWarning)

from llama_index.core.llms import CustomLLM, CompletionResponse, CompletionResponseGen, LLMMetadata
from llama_index.core.llms.callbacks import llm_completion_callback
from transformers import (
    AutoModel,
    AutoProcessor,
    AutoTokenizer,
    AutoModelForCausalLM,
    TextIteratorStreamer,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration,
    Qwen3OmniMoeForConditionalGeneration,
    Qwen3OmniMoeProcessor,
    BitsAndBytesConfig,
)
from transformers import Mistral3ForConditionalGeneration

try:
    from transformers import MistralCommonBackend
except Exception:
    MistralCommonBackend = None
import torch
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.schema import ImageDocument
import threading
import logging
import numpy as np
from PIL import Image

# API clients
from google import genai
from openai import OpenAI

# Diffusion models (optional - may fail on some platforms like macOS with torch.xpu issues)
try:
    from diffusers import DiffusionPipeline, QwenImageEditPlusPipeline
    import soundfile as sf
    DIFFUSERS_AVAILABLE = True
except (ImportError, AttributeError) as e:
    DIFFUSERS_AVAILABLE = False
    DiffusionPipeline, QwenImageEditPlusPipeline, sf  = None, None, None
    logging.getLogger(__name__).warning(f"Diffusers not available: {e}")

# Lock to prevent races when creating cached instances concurrently
_CACHE_LOCK = threading.Lock()
_logger = logging.getLogger(__name__)

# Module-level caches to avoid creating multiple heavyweight model instances
_EMBEDDER_CACHE = {}
_RERANKER_CACHE = {}
_LLM_CACHE = {}
_MINISTRAL_CACHE = {}
_API_CLIENT_CACHE = {}

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
    """Return cached Qwen3VLMultiModal or create one.

    Cached by (model_name, device). Defaults to the int4-friendly
    Qwen/Qwen3-VL-30B-A3B-Instruct checkpoint described in the docs.
    """
    key = (model_name or "Qwen/Qwen3-VL-30B-A3B-Instruct", device or "auto")
    inst = _LLM_CACHE.get(key)
    if inst is not None:
        return inst

    with _CACHE_LOCK:
        inst = _LLM_CACHE.get(key)
        if inst is not None:
            return inst

        _logger.info("Creating Qwen3VLMultiModal for key=%s", key)
        try:
            before_alloc = None
            try:
                if torch.cuda.is_available():
                    before_alloc = torch.cuda.memory_allocated()
            except Exception:
                before_alloc = None

            inst = Qwen3VLMultiModal(model_id=key[0], device_map=key[1])

            after_alloc = None
            try:
                if torch.cuda.is_available():
                    after_alloc = torch.cuda.memory_allocated()
            except Exception:
                after_alloc = None

            _logger.info(
                "Qwen3VLMultiModal created for key=%s (mem_before=%s, mem_after=%s)",
                key,
                before_alloc,
                after_alloc,
            )
        except Exception:
            _logger.exception("Failed to create Qwen3VLMultiModal for key=%s", key)
            raise

        _LLM_CACHE[key] = inst
        return inst

def get_or_create_ministral_llm(model_name: Optional[str] = None, device: Optional[str] = None):
    """Return cached MinistralMultiModal or create one.

    Defaults to Ministral-3-8B-Instruct-2512.
    """
    key = (model_name or "mistralai/Ministral-3-8B-Instruct-2512", device or "auto")
    inst = _MINISTRAL_CACHE.get(key)
    if inst is not None:
        return inst

    with _CACHE_LOCK:
        inst = _MINISTRAL_CACHE.get(key)
        if inst is not None:
            return inst

        _logger.info("Creating MinistralMultiModal for key=%s", key)
        try:
            before_alloc = None
            if torch.cuda.is_available():
                try:
                    before_alloc = torch.cuda.memory_allocated()
                except Exception:
                    pass

            inst = MinistralMultiModal(model_id=key[0], device_map=key[1])

            after_alloc = None
            if torch.cuda.is_available():
                try:
                    after_alloc = torch.cuda.memory_allocated()
                except Exception:
                    pass

            _logger.info(
                "MinistralMultiModal created (mem_before=%s, mem_after=%s)",
                before_alloc, after_alloc
            )
        except Exception:
            _logger.exception("Failed to create MinistralMultiModal for key=%s", key)
            raise

        _MINISTRAL_CACHE[key] = inst
        return inst

def get_or_create_devstral_llm(model_name: Optional[str] = None, device: Optional[str] = None):
    """Return cached Devstral LLM configured for agentic software engineering tasks.
    
    Uses Mistral3ForConditionalGeneration with Devstral-Small-2-24B-Instruct-2512.
    This model excels at using tools, exploring codebases, and editing multiple files.
    """
    key = (model_name or "mistralai/Devstral-Small-2-24B-Instruct-2512", device or "auto")
    inst = _LLM_CACHE.get(key)
    if inst is not None:
        return inst

    with _CACHE_LOCK:
        inst = _LLM_CACHE.get(key)
        if inst is not None:
            return inst
        
        _logger.info("Creating Devstral LLM for key=%s", key)
        try:
            before_alloc = None
            if torch.cuda.is_available():
                try:
                    before_alloc = torch.cuda.memory_allocated()
                except Exception:
                    pass

            inst = DevstralLLM(model_id=key[0], device_map=key[1])

            after_alloc = None
            if torch.cuda.is_available():
                try:
                    after_alloc = torch.cuda.memory_allocated()
                except Exception:
                    pass

            _logger.info(
                "DevstralLLM created for key=%s (mem_before=%s, mem_after=%s)",
                key,
                before_alloc,
                after_alloc,
            )
        except Exception:
            _logger.exception("Failed to create DevstralLLM for key=%s", key)
            raise

        _LLM_CACHE[key] = inst
        return inst

def unload_model_from_gpu(model):
    """Unload a model from GPU to CPU and clear cache."""
    try:
        if hasattr(model, '_model') and model._model is not None:
            _logger.info("Unloading model from GPU to CPU")
            model._model.to('cpu')
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        return True
    except Exception as e:
        _logger.warning(f"Failed to unload model: {e}")
        return False


def reload_model_to_gpu(model):
    """Reload a model from CPU to GPU."""
    try:
        if hasattr(model, '_model') and model._model is not None:
            _logger.info("Reloading model to GPU")
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            model._model.to(device)
        return True
    except Exception as e:
        _logger.warning(f"Failed to reload model: {e}")
        return False



_DEFAULT_QWEN_VL = "Qwen/Qwen3-VL-30B-A3B-Instruct"


class DevstralLLM(CustomLLM):
    """CustomLLM wrapper for Devstral-Small-2-24B-Instruct-2512 using Mistral3ForConditionalGeneration.

    Optimized for agentic software engineering tasks including tool use, code exploration, and multi-file editing.
    Model is in FP8 format by default - no additional quantization applied.
    """

    model_id: str = Field(default="mistralai/Devstral-Small-2-24B-Instruct-2512")
    max_new_tokens: int = Field(default=2048)
    temperature: float = Field(default=0.15)
    top_p: float = Field(default=0.9)
    device_map: str = Field(default="auto")
    use_flash_attn2: bool = Field(default=False)
    max_input_tokens: int = Field(default=32768)

    _tokenizer: Any = PrivateAttr(default=None)
    _model: Any = PrivateAttr(default=None)
    _hf_loaded: bool = PrivateAttr(default=False)
    _hf_lock: Any = PrivateAttr(default=None)

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        try:
            self._hf_lock = threading.Lock()
        except Exception:
            self._hf_lock = None
        self._hf_loaded = False

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            model_name=self.model_id,
            context_window=32768,
            num_output=self.max_new_tokens,
            is_chat_model=True,
            is_function_calling_model=True,
        )

    @property
    def model_name(self) -> str:
        return self.model_id

    def _init_hf(self) -> None:
        """Load tokenizer + Devstral model. No quantization - model is already FP8."""
        try:
            os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128,expandable_segments:True")
        except Exception:
            pass

        if MistralCommonBackend is None:
            raise RuntimeError("MistralCommonBackend not available; Mistral models cannot be initialized.")

        self._tokenizer = MistralCommonBackend.from_pretrained(self.model_id)

        # Model kwargs - no quantization for FP8 model
        model_kwargs = {
            "device_map": self.device_map,
            "torch_dtype": "auto",
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
        }

        try:
            offload_folder = os.path.abspath("./offload_devstral")
            os.makedirs(offload_folder, exist_ok=True)
            model_kwargs["offload_folder"] = offload_folder
        except Exception:
            pass

        if self.use_flash_attn2:
            model_kwargs["attn_implementation"] = "flash_attention_2"

        self._model = Mistral3ForConditionalGeneration.from_pretrained(self.model_id, **model_kwargs)
        self._model.eval()

    def _ensure_hf(self) -> None:
        if getattr(self, "_hf_loaded", False):
            return
        lock = getattr(self, "_hf_lock", None)
        if lock is not None:
            lock.acquire()
        try:
            if getattr(self, "_hf_loaded", False):
                return
            self._init_hf()
            self._hf_loaded = True
        finally:
            if lock is not None:
                try:
                    lock.release()
                except Exception:
                    pass

    def _truncate_input(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Truncate input to max_input_tokens"""
        if input_ids.shape[-1] > self.max_input_tokens:
            _logger.warning(
                "Truncating Devstral input from %d to %d tokens",
                input_ids.shape[-1], self.max_input_tokens
            )
            return input_ids[:, -self.max_input_tokens:]
        return input_ids

    @llm_completion_callback()
    def complete(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> CompletionResponse:
        self._ensure_hf()

        # Build messages using Mistral format
        messages = [{"role": "user", "content": prompt}]

        # Tokenize using apply_chat_template
        tokenized = self._tokenizer.apply_chat_template(
            conversation=messages,
            tools=kwargs.get("tools"),
            return_tensors="pt",
            return_dict=True,
        )

        input_ids = tokenized["input_ids"]
        if input_ids.shape[-1] > self.max_input_tokens:
            input_ids = self._truncate_input(input_ids)

        try:
            input_ids = input_ids.to(self._model.device)
        except Exception:
            pass

        # Generate
        gen_kwargs = dict(
            max_new_tokens=min(int(kwargs.get("max_new_tokens", self.max_new_tokens)), self.max_new_tokens),
            do_sample=(kwargs.get("temperature", self.temperature) or 0.0) > 0.0,
            temperature=kwargs.get("temperature", self.temperature) or 0.15,
            top_p=kwargs.get("top_p", self.top_p),
        )

        with torch.inference_mode():
            output_ids = self._model.generate(input_ids, **gen_kwargs)

        # Decode only new tokens
        new_tokens = output_ids[0][len(tokenized["input_ids"][0]):]
        text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)

        if stop:
            text = _truncate_on_stop(text, stop)

        return CompletionResponse(text=text.strip())

    @llm_completion_callback()
    def stream_complete(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> CompletionResponseGen:
        self._ensure_hf()

        # Build messages
        messages = [{"role": "user", "content": prompt}]

        # Tokenize
        tokenized = self._tokenizer.apply_chat_template(
            conversation=messages,
            tools=kwargs.get("tools"),
            return_tensors="pt",
            return_dict=True,
        )

        input_ids = tokenized["input_ids"]
        if input_ids.shape[-1] > self.max_input_tokens:
            input_ids = self._truncate_input(input_ids)

        try:
            input_ids = input_ids.to(self._model.device)
        except Exception:
            pass

        gen_kwargs = dict(
            max_new_tokens=min(int(kwargs.get("max_new_tokens", self.max_new_tokens)), self.max_new_tokens),
            do_sample=(kwargs.get("temperature", self.temperature) or 0.0) > 0.0,
            temperature=kwargs.get("temperature", self.temperature) or 0.15,
            top_p=kwargs.get("top_p", self.top_p),
        )

        streamer = TextIteratorStreamer(
            self._tokenizer.tokenizer if hasattr(self._tokenizer, "tokenizer") else self._tokenizer,
            skip_special_tokens=True,
            skip_prompt=True,
        )

        thread = threading.Thread(
            target=self._model.generate,
            kwargs={"input_ids": input_ids, **gen_kwargs, "streamer": streamer},
            daemon=True,
        )
        thread.start()

        text_accum = ""
        for delta in streamer:
            text_accum += delta
            if stop and any(s and s in text_accum for s in stop):
                trimmed = _truncate_on_stop(text_accum, stop)
                yield CompletionResponse(text=trimmed, delta="")
                break
            yield CompletionResponse(text=text_accum, delta=delta)


class Qwen3VLMultiModal(CustomLLM):
    """CustomLLM wrapper for Qwen3-VL-30B-A3B-Instruct with int4 quantization.

    The implementation mirrors the official Qwen3-VL documentation flow: prompts
    are built via AutoProcessor.apply_chat_template with tokenized inputs so
    llama-index can drive the model seamlessly. We quantize to 4-bit NF4 via
    BitsAndBytes to satisfy the memory constraints called out in the requirements.
    """

    model_id: str = Field(default=_DEFAULT_QWEN_VL)
    max_new_tokens: int = Field(default=4096)
    temperature: float = Field(default=0.2)
    top_p: float = Field(default=0.95)
    device_map: str = Field(default="auto")
    min_pixels: Optional[int] = Field(default=None)
    max_pixels: Optional[int] = Field(default=None)
    use_flash_attn2: bool = Field(default=False)
    max_input_tokens: int = Field(default=4096)

    _processor: Any = PrivateAttr(default=None)
    _model: Any = PrivateAttr(default=None)
    _hf_loaded: bool = PrivateAttr(default=False)
    _hf_lock: Any = PrivateAttr(default=None)

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        try:
            self._hf_lock = threading.Lock()
        except Exception:
            self._hf_lock = None
        self._hf_loaded = False

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            model_name=self.model_id,
            context_window=256000,  # Qwen3-VL advertises 256K context
            num_output=self.max_new_tokens,
            is_chat_model=True,
            is_function_calling_model=False,
        )

    @property
    def model_name(self) -> str:
        return self.model_id

    def _init_hf(self) -> None:
        """Load processor + Qwen3-VL model with 4-bit quantization."""
        try:
            os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128,expandable_segments:True")
        except Exception:
            pass

        processor_kwargs: Dict[str, Any] = {}
        if self.min_pixels is not None:
            processor_kwargs["min_pixels"] = self.min_pixels
        if self.max_pixels is not None:
            processor_kwargs["max_pixels"] = self.max_pixels
        self._processor = AutoProcessor.from_pretrained(self.model_id, **processor_kwargs)

        compute_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        quant_cfg = None
        try:
            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=compute_dtype,
            )
        except Exception as exc:
            _logger.warning("BitsAndBytes unavailable (%s); falling back to fp16 load for Qwen3-VL.", exc)

        model_kwargs: Dict[str, Any] = {
            "device_map": self.device_map,
            "low_cpu_mem_usage": True,
            "torch_dtype": compute_dtype,
            "trust_remote_code": True,
        }
        if quant_cfg is not None:
            model_kwargs["quantization_config"] = quant_cfg

        try:
            offload_folder = os.path.abspath("./offload_qwen3_vl")
            os.makedirs(offload_folder, exist_ok=True)
            model_kwargs["offload_folder"] = offload_folder
        except Exception:
            pass

        if self.use_flash_attn2:
            model_kwargs["attn_implementation"] = "flash_attention_2"

        model_cls = Qwen3VLMoeForConditionalGeneration if self._is_moe_model() else Qwen3VLForConditionalGeneration
        self._model = model_cls.from_pretrained(self.model_id, **model_kwargs)

    def _is_moe_model(self) -> bool:
        model_id = (self.model_id or "").lower()
        return "a3b" in model_id or "moe" in model_id

    def _ensure_hf(self) -> None:
        if getattr(self, "_hf_loaded", False):
            return
        lock = getattr(self, "_hf_lock", None)
        if lock is not None:
            lock.acquire()
        try:
            if getattr(self, "_hf_loaded", False):
                return
            self._init_hf()
            self._hf_loaded = True
        finally:
            if lock is not None:
                try:
                    lock.release()
                except Exception:
                    pass

    def _force_batch_size_one(self, inputs: Dict[str, Any]) -> Dict[str, Any]:

        trimmed = False

        def _slice_val(value):
            nonlocal trimmed
            if isinstance(value, torch.Tensor):
                if value.dim() >= 1 and value.shape[0] > 1:
                    trimmed = True
                    return value[:1].contiguous()
                return value
            if isinstance(value, dict):
                return {k: _slice_val(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                seq = [_slice_val(v) for v in value]
                return type(value)(seq) if isinstance(value, tuple) else seq
            return value

        out_inputs = {k: _slice_val(v) for k, v in inputs.items()}
        if trimmed:
            _logger.warning("Trimmed input batch dimension to 1 for Qwen3-VL inference.")
        return out_inputs

    def _build_user_messages(
        self, prompt: str, image_documents: Optional[Sequence[ImageDocument]]
    ) -> List[dict]:
        message: Dict[str, Any] = {"role": "user", "content": []}
        if image_documents:
            for img in image_documents:
                # Check extension to decide if it's image or video
                ext = os.path.splitext(img.image_path)[1].lower()
                if ext in ['.mp4', '.mkv', '.mov', '.avi']:
                    message["content"].append({"type": "video", "video": f"file://{img.image_path}"})
                else:
                    message["content"].append({"type": "image", "image": f"file://{img.image_path}"})
        message["content"].append({"type": "text", "text": prompt})
        return [message]

    def _prepare_inputs_from_messages(self, messages: List[dict]) -> Dict[str, Any]:
        self._ensure_hf()
        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        if not isinstance(inputs, dict):
            try:
                inputs = dict(inputs)
            except Exception:
                pass

        try:
            if "input_ids" in inputs:
                seq_len = int(inputs["input_ids"].shape[-1])
                if seq_len > int(self.max_input_tokens):
                    keep = int(self.max_input_tokens)
                    _logger.warning("Truncating Qwen3-VL input from %d to %d tokens", seq_len, keep)
                    for key, tensor in list(inputs.items()):
                        if isinstance(tensor, torch.Tensor) and tensor.dim() >= 2:
                            inputs[key] = tensor[:, -keep:]
        except Exception:
            pass

        inputs = self._force_batch_size_one(inputs)
        try:
            inputs = {k: (v.to(self._model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        except Exception:
            pass
        return inputs

    def _decode_new_tokens(self, inputs: Dict[str, Any], generated_ids) -> str:
        self._ensure_hf()
        trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        texts = self._processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        if isinstance(texts, list):
            return texts[0] if texts else ""
        return texts or ""

    def _apply_stop(self, text: str, stop: Optional[List[str]]) -> str:
        if not stop or not text:
            return text
        return _truncate_on_stop(text, stop)

    @llm_completion_callback()
    def complete(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> CompletionResponse:
        image_documents: Optional[Sequence[ImageDocument]] = kwargs.pop("image_documents", None)
        messages = self._build_user_messages(prompt, image_documents)
        inputs = self._prepare_inputs_from_messages(messages)
        gen_kwargs = dict(
            max_new_tokens=min(int(kwargs.get("max_new_tokens", self.max_new_tokens)), 64),
            do_sample=(kwargs.get("temperature", self.temperature) or 0.0) > 0.0,
            temperature=kwargs.get("temperature", self.temperature),
            top_p=kwargs.get("top_p", self.top_p),
            use_cache=False,
        )

        reserved = set(gen_kwargs.keys())
        merged_inputs = {k: v for k, v in inputs.items() if k not in reserved}
        out = self._model.generate(**merged_inputs, **gen_kwargs)
        text = self._apply_stop(self._decode_new_tokens(inputs, out), stop)
        return CompletionResponse(text=str(text))

    @llm_completion_callback()
    def stream_complete(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> CompletionResponseGen:
        image_documents: Optional[Sequence[ImageDocument]] = kwargs.pop("image_documents", None)
        messages = self._build_user_messages(prompt, image_documents)
        inputs = self._prepare_inputs_from_messages(messages)
        gen_kwargs = dict(
            max_new_tokens=min(int(kwargs.get("max_new_tokens", self.max_new_tokens)), 64),
            do_sample=(kwargs.get("temperature", self.temperature) or 0.0) > 0.0,
            temperature=kwargs.get("temperature", self.temperature),
            top_p=kwargs.get("top_p", self.top_p),
            use_cache=False,
        )
        streamer = TextIteratorStreamer(
            self._processor.tokenizer,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        reserved = set(gen_kwargs.keys())
        merged_inputs = {k: v for k, v in inputs.items() if k not in reserved}
        thread = threading.Thread(
            target=self._model.generate,
            kwargs={**merged_inputs, **gen_kwargs, "streamer": streamer},
            daemon=True,
        )
        thread.start()

        text_accum = ""
        for delta in streamer:
            text_accum += delta
            if stop and any(s and s in text_accum for s in stop):
                trimmed = self._apply_stop(text_accum, stop)
                yield CompletionResponse(text=trimmed, delta="")
                break
            yield CompletionResponse(text=text_accum, delta=delta)


# ---------------- Embedding / reranker helpers ---------------- #

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

            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,  # choose bfloat16 if instability observed
            )

            # Let HF place layers and offload automatically (prevents pinning on GPU:0)
            target_device = "cuda:0" if torch.cuda.is_available() else "cpu"

            self._model = AutoModel.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                device_map=target_device,
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
                base_kwargs = dict(
                    task=task,
                    return_multivector=return_multivector,
                    truncate_dim=eff_truncate_dim,
                    max_length=eff_max_len,
                    batch_size=eff_batch,
                )
                image_kwargs = dict(base_kwargs)
                if max_pixels is not None:
                    image_kwargs["max_pixels"] = max_pixels

                if img_path is not None and text:
                    img = Image.open(img_path).convert("RGB")
                    emb = model.encode(text=text, image=img, **image_kwargs)
                    results.append(_to_list_vector(emb))
                elif img_path is not None:
                    img = Image.open(img_path).convert("RGB")
                    emb = model.encode_image(images=[img], **image_kwargs)
                    results.append(_to_list_vector(emb))
                else:
                    ekw = {"texts": [text], **base_kwargs}
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
                # For CPU targets, use regular PyTorch load
                self._model = AutoModel.from_pretrained(
                    self.model_name,
                    torch_dtype="auto",
                    trust_remote_code=True,
                    device_map=device_map,
                )
            # If device_map is 'cpu', ensure model is on CPU. Otherwise device_map already placed weights.
            if device_map == 'cpu' and hasattr(self._model, 'to'):
                self._model.to('cpu')

            if hasattr(self._model, 'eval'):
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
                _logger.exception("Reranker lazy-load failed: %s", e)
                # If loading fails, return the original nodes limited to top_n
                return nodes[:self.top_n]

        query = getattr(query_bundle, "query_str", "") or ""
        _logger.debug("Reranker received %d nodes for query='%s'", len(nodes), query)

        # Prepare query-document pairs for reranking
        text_pairs = []
        text_indices = []
        image_pairs = []
        image_indices = []

        def _unwrap(node_like):
            return getattr(node_like, "node", None) or node_like

        for i, node_wrapper in enumerate(nodes):
            node = _unwrap(node_wrapper)
            has_image = self._node_has_image(node)

            if has_image:
                image_path = self._extract_image_path(node)
                if image_path:
                    image_pairs.append([query, image_path])
                    image_indices.append(i)
                else:
                    _logger.debug("Image node missing path for index %s", i)
            else:
                text_content = ""
                if hasattr(node, "get_content"):
                    try:
                        text_content = node.get_content() or ""
                    except Exception as err:
                        _logger.debug("get_content failed for node %s: %s", i, err)
                        text_content = ""
                if not text_content:
                    text_content = getattr(node, "text", "") or ""
                if text_content:
                    text_pairs.append([query, text_content])
                    text_indices.append(i)
                else:
                    _logger.debug("Skipping empty text node at index %s", i)

        all_scores = []

        try:
            if text_pairs:
                text_scores = self._model.compute_score(
                    text_pairs,
                    max_length=1024,
                    doc_type="text"
                )
                for score, idx in zip(text_scores, text_indices):
                    all_scores.append((score, idx))

            if image_pairs:
                image_scores = self._model.compute_score(
                    image_pairs,
                    max_length=2048,
                    doc_type="image"
                )
                for score, idx in zip(image_scores, image_indices):
                    all_scores.append((score, idx))

        except Exception as e:
            _logger.exception("Error during reranking: %s", e)
            return nodes[:self.top_n]

        if not all_scores:
            _logger.debug("No reranker scores computed; returning original nodes")
            return nodes[:self.top_n]

        all_scores.sort(key=lambda x: x[0], reverse=True)

        reranked_nodes = []
        for score, node_idx in all_scores[:self.top_n]:
            if node_idx < len(nodes):
                node = nodes[node_idx]
                if hasattr(node, 'score'):
                    node.score = score
                reranked_nodes.append(node)

        _logger.debug("Reranker returning %d nodes", len(reranked_nodes))
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

    def score_text_pairs(self, pairs: List[List[str]], max_length: int = 1024) -> List[float]:
        """Utility helper for non-llama-index callers: score query/document text pairs."""
        if not pairs:
            return []
        if not getattr(self, "_loaded", False):
            self._load_model()
            self._loaded = True
        return self._model.compute_score(pairs, max_length=max_length, doc_type="text")

# ============================================================================
# Ministral-3 Support (Mistral AI)
# ============================================================================

_MINISTRAL_CACHE = {}

class MinistralMultiModal(CustomLLM):
    """CustomLLM wrapper for Mistral Ministral-3 series (3B/8B/14B) in native FP8.

    Supports:
    - mistralai/Ministral-3-3B-Instruct-2512
    - mistralai/Ministral-3-8B-Instruct-2512
    - mistralai/Ministral-3-14B-Instruct-2512

    Mirrors Qwen3VLMultiModal implementation pattern.
    """

    model_id: str = Field(default="mistralai/Ministral-3-8B-Instruct-2512")
    max_new_tokens: int = Field(default=2048)
    temperature: float = Field(default=0.7)
    top_p: float = Field(default=0.95)
    top_k: int = Field(default=50)
    device_map: str = Field(default="auto")
    use_flash_attn2: bool = Field(default=False)
    max_input_tokens: int = Field(default=8192)

    _tokenizer: Any = PrivateAttr(default=None)
    _model: Any = PrivateAttr(default=None)
    _hf_loaded: bool = PrivateAttr(default=False)
    _hf_lock: Any = PrivateAttr(default=None)

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        try:
            self._hf_lock = threading.Lock()
        except Exception:
            self._hf_lock = None
        self._hf_loaded = False

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            model_name=self.model_id,
            context_window=32768,
            num_output=self.max_new_tokens,
            is_chat_model=True,
            is_function_calling_model=False,
        )

    @property
    def model_name(self) -> str:
        return self.model_id

    def _init_hf(self) -> None:
        """Load tokenizer + Ministral-3 model in native FP8"""

        try:
            os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128,expandable_segments:True")
        except Exception:
            pass

        if MistralCommonBackend is None:
            raise RuntimeError("MistralCommonBackend not available; Mistral models cannot be initialized.")

        self._tokenizer = MistralCommonBackend.from_pretrained(self.model_id)

        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # Quantization disabled by default - Ministral models already in FP8 format
        # quant_cfg = None
        # try:
        #     quant_cfg = BitsAndBytesConfig(
        #         load_in_4bit=True,
        #         bnb_4bit_quant_type="nf4",
        #         bnb_4bit_use_double_quant=True,
        #         bnb_4bit_compute_dtype=torch.bfloat16,
        #     )
        # except Exception as exc:
        #     _logger.warning("BitsAndBytes unavailable (%s); using fp16", exc)

        model_kwargs: Dict[str, Any] = {
            "device_map": self.device_map,
            "low_cpu_mem_usage": True,
            "torch_dtype": "auto",
            "trust_remote_code": True,
        }

        # No quantization config added - using native FP8 format
        # if quant_cfg is not None:
        #     model_kwargs["quantization_config"] = quant_cfg

        try:
            offload_folder = os.path.abspath("./offload_ministral")
            os.makedirs(offload_folder, exist_ok=True)
            model_kwargs["offload_folder"] = offload_folder
        except Exception:
            pass

        if self.use_flash_attn2:
            model_kwargs["attn_implementation"] = "flash_attention_2"

        self._model = Mistral3ForConditionalGeneration.from_pretrained(self.model_id, **model_kwargs)
        self._model.eval()

    def _ensure_hf(self) -> None:
        if getattr(self, "_hf_loaded", False):
            return
        lock = getattr(self, "_hf_lock", None)
        if lock is not None:
            lock.acquire()
        try:
            if getattr(self, "_hf_loaded", False):
                return
            self._init_hf()
            self._hf_loaded = True
        finally:
            if lock is not None:
                try:
                    lock.release()
                except Exception:
                    pass

    def _truncate_input(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Truncate input to max_input_tokens"""
        if input_ids.shape[-1] > self.max_input_tokens:
            _logger.warning(
                "Truncating Ministral input from %d to %d tokens",
                input_ids.shape[-1], self.max_input_tokens
            )
            return input_ids[:, -self.max_input_tokens:]
        return input_ids

    @llm_completion_callback()
    def complete(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> CompletionResponse:
        self._ensure_hf()

        messages = [{"role": "user", "content": prompt}]
        inputs = self._tokenizer.apply_chat_template(
            conversation=messages,
            return_tensors="pt",
            return_dict=True,
        )

        try:
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        except Exception:
            pass

        if "input_ids" in inputs:
            inputs["input_ids"] = self._truncate_input(inputs["input_ids"])
        if "attention_mask" in inputs:
            inputs["attention_mask"] = inputs["attention_mask"][:, -self.max_input_tokens:]

        # Generate
        gen_kwargs = dict(
            max_new_tokens=min(int(kwargs.get("max_new_tokens", self.max_new_tokens)), self.max_new_tokens),
            do_sample=(kwargs.get("temperature", self.temperature) or 0.0) > 0.0,
            temperature=kwargs.get("temperature", self.temperature) or 0.7,
            top_p=kwargs.get("top_p", self.top_p),
            top_k=self.top_k,
            pad_token_id=self._tokenizer.pad_token_id,
            eos_token_id=self._tokenizer.eos_token_id,
        )

        with torch.inference_mode():
            output_ids = self._model.generate(**inputs, **gen_kwargs)

        start = inputs["input_ids"].shape[-1] if "input_ids" in inputs else 0
        new_tokens = output_ids[0][start:]
        text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)

        if stop:
            text = _truncate_on_stop(text, stop)

        return CompletionResponse(text=text.strip())

    @llm_completion_callback()
    def stream_complete(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> CompletionResponseGen:
        self._ensure_hf()

        messages = [{"role": "user", "content": prompt}]
        inputs = self._tokenizer.apply_chat_template(
            conversation=messages,
            return_tensors="pt",
            return_dict=True,
        )

        try:
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        except Exception:
            pass

        if "input_ids" in inputs:
            inputs["input_ids"] = self._truncate_input(inputs["input_ids"])
        if "attention_mask" in inputs:
            inputs["attention_mask"] = inputs["attention_mask"][:, -self.max_input_tokens:]

        gen_kwargs = dict(
            max_new_tokens=min(int(kwargs.get("max_new_tokens", self.max_new_tokens)), self.max_new_tokens),
            do_sample=(kwargs.get("temperature", self.temperature) or 0.0) > 0.0,
            temperature=kwargs.get("temperature", self.temperature) or 0.7,
            top_p=kwargs.get("top_p", self.top_p),
            top_k=self.top_k,
            pad_token_id=self._tokenizer.pad_token_id,
            eos_token_id=self._tokenizer.eos_token_id,
        )

        tokenizer_ref = self._tokenizer.tokenizer if hasattr(self._tokenizer, "tokenizer") else self._tokenizer
        streamer = TextIteratorStreamer(
            tokenizer_ref, skip_special_tokens=True, skip_prompt=True
        )

        thread = threading.Thread(
            target=self._model.generate,
            kwargs={**inputs, **gen_kwargs, "streamer": streamer},
            daemon=True,
        )
        thread.start()

        text_accum = ""
        for delta in streamer:
            text_accum += delta
            if stop and any(s and s in text_accum for s in stop):
                trimmed = _truncate_on_stop(text_accum, stop)
                yield CompletionResponse(text=trimmed, delta="")
                break
            yield CompletionResponse(text=text_accum, delta=delta)
# This file contains new model classes to add to custom_models.py
# Copy these classes to the end of custom_models.py

# ============================================================================
# API-based LLM Clients (Gemini, OpenAI)
# ============================================================================

class GeminiMultimodalLLM(CustomLLM):
    """Native Gemini API client with multimodal support (text, image, audio, video)."""
    
    model_id: str = Field(default="gemini-3-pro-preview")
    max_new_tokens: int = Field(default=16000)
    temperature: float = Field(default=0.6)
    top_p: float = Field(default=0.95)
    top_k: int = Field(default=20)
    api_key: Optional[str] = Field(default=None)
    
    _client: Any = PrivateAttr(default=None)
    _previous_interaction_id: Optional[str] = PrivateAttr(default=None)
    
    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        
        api_key = self.api_key or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment")
        
        self._client = genai.Client(api_key=api_key, http_options={'api_version': 'v1alpha'})
        self._previous_interaction_id = None
    
    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            model_name=self.model_id,
            context_window=1000000,  # Gemini 3 supports 1M context
            num_output=self.max_new_tokens,
            is_chat_model=True,
            is_function_calling_model=True,
        )
    
    @property
    def model_name(self) -> str:
        return self.model_id
    
    def _infer_interaction_type(self, mime_type: str) -> str:
        """Map mime type to Gemini Interactions input type."""
        if mime_type.startswith("image/"):
            return "image"
        if mime_type.startswith("audio/"):
            return "audio"
        if mime_type.startswith("video/"):
            return "video"
        if mime_type == "application/pdf":
            return "document"
        return "document"

    def _prepare_interaction_input(
        self,
        prompt: str,
        image_documents: Optional[List[ImageDocument]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Prepare Interactions input with text and optional media."""
        inputs: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]

        if image_documents:
            for img_doc in image_documents:
                file_path = img_doc.image_path
                file_size = os.path.getsize(file_path)
                mime_type, _ = mimetypes.guess_type(file_path)
                mime_type = mime_type or "application/octet-stream"
                input_type = self._infer_interaction_type(mime_type)

                # Use Files API for large files (>20MB)
                if file_size > 20 * 1024 * 1024:
                    uploaded_file = self._client.files.upload(file=file_path)
                    inputs.append({
                        "type": input_type,
                        "uri": uploaded_file.uri,
                    })
                else:
                    with open(file_path, "rb") as f:
                        encoded = base64.b64encode(f.read()).decode("utf-8")
                    inputs.append({
                        "type": input_type,
                        "data": encoded,
                        "mime_type": mime_type,
                    })

        return inputs

    def _extract_text(self, interaction: Any) -> str:
        outputs = getattr(interaction, "outputs", None) or []
        for output in reversed(outputs):
            text = getattr(output, "text", None)
            if text:
                return text.strip()
            if isinstance(output, dict):
                text = output.get("text")
                if text:
                    return text.strip()
        return ""
    
    @llm_completion_callback()
    def complete(self, prompt: str, image_documents: Optional[List[ImageDocument]] = None, **kwargs: Any) -> CompletionResponse:
        inputs = self._prepare_interaction_input(prompt, image_documents, **kwargs)

        generation_config = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_output_tokens": self.max_new_tokens,
        }

        interaction = self._client.interactions.create(
            model=self.model_id,
            input=inputs,
            previous_interaction_id=self._previous_interaction_id,
            generation_config=generation_config,
        )

        self._previous_interaction_id = getattr(interaction, "id", None)
        text = self._extract_text(interaction)
        return CompletionResponse(text=text)
    
    @llm_completion_callback()
    def stream_complete(self, prompt: str, image_documents: Optional[List[ImageDocument]] = None, **kwargs: Any) -> CompletionResponseGen:
        inputs = self._prepare_interaction_input(prompt, image_documents, **kwargs)

        generation_config = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_output_tokens": self.max_new_tokens,
        }

        interaction = self._client.interactions.create(
            model=self.model_id,
            input=inputs,
            previous_interaction_id=self._previous_interaction_id,
            generation_config=generation_config,
        )

        # Interactions streaming isn't integrated here yet; return full response.
        self._previous_interaction_id = getattr(interaction, "id", None)
        text = self._extract_text(interaction)
        yield CompletionResponse(text=text, delta=text)


class OpenAIMultimodalLLM(CustomLLM):
    """Native OpenAI API client with multimodal support."""
    
    model_id: str = Field(default="gpt-4o")
    max_new_tokens: int = Field(default=4096)
    temperature: float = Field(default=0.7)
    api_key: Optional[str] = Field(default=None)
    
    _client: Any = PrivateAttr(default=None)
    _conversation_id: Optional[str] = PrivateAttr(default=None)
    
    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        
        api_key = self.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")
        
        self._client = OpenAI(api_key=api_key)
        self._conversation_id = None
    
    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            model_name=self.model_id,
            context_window=128000,
            num_output=self.max_new_tokens,
            is_chat_model=True,
            is_function_calling_model=True,
        )
    
    @property
    def model_name(self) -> str:
        return self.model_id
    
    def _ensure_conversation_id(self) -> Optional[str]:
        if self._conversation_id:
            return self._conversation_id

        try:
            convo = self._client.conversations.create()
        except Exception:
            return None

        convo_id = getattr(convo, "id", None)
        if convo_id:
            self._conversation_id = convo_id
        return self._conversation_id

    def _prepare_input(self, prompt: str, image_documents: Optional[List[ImageDocument]] = None) -> List[Dict]:
        """Prepare Responses API input with text and optional images."""
        content = [{"type": "input_text", "text": prompt}]

        if image_documents:
            for img_doc in image_documents:
                file_path = img_doc.image_path

                with open(file_path, "rb") as f:
                    image_data = base64.b64encode(f.read()).decode("utf-8")

                mime_type, _ = mimetypes.guess_type(file_path)
                mime_type = mime_type or "image/jpeg"

                content.append({
                    "type": "input_image",
                    "image_url": f"data:{mime_type};base64,{image_data}",
                })

        return [{"role": "user", "content": content}]

    def _extract_response_text(self, response: Any) -> str:
        text = getattr(response, "output_text", None)
        if text:
            return text.strip()

        output = getattr(response, "output", None) or getattr(response, "outputs", None) or []
        for item in output:
            content = getattr(item, "content", None) if not isinstance(item, dict) else item.get("content")
            if not content:
                continue
            for part in content:
                part_type = part.get("type") if isinstance(part, dict) else getattr(part, "type", None)
                if part_type in ("output_text", "text"):
                    text = part.get("text") if isinstance(part, dict) else getattr(part, "text", None)
                    if text:
                        return text.strip()
        return ""
    
    @llm_completion_callback()
    def complete(self, prompt: str, image_documents: Optional[List[ImageDocument]] = None, **kwargs: Any) -> CompletionResponse:
        payload = self._prepare_input(prompt, image_documents)
        convo_id = self._ensure_conversation_id()

        response_kwargs = {
            "model": self.model_id,
            "input": payload,
            "max_output_tokens": self.max_new_tokens,
            "temperature": self.temperature,
        }
        if convo_id:
            response_kwargs["conversation"] = convo_id

        response = self._client.responses.create(**response_kwargs)

        text = self._extract_response_text(response)
        return CompletionResponse(text=text)
    
    @llm_completion_callback()
    def stream_complete(self, prompt: str, image_documents: Optional[List[ImageDocument]] = None, **kwargs: Any) -> CompletionResponseGen:
        payload = self._prepare_input(prompt, image_documents)
        convo_id = self._ensure_conversation_id()

        response_kwargs = {
            "model": self.model_id,
            "input": payload,
            "max_output_tokens": self.max_new_tokens,
            "temperature": self.temperature,
        }
        if convo_id:
            response_kwargs["conversation"] = convo_id

        response = self._client.responses.create(**response_kwargs)

        text = self._extract_response_text(response)
        yield CompletionResponse(text=text, delta=text)


# ============================================================================
# Qwen3 Text-only Models (No VL)
# ============================================================================

class Qwen3TextLLM(CustomLLM):
    """Qwen3 text-only models (4B and 30B FP8 variants) without vision capabilities."""
    
    model_id: str = Field(default="Qwen/Qwen3-4B-Instruct-2507-FP8")
    max_new_tokens: int = Field(default=16384)
    temperature: float = Field(default=0.6)
    top_p: float = Field(default=0.95)
    device_map: str = Field(default="auto")
    max_input_tokens: int = Field(default=32768)
    
    _tokenizer: Any = PrivateAttr(default=None)
    _model: Any = PrivateAttr(default=None)
    _hf_loaded: bool = PrivateAttr(default=False)
    _hf_lock: Any = PrivateAttr(default=None)
    
    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        try:
            self._hf_lock = threading.Lock()
        except Exception:
            self._hf_lock = None
        self._hf_loaded = False
    
    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            model_name=self.model_id,
            context_window=32768,
            num_output=self.max_new_tokens,
            is_chat_model=True,
            is_function_calling_model=False,
        )
    
    @property
    def model_name(self) -> str:
        return self.model_id
    
    def _init_hf(self) -> None:
        """Load tokenizer + model. Models are already in FP8, no quantization needed."""
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=True
        )
        
        # Model is already FP8, load with auto dtype
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype="auto",
            device_map=self.device_map,
            trust_remote_code=True
        )
        self._model.eval()
    
    def _ensure_hf(self) -> None:
        if getattr(self, "_hf_loaded", False):
            return
        lock = getattr(self, "_hf_lock", None)
        if lock is not None:
            lock.acquire()
        try:
            if getattr(self, "_hf_loaded", False):
                return
            self._init_hf()
            self._hf_loaded = True
        finally:
            if lock is not None:
                try:
                    lock.release()
                except Exception:
                    pass
    
    @llm_completion_callback()
    def complete(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> CompletionResponse:
        self._ensure_hf()
        
        messages = [{"role": "user", "content": prompt}]
        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        model_inputs = self._tokenizer([text], return_tensors="pt").to(self._model.device)
        
        generated_ids = self._model.generate(
            **model_inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        content = self._tokenizer.decode(output_ids, skip_special_tokens=True)
        
        if stop:
            content = _truncate_on_stop(content, stop)
        
        return CompletionResponse(text=content.strip())
    
    @llm_completion_callback()
    def stream_complete(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> CompletionResponseGen:
        self._ensure_hf()
        
        messages = [{"role": "user", "content": prompt}]
        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        model_inputs = self._tokenizer([text], return_tensors="pt").to(self._model.device)
        
        streamer = TextIteratorStreamer(
            self._tokenizer,
            skip_special_tokens=True,
            skip_prompt=True,
        )
        
        thread = threading.Thread(
            target=self._model.generate,
            kwargs={
                **model_inputs,
                "max_new_tokens": self.max_new_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "streamer": streamer
            },
            daemon=True,
        )
        thread.start()
        
        text_accum = ""
        for delta in streamer:
            text_accum += delta
            if stop and any(s and s in text_accum for s in stop):
                trimmed = _truncate_on_stop(text_accum, stop)
                yield CompletionResponse(text=trimmed, delta="")
                break
            yield CompletionResponse(text=text_accum, delta=delta)


# ============================================================================
# Qwen3-Omni for Audio/Video + Text
# ============================================================================

class Qwen3OmniMultiModal(CustomLLM):
    """Qwen3-Omni-30B-A3B-Instruct for audio, video, and text processing."""
    
    model_id: str = Field(default="Qwen/Qwen3-Omni-30B-A3B-Instruct")
    max_new_tokens: int = Field(default=4096)
    temperature: float = Field(default=0.2)
    top_p: float = Field(default=0.95)
    device_map: str = Field(default="auto")
    use_flash_attn2: bool = Field(default=False)
    use_audio_in_video: bool = Field(default=True)
    speaker: str = Field(default="Ethan")
    
    _processor: Any = PrivateAttr(default=None)
    _model: Any = PrivateAttr(default=None)
    _hf_loaded: bool = PrivateAttr(default=False)
    _hf_lock: Any = PrivateAttr(default=None)
    
    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        try:
            self._hf_lock = threading.Lock()
        except Exception:
            self._hf_lock = None
        self._hf_loaded = False
    
    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            model_name=self.model_id,
            context_window=32768,
            num_output=self.max_new_tokens,
            is_chat_model=True,
            is_function_calling_model=False,
        )
    
    @property
    def model_name(self) -> str:
        return self.model_id
    
    def _init_hf(self) -> None:
        """Load processor + Qwen3-Omni model."""
        model_kwargs = {
            "dtype": "auto",
            "device_map": self.device_map,
        }
        
        if self.use_flash_attn2:
            model_kwargs["attn_implementation"] = "flash_attention_2"
        
        self._model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            self.model_id,
            **model_kwargs
        )
        
        self._processor = Qwen3OmniMoeProcessor.from_pretrained(self.model_id)
        self._model.eval()
    
    def _ensure_hf(self) -> None:
        if getattr(self, "_hf_loaded", False):
            return
        lock = getattr(self, "_hf_lock", None)
        if lock is not None:
            lock.acquire()
        try:
            if getattr(self, "_hf_loaded", False):
                return
            self._init_hf()
            self._hf_loaded = True
        finally:
            if lock is not None:
                try:
                    lock.release()
                except Exception:
                    pass
    
    def _prepare_conversation(self, prompt: str, image_documents: Optional[List[ImageDocument]] = None) -> List[Dict]:
        """Prepare conversation with text, images, audio, and video."""
        content = [{"type": "text", "text": prompt}]
        
        if image_documents:
            for img_doc in image_documents:
                file_path = img_doc.image_path
                mime_type, _ = mimetypes.guess_type(file_path)
                
                if mime_type and mime_type.startswith("image/"):
                    content.append({"type": "image", "image": file_path})
                elif mime_type and mime_type.startswith("audio/"):
                    content.append({"type": "audio", "audio": file_path})
                elif mime_type and mime_type.startswith("video/"):
                    content.append({"type": "video", "video": file_path})
        
        return [{"role": "user", "content": content}]
    
    @llm_completion_callback()
    def complete(self, prompt: str, image_documents: Optional[List[ImageDocument]] = None, **kwargs: Any) -> CompletionResponse:
        self._ensure_hf()
        
        try:
            from qwen_omni_utils import process_mm_info
        except ImportError:
            _logger.warning("qwen_omni_utils not available, multimodal features may be limited")
            # Fallback to text-only processing
            return CompletionResponse(text="Qwen Omni utils not available. Please install qwen_omni_utils.")
        
        conversation = self._prepare_conversation(prompt, image_documents)
        
        text = self._processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=self.use_audio_in_video)
        
        inputs = self._processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=self.use_audio_in_video
        )
        inputs = inputs.to(self._model.device).to(self._model.dtype)
        
        text_ids, audio = self._model.generate(
            **inputs,
            speaker=self.speaker,
            thinker_return_dict_in_generate=True,
            use_audio_in_video=self.use_audio_in_video
        )
        
        output_text = self._processor.batch_decode(
            text_ids.sequences[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        # Optionally save audio output if available
        if audio is not None:
            try:
                sf.write(
                    "omni_output.wav",
                    audio.reshape(-1).detach().cpu().numpy(),
                    samplerate=24000,
                )
                _logger.info("Audio output saved to omni_output.wav")
            except Exception as e:
                _logger.warning(f"Failed to save audio output: {e}")
        
        return CompletionResponse(text=output_text.strip())
    
    @llm_completion_callback()
    def stream_complete(self, prompt: str, image_documents: Optional[List[ImageDocument]] = None, **kwargs: Any) -> CompletionResponseGen:
        # Omni model doesn't support streaming well, return full response
        response = self.complete(prompt, image_documents, **kwargs)
        yield CompletionResponse(text=response.text, delta=response.text)


# ============================================================================
# Image Generation with Qwen-Image-2512
# ============================================================================

class QwenImageGenerator:
    """Image generation using Qwen-Image-2512 diffusion model."""
    
    def __init__(self, model_name: str = "Qwen/Qwen-Image-2512"):
        if not DIFFUSERS_AVAILABLE:
            raise RuntimeError("Diffusers library not available")
        self.model_name = model_name
        self.pipe = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    
    def _ensure_pipeline(self):
        """Lazy load the pipeline."""
        if self.pipe is None:
            _logger.info(f"Loading image generation pipeline: {self.model_name}")
            self.pipe = DiffusionPipeline.from_pretrained(
                self.model_name,
                torch_dtype=self.torch_dtype
            ).to(self.device)
    
    def generate(
        self,
        prompt: str,
        negative_prompt: str = "低分辨率，低画质，肢体畸形，手指畸形，画面过饱和，蜡像感，人脸无细节，过度光滑，画面具有AI感。构图混乱。文字模糊，扭曲。",
        aspect_ratio: str = "16:9",
        num_inference_steps: int = 50,
        true_cfg_scale: float = 4.0,
        seed: Optional[int] = None,
        output_path: str = "generated_image.png"
    ) -> str:
        """Generate image from text prompt."""
        self._ensure_pipeline()
        
        aspect_ratios = {
            "1:1": (1328, 1328),
            "16:9": (1664, 928),
            "9:16": (928, 1664),
            "4:3": (1472, 1104),
            "3:4": (1104, 1472),
            "3:2": (1584, 1056),
            "2:3": (1056, 1584),
        }
        
        width, height = aspect_ratios.get(aspect_ratio, (1664, 928))
        
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        
        image = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            true_cfg_scale=true_cfg_scale,
            generator=generator
        ).images[0]
        
        image.save(output_path)
        _logger.info(f"Image saved to {output_path}")
        
        return output_path


# ============================================================================
# Image Editing with Qwen-Image-Edit-2511
# ============================================================================

class QwenImageEditor:
    """Image editing using Qwen-Image-Edit-2511."""
    
    def __init__(self, model_name: str = "Qwen/Qwen-Image-Edit-2511"):
        if not DIFFUSERS_AVAILABLE:
            raise RuntimeError("Diffusers library not available")
        self.model_name = model_name
        self.pipeline = None
        self.torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    
    def _ensure_pipeline(self):
        """Lazy load the pipeline."""
        if self.pipeline is None:
            _logger.info(f"Loading image editing pipeline: {self.model_name}")
            self.pipeline = QwenImageEditPlusPipeline.from_pretrained(
                self.model_name,
                torch_dtype=self.torch_dtype
            )
            self.pipeline.to('cuda' if torch.cuda.is_available() else 'cpu')
            self.pipeline.set_progress_bar_config(disable=None)
    
    def edit(
        self,
        image_paths: Union[str, List[str]],
        prompt: str,
        negative_prompt: str = " ",
        num_inference_steps: int = 40,
        guidance_scale: float = 1.0,
        true_cfg_scale: float = 4.0,
        seed: int = 0,
        output_path: str = "edited_image.png"
    ) -> str:
        """Edit image(s) based on text prompt."""
        self._ensure_pipeline()
        
        # Load images
        if isinstance(image_paths, str):
            image_paths = [image_paths]
        
        images = [Image.open(path) for path in image_paths]
        
        inputs = {
            "image": images,
            "prompt": prompt,
            "generator": torch.manual_seed(seed),
            "true_cfg_scale": true_cfg_scale,
            "negative_prompt": negative_prompt,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "num_images_per_prompt": 1,
        }
        
        with torch.inference_mode():
            output = self.pipeline(**inputs)
            output_image = output.images[0]
            output_image.save(output_path)
        
        _logger.info(f"Edited image saved to {output_path}")
        return output_path


# ============================================================================
# Getter functions for new models
# ============================================================================

def get_or_create_qwen3_text_llm(model_name: Optional[str] = None, device: Optional[str] = None):
    """Return cached Qwen3TextLLM or create one."""
    key = (model_name or "Qwen/Qwen3-4B-Instruct-2507-FP8", device or "auto")
    inst = _LLM_CACHE.get(key)
    if inst is not None:
        return inst
    
    with _CACHE_LOCK:
        inst = _LLM_CACHE.get(key)
        if inst is not None:
            return inst
        
        _logger.info("Creating Qwen3TextLLM for key=%s", key)
        inst = Qwen3TextLLM(model_id=key[0], device_map=key[1])
        _LLM_CACHE[key] = inst
        return inst


def get_or_create_qwen3_omni_llm(model_name: Optional[str] = None, device: Optional[str] = None):
    """Return cached Qwen3OmniMultiModal or create one."""
    key = (model_name or "Qwen/Qwen3-Omni-30B-A3B-Instruct", device or "auto")
    inst = _LLM_CACHE.get(key)
    if inst is not None:
        return inst
    
    with _CACHE_LOCK:
        inst = _LLM_CACHE.get(key)
        if inst is not None:
            return inst
        
        _logger.info("Creating Qwen3OmniMultiModal for key=%s", key)
        inst = Qwen3OmniMultiModal(model_id=key[0], device_map=key[1])
        _LLM_CACHE[key] = inst
        return inst


def get_or_create_gemini_llm(
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    session_id: Optional[str] = None,
):
    """Return cached GeminiMultimodalLLM or create one."""
    key = (model_name or "gemini-3-pro-preview", api_key or os.environ.get("GOOGLE_API_KEY"), session_id)
    inst = _API_CLIENT_CACHE.get(key)
    if inst is not None:
        return inst
    
    with _CACHE_LOCK:
        inst = _API_CLIENT_CACHE.get(key)
        if inst is not None:
            return inst
        
        _logger.info("Creating GeminiMultimodalLLM for key=%s", key[0])
        inst = GeminiMultimodalLLM(model_id=key[0], api_key=key[1])
        _API_CLIENT_CACHE[key] = inst
        return inst


def get_or_create_openai_llm(
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    session_id: Optional[str] = None,
):
    """Return cached OpenAIMultimodalLLM or create one."""
    key = (model_name or "gpt-4o", api_key or os.environ.get("OPENAI_API_KEY"), session_id)
    inst = _API_CLIENT_CACHE.get(key)
    if inst is not None:
        return inst
    
    with _CACHE_LOCK:
        inst = _API_CLIENT_CACHE.get(key)
        if inst is not None:
            return inst
        
        _logger.info("Creating OpenAIMultimodalLLM for key=%s", key[0])
        inst = OpenAIMultimodalLLM(model_id=key[0], api_key=key[1])
        _API_CLIENT_CACHE[key] = inst
        return inst


def get_or_create_image_generator(model_name: Optional[str] = None):
    """Return cached QwenImageGenerator or create one."""
    if not DIFFUSERS_AVAILABLE:
        raise RuntimeError("Diffusers library not available. Cannot create image generator.")
    
    key = model_name or "Qwen/Qwen-Image-2512"
    inst = _API_CLIENT_CACHE.get(("img_gen", key))
    if inst is not None:
        return inst
    
    with _CACHE_LOCK:
        cache_key = ("img_gen", key)
        inst = _API_CLIENT_CACHE.get(cache_key)
        if inst is not None:
            return inst
        
        _logger.info("Creating QwenImageGenerator for model=%s", key)
        inst = QwenImageGenerator(model_name=key)
        _API_CLIENT_CACHE[cache_key] = inst
        return inst


def get_or_create_image_editor(model_name: Optional[str] = None):
    """Return cached QwenImageEditor or create one."""
    if not DIFFUSERS_AVAILABLE:
        raise RuntimeError("Diffusers library not available. Cannot create image editor.")
    
    key = model_name or "Qwen/Qwen-Image-Edit-2511"
    inst = _API_CLIENT_CACHE.get(("img_edit", key))
    if inst is not None:
        return inst
    
    with _CACHE_LOCK:
        cache_key = ("img_edit", key)
        inst = _API_CLIENT_CACHE.get(cache_key)
        if inst is not None:
            return inst
        
        _logger.info("Creating QwenImageEditor for model=%s", key)
        inst = QwenImageEditor(model_name=key)
        _API_CLIENT_CACHE[cache_key] = inst
        return inst
