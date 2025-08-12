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
from transformers import AwqConfig


class QwenVLCustomLLM(CustomLLM):
    model_name: str = Field(default="Qwen/Qwen2.5-VL-32B-Instruct-AWQ")
    context_window: int = Field(default=32768)
    num_output: int = Field(default=256)
    _model = PrivateAttr()
    _processor = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        quantization_config = AwqConfig(
            bits=4,
            fuse_max_seq_len=512,
            do_fuse=True,
        )
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
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
    vlm_model_name: str = Field(default="OpenGVLab/InternVL3-8B")
    context_window: int = Field(default=32768)
    num_output: int = Field(default=512)

    _gpt_model = PrivateAttr()
    _gpt_tokenizer = PrivateAttr()
    _vlm_model = PrivateAttr()
    _vlm_tokenizer = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel

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

        # Load InternVL3-8B (VLM) in 4-bit
        self._vlm_model = AutoModel.from_pretrained(
            self.vlm_model_name,
            load_in_4bit=True,
            torch_dtype="auto",
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).eval()
        self._vlm_tokenizer = AutoTokenizer.from_pretrained(
            self.vlm_model_name, trust_remote_code=True, use_fast=False
        )

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
        # Minimal single/multi-image handling using InternVL3 .chat API
        pixel_values = self._build_vlm_pixel_values(image_paths)
        question = ("<image>\n" if len(image_paths) > 0 else "") + (prompt or "Describe the image(s).")
        gen_cfg = dict(max_new_tokens=min(self.num_output, 1024), do_sample=False)
        # InternVL3 returns (response, history) when return_history=True
        try:
            response, _history = self._vlm_model.chat(
                self._vlm_tokenizer,
                pixel_values,
                question,
                gen_cfg,
                history=None,
                return_history=True,
            )
            return response if isinstance(response, str) else str(response)
        except TypeError:
            # Some versions return just the response string
            response = self._vlm_model.chat(
                self._vlm_tokenizer, pixel_values, question, gen_cfg
            )
            return response if isinstance(response, str) else str(response)

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

    
class JinaEmbeddingsV4(BaseEmbedding):
    """
    Multimodal embedding class using jinaai/jina-embeddings-v4 loaded in 4-bit with Transformers.
    Supports text-only, image-only, and text+image inputs with mean pooling.
    """

    model_name: str = Field(default="jinaai/jina-embeddings-v4")
    _model = PrivateAttr()
    _processor = PrivateAttr()

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        from transformers import AutoModel, AutoProcessor
        # Load in 4-bit with bitsandbytes
        self._model = AutoModel.from_pretrained(
            self.model_name,
            load_in_4bit=True,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
            task="retrieval",  # <-- set task
        ).eval()
        self._processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            task="retrieval",  # <-- set task
        )
    @classmethod
    def class_name(cls) -> str:
        return "jina_v4"



    def _embed(self, texts: List[str], image_paths: Optional[List[Optional[str]]] = None) -> List[List[float]]:
        # Use the official encode_text/encode_image API from the model for pooling and embedding
        image_paths = image_paths or [None] * len(texts)
        if len(image_paths) == 1 and len(texts) > 1:
            image_paths = image_paths * len(texts)

        # Load images (if any)
        images = []
        have_any_image = False
        for p in image_paths:
            if p:
                from PIL import Image
                img = Image.open(p).convert("RGB")
                images.append(img)
                have_any_image = True
            else:
                images.append(None)

        # Use encode_text for text-only, encode_image for image-only, and encode for multimodal
        model = self._model
        device = model.device
        results = []
        for text, img in zip(texts, images):
            with torch.no_grad():
                if img is not None and text:
                    # Multimodal: both text and image
                    emb = model.encode(text=text, image=img)
                elif img is not None:
                    # Image only
                    emb = model.encode_image(img)
                else:
                    # Text only
                    emb = model.encode_text(text)
                if isinstance(emb, torch.Tensor):
                    emb = emb.cpu().numpy()
                elif hasattr(emb, 'detach'):
                    emb = emb.detach().cpu().numpy()
                if emb.ndim == 2:
                    emb = emb[0]
                results.append(emb.tolist())
        return results

    def _get_query_embedding(self, query: str, image_path: Optional[str] = None) -> List[float]:
        return self._embed([query], [image_path] if image_path else None)[0]

    def _get_text_embedding(self, text: str, image_path: Optional[str] = None) -> List[float]:
        return self._embed([text], [image_path] if image_path else None)[0]

    def _get_text_embeddings(self, texts: List[str], image_paths: Optional[List[Optional[str]]] = None) -> List[List[float]]:
        return self._embed(texts, image_paths)

    async def _aget_query_embedding(self, query: str, image_path: Optional[str] = None) -> List[float]:
        return self._get_query_embedding(query, image_path)

    async def _aget_text_embedding(self, text: str, image_path: Optional[str] = None) -> List[float]:
        return self._get_text_embedding(text, image_path)
