# Standard library imports
import logging
import os
import sys
from typing import Dict, Any, List, Optional
from urllib.parse import urlparse
import asyncio
import json
import io
import contextlib

# Third-party imports
from custom_models import (
    get_or_create_jina_reranker,
    get_or_create_jina_embedder,
    get_or_create_qwen_vl_llm,
    get_or_create_qwen3_text_llm,
    get_or_create_devstral_llm,
    get_or_create_ministral_llm,
    get_or_create_gemini_llm,
    get_or_create_openai_llm,
    get_or_create_qwen3_omni_llm,
)

# LlamaIndex core imports
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.core.query_engine import BaseQueryEngine
from llama_index.core.tools import FunctionTool
from llama_index.core.agent.workflow import ReActAgent, AgentStream
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.core.workflow import Context
from llama_index.core.schema import ImageDocument
from llama_index.core.program import LLMTextCompletionProgram

# LlamaIndex specialized imports
from llama_index.readers.web import SimpleWebPageReader
from llama_index.readers.youtube_transcript import YoutubeTranscriptReader
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.readers.docling import DoclingReader
from utils.document_processor import UserAgentWebPageReader

# Import all required official LlamaIndex Readers
from llama_index.readers.file import (
    PandasCSVReader,
    PandasExcelReader,
)
from pydantic import BaseModel, Field
import weave
from ddgs import DDGS

## GPTOSSWrapper moved to custom_models; using get_or_create_gpt_llm()

weave.init("conversational-ai-agent")

# Initialize models based on API availability
@weave.op
def initialize_models(
    use_api_mode=False,
    model_suite="qwen",
    local_model_id: Optional[str] = None,
    api_model_name: Optional[str] = None,
    session_id: Optional[str] = None,
):
    """Initialize models for API or local mode.

    Args:
        use_api_mode: When True, use API-backed models (Gemini/OpenAI).
        model_suite: Local suite ("qwen" or "ministral") or API provider name.
        local_model_id: Optional local model override (HF model id).
        api_model_name: Optional API model name (e.g., "gemini-3-flash-preview").
        session_id: Optional session id for per-session API state.

    Returns:
        A tuple of initialized models. In API mode:
        (embed_model, proj_llm, code_llm). In local mode:
        (embed_model, proj_llm, code_llm, img_analysis_llm, media_analysis_llm,
        img_gen_model, img_edit_model).

    Raises:
        Exception: If local model initialization fails.
    """
    if use_api_mode:
        # API Mode - Using native multimodal API clients
        try:
            logger.info("Initializing models in API mode...")

            provider_order = []
            if model_suite in ("gemini", "openai"):
                provider_order.append(model_suite)
            for provider in ("gemini", "openai"):
                if provider not in provider_order:
                    provider_order.append(provider)

            for provider in provider_order:
                if provider == "gemini":
                    google_api_key = os.environ.get("GOOGLE_API_KEY")
                    if not google_api_key:
                        continue
                    logger.info("Using Gemini API with model: %s", api_model_name or "default")
                    proj_llm = get_or_create_gemini_llm(
                        model_name=api_model_name,
                        session_id=session_id,
                    )
                    code_llm = proj_llm  # Gemini is good at code
                    embed_model = get_or_create_jina_embedder()
                    return embed_model, proj_llm, code_llm

                if provider == "openai":
                    openai_api_key = os.environ.get("OPENAI_API_KEY")
                    if not openai_api_key:
                        continue
                    logger.info("Using OpenAI API with model: %s", api_model_name or "default")
                    proj_llm = get_or_create_openai_llm(
                        model_name=api_model_name,
                        session_id=session_id,
                    )
                    code_llm = proj_llm  # OpenAI is good at code
                    embed_model = get_or_create_jina_embedder()
                    return embed_model, proj_llm, code_llm
            
            # No API keys found, fall back to local mode
            logger.warning("No API keys found. Falling back to local mode...")
            return initialize_models(use_api_mode=False, model_suite=model_suite, local_model_id=local_model_id)
            
        except Exception as e:
            logger.exception("Error initializing API mode: %s", e)
            logger.info("Falling back to local mode...")
            return initialize_models(use_api_mode=False, model_suite=model_suite, local_model_id=local_model_id)
    else:
        logger.info("Initializing models in local mode...")
        try:
            embed_model = get_or_create_jina_embedder()

            # Determine main model and specialized models
            img_analysis_llm = None
            media_analysis_llm = None
            img_gen_model = None
            img_edit_model = None

            if model_suite == "ministral":
                model_id = local_model_id or "mistralai/Ministral-3-8B-Instruct-2512"
                logger.info("Initializing Ministral suite: %s", model_id)
                proj_llm = get_or_create_ministral_llm(model_name=model_id, device="auto")
                code_llm = get_or_create_devstral_llm()
                # Ministral doesn't have specialized image/media models
            else:
                # Qwen suite - main model is always regular Instruct, specialized models created separately
                model_id = local_model_id or "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
                
                # Main model: always a regular Qwen Instruct (not VL, not Omni)
                logger.info("Initializing Qwen main model: %s", model_id)
                proj_llm = get_or_create_qwen3_text_llm(model_name=model_id, device="auto")
                
                # Determine VL model size based on main model
                if "4B" in model_id:
                    vl_model = "Qwen/Qwen3-VL-4B-Instruct"
                    logger.info("Using Qwen3-VL-4B for image analysis")
                elif "30B" in model_id:
                    vl_model = "Qwen/Qwen3-VL-30B-A3B-Instruct"
                    logger.info("Using Qwen3-VL-30B for image analysis")
                else:
                    vl_model = "Qwen/Qwen3-VL-30B-A3B-Instruct"  # Default to 30B
                    logger.info("Using default Qwen3-VL-30B for image analysis")
                
                img_analysis_llm = get_or_create_qwen_vl_llm(model_name=vl_model, device="auto")
                
                # Media analysis: always Qwen-Omni-30B
                logger.info("Initializing Qwen3-Omni-30B for media analysis")
                media_analysis_llm = get_or_create_qwen3_omni_llm(
                    model_name="Qwen/Qwen3-Omni-30B-A3B-Instruct",
                    device="auto"
                )
                
                # Image generation/editing models (if diffusers available)
                try:
                    from custom_models import DIFFUSERS_AVAILABLE
                    if DIFFUSERS_AVAILABLE:
                        from custom_models import get_or_create_image_generator, get_or_create_image_editor
                        logger.info("Initializing Qwen image generation models")
                        img_gen_model = get_or_create_image_generator()
                        img_edit_model = get_or_create_image_editor()
                except Exception as e:
                    logger.warning("Image generation/editing models not available: %s", e)
                
                # Code model: always Devstral
                code_llm = get_or_create_devstral_llm()
            
            return embed_model, proj_llm, code_llm, img_analysis_llm, media_analysis_llm, img_gen_model, img_edit_model
        except Exception as e:
            logger.exception("Error initializing models: %s", e)
            raise


_ACTIVE_MODEL_CONFIG: Dict[str, Any] = {}


@weave.op
def configure_models(
    use_api_mode: bool,
    model_suite: str = "qwen",
    local_model_id: Optional[str] = None,
    session_id: Optional[str] = None,
):
    """Configure global model settings for the current agent session.

    Args:
        use_api_mode: Whether to use API-backed models.
        model_suite: Local suite or API provider name.
        local_model_id: Local model id or API model name depending on mode.
        session_id: Optional session id for per-session API state.
    """
    global USE_API_MODE, LOCAL_MODEL_SUITE, embed_model, proj_llm, code_llm, IMAGE_CAPTION_LLM, _ACTIVE_MODEL_CONFIG
    global img_analysis_llm, media_analysis_llm, img_gen_model, img_edit_model

    resolved_use_api = bool(use_api_mode)

    resolved_model_id = local_model_id
    config = {
        "use_api_mode": resolved_use_api,
        "model_suite": model_suite,
        "local_model_id": resolved_model_id,
        "session_id": session_id if resolved_use_api else None,
    }

    if _ACTIVE_MODEL_CONFIG == config:
        return

    USE_API_MODE = resolved_use_api
    LOCAL_MODEL_SUITE = model_suite
    # In API mode, local_model_id contains the API model name (e.g., gemini-3-flash-preview)
    # In local mode, it contains the HuggingFace model ID
    api_model_name = resolved_model_id if use_api_mode else None
    
    if use_api_mode:
        # API mode: only main models
        embed_model, proj_llm, code_llm = initialize_models(
            use_api_mode=USE_API_MODE,
            model_suite=LOCAL_MODEL_SUITE,
            local_model_id=None,
            api_model_name=api_model_name,
            session_id=session_id,
        )
        img_analysis_llm = None
        media_analysis_llm = None
        img_gen_model = None
        img_edit_model = None
    else:
        # Local mode: all specialized models
        result = initialize_models(
            use_api_mode=USE_API_MODE,
            model_suite=LOCAL_MODEL_SUITE,
            local_model_id=local_model_id,
            api_model_name=None,
            session_id=session_id,
        )
        embed_model, proj_llm, code_llm, img_analysis_llm, media_analysis_llm, img_gen_model, img_edit_model = result
    
    Settings.llm = proj_llm
    Settings.embed_model = embed_model
    # All models (API and Local) are now multimodal
    IMAGE_CAPTION_LLM = proj_llm
    _ACTIVE_MODEL_CONFIG = config

    logger.info(
        "Model configuration set: use_api_mode=%s model_suite=%s model_id=%s",
        USE_API_MODE,
        LOCAL_MODEL_SUITE,
        resolved_model_id,
    )

# Setup logging - force INFO logs to stdout so they are visible in notebooks/terminals
root_logger = logging.getLogger()
# Remove any existing StreamHandlers (they may be configured to stderr or filtered)
for h in list(root_logger.handlers):
    if isinstance(h, logging.StreamHandler):
        root_logger.removeHandler(h)

# Create a StreamHandler that writes to stdout and set it to INFO
sh = logging.StreamHandler(sys.stdout)
sh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
sh.setLevel(logging.INFO)
root_logger.addHandler(sh)
root_logger.setLevel(logging.INFO)
# Ensure basicConfig is set for libraries that check it
logging.basicConfig(level=logging.INFO)

# Make llama_index and related libraries more verbose for debugging (they will propagate to root handler)
logging.getLogger("llama_index").setLevel(logging.DEBUG)
logging.getLogger("llama_index.core.agent").setLevel(logging.DEBUG)
logging.getLogger("llama_index.llms").setLevel(logging.DEBUG)

# Module logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class LoggingQueryEngine(BaseQueryEngine):
    """Wrap a BaseQueryEngine to add structured logging."""

    def __init__(self, inner_engine: BaseQueryEngine, name: str = "dynamic_hybrid_multimodal_rag_tool"):
        self._inner_engine = inner_engine
        self._logger = logging.getLogger(__name__)
        self._tool_name = name

    def query(self, query: str, **kwargs):
        self._logger.info("%s query received: %s", self._tool_name, query)
        try:
            response = self._inner_engine.query(query, **kwargs)
            self._logger.info("%s query succeeded", self._tool_name)
            return response
        except Exception:
            self._logger.exception("%s query failed", self._tool_name)
            raise

    async def aquery(self, query: str, **kwargs):
        self._logger.info("%s async query received: %s", self._tool_name, query)
        try:
            response = await self._inner_engine.aquery(query, **kwargs)
            self._logger.info("%s async query succeeded", self._tool_name)
            return response
        except Exception:
            self._logger.exception("%s async query failed", self._tool_name)
            raise

    def _query(self, query: str, **kwargs):
        # BaseQueryEngine calls this; delegate to wrapped engine.
        if hasattr(self._inner_engine, "_query"):
            return self._inner_engine._query(query, **kwargs)
        return self._inner_engine.query(query, **kwargs)

    async def _aquery(self, query: str, **kwargs):
        # BaseQueryEngine calls this; delegate to wrapped engine.
        if hasattr(self._inner_engine, "_aquery"):
            return await self._inner_engine._aquery(query, **kwargs)
        return await self._inner_engine.aquery(query, **kwargs)

    def _get_prompt_modules(self):
        if hasattr(self._inner_engine, "_get_prompt_modules"):
            return self._inner_engine._get_prompt_modules()
        return {}

    def __getattr__(self, item):
        # Delegate everything else to the wrapped engine (streaming, metadata, etc.)
        return getattr(self._inner_engine, item)


# Use environment variables to determine API mode
USE_API_MODE = os.environ.get("USE_API_MODE", "false").lower() == "true"
LOCAL_MODEL_SUITE = os.environ.get("LOCAL_MODEL_SUITE", "qwen").lower()

# Lazy-initialized globals (configured in configure_models)
embed_model = None
proj_llm = None
code_llm = None
IMAGE_CAPTION_LLM = None
img_analysis_llm = None
media_analysis_llm = None
img_gen_model = None
img_edit_model = None

@weave.op
def read_and_parse_content(input_path: str) -> List[Document]:
    """Parse a file into LlamaIndex Documents.

    This handles Docling for PDFs and Office files, specialized readers for
    CSV/Excel/JSON/text, and multimodal LLM analysis for images and media.

    Args:
        input_path: Path to the file on disk.

    Returns:
        List of Documents. On error, returns a single Document with an error
        message in its text.
    """
    # Standard document parsing
    if not os.path.exists(input_path):
        return [Document(text=f"Error: File not found at {input_path}")]

    file_extension = os.path.splitext(input_path)[1].lower()

    # PDFs only: Use DoclingReader for layout-aware extraction
    if file_extension == '.pdf':
        try:
            logger.info(f"Using DoclingReader for PDF file: {input_path}")
            reader = DoclingReader(export_type=DoclingReader.ExportType.MARKDOWN)
            documents = reader.load_data(input_path)

            # Add metadata
            for doc in documents:
                doc.metadata["source"] = input_path
                doc.metadata["loader"] = "docling"
                doc.metadata["file_type"] = "pdf"

            logger.info(f"DoclingReader extracted {len(documents)} documents from PDF")
            return documents

        except Exception as e:
            logger.exception(f"DoclingReader failed for PDF {input_path}: {e}")
            return [Document(text=f"Error loading PDF with DoclingReader: {e}")]
    
    # Office documents: Use DoclingReader
    if file_extension in ['.docx', '.doc', '.pptx', '.ppt', '.html', '.htm']:
        try:
            logger.info(f"Using DoclingReader for {file_extension} file")
            reader = DoclingReader(export_type=DoclingReader.ExportType.MARKDOWN)
            documents = reader.load_data(input_path)

            # Add metadata
            for doc in documents:
                doc.metadata["source"] = input_path
                doc.metadata["loader"] = "docling"
                doc.metadata["file_type"] = file_extension[1:]

            logger.info(f"DoclingReader extracted {len(documents)} documents")
            return documents

        except Exception as e:
            logger.exception(f"DoclingReader failed for {input_path}: {e}")
            return [Document(text=f"Error loading file with DoclingReader: {e}")]

    # Readers map for structured data
    readers_map = {
        '.csv': PandasCSVReader(),
        '.xlsx': PandasExcelReader(),
    }

    # Images: Use multimodal LLM natively
    if file_extension in ['.jpg', '.jpeg', '.png', '.gif', '.webp']:
        try:
            image_llm = img_analysis_llm or IMAGE_CAPTION_LLM
            if image_llm is None:
                return [Document(text=f"Multimodal model not available for image processing")]
            
            logger.info(f"Using multimodal LLM to describe image: {input_path}")
            
            prompt = (
                "Provide a clear, objective description of the image. Include: main objects, "
                "their attributes, setting/context, notable text (if any), and overall scene summary."
            )
            
            # Use native multimodal completion
            img_doc = ImageDocument(image_path=input_path, metadata={"source": input_path, "type": "input_image"})
            resp = image_llm.complete(prompt, image_documents=[img_doc])
            description = getattr(resp, "text", str(resp)).strip()
            
            return [Document(
                text=description,
                metadata={
                    "source": input_path,
                    "type": "image_description",
                    "path": input_path
                }
            )]
        except Exception as e:
            logger.exception(f"Multimodal image processing failed: {e}")
            return [Document(text=f"Error processing image with LLM: {e}")]

    # Audio/Video files: Use multimodal LLM natively
    if file_extension in ['.mp3', '.mp4', '.wav', '.m4a']:
        try:
            media_llm = media_analysis_llm
            if USE_API_MODE and LOCAL_MODEL_SUITE == "gemini":
                media_llm = IMAGE_CAPTION_LLM
            elif USE_API_MODE and LOCAL_MODEL_SUITE == "openai":
                return [Document(text="OpenAI does not support audio/video inputs in this app.")]
            if media_llm is None:
                return [Document(text="Multimodal model not available for audio/video processing")]

            logger.info(f"Using multimodal LLM to process audio/video: {input_path}")

            is_video = file_extension in ['.mp4']
            prompt = (
                f"Provide a detailed {'transcription and summary' if not is_video else 'description and transcription'} "
                f"of this {'video' if is_video else 'audio'} file. "
                "Include key points, speakers (if identifiable), and any important context."
            )

            # Use native multimodal completion
            media_doc = ImageDocument(image_path=input_path, metadata={"source": input_path, "type": "input_media"})
            resp = media_llm.complete(prompt, image_documents=[media_doc])
            description = getattr(resp, "text", str(resp)).strip()

            return [Document(
                text=description,
                metadata={
                    "source": input_path,
                    "type": "audio_video_description",
                    "path": input_path
                }
            )]
        except Exception as e:
            logger.exception(f"Multimodal audio/video processing failed: {e}")
            return [Document(text=f"Error processing audio/video with LLM: {e}")]

    # Use appropriate reader for supported file types
    documents: List[Document] = []

    if file_extension == ".json":
        try:
            with open(input_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            content = json.dumps(data, ensure_ascii=True, indent=2, sort_keys=True)
            documents = [Document(text=content, metadata={"source": input_path})]
        except Exception as e:
            return [Document(text=f"Error loading JSON: {e}")]
    elif file_extension in readers_map:
        loader = readers_map[file_extension]
        try:
            documents = loader.load_data(input_path)
        except Exception as e:
            return [Document(text=f"Error loading file with reader: {e}")]
    else:
        # Fallback for text files
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                content = f.read()
            documents = [Document(text=content, metadata={"source": input_path})]
        except Exception as e:
            return [Document(text=f"Error reading file as plain text: {e}")]

    # Add source metadata
    for doc in documents:
        doc.metadata["source"] = input_path

    return documents

# ============================================================================
# Temporary Web Search RAG (In-Memory Only)
# ============================================================================

@weave.op
def create_temporary_web_search_index(documents: List[Document], llm=None) -> Optional[VectorStoreIndex]:
    """Create a temporary in-memory RAG index for web search results.

    The index is not persisted and is garbage collected after use.

    Args:
        documents: Documents extracted from web search.
        llm: Optional LLM instance for the query engine.

    Returns:
        VectorStoreIndex for querying, or None if creation fails.
    """
    if not documents:
        logger.warning("No documents provided for temporary index")
        return None
        
    try:
        embed_model = get_or_create_jina_embedder()
        
        # Simple hierarchical parsing for web content
        node_parser = HierarchicalNodeParser.from_defaults(
            chunk_sizes=[2048, 512, 256],
            chunk_overlap=128
        )
        
        all_nodes = node_parser.get_nodes_from_documents(documents)
        leaf_nodes = get_leaf_nodes(all_nodes)
        
        logger.info(f"Created temporary index with {len(leaf_nodes)} leaf nodes")
        
        # Create in-memory index (no persistence)
        index = VectorStoreIndex(
            nodes=leaf_nodes,
            embed_model=embed_model
        )
        
        return index
        
    except Exception as e:
        logger.exception("Failed to create temporary web search index: %s", e)
        return None

def _extract_urls_from_results(results: List[Dict[str, Any]], max_results: int) -> List[str]:
    urls: List[str] = []
    for result in results:
        url = result.get("href") or result.get("link") or result.get("url") or result.get("FirstURL") or result.get("first_url")
        if not url:
            continue
        url = str(url).rstrip(").,;:'\"")
        if url not in urls:
            urls.append(url)
        if len(urls) >= max_results:
            break
    return urls

def search_for_urls(query: str, max_results: int = 3) -> List[str]:
    logger.info("[web] start search: %s", query)
    ddgs_errors: List[str] = []
    backend_attempts = [
        ("google", {"backend": "google"}),
        ("default", {}),
    ]

    for backend_name, backend_kwargs in backend_attempts:
        try:
            kwargs = {"max_results": max_results}
            kwargs.update(backend_kwargs)
            with DDGS() as ddg:
                results = list(ddg.text(query, **kwargs))
            logger.info("[web] ddgs backend='%s' results: %d", backend_name, len(results))
            urls = _extract_urls_from_results(results, max_results)
            if urls:
                return urls
        except Exception as e:
            ddgs_errors.append(f"{backend_name}: {e}")
            logger.warning("[web] ddgs failed (backend=%s): %s", backend_name, e)
    if ddgs_errors:
        logger.warning("[web] ddgs attempts exhausted: %s", "; ".join(ddgs_errors))

    logger.info("[web] no URLs extracted (blocked/timeout/empty)")
    return []

def extract_documents_from_url(url: str) -> List[Document]:
    url = str(url).rstrip(").,;:'\"")
    logger.info("[web] fetching url: %s", url)
    try:
        netloc = urlparse(url).netloc.lower()
        if "youtube" in netloc or "youtu.be" in netloc:
            logger.info("[web] using YoutubeTranscriptReader()")
            loader = YoutubeTranscriptReader()
            documents = loader.load_data(youtubelinks=[url])
        else:
            logger.info("[web] using SimpleWebPageReader()")
            default_loader = SimpleWebPageReader(html_to_text=True, fail_on_error=True)
            ua_loader = UserAgentWebPageReader(html_to_text=True, fail_on_error=True)
            try:
                documents = default_loader.load_data(urls=[url])
            except Exception as exc:
                logger.warning("[web] default web reader failed; retrying with User-Agent: %s", exc)
                documents = ua_loader.load_data(urls=[url])

        for doc in documents:
            if not getattr(doc, "metadata", None):
                doc.metadata = {}
            doc.metadata["source"] = url
            doc.metadata["type"] = "web_text"

        logger.info(f"[web] extracted {len(documents)} documents from {url}")
        return documents
    except Exception as e:
        logger.warning(f"[web] fetch failed for {url}: {e}")
        return [Document(text=f"Error extracting content from URL: {e}",
                         metadata={"type": "web_text", "source": url, "error": True})]

def search_and_extract_content_from_url(query: str, max_results: int = 3) -> List[Document]:
    urls = search_for_urls(query, max_results=max_results)
    if not urls:
        return [Document(text="No URL could be extracted from the search results.",
                         metadata={"type": "web_text", "source": "search", "error": True})]

    documents: List[Document] = []
    for url in urls:
        documents.extend(extract_documents_from_url(url))
    return documents

def format_web_search_documents(documents: List[Document]) -> str:
    sources: List[str] = []
    text_by_source: Dict[str, List[str]] = {}
    for doc in documents:
        metadata = doc.metadata or {}
        source = metadata.get("source", "unknown")
        if source not in text_by_source:
            text_by_source[source] = []
            sources.append(source)
        text_by_source[source].append(doc.text or "")

    parts: List[str] = []
    for source in sources:
        body = "\n\n".join(text_by_source[source]).strip()
        if not body:
            continue
        parts.append(f"Source: {source}\n\n{body}")
    return "\n\n---\n\n".join(parts)

def enhanced_web_search_and_query(query: str) -> str:
    """Run web search and return either raw pages or a RAG answer.

    API mode returns raw page content for multiple results. Local mode builds
    a temporary in-memory index and answers the query.

    Args:
        query: The search query.

    Returns:
        A string containing either raw page content or an answer.
    """
    # Extract content from web search
    documents = search_and_extract_content_from_url(query, max_results=3)

    valid_documents = [
        doc for doc in documents
        if doc.text and not (doc.metadata or {}).get("error")
    ]
    if not valid_documents:
        error_msg = documents[0].text if documents else "No content extracted"
        logger.warning("Web search failed for query '%s': %s", query, error_msg)
        return f"Failed to extract web content: {error_msg}"

    if USE_API_MODE:
        logger.info("API mode web search: returning raw content for %d documents", len(valid_documents))
        return format_web_search_documents(valid_documents)

    logger.info("Creating temporary index for %d documents from web search", len(valid_documents))
    
    # Create temporary in-memory index
    temp_index = create_temporary_web_search_index(valid_documents, llm=proj_llm)
    
    if temp_index is None:
        return "Failed to create temporary search index"
    
    # Query the temporary index
    try:
        reranker = get_or_create_jina_reranker(model_name="jinaai/jina-reranker-m0", top_n=5)
        
        query_engine = temp_index.as_query_engine(
            similarity_top_k=10,
            node_postprocessors=[reranker],
            response_mode="tree_summarize",
            llm=proj_llm
        )
        
        response = query_engine.query(query)
        result_text = str(response)
        
        logger.info("Temporary web search RAG completed successfully")
        return result_text
        
    except Exception as e:
        logger.exception("Failed to query temporary index: %s", e)
        # Fallback: return raw document content
        text_docs = [doc for doc in documents if doc.metadata.get("type") == "web_text"]
        if text_docs:
            return text_docs[0].text[:2000]  # Return first 2000 chars
        return "Error processing web search results"

def make_enhanced_web_search_tool():
    """Create a web search tool with API/raw and local/RAG behavior.

    Returns:
        Callable tool function that accepts a query string.
    """
    def enhanced_web_search(query: str) -> str:
        """Perform web search and return raw pages (API) or a RAG answer (local)."""
        logger.info(f"enhanced_web_search called with query: {query}")
        return enhanced_web_search_and_query(query)

    enhanced_web_search.__name__ = "enhanced_web_search"
    return enhanced_web_search

def safe_import(module_name):
    """Import a module by name, returning None on ImportError.

    Args:
        module_name: Fully qualified module name.

    Returns:
        Imported module object, or None if import fails.
    """
    try:
        return __import__(module_name, fromlist=[''])
    except ImportError:
        return None

# Restricted global namespace for code execution
safe_globals = {
    "__builtins__": {
        k: v for k, v in __builtins__.items() 
        if k in {
            "len", "str", "int", "float", "list", "dict", "sum", "max", "min",
            "round", "abs", "sorted", "enumerate", "range", "zip", "map", "filter",
            "any", "all", "type", "isinstance", "print", "bool", "set", "tuple"
        }
    }
}

# Modules to make available in the execution environment
modules_to_load = {
    # Core
    "math": "math", "datetime": "datetime", "re": "re", "os": "os", "sys": "sys", 
    "json": "json", "csv": "csv", "random": "random", "collections": "collections",
    "itertools": "itertools", "functools": "functools", "pathlib": "pathlib",
    # Data Science / ML
    "np": "numpy", "pd": "pandas", "plt": "matplotlib.pyplot", "sns": "seaborn",
    "sklearn": "sklearn", "scipy": "scipy", "requests": "requests", "bs4": "bs4",
    "PIL": "PIL", "yaml": "yaml", "tqdm": "tqdm"
}

for alias, name in modules_to_load.items():
    mod = safe_import(name)
    if mod:
        safe_globals[alias] = mod
        if alias != name:
            safe_globals[name] = mod

# Special cases for common submodules/classes
if "bs4" in safe_globals:
    safe_globals["BeautifulSoup"] = safe_globals["bs4"].BeautifulSoup
if "PIL" in safe_globals:
    from PIL import Image
    safe_globals["Image"] = Image


def execute_python_code(code: str) -> str:
    """Execute Python code in a restricted global namespace.

    Captures stdout (print statements) and the value of a `result` variable,
    if defined in the executed code.

    Args:
        code: Python source code to execute.

    Returns:
        Combined stdout and result value, or an error message on failure.
    """
    output_buffer = io.StringIO()
    try:
        exec_locals = {}
        with contextlib.redirect_stdout(output_buffer):
            exec(code, safe_globals, exec_locals)
        
        stdout = output_buffer.getvalue().strip()
        result_val = exec_locals.get('result')
        
        response_parts = []
        if stdout:
            response_parts.append(stdout)
        if result_val is not None:
            response_parts.append(f"Result: {result_val}")
            
        if not response_parts:
            return "Code executed successfully (no output)"
        return "\n".join(response_parts)
        
    except Exception as e:
        return f"Code execution failed: {str(e)}"

# Pydantic model for structured output
class StructuredResponse(BaseModel):
    """Structured answer format with validation.

    Attributes:
        reasoning: Step-by-step reasoning used to derive the answer.
        final_answer: Exact answer in concise format.
        confidence: Confidence score between 0 and 1.
    """
    reasoning: str = Field(
        description="Step-by-step reasoning process used to arrive at the answer"
    )
    final_answer: str = Field(
        description="Exact answer in concise format"
    )
    confidence: float = Field(
        description="Confidence score between 0 and 1",
        ge=0.0,
        le=1.0,
        default=0.8
    )

def llm_reformat(response: str, question: str) -> str:
    """Extract a concise final answer from an LLM response.

    Args:
        response: Full model response text.
        question: Original user question.

    Returns:
        Extracted final answer, or the original response on failure.
    """

    format_prompt = """Extract the exact answer from the response below.

Now extract the exact answer:
Question: {query_str}
Response: {context}

Provide your reasoning, then the exact answer."""

    try:
        # Create Pydantic program
        program = LLMTextCompletionProgram.from_defaults(
            output_cls=StructuredResponse,
            prompt_template_str=format_prompt,
            llm=proj_llm,
            verbose=True
        )

        # Get structured response
        response_obj = program(
            query_str=question,
            context=response
        )

        logger.info(f"Structured response - Confidence: {response_obj.confidence:.2f}")
        logger.info(f"Reasoning: {response_obj.reasoning[:100]}...")

        return response_obj.final_answer

    except Exception as e:
        logger.exception("Pydantic parsing failed: %s", e)
        return response.strip()

class ConversationalAgent:
    def __init__(
        self,
        use_api_mode: Optional[bool] = None,
        model_suite: str = "qwen",
        local_model_id: Optional[str] = None,
        session_id: Optional[str] = None,
        media_analysis_enabled: bool = False,
        code_execution_enabled: bool = True,
        use_specialized_code_model: bool = False,
        img_generation_enabled: bool = False,
        img_editing_enabled: bool = False
    ):
        logger.info("Initializing Conversational Agent...")

        if use_api_mode is None:
            use_api_mode = USE_API_MODE
        if not model_suite:
            model_suite = LOCAL_MODEL_SUITE

        self.session_id = session_id

        configure_models(
            use_api_mode=use_api_mode,
            model_suite=model_suite,
            local_model_id=local_model_id,
            session_id=self.session_id,
        )

        # Vérification du token HuggingFace
        hf_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        if not hf_token:
            logger.warning("Warning: HUGGINGFACEHUB_API_TOKEN not found, some features may not work")

        # Initialize VectorStoreManager for conversation caches
        from utils.vector_store import VectorStoreManager
        self.vector_store_manager = VectorStoreManager(
            conversations_dir="./chroma_db/conversations"
        )
        
        # Web search tool (uses temporary in-memory RAG)
        self.web_tool = make_enhanced_web_search_tool()
        self.web_function_tool = FunctionTool.from_defaults(
            fn=self.web_tool,
            name="enhanced_web_search"
        )

        # Code execution tool
        self.exec_function_tool = FunctionTool.from_defaults(
            fn=execute_python_code,
            name="execute_python_code",
        )

        # Build agent list based on configuration
        agents_list = []
        
        # External knowledge agent (always present, root agent)
        external_knowledge_system_prompt = """You are an expert research assistant specialized in information gathering and knowledge retrieval. Your role is to help users find accurate information by searching the web, analyzing documents, and processing various types of content. Always provide comprehensive and well-researched answers."""
        
        self.external_knowledge_agent = ReActAgent(
            tools=[self.web_function_tool],
            llm=proj_llm,
            max_steps=8,
            system_prompt=external_knowledge_system_prompt,
            verbose=True,
            name="external_knowledge_agent",
            description="Handles external knowledge retrieval, web searches, and document processing"
        )
        agents_list.append(self.external_knowledge_agent)
        
        # Determine which specialized agents to activate
        use_api = use_api_mode
        
        # Code execution agent (conditional on user activation)
        if code_execution_enabled:
            # Determine which LLM to use for code execution
            if use_api:
                # API mode: use OpenAI gpt-5.2-codex if OpenAI AND use_specialized_code_model, else main model
                if model_suite == "openai" and use_specialized_code_model:
                    from custom_models import get_or_create_openai_llm
                    code_exec_llm = get_or_create_openai_llm(
                        model_name="gpt-5.2-codex",
                        session_id=self.session_id,
                    )
                    logger.info("Using OpenAI gpt-5.2-codex for code execution")
                else:
                    # Use main model (proj_llm for Gemini, or OpenAI if specialized model not requested)
                    code_exec_llm = proj_llm
                    logger.info(f"Using main model ({model_suite}) for code execution")
            else:
                # Local mode: always use Devstral (code_llm)
                code_exec_llm = code_llm
            
            code_system_prompt = """You are a skilled programming assistant and data analyst. Your expertise includes Python programming, data analysis, mathematical computations, and problem-solving through code execution. You can write, execute, and debug Python code to solve complex problems, perform calculations, analyze data, and generate insights. Always write clean, efficient, and well-documented code.

The following modules are already available in your execution environment (no need to import them):
- Data Science: np (numpy), pd (pandas), plt (matplotlib.pyplot), sns (seaborn), sklearn, scipy
- Utilities: requests, bs4 (BeautifulSoup), PIL (Image), yaml, tqdm
- Core: math, datetime, re, os, sys, json, csv, random, collections, itertools, functools, pathlib (and all other standard Python built-ins)

If you need to return a text result or a specific value, assign it to a variable named 'result'.
"""
            self.code_agent = ReActAgent(
                tools=[self.exec_function_tool],
                llm=code_exec_llm,
                max_steps=6,
                system_prompt=code_system_prompt,
                verbose=True,
                name="code_execution_agent",
                description="Executes Python code, performs calculations, data analysis, and mathematical operations"
            )
            agents_list.append(self.code_agent)
            logger.info("Code execution agent activated")
        
        # Image analysis agent (local mode, Qwen suite only - Ministral handles images natively)
        # In API mode, images are handled natively by external_knowledge_agent
        if not use_api and model_suite == "qwen" and img_analysis_llm is not None:
            self.img_analysis_agent = ReActAgent(
                tools=[],
                llm=img_analysis_llm,
                max_steps=4,
                system_prompt="You are an expert in visual content analysis. Analyze images and provide detailed descriptions.",
                verbose=True,
                name="img_analysis_agent",
                description="Analyzes visual content and images using Qwen-VL"
            )
            agents_list.append(self.img_analysis_agent)
            logger.info("Image analysis agent activated (Qwen-VL)")
        
        # Media analysis agent (local mode only, conditional on user activation)
        # In API mode, audio/video are handled natively by external_knowledge_agent
        if not use_api and media_analysis_enabled and media_analysis_llm is not None:
            self.med_analysis_agent = ReActAgent(
                tools=[],
                llm=media_analysis_llm,
                max_steps=4,
                system_prompt="You are an expert in audio and video analysis. Transcribe and analyze multimedia content.",
                verbose=True,
                name="med_analysis_agent",
                description="Analyzes audio and video content using Qwen-Omni"
            )
            agents_list.append(self.med_analysis_agent)
            logger.info("Media analysis agent activated (Qwen-Omni)")
        
        # Image generation agent (conditional on user activation)
        if img_generation_enabled:
            if use_api:
                # API mode: specialized models/tools per provider
                if model_suite == "gemini":
                    # Gemini: use gemini-3-pro-image-preview with image_config
                    from custom_models import get_or_create_gemini_llm
                    img_gen_llm = get_or_create_gemini_llm(model_name="gemini-3-pro-image-preview")
                    self.img_generation_agent = ReActAgent(
                        tools=[],
                        llm=img_gen_llm,
                        max_steps=3,
                        system_prompt="You are an expert in image generation. Create images based on user descriptions using Gemini's image generation capabilities.",
                        verbose=True,
                        name="img_generation_agent",
                        description="Generates images from text descriptions using Gemini"
                    )
                    agents_list.append(self.img_generation_agent)
                    logger.info("Image generation agent activated (Gemini gemini-3-pro-image-preview)")
                elif model_suite == "openai":
                    # OpenAI: use proj_llm with tools=[{"type": "image_generation"}]
                    # Note: The tools parameter will be passed when calling the API
                    self.img_generation_agent = ReActAgent(
                        tools=[],
                        llm=proj_llm,
                        max_steps=3,
                        system_prompt="You are an expert in image generation. Create images based on user descriptions using OpenAI's image_generation tool.",
                        verbose=True,
                        name="img_generation_agent",
                        description="Generates images from text descriptions using OpenAI"
                    )
                    # Store reference for OpenAI image generation tool
                    self.img_generation_agent._use_image_tool = True
                    agents_list.append(self.img_generation_agent)
                    logger.info(f"Image generation agent activated (OpenAI with image_generation tool)")
                else:
                    logger.warning("Image generation not supported for this provider, skipping")
            else:
                # Local mode: use Qwen image generator
                if img_gen_model is not None:
                    # Note: For local, we use the main LLM with access to img_gen_model
                    self.img_generation_agent = ReActAgent(
                        tools=[],
                        llm=proj_llm,
                        max_steps=3,
                        system_prompt="You are an expert in image generation. Create images based on user descriptions using Qwen image generation model.",
                        verbose=True,
                        name="img_generation_agent",
                        description="Generates images from text descriptions using Qwen"
                    )
                    # Store reference to actual generator
                    self.img_generation_agent._generator = img_gen_model
                    agents_list.append(self.img_generation_agent)
                    logger.info("Image generation agent activated (Qwen)")
        
        # Image editing agent (conditional on user activation)
        if img_editing_enabled:
            if use_api:
                # API mode: specialized models/tools per provider
                if model_suite == "gemini":
                    # Gemini: use gemini-3-pro-image-preview with image_config
                    from custom_models import get_or_create_gemini_llm
                    img_edit_llm = get_or_create_gemini_llm(model_name="gemini-3-pro-image-preview")
                    self.img_editing_agent = ReActAgent(
                        tools=[],
                        llm=img_edit_llm,
                        max_steps=3,
                        system_prompt="You are an expert in image editing. Modify images based on user instructions using Gemini's image editing capabilities.",
                        verbose=True,
                        name="img_editing_agent",
                        description="Edits and modifies existing images using Gemini"
                    )
                    agents_list.append(self.img_editing_agent)
                    logger.info("Image editing agent activated (Gemini gemini-3-pro-image-preview)")
                elif model_suite == "openai":
                    # OpenAI: use proj_llm with tools (image editing may use different tool/approach)
                    self.img_editing_agent = ReActAgent(
                        tools=[],
                        llm=proj_llm,
                        max_steps=3,
                        system_prompt="You are an expert in image editing. Modify images based on user instructions using OpenAI's image tools.",
                        verbose=True,
                        name="img_editing_agent",
                        description="Edits and modifies existing images using OpenAI"
                    )
                    self.img_editing_agent._use_image_tool = True
                    agents_list.append(self.img_editing_agent)
                    logger.info("Image editing agent activated (OpenAI)")
                else:
                    logger.warning("Image editing not supported for this provider, skipping")
            else:
                # Local mode: use Qwen image editor
                if img_edit_model is not None:
                    self.img_editing_agent = ReActAgent(
                        tools=[],
                        llm=proj_llm,
                        max_steps=3,
                        system_prompt="You are an expert in image editing. Modify images based on user instructions using Qwen image editing model.",
                        verbose=True,
                        name="img_editing_agent",
                        description="Edits and modifies existing images using Qwen"
                    )
                    # Store reference to actual editor
                    self.img_editing_agent._editor = img_edit_model
                    agents_list.append(self.img_editing_agent)
                    logger.info("Image editing agent activated (Qwen)")

        # Fixed indentation: coordinator initialization inside __init__
        self.coordinator = AgentWorkflow(
            agents=agents_list,
            root_agent="external_knowledge_agent",
        )
        
        logger.info(f"Coordinator initialized with {len(agents_list)} agents")

    @weave.op
    def _record_steps(self, deltas: List[str]) -> int:
        """Record streamed deltas as a single Weave op.

        Args:
            deltas: Streamed token deltas.

        Returns:
            Count of deltas recorded.
        """
        return len(deltas)

    @weave.op
    def run(self, query: str, max_steps: Optional[int] = None) -> str:
        """Run the agent on a user query.

        Args:
            query: User question or instruction.
            max_steps: Maximum reasoning steps (kept for API compatibility).

        Returns:
            Agent response as a string.
        """
        try:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                import nest_asyncio
                nest_asyncio.apply()
                response = loop.run_until_complete(self.process_query(query))
            else:
                response = asyncio.run(self.process_query(query))

            return str(response)
        except Exception as e:
            error_msg = f"Error during agent execution: {e}"
            logger.exception(error_msg)
            return error_msg

    async def process_query(self, query: str) -> str:
        """Process a query with knowledge base integration.

        Uses a non-streaming await pattern for workflow execution.

        Args:
            query: User question or instruction.

        Returns:
            Final answer string formatted by the post-processor.
        """
        context_prompt = f"""Question: {query}

    You are a helpful AI assistant. I will ask you a question.

    IMPORTANT INSTRUCTIONS:
    1. Think through this STEP BY STEP, carefully analyzing all aspects of the question.
    2. Pay special attention to specific qualifiers like dates, types, categories, or locations.
    3. Make sure your searches include ALL important details from the question.
    4. Report your thoughts and reasoning process clearly.
    5. Finish your answer with: FINAL ANSWER: [YOUR FINAL ANSWER]
    """.strip()

        try:
            ctx = Context(self.coordinator)
            logger.info("=== AGENT REASONING STEPS ===")
            stats = self.vector_store_manager.get_stats()
            logger.info("Cached sources available: %s", stats.get("library_sources", 0))

            handler = self.coordinator.run(ctx=ctx, user_msg=context_prompt)
            full_response = ""
            step_deltas: List[str] = []

            async for event in handler.stream_events():
                if isinstance(event, AgentStream):
                    # preserve streaming behavior (no newline, immediate flush)
                    sys.stdout.write(event.delta)
                    sys.stdout.flush()
                    full_response += event.delta
                    step_deltas.append(event.delta)

            final_response = await handler
            logger.info("\n=== END REASONING ===")

            # Extract the final formatted answer
            final_answer = llm_reformat(str(final_response), query)
            self._record_steps(step_deltas)

            return final_answer
        except Exception as e:
            error_msg = f"Error processing question: {str(e)}"
            logger.exception(error_msg)
            return error_msg

    def get_knowledge_base_stats(self):
        """Return statistics about the current knowledge base."""
        return self.vector_store_manager.get_stats()
