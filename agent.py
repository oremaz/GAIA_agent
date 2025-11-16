# Standard library imports
import logging
import os
import sys
import re
import hashlib
from typing import Dict, Any, List, Optional, Set
from urllib.parse import urlparse
import torch
import asyncio
import mimetypes
# Using PyMuPDFReader from llama_index.readers.file instead of direct fitz usage

# Third-party imports
import requests
# Third-party imports
from custom_models import (
    get_or_create_jina_reranker,
    get_or_create_jina_embedder,
    get_or_create_qwen_vl_llm,
    get_or_create_qwen_coder_llm,
)

# LlamaIndex core imports
from llama_index.core import VectorStoreIndex, Document, Settings, PromptTemplate
from llama_index.core.tools import FunctionTool
from llama_index.core.agent.workflow import ReActAgent, AgentStream
from llama_index.core.node_parser import UnstructuredElementNodeParser, SentenceSplitter
from llama_index.core.workflow import Context
from llama_index.core.schema import ImageNode, TextNode, ImageDocument

# LlamaIndex specialized imports
from llama_index.readers.assemblyai import AssemblyAIAudioTranscriptReader
from llama_index.readers.json import JSONReader
from llama_index.readers.web import BeautifulSoupWebReader
from llama_index.readers.youtube_transcript import YoutubeTranscriptReader
from llama_index.core.agent.workflow import AgentWorkflow

# Optional utilities that may come from external packages or later imports
try:
    from llama_index.text_splitter import RecursiveCharacterTextSplitter
except Exception:
    RecursiveCharacterTextSplitter = None

try:
    from llama_index.tools.bm25 import BM25RetrieverTool
except Exception:
    BM25RetrieverTool = None

# Import all required official LlamaIndex Readers
from llama_index.readers.file import (
    PyMuPDFReader,
    DocxReader,
    CSVReader,
    PandasExcelReader,
    VideoAudioReader  # Adding VideoAudioReader for handling audio/video without API
)
from llama_index.readers.smart_pdf_loader import SmartPDFLoader
import weave
from ddgs import DDGS
from transformers import AutoTokenizer, AutoModelForCausalLM
import json


class MultimodalPDFReader:
    """Reader that combines SmartPDFLoader (layout-aware text/tables) with
    PyMuPDF (fitz) image extraction. Falls back to PyMuPDFReader if SmartPDFLoader
    is not available, and to a plain-text fallback as last resort.
    """

    def __init__(self):
        pass

    def load_data(self, file_path: str) -> List[Document]:
        docs: List[Document] = []

        # 1) Text / layout-aware parsing: SmartPDFLoader
        loader = SmartPDFLoader()
        text_docs = loader.load_data(file_path)
        if text_docs:
            docs.extend(text_docs)

        # 2) Use the LlamaIndex PyMuPDFReader for page-level text (and any image
        #    documents it may provide). PyMuPDFReader returns Documents where
        #    page text is often bytes and extra_info holds metadata; normalize
        #    both metadata and extra_info here.
        py_reader = PyMuPDFReader()
        py_docs = py_reader.load_data(file_path)

        for d in py_docs:
            # Normalize metadata and extra_info into a single dict
            md = getattr(d, "metadata", {}) or {}
            extra = getattr(d, "extra_info", {}) or {}
            combined = {**extra, **md}

            # Decode bytes text if necessary
            text = d.text
            if isinstance(text, (bytes, bytearray)):
                try:
                    text = text.decode("utf-8")
                except Exception:
                    text = text.decode("latin-1", errors="ignore")

            # Determine if this doc represents an image (some readers may
            # surface image docs via extra_info or metadata)
            is_image = combined.get("type") in ("image", "web_image") or "image_data" in combined

            if is_image:
                # Preserve any image_data if present
                docs.append(Document(text=text or "IMAGE_CONTENT_BINARY", metadata=combined))
        # Ensure source metadata exists
        for d in docs:
            if not d.metadata.get("source"):
                d.metadata["source"] = file_path

        return docs

## GPTOSSWrapper moved to custom_models; using get_or_create_gpt_llm()

# Optional API-based imports (conditionally loaded)
try:
    # Gemini (for API mode)
    from llama_index.llms.gemini import Gemini
    from llama_index.embeddings.gemini import GeminiEmbedding
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    # LlamaParse for document parsing (API mode)
    from llama_cloud_services import LlamaParse
    LLAMAPARSE_AVAILABLE = True
except ImportError:
    LLAMAPARSE_AVAILABLE = False


weave.init("gaia-llamaindex-agents")

def get_max_memory_config(max_memory_per_gpu):
    """Generate max_memory config for available GPUs"""
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        max_memory = {}
        for i in range(num_gpus):
            max_memory[i] = max_memory_per_gpu
        return max_memory
    return None

# Initialize models based on API availability
def initialize_models(use_api_mode=False):
    """Initialize LLM, Code LLM, and Embed models.

    Args:
        use_api_mode: when True use API-backed models (Gemini/GeminiEmbedding)
    """
    if use_api_mode and GEMINI_AVAILABLE:
        # API Mode - Using Google's Gemini models
        try:
            logger.info("Initializing models in API mode with Gemini...")
            google_api_key = os.environ.get("GOOGLE_API_KEY")
            if not google_api_key:
                logger.warning("WARNING: GOOGLE_API_KEY not found. Falling back to non-API mode.")
                return initialize_models(use_api_mode=False)

            # Main LLM - Gemini 2.5 Flash
            proj_llm = Gemini(
                model="models/gemini-2.5-flash",
                api_key=google_api_key,
                max_tokens=16000,
                temperature=0.6,
                top_p=0.95,
                top_k=20
            )

            # Same model for code since Gemini is good at code
            code_llm = proj_llm

            # Vertex AI multimodal embedding
            embed_model = GeminiEmbedding(
                model_name="gemini-embedding-001",
                api_key=google_api_key,
                task_type="retrieval_document"
            )

            return proj_llm, code_llm, embed_model
        except Exception as e:
            logger.exception("Error initializing API mode: %s", e)
            logger.info("Falling back to non-API mode...")
            return initialize_models(use_api_mode=False)
    else:
        logger.info("Initializing models in non-API mode with local models...")
        try:
            logger.info("Initializing Qwen3-VL multimodal pipeline via get_or_create_qwen_vl_llm")
            proj_llm = get_or_create_qwen_vl_llm(
                model_name="Qwen/Qwen3-VL-30B-A3B-Instruct",
                device="auto"
            )
            code_llm = get_or_create_qwen_coder_llm(
                model_name="Qwen/Qwen2.5-Coder-3B-Instruct",
                device="auto"
            )
            embed_model = get_or_create_jina_embedder()
            return proj_llm, code_llm, embed_model
        except Exception as e:
            logger.exception("Error initializing models: %s", e)
            raise

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


# Use environment variables to determine API mode
USE_API_MODE = os.environ.get("USE_API_MODE", "false").lower() == "true"

# Initialize models based on API mode and multimodal setting
proj_llm, code_llm, embed_model = initialize_models(use_api_mode=USE_API_MODE)

Settings.llm = proj_llm
Settings.embed_model = embed_model

# Add debug information to verify model configuration
logger.info(f"proj_llm type: {type(proj_llm)}")
logger.info(f"proj_llm model_name: {getattr(proj_llm, 'model_name', 'N/A')}")
logger.info(f"embed_model type: {type(embed_model)}")


IMAGE_CAPTION_LLM = proj_llm if not USE_API_MODE else None

def process_image(image_path: str) -> str:
    """Describe an image using the primary Qwen3-VL-30B-A3B-Instruct model."""
    import os
    if IMAGE_CAPTION_LLM is None:
        return "Image captioning model not available."
    if not image_path:
        return "No image path provided."
    if not os.path.exists(image_path):
        return f"Image file not found: {image_path}"
    try:
        prompt = (
            "Provide a clear, objective description of the image. Include: main objects, "
            "their attributes, setting/context, notable text (if any), and overall scene summary."
        )
        img_doc = ImageDocument(image_path=image_path, metadata={"source": image_path, "type": "input_image"})
        # Use the model's completion interface with an image document
        resp = IMAGE_CAPTION_LLM.complete(prompt, image_documents=[img_doc])
        text = getattr(resp, "text", str(resp))
        return text.strip() if text else "(empty description)"
    except Exception as e:
        logger.exception("process_image failed: %s", e)
        return f"Failed to describe image: {e}" 

def read_and_parse_content(input_path: str) -> List[Document]:
    """
    Reads and parses content from a local file path into Document objects.
    """
    # Check if API mode and LlamaParse is available for enhanced document parsing
    if USE_API_MODE and LLAMAPARSE_AVAILABLE:
        try:
            llamacloud_api_key = os.environ.get("LLAMA_CLOUD_API_KEY")
            if llamacloud_api_key:
                # Use LlamaParse for enhanced document parsing
                logger.info(f"Using LlamaParse to extract content from {input_path}")
                parser = LlamaParse(api_key=llamacloud_api_key)
                return parser.load_data(input_path)
        except Exception as e:
            logger.exception("Error using LlamaParse: %s", e)
            logger.info("Falling back to standard document parsing...")

    # Standard document parsing (fallback)
    if not os.path.exists(input_path):
        return [Document(text=f"Error: File not found at {input_path}")]

    file_extension = os.path.splitext(input_path)[1].lower()

    # Readers map - note: doc/docx/pptx etc are converted to PDF first (see convert_to_pdf)
    readers_map = {
        '.csv': CSVReader(),
        '.json': JSONReader(),
        '.xlsx': PandasExcelReader(),
        '.pdf': MultimodalPDFReader(),
    }

    # Audio/Video files using the appropriate reader based on mode
    if file_extension in ['.mp3', '.mp4', '.wav', '.m4a', '.flac']:
        try:
            if USE_API_MODE:
                # Use AssemblyAI with API mode
                loader = AssemblyAIAudioTranscriptReader(file_path=input_path)
                documents = loader.load_data()
            else:
                # Use VideoAudioReader without API
                loader = VideoAudioReader()
                documents = loader.load_data(input_path)
            return documents
        except Exception as e:
            return [Document(text=f"Error transcribing audio: {e}")]

    # Handle image files
    if file_extension in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']:
        try:
            with open(input_path, 'rb') as f:
                image_data = f.read()
            return [Document(
                text=f"IMAGE_CONTENT_BINARY",
                metadata={
                    "source": input_path,
                    "type": "image",
                    "path": input_path,
                    "image_data": image_data
                }
            )]
        except Exception as e:
            return [Document(text=f"Error reading image: {e}")]

    # Helper: try to convert office docs to PDF first, then use the PDF reader
    def convert_to_pdf(path: str) -> str:
        """Convert office documents to PDF using LibreOffice headless (soffice).

        This implementation intentionally does NOT use pypandoc. It will raise
        a RuntimeError on failure (fail-fast) with an informative message so
        callers can handle the error appropriately.

        Returns:
            Absolute path to the generated PDF file on success.

        Raises:
            RuntimeError: if LibreOffice is not available or conversion fails.
        """
        ext = os.path.splitext(path)[1].lower()
        if ext not in ['.doc', '.docx', '.ppt', '.pptx', '.odt']:
            return path

        try:
            import tempfile
            import subprocess
            import shutil
            import os as _os

            # Locate the LibreOffice binary (soffice preferred)
            soffice = shutil.which("soffice") or shutil.which("libreoffice")
            if not soffice:
                raise RuntimeError("LibreOffice executable 'soffice' or 'libreoffice' not found in PATH. Please install LibreOffice and ensure it's on PATH.")

            outdir = tempfile.mkdtemp(prefix="office_to_pdf_")

            # Run headless conversion
            proc = subprocess.run(
                [soffice, "--headless", "--convert-to", "pdf", path, "--outdir", outdir],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            if proc.returncode != 0:
                stderr = proc.stderr.decode('utf-8', errors='ignore') if proc.stderr else ''
                raise RuntimeError(f"LibreOffice conversion failed (returncode={proc.returncode}): {stderr}")

            base = _os.path.basename(path)
            pdfname = _os.path.splitext(base)[0] + '.pdf'
            candidate = _os.path.join(outdir, pdfname)
            if _os.path.exists(candidate):
                return candidate

            # If expected filename not found, try to find any PDF in outdir
            for f in _os.listdir(outdir):
                if f.lower().endswith('.pdf'):
                    return _os.path.join(outdir, f)

            raise RuntimeError(f"LibreOffice conversion completed but no PDF was produced in {outdir}.")

        except RuntimeError:
            # Re-raise RuntimeError as-is
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to convert {path} to PDF using LibreOffice: {e}")

    # Use appropriate reader for supported file types
    documents: List[Document] = []

    # Use the module-level MultimodalPDFReader implementation above via readers_map

    # PDF handling is delegated to MultimodalPDFReader via readers_map

    # If office docs, convert to PDF first then fall through to PDF reader
    if file_extension in ['.doc', '.docx', '.ppt', '.pptx', '.odt']:
        pdf_path = convert_to_pdf(input_path)
        # If conversion returned same path or failed, keep going; otherwise update
        if pdf_path and os.path.exists(pdf_path) and pdf_path != input_path:
            input_path = pdf_path
            file_extension = os.path.splitext(pdf_path)[1].lower()

    if file_extension in readers_map:
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

class DynamicQueryEngineManager:
    """Manager supporting incremental document insertion without full index rebuild each time.

    An explicit LLM reference is stored (self.llm) since global Settings.llm was removed
    to allow coexistence of text-only and multimodal models. All query engines and any
    summarization should therefore use this instance LLM explicitly.
    """

    def __init__(self, initial_documents: Optional[List[str]] = None, llm=None):
        self.documents: List[Document] = []
        self.query_engine_tool = None
        self.index: Optional[VectorStoreIndex] = None
        self.vector_store = None  # ChromaVectorStore instance
        self.element_parser = UnstructuredElementNodeParser()
        # Store explicit llm reference (fallback to module-level proj_llm)
        self.llm = llm or proj_llm
        if RecursiveCharacterTextSplitter is not None:
            self.splitter = RecursiveCharacterTextSplitter(chunk_size=4096, chunk_overlap=512)
        else:
            self.splitter = SentenceSplitter(chunk_size=4096, chunk_overlap=512)
        self._text_hashes: Set[str] = set()  # simple dedup on chunk text
        self._hybrid_reranker = None

        if initial_documents:
            self._load_initial_documents(initial_documents)

        self._ensure_index_initialized()

    # ---- Initialization helpers ----
    def _load_initial_documents(self, document_paths: List[str]):
        for path in document_paths:
            docs = read_and_parse_content(path)
            self.documents.extend(docs)
        logger.info("Loaded %d initial documents", len(self.documents))

    def _build_hybrid_reranker(self):
        class HybridReranker:
            def __init__(self):
                self.jina_reranker = get_or_create_jina_reranker(
                    model_name="jinaai/jina-reranker-m0", top_n=5, device="cuda:1"
                )

            def postprocess_nodes(self, nodes, query_bundle):
                try:
                    q = getattr(query_bundle, 'query_str', None)
                except Exception:
                    q = None
                logger.debug("HybridReranker.postprocess_nodes: nodes=%s query=%s", len(nodes) if nodes else 0, q)
                return self.jina_reranker.postprocess_nodes(nodes, query_bundle)
        self._hybrid_reranker = HybridReranker()

    def _ensure_index_initialized(self):
        if self.index is not None:
            return
        if not self.documents:
            self.documents = [Document(text="No documents loaded yet. Use web search to add content.")]
        logger.info("Initializing index with %d documents", len(self.documents))
        nodes = self._parse_documents_to_nodes(self.documents)
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
            from llama_index.vector_stores.chroma import ChromaVectorStore
            from inspect import signature
            Client = chromadb.Client
            chroma_client = None
            try:
                params = signature(Client).parameters
                if 'settings' in params:
                    chroma_client = Client(settings=ChromaSettings(chroma_db_impl="duckdb+parquet", persist_directory="./chroma_db"))
                elif 'persist_directory' in params or 'chroma_db_impl' in params:
                    chroma_client = Client(chroma_db_impl="duckdb+parquet", persist_directory="./chroma_db")
                else:
                    chroma_client = Client()
            except Exception:
                try:
                    chroma_client = chromadb.Client(chroma_db_impl="duckdb+parquet", persist_directory="./chroma_db")
                except Exception:
                    chroma_client = chromadb.Client()
            collection = chroma_client.get_or_create_collection(name="gaia_collection")
            self.vector_store = ChromaVectorStore(chroma_collection=collection)
            self.index = VectorStoreIndex(nodes, vector_store=self.vector_store)
            logger.info("Chroma vector store initialized (initial nodes=%d)", len(nodes))
        except Exception as e:
            logger.exception("Chroma init failed, using in-memory store: %s", e)
            self.index = VectorStoreIndex(nodes)
        self._build_hybrid_reranker()
        self._create_or_update_query_engine_tool(rebuild=True)

    # ---- Parsing ----
    def _parse_documents_to_nodes(self, documents: List[Document]):
        text_docs: List[Document] = []
        image_docs: List[Document] = []
        for doc in documents:
            doc_type = doc.metadata.get("type", "")
            source = doc.metadata.get("source", "").lower()
            file_type = doc.metadata.get("file_type", "")
            if (doc_type in ["image", "web_image"] or file_type in ['jpg', 'png', 'jpeg', 'gif', 'bmp', 'webp'] or any(ext in source for ext in ['.jpg', '.png', '.jpeg', '.gif', '.bmp', '.webp'])):
                image_docs.append(doc)
            else:
                text_docs.append(doc)
        nodes = []
        if text_docs:
            initial_nodes = self.element_parser.get_nodes_from_documents(text_docs)
            try:
                split_nodes = self.splitter.get_nodes_from_documents(initial_nodes)
            except Exception:
                split_nodes = self.splitter.get_nodes_from_documents(initial_nodes)
            # Deduplicate by text hash
            for n in split_nodes:
                txt = getattr(n, 'text', '') or ''
                h = hashlib.sha1(txt.strip().lower().encode('utf-8')).hexdigest()
                if h in self._text_hashes:
                    continue
                self._text_hashes.add(h)
                nodes.append(n)
        if image_docs:
            for img_doc in image_docs:
                try:
                    image_node = ImageNode(
                        text=img_doc.text or f"Image content from {img_doc.metadata.get('source', 'unknown')}",
                        metadata=img_doc.metadata,
                        image_path=img_doc.metadata.get("path"),
                        image=img_doc.metadata.get("image_data")
                    )
                    nodes.append(image_node)
                except Exception as e:
                    logger.exception("Error creating ImageNode: %s", e)
                    nodes.append(TextNode(text=img_doc.text or "Image node fallback", metadata=img_doc.metadata))
        return nodes

    # ---- Query Engine Tool ----
    def _create_or_update_query_engine_tool(self, rebuild: bool = False):
        if self.index is None:
            return
        query_engine = self.index.as_query_engine(
            similarity_top_k=20,
            node_postprocessors=[self._hybrid_reranker] if self._hybrid_reranker else [],
            response_mode="tree_summarize",
            llm=self.llm  # explicit LLM (Settings.llm not set globally)
        )
        from llama_index.core.tools import QueryEngineTool
        if self.query_engine_tool is None:
            self.query_engine_tool = QueryEngineTool.from_defaults(
                query_engine=query_engine,
                name="dynamic_hybrid_multimodal_rag_tool",
                description=(
                    "Advanced dynamic knowledge base with multimodal reranking. "
                    "Uses Jina reranker-m0 for unified text and visual content reranking. "
                    "Supports text-to-text, text-to-image, image-to-text, and image-to-image ranking. "
                    "Automatically updated with web search content."
                )
            )
        else:
            # Only swap the underlying query_engine if a rebuild occurred
            if rebuild:
                self.query_engine_tool.query_engine = query_engine

    # ---- Public API ----
    def add_documents(self, new_documents: List[Document]):
        self._ensure_index_initialized()
        if not new_documents:
            return
        logger.info("Incremental add: %d raw documents", len(new_documents))
        self.documents.extend(new_documents)
        new_nodes = self._parse_documents_to_nodes(new_documents)
        if not new_nodes:
            logger.info("No new unique nodes to insert (all duplicates)")
            return
        # Insert nodes (embeddings computed automatically by index)
        try:
            self.index.insert_nodes(new_nodes)
            logger.info("Inserted %d new nodes (total docs=%d hashes=%d)", len(new_nodes), len(self.documents), len(self._text_hashes))
        except Exception as e:
            logger.exception("Incremental insert failed; consider rebuild: %s", e)
            self.rebuild_all()

    def rebuild_all(self):
        """Full rebuild (e.g., after splitter config change)."""
        logger.info("Starting full rebuild of index (%d documents)", len(self.documents))
        self.index = None
        self._text_hashes.clear()
        self._ensure_index_initialized()
        self._create_or_update_query_engine_tool(rebuild=True)
        logger.info("Rebuild complete")

    def get_tool(self):
        return self.query_engine_tool

def search_and_extract_content_from_url(query: str) -> List[Document]:
    logger.info(f"[web] start search: {query}")
    url = None
    try:
        # Add a short timeout using DDGS context + our own watchdog
        with DDGS() as ddg:
            # DDGS has no built-in timeout; consider running in a thread with timeout or replacing with requests to a known endpoint when testing
            results = list(ddg.text(query, max_results=1, backend="google"))
        logger.info(f"[web] ddgs results: {len(results)}")
        if results:
            first = results[0]
            url = first.get("href") or first.get("link") or first.get("url") or first.get("FirstURL") or first.get("first_url")
    except Exception as e:
        logger.warning(f"[web] ddgs failed: {e}")

    if not url:
        logger.info("[web] no URL extracted (blocked/timeout/empty)")
        return [Document(text="No URL could be extracted from the search results.",
                         metadata={"type": "web_text", "source": "search"})]

    url = str(url).rstrip(").,;:'\"")
    logger.info(f"[web] fetching url: {url}")

    try:
        netloc = urlparse(url).netloc.lower()
        if "youtube" in netloc or "youtu.be" in netloc:
            logger.info("[web] using YoutubeTranscriptReader()")
            loader = YoutubeTranscriptReader()
            documents = loader.load_data(youtubelinks=[url])
        else:
            logger.info("[web] using BeautifulSoupWebReader()")
            # If your BS reader supports it, pass a timeout; otherwise wrap requests globally
            loader = BeautifulSoupWebReader()
            documents = loader.load_data(urls=[url])

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
                         metadata={"type": "web_text", "source": url})]

def enhanced_web_search_and_update(query: str, manager: DynamicQueryEngineManager = None) -> str:
    """
    Performs web search, extracts content, and adds it to the dynamic query engine.
    """
    # Extract content from web search
    documents = search_and_extract_content_from_url(query)

    # Add documents to the dynamic query engine
    if documents and not any("Error" in doc.text for doc in documents):
        # Log before adding
        logger.info("enhanced_web_search_and_update: adding %d documents from search '%s'", len(documents), query)

        if manager is None:
            # Create a short-lived manager when caller didn't provide one
            temp_manager = DynamicQueryEngineManager()
            temp_manager.add_documents(documents)
        else:
            manager.add_documents(documents)

        # Return summary of what was added
        text_docs = [doc for doc in documents if doc.metadata.get("type") == "web_text"]
        image_docs = [doc for doc in documents if doc.metadata.get("type") == "web_image"]

        summary = f"Successfully added web content to knowledge base:\n"
        summary += f"- {len(text_docs)} text documents\n"
        summary += f"- {len(image_docs)} images\n"
        summary += f"Source: {documents[0].metadata.get('source', 'Unknown')}"

        return summary
    else:
        error_msg = documents[0].text if documents else "No content extracted"
        logger.warning("enhanced_web_search_and_update: failed to extract content for query '%s': %s", query, error_msg)
        return f"Failed to extract web content: {error_msg}"

def make_enhanced_web_search_tool(manager: DynamicQueryEngineManager):
    def enhanced_web_search(query: str) -> str:
        "Perform a web search for the provided query, extract textual (and when present image) content, add it to the dynamic knowledge base, and return a brief summary of what was added. Returns an error message string if extraction fails."
        logger.info(f"enhanced_web_search called with query: {query}")
        return enhanced_web_search_and_update(query, manager=manager)

    # Provide a stable function name for tool display
    enhanced_web_search.__name__ = "enhanced_web_search"
    return enhanced_web_search

def safe_import(module_name):
    """Safely import a module, return None if not available"""
    try:
        return __import__(module_name)
    except ImportError:
        return None

safe_globals = {
    "__builtins__": {
        "len": len, "str": str, "int": int, "float": float,
        "list": list, "dict": dict, "sum": sum, "max": max, "min": min,
        "round": round, "abs": abs, "sorted": sorted, "enumerate": enumerate,
        "range": range, "zip": zip, "map": map, "filter": filter,
        "any": any, "all": all, "type": type, "isinstance": isinstance,
        "print": print, "open": open, "bool": bool, "set": set, "tuple": tuple
    }
}

# Core modules (always available)
core_modules = [
    "math", "datetime", "re", "os", "sys", "json", "csv", "random",
    "itertools", "collections", "functools", "operator", "copy",
    "decimal", "fractions", "uuid", "typing", "statistics", "pathlib",
    "glob", "shutil", "tempfile", "pickle", "gzip", "zipfile", "tarfile",
    "base64", "hashlib", "secrets", "hmac", "textwrap", "string",
    "difflib", "socket", "ipaddress", "logging", "warnings", "traceback",
    "pprint", "threading", "queue", "sqlite3", "urllib", "html", "xml",
    "configparser"
]

for module in core_modules:
    imported = safe_import(module)
    if imported:
        safe_globals[module] = imported

# Data science modules (may not be available)
optional_modules = {
    "numpy": "numpy",
    "np": "numpy",
    "pandas": "pandas",
    "pd": "pandas",
    "scipy": "scipy",
    "matplotlib": "matplotlib",
    "plt": "matplotlib.pyplot",
    "seaborn": "seaborn",
    "sns": "seaborn",
    "plotly": "plotly",
    "sklearn": "sklearn",
    "statsmodels": "statsmodels",
    "PIL": "PIL",
    "skimage": "skimage",
    "pytz": "pytz",
    "requests": "requests",
    "bs4": "bs4",
    "sympy": "sympy",
    "tqdm": "tqdm",
    "yaml": "yaml",
    "toml": "toml"
}

for alias, module_name in optional_modules.items():
    imported = safe_import(module_name)
    if imported:
        safe_globals[alias] = imported

# Special cases
if safe_globals.get("bs4"):
    safe_globals["BeautifulSoup"] = safe_globals["bs4"].BeautifulSoup

if safe_globals.get("PIL"):
    image_module = safe_import("PIL.Image")
    if image_module:
        safe_globals["Image"] = image_module

def execute_python_code(code: str) -> str:
    """python_code_execution(code: str) -> str

    Execute arbitrary Python code inside a restricted global namespace containing only
    a curated subset of safe builtins and optionally-available data/ML libraries.
    If the executed code defines a variable named 'result', its string value is returned;
    otherwise a generic success message is returned. Any exception is caught and its
    message returned prefixed with 'Code execution failed'.
    """
    try:
        exec_locals = {}
        exec(code, safe_globals, exec_locals)
        if 'result' in exec_locals:
            return str(exec_locals['result'])
        return "Code executed successfully"
    except Exception as e:
        return f"Code execution failed: {str(e)}"

def clean_response(response: str) -> str:
    """Clean response by removing common prefixes before GAIA formatting."""
    response_clean = response.strip()

    prefixes_to_remove = [
        "FINAL ANSWER:", "Answer:", "The answer is:",
        "Based on my analysis,", "After reviewing,",
        "The result is:", "Final result:", "According to",
        "In conclusion,", "Therefore,", "Thus,"
    ]

    for prefix in prefixes_to_remove:
        if response_clean.startswith(prefix):
            response_clean = response_clean[len(prefix):].strip()

    return response_clean

def llm_reformat(response: str, question: str) -> str:
    """Use LLM to reformat the response according to GAIA requirements"""
    format_prompt = f"""Extract the exact answer from the response below. Follow GAIA formatting rules strictly.

GAIA Format Rules:
- ONLY the precise answer, no explanations
- No prefixes like "Answer:", "The result is:", etc.
- For numbers: just the number (e.g., "156", "3.14e+8")
- For names: just the name (e.g., "Martinez", "Sarah")
- For lists: comma-separated (e.g., "C++, Java, Python")
- For country codes: just the code (e.g., "FRA", "US")
- For yes/no: just "Yes" or "No"

Examples:
Question: "How many papers were published?"
Response: "The analysis shows 156 papers were published in total."
Answer: 156

Question: "What is the last name of the developer?"
Response: "The developer mentioned is Dr. Sarah Martinez from the AI team."
Answer: Martinez

Question: "List programming languages, alphabetized:"
Response: "The languages mentioned are Python, Java, and C++. Alphabetized: C++, Java, Python"
Answer: C++, Java, Python

Now extract the exact answer:
Question: {question}
Response: {response}

Answer:"""

    try:
        # Use the global LLM instance
        formatting_response = proj_llm.complete(format_prompt)
        answer = str(formatting_response).strip()

        # Extract just the answer after "Answer:"
        if "Answer:" in answer:
            answer = answer.split("Answer:")[-1].strip()

        return answer
    except Exception as e:
        logger.exception("LLM reformatting failed: %s", e)
        return response

def final_answer_tool(agent_response: str, question: str) -> str:
    """
    Final answer tool with conditional LLM reformatting.
    Args:
        agent_response: The raw response from agent reasoning
        question: The original question for context
    Returns:
        Exact answer in GAIA format
    """
    # Step 1: Clean the response
    cleaned_response = clean_response(agent_response)

    formatted_answer = llm_reformat(cleaned_response, question)
    logger.info(f"Original response cleaned: {cleaned_response[:100]}...")
    logger.info(f"LLM formatted answer: {formatted_answer}")
    return formatted_answer

class EnhancedGAIAAgent:
    def __init__(self):
        logger.info("Initializing Enhanced GAIA Agent...")

        # Vérification du token HuggingFace
        hf_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        if not hf_token:
            logger.warning("Warning: HUGGINGFACEHUB_API_TOKEN not found, some features may not work")

        # Initialize the dynamic query engine manager
        self.dynamic_qe_manager = DynamicQueryEngineManager()

        # Raw callable web search function
        self.web_tool = make_enhanced_web_search_tool(self.dynamic_qe_manager)
        # Wrap web search in FunctionTool for consistent tool interface
        self.web_function_tool = FunctionTool.from_defaults(
            fn=self.web_tool,
            name="enhanced_web_search"
        )

        # Wrap execute_python_code in FunctionTool
        self.exec_function_tool = FunctionTool.from_defaults(
            fn=execute_python_code,
            name="execute_python_code",
        )

        # Assemble tool list (web search, RAG query engine, code exec)
        tool_list = [
            self.web_function_tool,
            self.dynamic_qe_manager.get_tool(),
            self.exec_function_tool,
        ]

        # Optional image captioning tool (7B VL model) if available
        if IMAGE_CAPTION_LLM is not None:
            try:
                self.image_caption_tool = FunctionTool.from_defaults(
                    fn=process_image,
                    name="process_image",
                    description="Convert a local image path into a detailed textual description/caption."
                )
                tool_list.append(self.image_caption_tool)
                logger.info("Registered process_image tool in EnhancedGAIAAgent")
            except Exception as e:
                logger.warning(f"Failed to register process_image tool: {e}")


        external_knowledge_system_prompt = f"""You are an expert research assistant specialized in information gathering and knowledge retrieval. Your role is to help users find accurate information by searching the web, analyzing documents, and processing various types of content including text, images, and multimedia files. You have access to advanced tools for web search, document analysis, and knowledge base queries. Always provide comprehensive and well-researched answers.
"""
        
        code_system_prompt = f"""You are a skilled programming assistant and data analyst. Your expertise includes Python programming, data analysis, mathematical computations, and problem-solving through code execution. You can write, execute, and debug Python code to solve complex problems, perform calculations, analyze data, and generate insights. Always write clean, efficient, and well-documented code.

"""
        self.external_knowledge_agent = ReActAgent(
            tools=tool_list,
            llm=proj_llm,
            max_steps=8,
            system_prompt=external_knowledge_system_prompt,
            verbose=True,
            name="external_knowledge_agent",
            description="Handles external knowledge retrieval, web searches, document processing, and information gathering tasks"
        )
        
        self.code_agent = ReActAgent(
            tools=[self.exec_function_tool],
            llm=code_llm,
            max_steps=6,
            system_prompt=code_system_prompt,
            verbose=True,
            name="code_execution_agent",
            description="Executes Python code, performs calculations, data analysis, and mathematical operations"
        )
        

        # Fixed indentation: coordinator initialization inside __init__
        self.coordinator = AgentWorkflow(
            agents=[self.external_knowledge_agent, self.code_agent],
            root_agent="external_knowledge_agent",
        )

    def load_documents_from_file(self, file_path: str):
        """Load and process text documents for BM25, or return raw content for media files."""
        try:
            # Devine le type MIME du fichier
            mime_type, _ = mimetypes.guess_type(file_path)
            if mime_type is None:
                mime_type = ""
            logger.info(f"Detected MIME type: {mime_type} for file {file_path}")
            # Traitement selon le type de fichier
            if mime_type.startswith("image") or mime_type.startswith("video") or mime_type.startswith("audio"):
                with open(file_path, "rb") as f:
                    binary_content = f.read()
                logger.info(f"Loaded {mime_type} file: {file_path}")
                return binary_content
            else : 
            # Si fichier texte → traitement BM25
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            # Use semantic-first chunking for file loader as well
            if RecursiveCharacterTextSplitter is not None:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=4096,
                    chunk_overlap=512,
                )
            else:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=4096,
                    chunk_overlap=512,
                )

            chunks = text_splitter.split_text(content)
            docs = [Document(page_content=chunk, metadata={"source": file_path})
                    for chunk in chunks]

            # BM25 + agent
            self.retriever_tool = BM25RetrieverTool(docs)
            self._create_agent()

            logger.info(f"Loaded {len(docs)} document chunks from {file_path}")
            return True

        except Exception as e:
            logger.exception("Error loading documents from %s: %s", file_path, e)
            return False


    def download_gaia_file(self, task_id: str, api_url: str = "https://agents-course-unit4-scoring.hf.space") -> str:
        """Download file associated with GAIA task_id and return its path"""
        try:
            response = requests.get(f"{api_url}/files/{task_id}", timeout=30)
            response.raise_for_status()

            # Try to get filename from headers
            logger.debug(f"Response headers: {response.headers}")
            content_disp = response.headers.get("content-disposition", "")
            match = re.search(r'filename="(.+)"', content_disp)
            if match:
                filename = match.group(1)
            else:
                # Error
                raise ValueError("Filename not found in response headers")

            # Save the file
            with open(filename, 'wb') as f:
                f.write(response.content)

            logger.info(f"Downloaded file saved as {filename}")
            return os.path.abspath(filename)  # Or just return `filename` for relative path

        except Exception as e:
            logger.exception("Failed to download file for task %s: %s", task_id, e)
            return None

    def add_documents_to_knowledge_base(self, file_path: str):
        """Add downloaded GAIA documents to the dynamic knowledge base"""
        try:
            documents = read_and_parse_content(file_path)
            if documents:
                self.dynamic_qe_manager.add_documents(documents)
                logger.info(f"Added {len(documents)} documents from {file_path} to dynamic knowledge base")
                # No need to replace the tools list; QueryEngineTool is updated in-place.
                return True
        except Exception as e:
            logger.exception("Failed to add documents from %s: %s", file_path, e)
            return False

    async def solve_gaia_question(self, question_data: Dict[str, Any]) -> str:
        """
        Solve GAIA question with dynamic knowledge base integration.
        Uses a non-streaming await pattern for the workflow execution.
        """
        question = question_data.get("Question", "")
        task_id = question_data.get("task_id", "")

        # Try to download and add file to knowledge base if task_id provided
        file_path = None
        if task_id:
            try:
                file_path = self.download_gaia_file(task_id)
                if file_path:
                    # Add documents to dynamic knowledge base
                    self.add_documents_to_knowledge_base(file_path)
                    logger.info("Successfully integrated GAIA file into dynamic knowledge base")
            except Exception as e:
                logger.exception("Failed to download/process file for task %s: %s", task_id, e)

        # Multimodal mode - no reasoning level prefix
        gaia_format_instructions = """YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings.
If you are asked for a number, don't use comma to write your number neither use units such as $ or percent sign unless specified otherwise.
If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise.
If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string."""

        context_prompt = f"""GAIA Task ID: {task_id}
    Question: {question}
    {f'File processed and added to knowledge base: {file_path}' if file_path else 'No additional files'}
    You are a general AI assistant. I will ask you a question.

    IMPORTANT INSTRUCTIONS:
    1. Think through this STEP BY STEP, carefully analyzing all aspects of the question.
    2. Pay special attention to specific qualifiers like dates, types, categories, or locations.
    3. Make sure your searches include ALL important details from the question.
    4. Report your thoughts and reasoning process clearly.
    5. Finish your answer with: FINAL ANSWER: [YOUR FINAL ANSWER]

    {gaia_format_instructions}
    """.strip()

        try:
            ctx = Context(self.coordinator)
            logger.info("=== AGENT REASONING STEPS ===")
            logger.info(f"Dynamic knowledge base contains {len(self.dynamic_qe_manager.documents)} documents")

            handler = self.coordinator.run(ctx=ctx, user_msg=context_prompt)
            full_response = ""

            async for event in handler.stream_events():
                if isinstance(event, AgentStream):
                    # preserve streaming behavior (no newline, immediate flush)
                    sys.stdout.write(event.delta)
                    sys.stdout.flush()
                    full_response += event.delta

            final_response = await handler
            logger.info("\n=== END REASONING ===")

            # Extract the final formatted answer
            final_answer = final_answer_tool(str(final_response), question)

            return final_answer
        except Exception as e:
            error_msg = f"Error processing question: {str(e)}"
            logger.exception(error_msg)
            return error_msg

    def get_knowledge_base_stats(self):
        """Get statistics about the current knowledge base"""
        return {
            "total_documents": len(self.dynamic_qe_manager.documents),
            "document_sources": [doc.metadata.get("source", "Unknown") for doc in self.dynamic_qe_manager.documents]
        }


class GAIAAgent:
    """Backwards-compatible wrapper that always instantiates the multimodal agent."""

    def __init__(self):
        logger.info("Initializing GAIAAgent (multimodal)")
        self._agent = EnhancedGAIAAgent()
        self.mode = "multimodal"

    async def solve_gaia_question(self, question_data: Dict[str, Any]) -> str:
        return await self._agent.solve_gaia_question(question_data)

    def add_documents_to_knowledge_base(self, file_path: str):
        return self._agent.add_documents_to_knowledge_base(file_path)

    def get_knowledge_base_stats(self):
        return self._agent.get_knowledge_base_stats()

    def download_gaia_file(self, task_id: str, api_url: str = "https://agents-course-unit4-scoring.hf.space"):
        return self._agent.download_gaia_file(task_id, api_url=api_url)

async def main():

    query = "Mercedes Sosa studio albums 2000-2009"

    logger.info("Starting targeted tests for query: %s", query)

    # 2) Test the full agent pipeline: create an agent router and ask it to solve the GAIA question

    try:
        agent_for_tool = GAIAAgent()
        question_data = {"Question": query, "task_id": ""}
        try:
            final_response = await agent_for_tool.solve_gaia_question(question_data)
            tool_result = final_response
            logger.info("solve_gaia_question -> result: %s", str(tool_result))
        except Exception as e:
            tool_result = f"solve_gaia_question failed or timed out: {e}"
            logger.exception("solve_gaia_question invocation failed: %s", e)
    except Exception as e:
        tool_result = f"Tool invocation failed: {e}"
        logger.exception("solve_gaia_question invocation failed: %s", e)

    # Print concise summary for quick inspection
    print("=== TARGETED TEST SUMMARY ===")
    print("query:", query)
    print("enhanced_web_search_tool -> sample:", str(tool_result)[:400])


if __name__ == '__main__':
    asyncio.run(main())
