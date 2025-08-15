# Standard library imports
import logging
import os
import re
from typing import Dict, Any, List
from urllib.parse import urlparse
import torch
import asyncio
import mimetypes
# Using PyMuPDFReader from llama_index.readers.file instead of direct fitz usage

# Third-party imports
import requests
# Third-party imports
from custom_models import QwenVLCustomLLM, JinaEmbeddingsV4, JinaMultimodalReranker

# LlamaIndex core imports
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.core.agent.workflow import ReActAgent, AgentStream
from llama_index.core.node_parser import UnstructuredElementNodeParser, SentenceSplitter
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.tools import FunctionTool
from llama_index.core.workflow import Context
from llama_index.core.schema import ImageNode, TextNode

# LlamaIndex specialized imports
from llama_index.llms.huggingface import HuggingFaceLLM
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
    """Initialize LLM, Code LLM, and Embed models based on mode"""
    if use_api_mode and GEMINI_AVAILABLE:
        # API Mode - Using Google's Gemini models
        try:
            print("Initializing models in API mode with Gemini...")
            google_api_key = os.environ.get("GOOGLE_API_KEY")
            if not google_api_key:
                print("WARNING: GOOGLE_API_KEY not found. Falling back to non-API mode.")
                return initialize_models(use_api_mode=False)

            # Main LLM - Gemini 2.0 Flash
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
                model_name="models/embedding-005",
                api_key=google_api_key,
                task_type="retrieval_document"
            )

            return proj_llm, code_llm, embed_model
        except Exception as e:
            print(f"Error initializing API mode: {e}")
            print("Falling back to non-API mode...")
            return initialize_models(use_api_mode=False)
    else:
        # Non-API Mode - Using HuggingFace models
        print("Initializing models in non-API mode with local models...")

        try:
            # Main LLM: Qwen2.5-VL (Vision-Language)
            proj_llm = QwenVLCustomLLM()

            # Embedding model: Jina AI v4 (multimodal)
            embed_model = JinaEmbeddingsV4()

            # Code LLM (unchanged)
            code_llm = HuggingFaceLLM(
                model_name="Qwen/Qwen2.5-Coder-3B-Instruct",
                tokenizer_name="Qwen/Qwen2.5-Coder-3B-Instruct",
                device_map="auto",
                model_kwargs={
                    "torch_dtype": "auto",
                    "load_in_4bit": True
                },
                generate_kwargs={"do_sample": False}
            )
            return proj_llm, code_llm, embed_model
        except Exception as e:
            print(f"Error initializing models: {e}")
            raise

# Setup logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("llama_index.core.agent").setLevel(logging.DEBUG)
logging.getLogger("llama_index.llms").setLevel(logging.DEBUG)
# Module logger
logger = logging.getLogger(__name__)

# Use environment variable to determine API mode
USE_API_MODE = os.environ.get("USE_API_MODE", "false").lower() == "true"

# Initialize models based on API mode setting
proj_llm, code_llm, embed_model = initialize_models(use_api_mode=USE_API_MODE)

# Set global settings
Settings.llm = proj_llm
Settings.embed_model = embed_model

def read_and_parse_content(input_path: str) -> List[Document]:
    """
    Reads and parses content from a local file path into Document objects.
    URL handling has been moved to search_and_extract_top_url.
    """
    # Check if API mode and LlamaParse is available for enhanced document parsing
    if USE_API_MODE and LLAMAPARSE_AVAILABLE:
        try:
            llamacloud_api_key = os.environ.get("LLAMA_CLOUD_API_KEY")
            if llamacloud_api_key:
                # Use LlamaParse for enhanced document parsing
                print(f"Using LlamaParse to extract content from {input_path}")
                parser = LlamaParse(api_key=llamacloud_api_key)
                return parser.load_data(input_path)
        except Exception as e:
            print(f"Error using LlamaParse: {e}")
            print("Falling back to standard document parsing...")

    # Standard document parsing (fallback)
    if not os.path.exists(input_path):
        return [Document(text=f"Error: File not found at {input_path}")]

    file_extension = os.path.splitext(input_path)[1].lower()

    # Readers map
    readers_map = {
        '.docx': DocxReader(),
        '.doc': DocxReader(),
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

    # Use appropriate reader for supported file types
    documents: List[Document] = []

    # Use the module-level MultimodalPDFReader implementation above via readers_map

    # PDF handling is delegated to MultimodalPDFReader via readers_map

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
    """Single unified manager for all RAG operations - replaces the entire static approach."""

    def __init__(self, initial_documents: List[str] = None):
        self.documents = []
        self.query_engine_tool = None

        # Load initial documents if provided
        if initial_documents:
            self._load_initial_documents(initial_documents)

        self._create_rag_tool()

    def _load_initial_documents(self, document_paths: List[str]):
        """Load initial documents using read_and_parse_content."""
        for path in document_paths:
            docs = read_and_parse_content(path)
            self.documents.extend(docs)
        logger.info("Loaded %d initial documents", len(self.documents))

    def _create_rag_tool(self):
        """Create RAG tool using multimodal-aware parsing."""
        documents = self.documents if self.documents else [
            Document(text="No documents loaded yet. Use web search to add content.")
        ]
        logger.info("_create_rag_tool: starting with %d documents", len(documents))

        # Separate text and image documents for proper processing
        text_documents = []
        image_documents = []

        for doc in documents:
            doc_type = doc.metadata.get("type", "")
            source = doc.metadata.get("source", "").lower()
            file_type = doc.metadata.get("file_type", "")

            # Identify image documents
            if (doc_type in ["image", "web_image"] or
                file_type in ['jpg', 'png', 'jpeg', 'gif', 'bmp', 'webp'] or
                any(ext in source for ext in ['.jpg', '.png', '.jpeg', '.gif', '.bmp', '.webp'])):
                image_documents.append(doc)
            else:
                text_documents.append(doc)

        logger.info("_create_rag_tool: %d text_documents, %d image_documents", len(text_documents), len(image_documents))

        # Use UnstructuredElementNodeParser for text content with multimodal awareness
        element_parser = UnstructuredElementNodeParser()
        splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=200)
        nodes = []

        # Process text documents with UnstructuredElementNodeParser
        if text_documents:
            initial_nodes = element_parser.get_nodes_from_documents(text_documents)
            final_nodes = splitter.get_nodes_from_documents(initial_nodes)
            nodes.extend(final_nodes)

        # Process image documents as ImageNodes
        if image_documents:
            for img_doc in image_documents:
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
                    # Fallback to regular TextNode for images
                    text_node = TextNode(
                        text=img_doc.text or f"Image content from {img_doc.metadata.get('source', 'unknown')}",
                        metadata=img_doc.metadata
                    )
                    nodes.append(text_node)

        logger.info("_create_rag_tool: built %d nodes", len(nodes))
        index = VectorStoreIndex(nodes)

        class HybridReranker:
            def __init__(self):
                # Use the new Jina multimodal reranker
                self.jina_reranker = JinaMultimodalReranker(
                    model_name="jinaai/jina-reranker-m0",
                    top_n=5,
                    device="auto"
                )

            def postprocess_nodes(self, nodes, query_bundle):
                # Use Jina multimodal reranker for all content types
                try:
                    q = getattr(query_bundle, 'query_str', None)
                except Exception:
                    q = None
                logger.info("HybridReranker.postprocess_nodes: called with %d nodes, query=%s", len(nodes) if nodes is not None else 0, str(q))
                res = self.jina_reranker.postprocess_nodes(nodes, query_bundle)
                logger.info("HybridReranker.postprocess_nodes: returned %d nodes", len(res) if res is not None else 0)
                return res

        hybrid_reranker = HybridReranker()

        query_engine = index.as_query_engine(
            similarity_top_k=20,
            node_postprocessors=[hybrid_reranker],
            response_mode="tree_summarize"
        )

        # Create QueryEngineTool
        from llama_index.core.tools import QueryEngineTool

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

    def add_documents(self, new_documents: List[Document]):
        """Add documents from web search and recreate tool."""
        self.documents.extend(new_documents)
        self._create_rag_tool()  # Recreate with ALL documents
        logger.info("Added %d documents. Total: %d", len(new_documents), len(self.documents))

    def get_tool(self):
        return self.query_engine_tool

# Global instance
dynamic_qe_manager = DynamicQueryEngineManager()

def search_and_extract_content_from_url(query: str) -> List[Document]:
    """
    Searches web, gets top URL, and extracts both text content and images.
    Returns a list of Document objects containing the extracted content.
    """
    # Try using ddgs (DDGS) with the 'google' backend first (preferred per request)
    url = None
    try:
        with DDGS() as ddg:
            results = list(ddg.text(query, max_results=1, backend="google"))
        if results and len(results) > 0:
            first = results[0]
            # ddgs may expose the link under several keys depending on backend/version
            url = first.get("href") or first.get("link") or first.get("url") or first.get("FirstURL") or first.get("first_url")
            # sometimes ddgs returns a dict-like string; coerce
            if not url:
                # try common keys by inspecting full dict
                for k in ("href", "link", "url", "FirstURL", "first_url"):
                    if k in first:
                        url = first[k]
                        break

    except Exception as e:
        url = None

    if not url:
        return [Document(text="No URL could be extracted from the search results.")]

    # sanitize URL
    url = str(url).rstrip(").,;:'\"")
    documents: List[Document] = []

    try:
        netloc = urlparse(url).netloc.lower()
        if "youtube" in netloc or "youtu.be" in netloc:
            loader = YoutubeTranscriptReader()
            documents = loader.load_data(youtubelinks=[url])
        else:
            loader = BeautifulSoupWebReader()
            documents = loader.load_data(urls=[url])

        for doc in documents:
            if not getattr(doc, "metadata", None):
                doc.metadata = {}
            doc.metadata["source"] = url
            doc.metadata["type"] = "web_text"

        return documents
    except Exception as e:
        return [Document(text=f"Error extracting content from URL: {str(e)}")]

def enhanced_web_search_and_update(query: str) -> str:
    """
    Performs web search, extracts content, and adds it to the dynamic query engine.
    """
    # Extract content from web search
    documents = search_and_extract_content_from_url(query)

    # Add documents to the dynamic query engine
    if documents and not any("Error" in doc.text for doc in documents):
        dynamic_qe_manager.add_documents(documents)

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
        return f"Failed to extract web content: {error_msg}"

# Create the enhanced web search tool
enhanced_web_search_tool = FunctionTool.from_defaults(
    fn=enhanced_web_search_and_update,
    name="enhanced_web_search",
    description="Search the web, extract content and images, and add them to the knowledge base for future queries."
)

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
    try:
        exec_locals = {}
        exec(code, safe_globals, exec_locals)
        if 'result' in exec_locals:
            return str(exec_locals['result'])
        else:
            return "Code executed successfully"
    except Exception as e:
        return f"Code execution failed: {str(e)}"

code_execution_tool = FunctionTool.from_defaults(
    fn=execute_python_code,
    name="Python Code Execution",
    description="Executes Python code safely for calculations and data processing"
)

def clean_response(response: str) -> str:
    """Clean response by removing common prefixes"""
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
        print(f"LLM reformatting failed: {e}")
        return response

def final_answer_tool(agent_response: str, question: str) -> str:
    """
    Simplified final answer tool using only LLM reformatting.
    Args:
        agent_response: The raw response from agent reasoning
        question: The original question for context
    Returns:
        Exact answer in GAIA format
    """
    # Step 1: Clean the response
    cleaned_response = clean_response(agent_response)

    # Step 2: Use LLM reformatting
    formatted_answer = llm_reformat(cleaned_response, question)

    print(f"Original response cleaned: {cleaned_response[:100]}...")
    print(f"LLM formatted answer: {formatted_answer}")

    return formatted_answer

class EnhancedGAIAAgent:
    def __init__(self):
        print("Initializing Enhanced GAIA Agent...")

        # Vérification du token HuggingFace
        hf_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        if not hf_token:
            print("Warning: HUGGINGFACEHUB_API_TOKEN not found, some features may not work")

        # Initialize the dynamic query engine manager
        self.dynamic_qe_manager = DynamicQueryEngineManager()

        # Create enhanced agents with dynamic tools
        self.external_knowledge_agent = ReActAgent(
            name="external_knowledge_agent",
            description="Advanced information retrieval with dynamic knowledge base",
            system_prompt="""You are an advanced information specialist with a sophisticated RAG system.
Your knowledge base uses hybrid reranking and grows dynamically with each web search and document addition.

IMPORTANT INSTRUCTIONS FOR YOUR REASONING PROCESS:
1. Pay careful attention to ALL details in the user's question.
2. Think step by step about what is being asked, breaking down the requirements.
3. Identify specific qualifiers (e.g., "studio albums" vs just "albums", "between 2000-2010" vs "all time").
4. If searching for information, include ALL important details in your search query.
5. Double-check that your final answer addresses the EXACT question asked, not a simplified version.

For example:
- If asked "How many studio albums did Taylor Swift release between 2006-2010?", don't just search for 
  "Taylor Swift albums" - include "studio albums" AND the specific date range in your search.
- If asked about "Fortune 500 companies headquartered in California", don't just search for 
  "Fortune 500 companies" - include the location qualifier.

Always add relevant content to your knowledge base, then query it for answers.""",
            tools=[
                enhanced_web_search_tool,
                self.dynamic_qe_manager.get_tool(),
                code_execution_tool
            ],
            llm=proj_llm,
            max_steps=8,
            verbose=True
        )

        self.code_agent = ReActAgent(
            name="code_agent",
            description="Handles Python code for calculations and data processing",
            system_prompt="You are a Python programming specialist. You work with Python code to perform calculations, data analysis, and mathematical operations.",
            tools=[code_execution_tool],
            llm=code_llm,
            max_steps=6,
            verbose=True
        )

        # Fixed indentation: coordinator initialization inside __init__
        self.coordinator = AgentWorkflow(
            agents=[self.external_knowledge_agent, self.code_agent],
            root_agent="external_knowledge_agent"
        )

    def load_documents_from_file(self, file_path: str):
        """Load and process text documents for BM25, or return raw content for media files."""
        try:
            # Devine le type MIME du fichier
            mime_type, _ = mimetypes.guess_type(file_path)
            if mime_type is None:
                mime_type = ""
            print(f"Detected MIME type: {mime_type} for file {file_path}")
            # Traitement selon le type de fichier
            if mime_type.startswith("image") or mime_type.startswith("video") or mime_type.startswith("audio"):
                with open(file_path, "rb") as f:
                    binary_content = f.read()
                print(f"Loaded {mime_type} file: {file_path}")
                return binary_content
            else : 
            # Si fichier texte → traitement BM25
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            # Split en chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", ".", " ", ""]
            )

            chunks = text_splitter.split_text(content)
            docs = [Document(page_content=chunk, metadata={"source": file_path})
                    for chunk in chunks]

            # BM25 + agent
            self.retriever_tool = BM25RetrieverTool(docs)
            self._create_agent()

            print(f"Loaded {len(docs)} document chunks from {file_path}")
            return True

        except Exception as e:
            print(f"Error loading documents from {file_path}: {e}")
            return False


    def download_gaia_file(self, task_id: str, api_url: str = "https://agents-course-unit4-scoring.hf.space") -> str:
        """Download file associated with GAIA task_id and return its path"""
        try:
            response = requests.get(f"{api_url}/files/{task_id}", timeout=30)
            response.raise_for_status()

            # Try to get filename from headers
            print(f"Response headers: {response.headers}")
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

            print(f"Downloaded file saved as {filename}")
            return os.path.abspath(filename)  # Or just return `filename` for relative path

        except Exception as e:
            print(f"Failed to download file for task {task_id}: {e}")
            return None

    def add_documents_to_knowledge_base(self, file_path: str):
        """Add downloaded GAIA documents to the dynamic knowledge base"""
        try:
            documents = read_and_parse_content(file_path)
            if documents:
                self.dynamic_qe_manager.add_documents(documents)
                print(f"Added {len(documents)} documents from {file_path} to dynamic knowledge base")

                # Update the agent's tools with the refreshed query engine
                self.external_knowledge_agent.tools = [
                    enhanced_web_search_tool,
                    self.dynamic_qe_manager.get_tool(),  # Get the updated tool
                    code_execution_tool
                ]

                return True
        except Exception as e:
            print(f"Failed to add documents from {file_path}: {e}")
            return False

    async def solve_gaia_question(self, question_data: Dict[str, Any]) -> str:
        """
        Solve GAIA question with dynamic knowledge base integration
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
                    print(f"Successfully integrated GAIA file into dynamic knowledge base")
            except Exception as e:
                print(f"Failed to download/process file for task {task_id}: {e}")

        # Enhanced context prompt with dynamic knowledge base awareness and step-by-step reasoning
        context_prompt = f"""
GAIA Task ID: {task_id}
Question: {question}
{f'File processed and added to knowledge base: {file_path}' if file_path else 'No additional files'}

You are a general AI assistant. I will ask you a question. 

IMPORTANT INSTRUCTIONS:
1. Think through this STEP BY STEP, carefully analyzing all aspects of the question.
2. Pay special attention to specific qualifiers like dates, types, categories, or locations.
3. Make sure your searches include ALL important details from the question.
4. Report your thoughts and reasoning process clearly.
5. Finish your answer with: FINAL ANSWER: [YOUR FINAL ANSWER]

YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings. 
If you are asked for a number, don't use comma to write your number neither use units such as $ or percent sign unless specified otherwise. 
If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise. 
If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string."""

        try:
            ctx = Context(self.coordinator)
            print("=== AGENT REASONING STEPS ===")
            print(f"Dynamic knowledge base contains {len(self.dynamic_qe_manager.documents)} documents")

            handler = self.coordinator.run(ctx=ctx, user_msg=context_prompt)
            full_response = ""

            async for event in handler.stream_events():
                if isinstance(event, AgentStream):
                    print(event.delta, end="", flush=True)
                    full_response += event.delta

            final_response = await handler
            print("\n=== END REASONING ===")

            # Extract the final formatted answer
            #final_answer = final_answer_tool(str(final_response), question)
            #print(f"Final GAIA formatted answer: {final_answer}")
            #print(f"Knowledge base now contains {len(self.dynamic_qe_manager.documents)} documents")

            return final_response
        except Exception as e:
            error_msg = f"Error processing question: {str(e)}"
            print(error_msg)
            return error_msg

    def get_knowledge_base_stats(self):
        """Get statistics about the current knowledge base"""
        return {
            "total_documents": len(self.dynamic_qe_manager.documents),
            "document_sources": [doc.metadata.get("source", "Unknown") for doc in self.dynamic_qe_manager.documents]
        }

async def main():

    agent = EnhancedGAIAAgent()

    question_data = {
        "Question": "How many studio albums were published by Mercedes Sosa between 2000 and 2009 (included)? List them !",
        "task_id": ""
    }

    print(question_data)

   
    answer = await agent.solve_gaia_question(question_data)   
    print(f"Answer: {answer}")

if __name__ == '__main__':
    asyncio.run(main())