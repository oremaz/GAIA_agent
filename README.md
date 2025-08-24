# GAIA Agent — Final Assignment Template

This repository is a working template for the Hugging Face Agents Course Unit 4 hands-on/leaderboard (GAIA). It contains two agent variants and supporting utilities: a Kaggle-oriented notebook + non-API local LlamaIndex pipeline, and a local/remote API-based agent built with smolagents/Gemini. This README explains the implementation choices, hardware and environment requirements, and step-by-step instructions to run the project in the two supported environments.

## Quick map
- Kaggle (notebook): `hf-agents.ipynb` — uses `agent.py`, `custom_models.py`, `requirements.txt`, `appasync.py`.
- Local / development: `agent2.py`, `app.py`, `requirements2.txt` — simplified local runner using smolagents/Gemini and Langfuse integration.

I read the following sources from this repository and the Unit4 hands-on page to create this README:
- Web page: https://huggingface.co/learn/agents-course/unit4/hands-on (Unit4 hands-on description and API routes)
- Python files (read): `agent.py`, `agent2.py`, `custom_models.py`, `app.py`, `appasync.py`
- Notebooks (read): `hf-agents.ipynb`
- Requirements (read): `requirements.txt`, `requirements2.txt`

The rest of this README documents how the code maps to the Unit4 challenge and how to run and validate the agent locally and on Kaggle.

## Goals implemented (short)
- Implements the GAIA evaluation flow: fetch questions from the Unit4 scoring API, optionally download task files, extract and index content, answer questions and submit answers via POST /submit.
- Two runtime modes:
	- API mode (USE_API_MODE=true): uses cloud API models (Gemini or remote models) — CPU is acceptable.
	- Non-API mode (USE_API_MODE=false): local models. The repo supports both multimodal (images + text) and text-only local pipelines via the `NONAPI_MULTIMODAL` flag.
- Document handling: office files (docx/doc/pptx) are converted to PDF using LibreOffice headless (`soffice`) and parsed via `SmartPDFLoader` + `PyMuPDFReader` for images.
- Semantic-first chunking (preferred): `RecursiveCharacterTextSplitter(chunk_size=4096, chunk_overlap=512)` with fallback to sentence-based splitting.
- Local custom model wrappers in `custom_models.py`: Qwen multimodal LLM, Qwen3 GGUF embedding wrapper, Gemma3 4-bit quantized loader, and Jina embeddings/reranker wrappers.
- Persistent vector backend: optional Chroma integration (falls back to in-memory index if `chromadb` is unavailable).

## Requirements and environment

- Kaggle/notebook environment (recommended for running `hf-agents.ipynb`):
	- Python 3.10+ (the notebook uses Kaggle image by default)
	- Install `requirements.txt` as-is in the notebook cell: `!pip install -r requirements.txt` (the included notebook already runs this).
	- LibreOffice CLI (`soffice`) available — the notebook includes a command to install it (`!brew install --cask libreoffice` in the provided notebook; on Linux you may prefer `apt-get install libreoffice-common libreoffice-writer libreoffice-core` in a container/VM).
	- Hardware:
		- API mode: CPU is OK.
		- Non-API multimodal: 2 GPUs (NVIDIA T4 recommended) for stable multimodal model loading (Qwen/Gemma etc.).
		- Non-API text-only: 1 GPU (NVIDIA T4 recommended).

- Local / development environment (for `agent2.py`, `app.py`):
	- Install `requirements2.txt` (these packages are lighter and align with the smolagents + local evaluation flow): `pip install -r requirements2.txt`.
	- The `agent2.py` code expects access to Gemini via `GOOGLE_API_KEY` or an alternative model configured with smolagents; it uses Langfuse for observability when configured.

Environment variables used by the code
- `USE_API_MODE` (true/false) — when true the code prefers API/cloud models (Gemini). Default: false.
- `NONAPI_MULTIMODAL` (true/false) — when false + non-API => text-only pipeline (GPT-OSS + Qwen3-embeddings). Default: true.
- `GOOGLE_API_KEY` — used for Gemini API routes (API mode).
- `HUGGINGFACEHUB_API_TOKEN` / `HF_TOKEN` — used for HF Hub downloads where needed.
- `LLAMA_CLOUD_API_KEY` — optional, used for Llama Cloud services like LlamaParse when available.

## How the Kaggle notebook flow works (hf-agents.ipynb)

Main files involved: `hf-agents.ipynb` (runner), `agent.py` (main agent implementation), `custom_models.py` (local model wrappers), `appasync.py` (async Gradio variant). Key steps in the notebook:

1. Clone the repository and install dependencies: the notebook runs `!pip install -r requirements.txt`.
2. (Notebook provides) `!brew install --cask libreoffice` — ensures `soffice` exists for office→PDF conversion.
3. The notebook sets secrets for the environment (HF token, Gemini key, LLAMA_CLOUD_API_KEY) through Kaggle Secrets. The notebook then sets `USE_API_MODE` and `NONAPI_MULTIMODAL` environment variables.
4. Run the `agent.py` main or launch the Gradio UI using `appasync.py` — the notebook demonstrates both running the agent script and launching the UI.

Key implementation notes (Kaggle / `agent.py` / `custom_models.py`):
- initialize_models(use_api_mode, multimodal): selects models depending on `USE_API_MODE` and `NONAPI_MULTIMODAL`:
	- API mode + Gemini available -> uses `Gemini` LLM & `GeminiEmbedding`.
	- Non-API multimodal -> `QwenVLCustomLLM` + `JinaEmbeddingsV4` + a Qwen coder for code LLM (HF loader with quantization flags).
	- Non-API text-only -> `HuggingFaceLLM` using `openai/gpt-oss-20b` as the main LLM (and code LLM), and `Qwen3GGUFEmbedding()` for CPU GGUF embeddings.
- Document ingestion: `read_and_parse_content()` handles file download/format detection. Office documents (.doc/.docx/.pptx/.ppt/.odt) are converted to PDF using LibreOffice headless (`convert_to_pdf` helper) and then parsed via `MultimodalPDFReader` (which combines `SmartPDFLoader` for layout-text and `PyMuPDFReader` for images). In text-only mode the pipeline will prefer `SmartPDFLoader` only (no image nodes) while PyMuPDF is used only for image extraction in multimodal mode.
- Semantic chunking: `RecursiveCharacterTextSplitter(chunk_size=4096, chunk_overlap=512)` is used where available and falls back to `SentenceSplitter`. This implements the requested semantic-first chunking.
- Vector store / RAG: `DynamicQueryEngineManager._create_rag_tool()` converts documents to nodes using `UnstructuredElementNodeParser`, splits them into large semantic chunks, creates `ImageNode` objects for image docs, and attempts to wire a Chroma-backed persistent collection (via `chromadb`) with a fallback to the default in-memory `VectorStoreIndex` when Chroma is unavailable.
- Reranker: a HybridReranker wraps `JinaMultimodalReranker` (the reranker model selection depends on `NONAPI_MULTIMODAL`) and runs on CPU by default.

## Architectural comparison — smolagents vs LlamaIndex (and how we use them)

Concise, code-driven comparison (accurate to the repository):

- Both implementations are agentic: each exposes agents and tools, but the placement of "knowledge" vs "tools" differs.

- LlamaIndex variant (files: `agent.py`, `custom_models.py`)
	- Pattern: RAG-first, agent-enabled. The code builds a semantic knowledge base (nodes -> embeddings -> vector index) and exposes that KB as a QueryEngineTool which a ReAct-style agent calls.
	- Key components (exact symbols in `agent.py`):
		- Ingestion: `MultimodalPDFReader`, `SmartPDFLoader`, `PyMuPDFReader` (collects text + image bytes into `Document`).
		- Nodeization & chunking: `UnstructuredElementNodeParser` + `RecursiveCharacterTextSplitter` (4096/512) or `SentenceSplitter` fallback.
		- Indexing: `VectorStoreIndex(nodes)` with an attempted persistent backend using `ChromaVectorStore` (via `chromadb.Client`) and safe fallback to in-memory index.
		- Query tool: `QueryEngineTool.from_defaults(...)` created in `DynamicQueryEngineManager._create_rag_tool()` and stored on `dynamic_qe_manager`.
		- Agent control: `ReActAgent` + `AgentWorkflow` (the repo wires `external_knowledge_agent`, `code_agent`, and a coordinator `AgentWorkflow`). The agent calls `dynamic_qe_manager.get_tool()` and `code_execution_tool` during reasoning.
		- Reranking: `JinaMultimodalReranker` wrapped in the local `HybridReranker` class and applied as a node postprocessor on the query engine.
	- Behavioral notes: ingest-first (files → KB), then the agent queries the KB. The repo also supports 'enhanced_web_search' (`enhanced_web_search_tool`) which appends web documents into the same KB and recreates the index.

- smolagents variant (file: `agent2.py`)
	- Pattern: Tool-first, multi-agent orchestration. Agents are composed from explicit tools and managed agents rather than implicitly querying a central vector KB.
	- Key components (exact symbols in `agent2.py`):
	- Tools: `WebSearchTool`, `visit_webpage`, `get_youtube_transcript`, `UnifiedMultimodalTool`, `FinalAnswerTool`, `ChromaBM25HybridRetrieverTool` (uses LangChain's `EnsembleRetriever` to combine Chroma and BM25 retrievers; alpha=0.5 by default).
		- Agents: `CodeAgent` (coder), `ToolCallingAgent` (retriever/runner) and a high-level `ToolCallingAgent` that can manage `managed_agents` like the coder and retriever.
		- Observability: Langfuse/OpenTelemetry setup in `_setup_langfuse_observability()` and instrumentation via `SmolagentsInstrumentor`.


Short control-flow sketches (textual):
- LlamaIndex (RAG-first): [Ingest files/web -> Parse -> Nodes -> Embed -> VectorStore] -> QueryEngineTool -> ReActAgent -> LLM (+ code tool)
- smolagents (Tool-first): Agent(s) [ToolCallingAgent, CodeAgent] -> call tools {WebSearch, UnifiedMultimodalTool, HybridRetriever} -> LLMs or subprocess tools

Implementation pointers (from code):
- The LlamaIndex flow recreates its index when `dynamic_qe_manager.add_documents()` is called (used by `enhanced_web_search_and_update`).
- The smolagents flow creates BM25 chunks inside `GAIAAgent.load_documents_from_file()` and replaces/attaches `BM25RetrieverTool` to the agent.

This updated section is intentionally precise and tied to the actual symbols and functions used in the repo so readers can quickly map design decisions to the concrete code.

## Agent architectures (detailed per runtime case)

Below are concise architecture diagrams and component roles for each supported case in this repo.

1) API mode (Kaggle / `agent.py`, `USE_API_MODE=true`)
- Core LLM: Gemini (via `llama_index.llms.gemini.Gemini`) — API-backed model for reasoning and generation.
- Embeddings: `GeminiEmbedding` (API) when available.
- Ingestion: `MultimodalPDFReader` or LlamaParse (if `LLAMAPARSE_AVAILABLE`) to parse files.
- Indexing: lightweight `VectorStoreIndex` built from parsed nodes. Optionally persistent Chroma if `chromadb` is configured.
- Reranking: `JinaMultimodalReranker` (CPU) if enabled.
- Agent loop: ReAct-style `ReActAgent` / `AgentWorkflow` (LlamaIndex agent) handles step-by-step reasoning, tool calls and code execution via `code_execution_tool`.
- When to use: CPU-only environments where cloud LLMs handle heavy lifting (recommended for small GPU or CPU-only Kaggle instances).

2) Non-API multimodal (Kaggle notebook, `USE_API_MODE=false` and `NONAPI_MULTIMODAL=true`)
- Core LLM: local multimodal LLM wrapper `QwenVLCustomLLM` or `Gemma3CustomLLM` (if available and GPU resources permit). These load large multimodal weights and use `AutoProcessor` for images+text.
- Embeddings: `JinaEmbeddingsV4` (multimodal embeddings) — can encode text, images, or text+image.
- Ingestion: `MultimodalPDFReader` extracts text and images from PDFs; image nodes become `ImageNode` objects.
- Chunking & nodes: `UnstructuredElementNodeParser` → `RecursiveCharacterTextSplitter` (4096/512) produces semantic nodes.
- Vector store: `ChromaVectorStore` preferred (with persistent `chroma_db`) and fallback to in-memory `VectorStoreIndex`.
- Reranker: `JinaMultimodalReranker` applied as a node postprocessor to support text↔image ranking.
- Agent loop: LlamaIndex-based `ReActAgent` (external_knowledge_agent) uses the dynamic query engine and code agent; tools include `enhanced_web_search_tool`, query engine, and `code_execution_tool`.
- Hardware note: requires multiple GPUs (2x T4 recommended) for stable device_map="auto" multimodal loads.

3) Non-API text-only (Kaggle notebook, `USE_API_MODE=false` and `NONAPI_MULTIMODAL=false`)
- Core LLM: `HuggingFaceLLM` wrapping `openai/gpt-oss-20b` configured for GPU if available (single GPU supported).
- Code LLM: same as core LLM (requested behavior).
- Embeddings: `Qwen3GGUFEmbedding` (GGUF via `llama_cpp` when available) running on CPU.
- Ingestion: PDFs parsed with `SmartPDFLoader` only (no image nodes). PyMuPDF used only if image extraction is explicitly enabled.
- Chunking & nodes: same semantic-first chunker (4096/512) but image nodes omitted.
- Vector store & Reranking: Chroma optional; reranker defaults to Jina reranker v2 (CPU) for text-only.
- Agent loop: LlamaIndex ReAct agent + code agent tools.
- Hardware note: single GPU (T4) recommended for GPT-OSS text-only.

4) Local smolagents flow (`agent2.py`, `app.py`) — tool-calling architecture
- Model: smolagents `OpenAIServerModel` or `genai.Client` driven models (Gemini) via API key — this flow is oriented around API calls.
- Agent composition:
	- `CodeAgent` — dedicated to performing Python code execution tasks via `PythonInterpreterTool`.
	- `ToolCallingAgent` / `ToolCallingAgent` (higher-level agent) — routes queries to tools like `BM25RetrieverTool`, `WebSearchTool`, `UnifiedMultimodalTool` and the coder agent.
	- `UnifiedMultimodalTool` — handles audio/video/image processing using the cloud files API or inline processing.
	- Retrieval: Chroma + BM25 hybrid retriever in `agent2.py` (`ChromaBM25HybridRetrieverTool`). The tool composes the Chroma retriever (via `chroma.as_retriever(...)`) and the `BM25Retriever` into a LangChain `EnsembleRetriever` and uses the ensemble to fetch relevant Documents. The `alpha` parameter (default 0.5) is used to build ensemble weights (dense weight = alpha, lexical weight = 1-alpha).
- Observability: Langfuse + OpenTelemetry added to trace agent flows and outputs.
- When to use: local dev or API-first setups where you prefer explicit tool contracts, multi-agent orchestration, and observability.

Small diagram (textual):
- LlamaIndex variant (RAG-first): [Ingest -> Parse -> Node/Chunks -> Embed -> VectorStore -> QueryEngine] -> Agent (ReAct) -> LLM
- smolagents variant (Tool-first): [Tools: WebSearch, BM25, MultimodalTool, CodeTool] <- Agent(s) (ToolCalling / CodeAgent) -> LLM(s)

## Engineering notes and trade-offs
- LlamaIndex centralizes document semantics, making it easier to scale retrieval and reranking for a large knowledge base. It fits the GAIA homework where file attachments are common and a searchable KB is useful.
- smolagents makes it straightforward to attach deterministic tools (code execution, exact web scraping, audio transcription) into an agent workflow and manage multiple collaborating agents; it is well-suited for orchestrating tool-heavy pipelines and for integrating observability.
- Both patterns can coexist: in this repo the primary Kaggle flow uses LlamaIndex for RAG, while `agent2.py` demonstrates a smolagents-driven pattern for local/API-first experiments.

How to run the Kaggle notebook (short checklist)
1. Open `hf-agents.ipynb` in Kaggle or your notebook environment.
2. Run the cells in order: clone repo, pip install `requirements.txt`, install LibreOffice (provided in the notebook), set environment secrets (HF/Gemini/LLAMA_CLOUD keys) via Kaggle secrets, set `USE_API_MODE`/`NONAPI_MULTIMODAL` as needed.
3. Run the test cell that executes `python agent.py` or launch Gradio with `python appasync.py`.

Notes about hardware on Kaggle
- For non-API multimodal runs you will need at least 2 GPUs (NVIDIA T4 recommended) because the multimodal models try to use device_map="auto" and may place large shards across multiple devices.
- For text-only (non-API) a single GPU (T4) is typically sufficient.
- API mode uses cloud models and is tolerant of CPU-only environments.

## How the Local flow works (`agent2.py`, `app.py`, `requirements2.txt`)

Main idea: `agent2.py` provides a more opinionated local agent using smolagents/OpenInference and Gemini API integration for the Unit4 benchmark flows. `app.py` is a simple Gradio wrapper that imports the local `GAIAAgent` class from `agent2.py` and exposes the Run+Submit UI.

Key points:
- `agent2.py` expects `GOOGLE_API_KEY` for Gemini API usage (it uses `genai.Client` or smolagents OpenAIServerModel depending on configuration).
- Observability: optional Langfuse / OpenTelemetry integration — configure `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` to enable traces.
- The local runner uses `requirements2.txt` (lighter set) — install via `pip install -r requirements2.txt`.

How to run locally (short checklist)
1. Create a Python venv and activate it.
2. Install dependencies: `pip install -r requirements2.txt`.
3. Export required env vars: at minimum `GOOGLE_API_KEY` (for Gemini usage) and optionally `HUGGINGFACEHUB_API_TOKEN`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`.
4. Run `python app.py` to launch the Gradio UI for local evaluation (it fetches the question list and submits answers to the Unit4 scoring API as the notebook-based flow does).

## Important runtime behaviors and guards
- LibreOffice conversion (`soffice`) is required for office → PDF conversion. `convert_to_pdf` fails fast with a RuntimeError if `soffice` is not on PATH — this makes failures explicit in CI and notebook runs.
- Chroma integration: the code attempts to import and initialize `chromadb` and create a persistent collection at `./chroma_db`. If chroma is not installed or initialization fails, the code logs the error and falls back to the default in-memory `VectorStoreIndex`.
- Model loads and quantization: many of the local model paths use 4-bit quantization (BitsAndBytes) and device_map="auto". Make sure matching CUDA + bitsandbytes + transformers versions are present when running non-API local multimodal pipelines.

## Troubleshooting / common errors
- Missing `soffice` / LibreOffice: install LibreOffice and ensure `soffice` is on PATH. On Debian/Ubuntu: `sudo apt-get update && sudo apt-get install -y libreoffice`.
- bitsandbytes / 4-bit loads failing: verify CUDA version compatibility and reinstall `bitsandbytes` compiled for your CUDA.
- `chromadb` import errors: install `chromadb==0.4.0` (already present in `requirements.txt`) or run in environments where chroma is not required — the code will fallback.
- llama_cpp / GGUF usage: `Qwen3GGUFEmbedding` tries to use `llama_cpp` when available and may download GGUF artifacts from HF; ensure `llama-cpp-python` and related dependencies are installed for GGUF runtime.

## How this solves the Unit4 hands-on
- Implements the same API routes described in the Unit4 hands-on: `GET /questions` (fetch), `GET /files/{task_id}` (download files), and `POST /submit` (submit results). The template demonstrates the full loop: fetch questions, download files where necessary, extract and index file content, run the agent to answer, and submit answers in GAIA format.
- The repository includes both an API/cloud friendly path (Gemini) and a local, self-contained path (Qwen/Gemma local wrappers + GGUF embedding) so you can iterate in CPU-only and GPU-enabled environments.

## Checklist (requirements coverage)
- Add multimodal boolean and USE_API_MODE branching -> Done (see `initialize_models` in `agent.py`).
- Text-only pipeline uses GPT-OSS on GPU and Qwen3 GGUF embeddings on CPU -> Implemented in `initialize_models` and `custom_models.py`.
- Office files converted to PDF using LibreOffice (`convert_to_pdf`) -> Done (LibreOffice-only, fail-fast).
- Semantic-first chunking (chunk_size=4096, overlap=512) -> Done (RecursiveCharacterTextSplitter fallback present).
- Gemma3 27B int4 model wrapper and Qwen3 GGUF embedding wrapper -> Done in `custom_models.py`.
- Chroma persistent backend with fallback -> Attempted in `_create_rag_tool` (falls back to in-memory `VectorStoreIndex`).

If you want, I can now:
1. Add a small `USE_CHROMA` env toggle to avoid chroma import attempts in dev environments.
2. Add a smoke test script that validates `convert_to_pdf` error behavior (no soffice) and Chroma fallback behavior.
3. Help you prepare a minimal Dockerfile / Kaggle environment YAML to speed up reproducible runs.

---

Short completion summary: I read the Unit4 hands-on page, `agent.py`, `agent2.py`, `custom_models.py`, `app.py`, `appasync.py`, `hf-agents.ipynb`, `requirements.txt` and `requirements2.txt`, and produced this README that explains how to run the Kaggle notebook variant and the local variant and maps code regions to the Unit4 challenge requirements.

