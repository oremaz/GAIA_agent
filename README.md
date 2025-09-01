# GAIA Agent — Final Assignment Template

This repository provides a GAIA (Unit 4) agent implementation with two main technical pillars:
1. A LlamaIndex-driven RAG + tool ReAct workflow (`agent.py` + `custom_models.py`).
2. A smolagents / Gemini local or API-first workflow (`agent2.py`, not expanded here per current task scope).

This README (updated to reflect ONLY the code paths present in `agent.py` and `custom_models.py`) documents what is actually implemented, clarifies prior placeholder claims, and lists remaining gaps if you intend to use this as a full leaderboard submission template.

## Quick map
- Kaggle (notebook): `hf-agents.ipynb` — uses `agent.py`, `custom_models.py`, `requirements.txt`, `appasync.py`.
- Local / development: `agent2.py`, `app.py`, `requirements2.txt` — simplified local runner using smolagents/Gemini and Langfuse integration.

I read the following sources from this repository and the Unit4 hands-on page to create this README:
- Web page: https://huggingface.co/learn/agents-course/unit4/hands-on (Unit4 hands-on description and API routes)
- Python files (read): `agent.py`, `agent2.py`, `custom_models.py`, `app.py`, `appasync.py`
- Notebooks (read): `hf-agents.ipynb`
- Requirements (read): `requirements.txt`, `requirements2.txt`

The rest of this README documents how the code maps to the Unit4 challenge and how to run and validate the agent locally and on Kaggle.

## Current capabilities (based on `agent.py` + `custom_models.py`)
Implemented:
1. Dual runtime modes selected by env vars:
	- `USE_API_MODE=true` → Gemini (LLM + embedding) if Google key present, with fallback to local mode on failure.
	- `USE_API_MODE=false` → Local models. Branching again on `NONAPI_MULTIMODAL`:
	  * `NONAPI_MULTIMODAL=true` (multimodal local) loads a large Hugging Face model: `Qwen/Qwen3-30B-A3B-Instruct-2507` via `HuggingFaceLLM` plus Jina V4 embeddings; adds an optional lightweight captioning model (`Qwen/Qwen2.5-VL-7B-Instruct-AWQ`) only for image description.
	  * `NONAPI_MULTIMODAL=false` (text-only local) uses a custom `GPTOSSWrapper` (model: `openai/gpt-oss-20b`) for both reasoning + code, and a `Qwen3GGUFEmbedding` (GGUF / llama.cpp fallback) for embeddings.
2. Incremental multimodal RAG:
	- Parsing: `MultimodalPDFReader` merges `SmartPDFLoader` (layout text) with `PyMuPDFReader` (images) — current implementation only appends image documents from PyMuPDF (text from SmartPDFLoader already loaded).
	- Office docs auto-converted to PDF via LibreOffice (`soffice`) in `convert_to_pdf` (hard fail with explicit RuntimeError if absent).
	- Audio/video handled with `VideoAudioReader` (non‑API) or `AssemblyAIAudioTranscriptReader` (API mode). Images become `Document` objects with binary bytes in metadata.
3. Node pipeline: `UnstructuredElementNodeParser` → `RecursiveCharacterTextSplitter(4096, 512)` (or `SentenceSplitter` fallback) → dedup by SHA1 of normalized text.
4. Vector store: attempts persistent Chroma (`duckdb+parquet` at `./chroma_db`) else falls back to in-memory `VectorStoreIndex` seamlessly.
5. Reranking: dynamic `HybridReranker` chooses `jinaai/jina-reranker-m0` (multimodal) or `jinaai/jina-reranker-v2-base-multilingual` (text-only). Applied as node postprocessor.
6. Dynamic knowledge growth: web search tool (`enhanced_web_search`) scrapes a top result (DuckDuckGo Search → first link via `ddgs`) using `BeautifulSoupWebReader` or `YoutubeTranscriptReader`, auto-inserts docs, and reuses the same index incrementally (`insert_nodes`).
7. Safe Python execution: sandboxed `execute_python_code` with curated builtins + optional scientific libs if available (no filesystem/network escalation safeguards beyond import allowlist—treat as semi‑trusted, not fully sandboxed for multi-user).
8. Tool-enabled agents:
	- Multimodal branch: Two `ReActAgent`s (`external_knowledge_agent` + `code_execution_agent`) orchestrated by an `AgentWorkflow` (root = external knowledge). Tools: dynamic RAG query engine, web search, code exec, optional image captioning.
	- Text-only branch: Custom iterative reasoning + tool loop inside `GPTOSSWrapper.solve` (regex extracts `TOOL_CALL:` directives) enabling web search, RAG queries, and code execution.
9. Final answer normalization: `final_answer_tool` cleans model output and optionally re-prompts the main LLM to extract a GAIA-compliant concise answer (numbers / short strings / CSV list) using explicit format rules.

Optional / available but unused by default path choices:
- `Qwen25VLMultiModal` (custom wrapper) exists in `custom_models.py` but is only invoked for the lightweight 7B captioning model via `get_or_create_qwen_vl_llm`, not for the main 30B reasoning LLM in current code.
- `QwenCoderGGUFLLM` helper present but not wired into `initialize_models`.

## Requirements and environment

- Kaggle/notebook environment (recommended for running `hf-agents.ipynb`):
	- Python 3.10+ (the notebook uses Kaggle image by default)
	- Install `requirements.txt` as-is in the notebook cell: `!pip install -r requirements.txt` (the included notebook already runs this).
	- Hardware:
		- API mode: CPU is OK.
		- Non-API multimodal: 2 GPUs (NVIDIA T4 recommended) for stable multimodal model loading (Qwen models).
		- Non-API text-only: 1 GPU (NVIDIA T4 recommended).

- Local / development environment (for `agent2.py`, `app.py`):
	- Install `requirements2.txt` (these packages are lighter and align with the smolagents + local evaluation flow): `pip install -r requirements2.txt`.
	- The `agent2.py` code expects access to Gemini via `GOOGLE_API_KEY` or an alternative model configured with smolagents; it uses Langfuse for observability when configured.

Environment variables used by the code
- `USE_API_MODE` (true/false) — when true the code prefers API/cloud models (Gemini). Default: false.
- `NONAPI_MULTIMODAL` (true/false) — when false + non-API => text-only pipeline (GPT-OSS + Qwen3 GGUF embeddings). Default: true.
- `GOOGLE_API_KEY` — used for Gemini API routes (API mode).
- `HUGGINGFACEHUB_API_TOKEN` / `HF_TOKEN` — used for HF Hub downloads where needed.
- `LLAMA_CLOUD_API_KEY` — optional, used for Llama Cloud services like LlamaParse when available.

## How the Kaggle notebook flow works (hf-agents.ipynb)

Main files involved: `hf-agents.ipynb` (runner), `agent.py` (main agent implementation), `custom_models.py` (local model wrappers), `appasync.py` (async Gradio variant). Key steps in the notebook:

1. Clone the repository and install dependencies: the notebook runs `!pip install -r requirements.txt`.
2. The notebook sets secrets for the environment (HF token, Gemini key, LLAMA_CLOUD_API_KEY) through Kaggle Secrets. The notebook then sets `USE_API_MODE` and `NONAPI_MULTIMODAL` environment variables.
3. Run the `agent.py` main or launch the Gradio UI using `appasync.py` — the notebook demonstrates both running the agent script and launching the UI.

Key implementation notes (reflecting current code):
- Model init (`initialize_models` in `agent.py`):
	* API mode: Gemini LLM (`models/gemini-2.5-flash`) + `GeminiEmbedding` (document retrieval embedding).
	* Non-API multimodal: `HuggingFaceLLM` wrapping `Qwen/Qwen3-30B-A3B-Instruct-2507` + `JinaEmbeddingsV4` (multimodal). 7B captioner (`Qwen2.5-VL-7B-Instruct-AWQ`) for image description only.
	* Non-API text-only: `GPTOSSWrapper` (custom iterative tool loop) for `openai/gpt-oss-20b` + `Qwen3GGUFEmbedding` (GGUF / llama.cpp fallback). Same model reused as code LLM.
- Ingestion: `read_and_parse_content` detects type; office docs converted via LibreOffice; PDFs go through `MultimodalPDFReader` (SmartPDFLoader text + PyMuPDF image docs). Audio handled differently API vs local. Images loaded as binary metadata.
- Chunking: Semantic-first splitter (4096 / 512) with fallback.
- Index: Chroma attempt with compatibility handling (different constructor signatures) → fallback to in-memory.
- Reranking: hybrid wrapper with Jina reranker variant selection.
- Dynamic updates: `enhanced_web_search_and_update` inserts new docs; index updated via `insert_nodes` (no full rebuild unless insert fails).
- Final answer formatting: cleaning + LLM reformat pass for GAIA answer constraints.

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

2) Non-API multimodal (Kaggle notebook, `USE_API_MODE=false`, `NONAPI_MULTIMODAL=true`)
- Core LLM: `HuggingFaceLLM` wrapping `Qwen/Qwen3-30B-A3B-Instruct-2507` (quantized). A separate lightweight `Qwen2.5-VL-7B-Instruct-AWQ` captioner may initialize for image → text descriptions.
- Embeddings: `JinaEmbeddingsV4` (text / image / cross).
- Others: (same as earlier list).

3) Non-API text-only (Kaggle notebook, `USE_API_MODE=false`, `NONAPI_MULTIMODAL=false`)
- Core & Code LLM: `GPTOSSWrapper` (iterative tool calling over `openai/gpt-oss-20b`).
- Embeddings: `Qwen3GGUFEmbedding` (CPU, llama.cpp if available; zero-vector fallback otherwise).
- Agent Loop: Iterative GPT-OSS tool loop: call tools for facts, then return FINAL ANSWER.

4) Local smolagents flow (`agent2.py`, `app.py`) — tool-calling architecture
- Model: smolagents `OpenAIServerModel` or `genai.Client` driven models (Gemini) via API key — this flow is oriented around API calls.
- Agent composition:
	- `CodeAgent` — dedicated to performing Python code execution tasks via `PythonInterpreterTool`.
	- `ToolCallingAgent` / `ToolCallingAgent` (higher-level agent) — routes queries to tools like `BM25RetrieverTool`, `WebSearchTool`, `UnifiedMultimodalTool` and the coder agent.
	- `UnifiedMultimodalTool` — handles audio/video/image processing using the cloud files API or inline processing.
	- Retrieval: Chroma + BM25 hybrid retriever in `agent2.py` (`ChromaBM25HybridRetrieverTool`). The tool composes the Chroma retriever (via `chroma.as_retriever(...)`) and the `BM25Retriever` into a LangChain `EnsembleRetriever` and uses the ensemble to fetch relevant Documents. The `alpha` parameter (default 0.5) is used to build ensemble weights (dense weight = alpha, lexical weight = 1-alpha).
- Observability: Langfuse + OpenTelemetry added to trace agent flows and outputs.
- When to use: local dev or API-first setups where you prefer explicit tool contracts, multi-agent orchestration, and observability.

Small diagram:
- LlamaIndex variant (RAG-first): [Ingest -> Parse -> Node/Chunks -> Embed -> VectorStore -> QueryEngine] -> Agent (ReAct) -> LLM
- smolagents variant (Tool-first): [Tools: WebSearch, BM25, MultimodalTool, CodeTool] <- Agent(s) (ToolCalling / CodeAgent) -> LLM(s)

## Engineering notes and trade-offs
- `insert_nodes` incremental growth avoids full index rebuilds; dedup by content hash prevents ballooning duplicates when repeatedly searching the same URL.
- GPT-OSS tool loop uses regex on raw generations; robustness relies on constraining the prompt to elicit `TOOL_CALL:` lines—could be hardened with a function-calling wrapper.
- Safety: Python execution environment is permissive for local analytics; do not expose it publicly without additional sandboxing.
- Reranker lazy loading (and GGUF fallback) reduces cold start cost but adds first-query latency.

How to run the Kaggle notebook (short checklist)
1. Open `hf-agents.ipynb` in Kaggle or your notebook environment.
2. Run the cells in order: clone repo, pip install `requirements.txt`, set environment secrets (HF/Gemini/LLAMA_CLOUD keys) via Kaggle secrets, set `USE_API_MODE`/`NONAPI_MULTIMODAL` as needed.
3. Run the test cell that executes `python agent.py` or launch Gradio with `python appasync.py`.

Notes about hardware on Kaggle
- For non-API multimodal runs you will need 2  NVIDIA T4 GPUs.
- For text-only (non-API) a single GPU (T4) is typically sufficient.
- API mode uses cloud models and is tolerant of CPU-only environments.

## How the Local flow works (`agent2.py`, `app.py`, `requirements2.txt`)

Key points:
- `agent2.py` expects `GOOGLE_API_KEY` for Gemini API usage (it uses `genai.Client` or smolagents OpenAIServerModel depending on configuration).
- Observability: Langfuse / OpenTelemetry integration — configure `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` to enable traces.
- The local runner uses `requirements2.txt` (lighter set) — install via `pip install -r requirements2.txt`.

How to run locally (short checklist)
1. Create a Python venv and activate it.
2. Install dependencies: `pip install -r requirements2.txt`.
3. Export required env vars: at minimum `GOOGLE_API_KEY` (for Gemini usage), `HUGGINGFACEHUB_API_TOKEN`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`.
4. Run `python app.py` to launch the Gradio UI for local evaluation (it fetches the question list and submits answers to the Unit4 scoring API as the notebook-based flow does).

## Important runtime behaviors and guards
- Chroma integration: the code attempts to import and initialize `chromadb` and create a persistent collection at `./chroma_db`. If chroma is not installed or initialization fails, the code logs the error and falls back to the default in-memory `VectorStoreIndex`.
- Model loads and quantization: many of the local model paths use 4-bit quantization (BitsAndBytes) and device_map="auto". Make sure matching CUDA + bitsandbytes + transformers versions are present when running non-API local multimodal pipelines.


## Checklist (concise mapping to current code)
- Environment mode switches (`USE_API_MODE`, `NONAPI_MULTIMODAL`) → Yes (`initialize_models`).
- Large local multimodal model (Qwen3 30B) or API Gemini → Yes.
- Text-only local path with GPT-OSS + Qwen3 GGUF embeddings → Yes.
- LibreOffice headless conversion (fail-fast) → Yes (`convert_to_pdf`).
- Semantic-first chunking (4096/512) with fallback → Yes.
- Incremental dynamic RAG with reranking → Yes.
- Image caption tool (optional 7B) → Yes (registered only if init succeeds).
- Final answer normalization for GAIA format → Yes (`final_answer_tool`).
- Full GAIA fetch/submit automation → Not yet (manual driver needed).
