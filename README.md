# GAIA Agent — Final Assignment Template

This repository provides a GAIA (Unit 4) agent implementation with two main technical pillars:
1. A LlamaIndex-driven RAG + tool ReAct workflow (`agent.py` + `custom_models.py`).
2. A smolagents / Gemini local or API-first workflow (`agent2.py`, not expanded here per current task scope).

This README (updated to reflect ONLY the code paths present in `agent.py` and `custom_models.py`) documents what is actually implemented, clarifies prior placeholder claims, and lists remaining gaps if you intend to use this as a full leaderboard submission template.

## Quick map
- Kaggle / notebook: `hf-agents.ipynb` — uses the main LlamaIndex stack (`agent.py`, `custom_models.py`, `requirements.txt`, `appasync.py`).
- Local / dev UI: `agent2.py`, `app.py`, `requirements2.txt` — simplified smolagents/Gemini runner (not covered in this doc).
- **Text-only CLI**: `agent3.py` — lightweight GPT‑OSS + Qwen3 embedding workflow for CPU/GGUF environments.

I read the following sources from this repository and the Unit4 hands-on page to create this README:
- Web page: https://huggingface.co/learn/agents-course/unit4/hands-on (Unit4 hands-on description and API routes)
- Python files (read): `agent.py`, `agent2.py`, `custom_models.py`, `app.py`, `appasync.py`
- Notebooks (read): `hf-agents.ipynb`
- Requirements (read): `requirements.txt`, `requirements2.txt`

The rest of this README documents how the code maps to the Unit4 challenge and how to run and validate the agent locally and on Kaggle.

## Current capabilities

### 1. Runtime modes
- `USE_API_MODE=true` → Gemini (LLM + embedding); falls back to local mode automatically on failures.
- `USE_API_MODE=false` → Local models:
  * `agent.py` (requires `NONAPI_MULTIMODAL=true`) loads the **new `Qwen/Qwen3-VL-30B-A3B-Instruct`** custom wrapper with NF4 int4 quantization plus `jinaai/jina-embeddings-v4`, and pairs it with `Qwen/Qwen2.5-Coder-3B-Instruct-AWQ` for the dedicated code executor agent.
  * `agent3.py` provides the **text-only GPT-OSS route** (see “Text-only agent” below). Set `NONAPI_MULTIMODAL=false` only if you plan to drive `agent3.TextOnlyGptOssAgent`; `agent.py` now raises if you try to run without the multimodal stack.

### 2. Retrieval / ingestion (shared with previous README)
1. **Document ingestion** — `MultimodalPDFReader` (SmartPDFLoader + PyMuPDF) for PDFs; LibreOffice conversion for Office formats; audio/video Readers; plain image docs stored with binary in metadata.
2. **Node pipeline** — `UnstructuredElementNodeParser` → `RecursiveCharacterTextSplitter(4096/512 overlap)` (SentenceSplitter fallback) → SHA1 dedup.
3. **Vector store** — persistent `ChromaVectorStore` at `./chroma_db` with in-memory fallback.
4. **Embeddings + reranker** — `jinaai/jina-embeddings-v4` (4-bit NF4) + `jinaai/jina-reranker-v2-base-multilingual` (quantized when CUDA available; llama.cpp GGUF fallback on CPU). The reranker exposes a helper for non–llama-index callers used by `agent3`.
5. **Dynamic knowledge growth** — `enhanced_web_search_and_update` uses DuckDuckGo Search (DDGS) and `BeautifulSoupWebReader`/`YoutubeTranscriptReader`, automatically appending new docs to the current vector index.

### 3. Tooling and orchestration
- **Multimodal agent (`agent.py`)** — ReAct workflow with two sub-agents (knowledge + code). Tools include RAG query engine, web search, Python execution, and the Qwen3-VL image caption hook. Final answers are cleaned and reformatted via the same Qwen3-VL model inside `final_answer_tool`.
- Code sub-agent specifically uses `Qwen/Qwen2.5-Coder-3B-Instruct-AWQ` via LlamaIndex’s `HuggingFaceLLM`, matching the upstream AWQ loading guidance.
- **Text-only agent (`agent3.py`)** — Standalone pipeline using GPT‑OSS‑20B (transformers), `Qwen3GGUFEmbedding`, and the Jina reranker. It keeps a lightweight knowledge base (chunked GGUF embeddings), an integrated DDGS scraper, and a Python execution sandbox, all orchestrated through an iterative `TOOL_CALL:` loop similar to the old wrapper.

### 4. Final answer handling
- `final_answer_tool` still normalizes the response for GAIA rules. In text-only mode, `agent3` relies on prompt discipline plus the same regex cleanup logic from `agent.py`.

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
- `NONAPI_MULTIMODAL` (true/false) — keep `true` for `agent.py`. Set to `false` only if you are instantiating `TextOnlyGptOssAgent` from `agent3.py`. Default: true.
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

## Text-only agent (`agent3.py`)

`agent3.py` is the new text-only pipeline requested in the spec. Highlights:

- **LLM**: `openai/gpt-oss-20b` via `transformers` (Harmony chat template + iterative tool loop).
- **Embeddings / reranker**: `Qwen3GGUFEmbedding` (llama.cpp-backed GGUF download) and `jinaai/jina-reranker-v2-base-multilingual`. Both have CPU fallbacks, so this route can run fully offline.
- **Knowledge base**: `SimpleKnowledgeBase` chunks every ingested document, stores embeddings in memory, and surfaces top passages both via cosine similarity and Jina reranking.
- **Tools wired into GPT‑OSS**: DDGS-powered `enhanced_web_search`, RAG query, and `execute_python_code` sharing the same sandbox allowlist as `agent.py`.

Usage example:

```python
import asyncio
from agent3 import TextOnlyGptOssAgent

agent = TextOnlyGptOssAgent()
question = {"Question": "Who won the 2012 Nobel Prize in Literature?", "task_id": ""}
print(asyncio.run(agent.solve_gaia_question(question)))
```

Tips:
- Keep `NONAPI_MULTIMODAL=true` when running `agent.py`. Use `TextOnlyGptOssAgent` directly instead of toggling the env var.
- First run may download GGUF models (Qwen3 embedding + GPT‑OSS weights); ensure you have enough disk space and set `HF_HOME` if necessary.
