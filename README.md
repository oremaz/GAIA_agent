# GAIA Agent — Final Assignment Template

This repository offers a complete Unit 4 GAIA agent with three practical entry points:

- **Multimodal RAG agent (`agent.py` + `custom_models.py`)** – LlamaIndex-based ReAct workflow that ingests local/web data, reranks with Jina models, and answers with Qwen3-VL plus a dedicated Qwen2.5 Coder LLM for code execution. `initialize_models` compares the Gemini API path (`USE_API_MODE=true`) with the local Qwen stack (`USE_API_MODE=false`) so you can benchmark both approaches.
- **Text-only agent (`agent3.py`)** – GPT‑OSS pipeline (GGUF embeddings + Jina reranker) designed for single-GPU runs (T4-class). Set `TEXT_ONLY=true` when running `appasync.py` to switch to this implementation automatically.
- **smolagents/Gemini runner (`agent2.py`, `app.py`)** – Tool-first architecture for developers who prefer Gemini APIs and Langfuse observability.

## Components at a glance
- `agent.py`: main multimodal GAIA agent wired through LlamaIndex.
- `custom_models.py`: cached loaders for Jina embeddings/reranker, Qwen3-VL multimodal wrapper, and the Qwen2.5 coder LLM.
- `agent3.py`: standalone GPT‑OSS–based agent with its own knowledge base and tool loop.
- `appasync.py`: Gradio runner that imports `GAIAAgent` unless `TEXT_ONLY=true`, in which case it instantiates the text-only agent.
- `agent2.py` / `app.py`: smolagents workflow (Gemini + Langfuse logging).

## Runtime modes (used by `agent.py`, `agent3.py`, and `appasync.py`)
| Mode | Env vars | LLMs | Notes |
|------|----------|------|-------|
| API (Gemini) | `USE_API_MODE=true` | `Gemini` LLM + `GeminiEmbedding` | Falls back to local mode if credentials missing. |
| Local multimodal | `USE_API_MODE=false` | `Qwen/Qwen3-VL-30B-A3B-Instruct` + `Qwen/Qwen2.5-Coder-3B-Instruct-AWQ` | Default path in `agent.py`; image captioning reuses the same Qwen3-VL instance. |
| Text-only CLI | `TEXT_ONLY=true` (for `appasync.py`) or import `TextOnlyGptOssAgent` directly | `openai/gpt-oss-20b` + `Qwen3GGUFEmbedding` | Requires one GPU (T4+). No llama-index dependency; dedicated knowledge base and reranker in `agent3.py`. |

## Multimodal pipeline (`agent.py`)
1. **Document ingestion** – `MultimodalPDFReader` combines `SmartPDFLoader` (layout text) with `PyMuPDFReader` (images). Office docs auto-convert to PDF via LibreOffice. Audio/video readers switch between AssemblyAI and `VideoAudioReader` depending on API mode.
2. **Chunking** – `UnstructuredElementNodeParser` followed by `RecursiveCharacterTextSplitter(4096/512)` or `SentenceSplitter` fallback. Dedup via SHA‑1 hashes on normalized text.
3. **Index & embeddings** – Attempts persistent `ChromaVectorStore` (`./chroma_db`) with graceful fallback to in-memory. Embeddings come from `jinaai/jina-embeddings-v4` (4‑bit NF4) and reranking via `jinaai/jina-reranker-v2-base-multilingual` (quantized on CUDA, GGUF fallback on CPU).
4. **Dynamic updates** – `enhanced_web_search_and_update` (DuckDuckGo Search via DDGS + BeautifulSoup/Youtube reader) inserts new documents and rebuilds the query tool incrementally.
5. **Agents & tools** – Two LlamaIndex `ReActAgent`s (knowledge + code) orchestrated by `AgentWorkflow`. Tools include the RAG query engine, web search, safe Python execution (`execute_python_code` with a curated builtins set), and image captioning through Qwen3-VL. The code agent explicitly uses the Qwen2.5 coder model for deterministic scripting.
6. **Final answer formatting** – `final_answer_tool` strips boilerplate and (optionally) re-prompts the main LLM to emit GAIA-compliant concise answers.

## Text-only pipeline (`agent3.py`)
- **Model stack** – GPT‑OSS‑20B (Harmony chat template) for reasoning, `Qwen3GGUFEmbedding` for vectorization (llama.cpp-backed GGUF download), and the same Jina reranker logic exposed as `score_text_pairs`.
- **Knowledge base** – `SimpleKnowledgeBase` chunks every ingested file (supports txt/json/csv/pdf/docx) and stores embeddings in memory (`KnowledgeChunk`). Retrieval combines cosine similarity with reranker scores.
- **Tools** – DDGS web search scraper, knowledge query, and the same Python execution sandbox. `GptOssReasoner` implements the iterative `TOOL_CALL: name("...")` loop.
- **API** – `TextOnlyGptOssAgent.solve_gaia_question` remains async to match the multimodal interface, even though its inner operations are synchronous today.

## Running the Kaggle notebook (`hf-agents.ipynb`)
1. Clone the repo and `pip install -r requirements.txt`.
2. Set secrets (`HUGGINGFACEHUB_API_TOKEN`, `GOOGLE_API_KEY`, `LLAMA_CLOUD_API_KEY`) via Kaggle secrets.
3. Choose `USE_API_MODE=true` for Gemini or `false` for local Qwen. Set `TEXT_ONLY=true` only if you plan to run the GPT‑OSS CLI agent instead of the multimodal pipeline.
4. Execute the test cell (runs `python agent.py`) or launch the async Gradio app (`python appasync.py`). `agent2.py`/`app.py` rely on local servers and should be run on your own machine rather than within the Kaggle notebook.

Hardware tips:
- API mode is CPU-friendly.
- Local multimodal runs expect at least two T4s (Qwen3-VL + coder).
- Text-only runs generally need one T4 (GGUF on CPU runs but is significantly slower).

## Environment variables
- `USE_API_MODE` – switch between Gemini (true) and local Qwen models (false). Default: false.
- `TEXT_ONLY` – when true, `appasync.py` imports `TextOnlyGptOssAgent` and ignores `USE_API_MODE`. Default: false.
- `GOOGLE_API_KEY`, `HUGGINGFACEHUB_API_TOKEN`, `LLAMA_CLOUD_API_KEY` – optional credentials for external services.

## smolagents workflow (`agent2.py`, `app.py`) – run locally
- Tool-first design: `WebSearchTool`, `UnifiedMultimodalTool`, `ChromaBM25HybridRetrieverTool`, `FinalAnswerTool`, etc.
- Agents include a coder (`CodeAgent`) and a top-level `ToolCallingAgent` that routes to managed agents/tools.
- Retrieval uses LangChain’s `EnsembleRetriever` to blend Chroma embeddings with BM25; `alpha=0.5` weights lexical vs dense matches.
- Observability via Langfuse/OpenTelemetry (`_setup_langfuse_observability`).
- Use this path when you prefer Gemini APIs or Langfuse instrumentation; otherwise `agent.py` is the default for leaderboard-style evaluations.

## Engineering notes
- Incremental updates use `insert_nodes` to avoid rebuilding the entire index; SHA‑1 dedup prevents repeated web content explosion.
- The reranker lazily loads (and uses GGUF fallback) to keep cold-start memory low, at the cost of first-query latency.
- Safe-code execution is intentionally permissive for local analytics—add further sandboxing before exposing it publicly.
- `appasync.py` runs Gradio in queue mode (`demo.queue().launch(...)`) so long-running async calls don’t block concurrent users.

## Checklist (current coverage)
- Environment switches (`USE_API_MODE`, `TEXT_ONLY`) → Yes.
- Large local multimodal stack (Qwen3-VL + Qwen2.5 coder) → Yes.
- Text-only GGUF path (`agent3.py`) → Yes.
- LibreOffice conversion / SmartPDFLoader ingestion → Yes.
- Semantic chunking (4096/512) and reranking → Yes.
- Web search ingestion + incremental KB → Yes.
- GAIA final-answer formatting → Yes.
- Full GAIA submission automation → Manual driver (the provided apps fetch/submit but automation scripts are not included).
