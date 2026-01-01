# Conversational AI Agent

Production-grade conversational AI agent with a Streamlit UI, LlamaIndex-based multi-agent workflows, and a smolagents tool-calling runner. This repository focuses on multimodal reasoning (text, images, audio, video), document-grounded answers, and configurable provider/model choices.

## What is implemented

### 1) LlamaIndex conversational agent (`agent.py`)
- **Multi-agent workflow** (`ReActAgent` + `AgentWorkflow`) with a research agent and optional specialized agents (code execution, image analysis, media analysis, image generation/editing).
- **Modes**:
  - **API mode**: Gemini or OpenAI (selected per chat in the UI). Uses Jina embeddings for any indexing.
  - **Local mode**: Qwen or Ministral suites with specialized models (text, vision, audio/video, code, optional image generation/editing if diffusers are available).
- **Document handling**:
  - Docling for PDFs, Office files, and HTML.
  - CSV/Excel/JSON/text readers for structured data.
  - Images/audio/video processed through the active multimodal LLM in local mode.
- **Web search RAG**:
  - DDGS search + content extraction.
  - Temporary in-memory `VectorStoreIndex` with hierarchical chunking and Jina reranking.
- **Code execution** with a restricted global namespace.
- **Structured answer extraction** via Pydantic `LLMTextCompletionProgram`.

### 2) smolagents GAIA runner (`agent2.py`)
- **smolagents agent** built on `CodeAgent` with:
  - Web search (DuckDuckGo), page visiting, YouTube transcript, Python interpreter, and optional multimodal tool.
  - Optional MCP tool collections (filesystem, GitHub, Brave Search, Slack, Postgres, PubMed).
- **Used for the Hugging Face GAIA challenge** as the final assignment for the AI Agent course.
- Supports **Gemini** and **OpenAI** via `OpenAIServerModel`.
- Optional multimodal tool:
  - **Gemini**: images/audio/video via Gemini SDK.
  - **OpenAI**: images via Responses API; audio/video via transcription models.
- **Document loading** in memory (Docling-first). If Docling fails, the document is skipped.
- **GAIA task files**: documents are injected into the prompt via Docling; media files are passed as file paths with explicit tool instructions to call `multimodal_processor`.

### 3) Streamlit UI (`app.py`)
- Per-chat configuration (framework, mode, provider, model, and specialized features).
- Session persistence to `.chat_sessions`.
- LlamaIndex local mode: conversation stores with cached sources you can link into a chat.
- Prompt media uploaders for LlamaIndex (API + local) and smolagents (multimodal tool).
- API-mode document injection (full content inserted into prompt memory).

### 4) Utilities
- `utils/vector_store.py`: Chroma-based conversation stores plus a shared library cache, all using Jina embeddings.
- `utils/session_manager.py`: session persistence (messages, metadata, agent config).
- `utils/document_processor.py`: Docling-driven file parsing + web/YT helpers.
- `mcp_connectors.py`: MCP server registration and loader helpers.

## How it works

### LlamaIndex local mode
- Uses **Jina embeddings** and **Chroma** conversation stores.
- Hierarchical chunking (`HierarchicalNodeParser`) and leaf-node indexing.
- Optional specialized agents for code, media analysis, image generation/editing.
- Chat uploads are indexed into Chroma and tracked in the shared library cache.

### LlamaIndex API mode
- Uses **Gemini/OpenAI** for reasoning.
- Document uploads are inserted as full text into prompt memory (no vector store).

### smolagents
- CodeAgent-based assistant designed for tool-heavy tasks.
- API only; no local-mode execution.
- MCP tools can be enabled per chat.

## Best practices

- **Local multi-agent workflows** are GPU-intensive. For Qwen 30B + vision + code models, plan for **L4-class GPU or better**.
- **Smaller local stacks**: Qwen 4B text-only is the most GPU-friendly option.
- **LlamaIndex local mode** is best for large document sets and offline workflows.
- **API mode** is best for quick setup and long-context analysis without local hardware requirements.

## Getting started

### Install dependencies
Use your existing environment and install with:
```
pip install -r requirements.txt
```

### Run the app
```
streamlit run app.py
```
The UI opens at `http://localhost:8501`.

## Using the Streamlit UI

### Create a chat
1. Click **New Chat** in the sidebar.
2. Choose:
   - **Framework**: `llamaindex` or `smolagents`.
   - **Mode**: `API` or `Local` (smolagents requires API).
   - **Provider/Model** and specialized feature toggles.
3. Click **Create Chat**.

### Upload documents
- **Local LlamaIndex**: documents are indexed into a per-chat vector store and cached in the shared library.
- **API mode**: full document content is added to prompt memory.

### Multimodal attachments
- **LlamaIndex**: use **Prompt Media** to attach images/audio/video to the next prompt (local audio/video requires Media Analysis enabled).
- **smolagents**: use **Prompt media (multimodal)** to attach images/audio/video for the next prompt; the model is instructed to call `multimodal_processor`.
- PDFs and office docs should be uploaded via the document upload sections (chat files), not prompt media.

## Configuration

### Environment variables
**Core keys**:
```
export GOOGLE_API_KEY="your-google-api-key"
export OPENAI_API_KEY="your-openai-api-key"
```

**Langfuse**:
```
export LANGFUSE_SECRET_KEY="your-langfuse-secret-key"
export LANGFUSE_PUBLIC_KEY="your-langfuse-public-key"
export LANGFUSE_HOST="https://cloud.langfuse.com"
```

**MCP servers (optional)**:
```
export GITHUB_PERSONAL_ACCESS_TOKEN="your-github-token"
export BRAVE_API_KEY="your-brave-api-key"
export SLACK_BOT_TOKEN="your-slack-bot-token"
export POSTGRES_CONNECTION_STRING="your-postgres-uri"
```

**OpenAI transcription model (optional)**:
```
export OPENAI_TRANSCRIBE_MODEL="gpt-4o-mini-transcribe"
```

### Local model selection
You can override local models using:
```
export LOCAL_MODEL_ID="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
```

### Streamlit settings
Configured in `.streamlit/config.toml` (port, theme, uploads, logging).

## Repository structure

- `agent.py`: LlamaIndex multi-agent conversational runtime (local and API modes).
- `agent2.py`: smolagents tool-calling agent (used for HF GAIA challenge).
- `custom_models.py`: cached model wrappers (Qwen, Ministral, Devstral, Jina embeddings/reranker, API clients).
- `app.py`: Streamlit UI entry point.
- `utils/`: session management, vector store manager, document processing.
- `mcp_connectors.py`: MCP server definitions and loader.

## Notes
- **Embeddings are always Jina** (API and local).
- **Vector stores are only used for LlamaIndex local mode**.
- **smolagents has no local mode**.

## Troubleshooting

- **Model errors**: confirm GPU memory and correct model IDs.
- **API failures**: verify the API keys are set.
- **Docling errors**: validate file type and ensure docling dependencies are installed.
- **MCP tools not loading**: verify required environment variables for each server.
