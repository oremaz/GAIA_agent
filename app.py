"""Streamlit UI for the Conversational AI Agent."""

import streamlit as st
import os
import sys
import hashlib
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import utilities
from utils.session_manager import SessionManager, ChatSession
from utils.vector_store import VectorStoreManager
from utils.document_processor import DocumentProcessor

# Import agents
from agent import ConversationalAgent
from agent2 import GAIAAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="AI Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# Session State Initialization
# ============================================================================

def initialize_session_state():
    """Initialize Streamlit session state variables."""

    # Session management
    if 'session_manager' not in st.session_state:
        st.session_state.session_manager = SessionManager()

    if 'current_session' not in st.session_state:
        st.session_state.current_session = None

    # Vector store management
    if 'vector_store_manager' not in st.session_state:
        st.session_state.vector_store_manager = None

    if 'conversation_index' not in st.session_state:
        st.session_state.conversation_index = None

    # Document processor
    if 'doc_processor' not in st.session_state:
        st.session_state.doc_processor = DocumentProcessor()

    # Agent state
    if 'agent' not in st.session_state:
        st.session_state.agent = None

    # MCP servers
    if 'mcp_servers' not in st.session_state:
        st.session_state.mcp_servers = []

    if "show_new_chat_config" not in st.session_state:
        st.session_state.show_new_chat_config = False



# ============================================================================
# Agent Initialization
# ============================================================================

def initialize_agent_for_session(session: ChatSession):
    """Initialize the agent for a session configuration.

    Args:
        session: Chat session with stored configuration.
    """

    agent_config = session.agent_config

    with st.spinner("Initializing agent..."):
        try:
            if agent_config["framework"] == "llamaindex":
                # Initialize LlamaIndex agent
                use_api_mode = agent_config["mode"] == "api"
                provider = agent_config.get("llm_provider", "gemini")

                # Get model ID for both API and local modes
                local_model_id = agent_config.get("llm_model")

                agent = ConversationalAgent(
                    use_api_mode=use_api_mode,
                    model_suite=agent_config.get("model_suite", provider if use_api_mode else "qwen"),
                    local_model_id=local_model_id,
                    session_id=session.session_id,
                    media_analysis_enabled=agent_config.get("media_analysis_enabled", False),
                    code_execution_enabled=agent_config.get("code_execution_enabled", True),
                    use_specialized_code_model=agent_config.get("use_specialized_code_model", False),
                    img_generation_enabled=agent_config.get("img_generation_enabled", False),
                    img_editing_enabled=agent_config.get("img_editing_enabled", False),
                )

                st.session_state.agent = agent

                # Initialize vector store ONLY for local mode
                if agent_config["mode"] == "local":
                    if st.session_state.vector_store_manager is None:
                        st.session_state.vector_store_manager = VectorStoreManager()

                    st.session_state.conversation_index = None
                    st.success("LlamaIndex initialized with Jina RAG")
                else:
                    st.session_state.vector_store_manager = None
                    st.session_state.conversation_index = None
                    st.success("LlamaIndex initialized (API mode)")

            elif agent_config["framework"] == "smolagents":
                # smolagents: API only, no vector stores
                if agent_config["mode"] != "api":
                    st.error("smolagents only supports API mode")
                    return

                agent = GAIAAgent(
                    user_id="streamlit_user",
                    session_id=session.session_id,
                    provider=agent_config["llm_provider"],
                    model_name=agent_config["llm_model"],
                    mcp_servers=st.session_state.mcp_servers
                )

                st.session_state.agent = agent
                st.session_state.vector_store_manager = None
                st.session_state.conversation_index = None

                st.success("smolagents initialized (API mode)")

        except Exception as e:
            st.error(f"Initialization failed: {e}")
            logger.exception("Agent initialization error")


# ============================================================================
# Session Management UI
# ============================================================================

def render_session_management():
    """Render the session management sidebar."""

    st.sidebar.header("Chat Sessions")

    # Create new session button
    if st.sidebar.button("New Chat", type="primary"):
        st.session_state.show_new_chat_config = True

    if st.session_state.show_new_chat_config:
        show_new_chat_config()

    # List existing sessions
    sessions = st.session_state.session_manager.list_sessions(limit=20)

    if sessions:
        st.sidebar.subheader("Recent Chats")

        for sess in sessions:
            col1, col2 = st.sidebar.columns([4, 1])

            with col1:
                if st.button(
                    f"{sess['title'][:30]}",
                    key=f"load_{sess['session_id']}",
                    use_container_width=True
                ):
                    load_session(sess['session_id'])

            with col2:
                if st.button(
                    "🗑️",
                    key=f"del_{sess['session_id']}",
                    help="Delete chat"
                ):
                    delete_session(sess['session_id'])


def show_new_chat_config():
    """Show the new chat configuration panel."""

    with st.sidebar.expander("Configure New Chat", expanded=True):
        # Framework selection
        framework = st.selectbox(
            "Framework",
            ["llamaindex", "smolagents"],
            key="new_chat_framework",
            help="LlamaIndex: Multimodal RAG | smolagents: Tool-calling"
        )

        # Mode selection
        mode = st.radio(
            "Mode",
            ["API", "Local"],
            key="new_chat_mode",
            help="API: Cloud providers | Local: On-device models"
        )

        use_api = (mode == "API")

        # Disable local for smolagents
        if framework == "smolagents" and not use_api:
            st.error("smolagents only supports API mode")
            return

        if use_api:
            # Provider selection
            provider = st.selectbox(
                "Provider",
                ["gemini", "openai"],
                key="new_chat_provider"
            )

            # Model selection based on provider
            if provider == "gemini":
                models = ["gemini-3-pro-preview", "gemini-3-flash-preview", "gemini-2.5-flash-lite", "gemini-2.5-flash"]
            elif provider == "openai":
                models = ["gpt-5.2", "gpt-5.2-pro", "gpt-5-mini", "gpt-5-nano"]

            model = st.selectbox("Model", models, key="new_chat_model")
            
            # Specialized agents toggles (API mode) - conditional on provider
            use_specialized_code_model = False
            use_image_tools = False
            
            st.markdown("**Specialized Agents**")

            if provider == "openai":
                use_specialized_code_model = st.checkbox(
                    "Specialized Model for Code Agent (gpt-5.2-codex)",
                    value=True,
                    key="new_chat_code_agent",
                    help="Use specialized gpt-5.2-codex model for code execution. If disabled, uses the selected model above."
                )

            use_image_tools = st.checkbox(
                "Image Tools (Generation + Editing)",
                value=False,
                key="new_chat_image_tools_api",
                help=f"Enable image generation and editing with {model}"
            )

            agent_config = {
                "framework": framework,
                "mode": "api",
                "llm_provider": provider,
                "llm_model": model,
                "code_execution_enabled": True,  # Always enabled in API mode
                "use_specialized_code_model": use_specialized_code_model,
                "img_generation_enabled": use_image_tools,
                "img_editing_enabled": use_image_tools,  # Same toggle for both
                "media_analysis_enabled": False  # Audio/video natively handled in API mode
            }

        else:
            suite = st.selectbox(
                "Model Suite",
                ["qwen", "ministral"],
                key="new_chat_suite",
                help="Qwen: Multimodal models | Ministral: Fast lightweight models"
            )

            if suite == "qwen":
                models = [
                    "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",  # Text-only 30B
                    "Qwen/Qwen3-4B-Instruct-2507-FP8",  # Text-only 4B
                ]
            else:
                models = [
                    "mistralai/Ministral-3-3B-Instruct-2512",
                    "mistralai/Ministral-3-8B-Instruct-2512",
                    "mistralai/Ministral-3-14B-Instruct-2512"
                ]

            model = st.selectbox(
                "Main Model", 
                models, 
                key="new_chat_model",
                help="Main LLM for general tasks"
            )

            # Specialized agents/models toggles (local mode only)
            st.markdown("**Specialized Agents**")
            
            use_code_llm = st.checkbox(
                "Code Agent (Devstral 24B)",
                value=True,
                key="new_chat_code_llm",
                help="Enable Devstral-Small-2-24B for software engineering tasks"
            )
            
            use_media_analysis = st.checkbox(
                "Media Analysis (Audio/Video)",
                value=False,
                key="new_chat_media_analysis",
                help="Enable Qwen-Omni for audio/video transcription and analysis"
            )
            
            use_image_gen = st.checkbox(
                "Image Generation (Qwen-Image-2512)",
                value=False,
                key="new_chat_image_gen",
                help="Enable text-to-image generation"
            )
            
            use_image_edit = st.checkbox(
                "Image Editing (Qwen-Image-Edit-2511)",
                value=False,
                key="new_chat_image_edit",
                help="Enable image editing capabilities"
            )

            agent_config = {
                "framework": framework,
                "mode": "local",
                "model_suite": suite,
                "llm_model": model,
                "media_analysis_enabled": use_media_analysis,
                "code_execution_enabled": use_code_llm,
                "img_generation_enabled": use_image_gen,
                "img_editing_enabled": use_image_edit
            }

        # MCP Servers (smolagents only)
        if framework == "smolagents":
            from mcp_connectors import get_available_mcp_servers, check_mcp_server_requirements

            st.subheader("MCP Servers")
            available_servers = get_available_mcp_servers()

            selected_servers = st.multiselect(
                "Enable MCP Tools",
                available_servers,
                key="new_chat_mcp",
                help="Model Context Protocol servers"
            )

            # Check requirements
            for server in selected_servers:
                met, missing = check_mcp_server_requirements(server)
                if not met:
                    st.warning(f"{server}: Missing {', '.join(missing)}")

            st.session_state.mcp_servers = selected_servers

        # Create chat button
        if st.button("Create Chat", key="create_chat_btn", type="primary"):
            create_chat_with_config(agent_config)


def create_chat_with_config(agent_config: Dict[str, Any]):
    """Create a new chat session with a specific configuration.

    Args:
        agent_config: Configuration dictionary for the new session.
    """

    # Generate title
    framework = agent_config["framework"]
    model = agent_config.get("llm_model", "")
    title = f"{framework.capitalize()} - {model.split('/')[-1][:20]}"

    # Create session
    new_session = st.session_state.session_manager.create_session(
        title=title,
        agent_config=agent_config
    )

    st.session_state.current_session = new_session
    st.session_state.show_new_chat_config = False

    # Initialize agent
    initialize_agent_for_session(new_session)

    st.success("Chat created")
    st.rerun()


def load_session(session_id: str):
    """Load an existing session and initialize its agent.

    Args:
        session_id: Session identifier to load.
    """

    session = st.session_state.session_manager.load_session(session_id)

    if session:
        st.session_state.current_session = session

        # Initialize agent for this session
        initialize_agent_for_session(session)

        st.rerun()
    else:
        st.error("Failed to load session")


def delete_session(session_id: str):
    """Delete a session and its associated data.

    Args:
        session_id: Session identifier to delete.
    """

    if st.session_state.session_manager.delete_session(session_id, cleanup_vector_store=True):

        # If deleted session was current, clear it
        if st.session_state.current_session and st.session_state.current_session.session_id == session_id:
            st.session_state.current_session = None
            st.session_state.agent = None

        st.success("Chat deleted")
        st.rerun()
    else:
        st.error("Failed to delete chat")


# ============================================================================
# Vector Store Management UI (ONLY for LlamaIndex + Local)
# ============================================================================

def render_vector_store_section():
    """Render the vector store management section."""

    if not st.session_state.current_session:
        st.sidebar.info("Create or select a chat first")
        return

    agent_config = st.session_state.current_session.agent_config

    # ONLY show for LlamaIndex + Local mode
    if agent_config.get("framework") != "llamaindex" or agent_config.get("mode") != "local":
        st.sidebar.info("Vector stores available only with LlamaIndex + Local mode")
        return

    session = st.session_state.current_session
    st.sidebar.header("Conversation Knowledge Base")
    st.sidebar.caption("Sources are cached and can be reused across chats")

    # Upload to conversation store
    st.sidebar.subheader("Upload Documents")

    uploaded_file = st.sidebar.file_uploader(
        "Add to this chat",
        type=["pdf", "docx", "txt", "csv", "xlsx", "json", "md", "pptx"],
        help="Documents are cached and linked to this chat",
        key="conversation_upload"
    )

    if uploaded_file:
        if st.sidebar.button("Upload to Chat", key="upload_conversation_btn"):
            upload_to_conversation_store(uploaded_file)

    # Add URL
    st.sidebar.subheader("Add Web Content")

    url_input = st.sidebar.text_input(
        "URL",
        placeholder="https://example.com",
        key="conversation_url"
    )

    if url_input and st.sidebar.button("Fetch and Add to Chat", key="fetch_conversation_btn"):
        fetch_url_to_conversation(url_input)

    # Library sources
    if st.session_state.vector_store_manager:
        st.sidebar.subheader("Library")
        sources = st.session_state.vector_store_manager.list_library_sources()
        linked_sources = set(session.metadata.get("linked_sources", []))

        if sources:
            source_lookup = {entry["source_id"]: entry for entry in sources}
            available_sources = [
                entry["source_id"]
                for entry in sources
                if entry["source_id"] not in linked_sources
            ]

            if available_sources:
                selected_sources = st.sidebar.multiselect(
                    "Add cached sources",
                    options=available_sources,
                    format_func=lambda sid: _format_library_label(source_lookup.get(sid, {})),
                    key="library_selection"
                )

                if selected_sources and st.sidebar.button("Link Selected to Chat", key="link_library_btn"):
                    add_library_sources_to_conversation(selected_sources)
            else:
                st.sidebar.caption("All cached sources are already linked.")
        else:
            st.sidebar.caption("No cached documents yet.")

        if linked_sources:
            st.sidebar.subheader("Linked to This Chat")
            for source_id in linked_sources:
                entry = st.session_state.vector_store_manager.library_index.get("sources", {}).get(source_id, {})
                st.sidebar.caption(f"- {_format_library_label(entry)}")

        stats = st.session_state.vector_store_manager.get_stats()
        chat_doc_count = st.session_state.vector_store_manager.get_conversation_document_count(session.session_id)
        st.sidebar.subheader("Store Statistics")
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.metric("Chat Docs", chat_doc_count)
        with col2:
            st.metric("Cached Sources", stats.get("library_sources", 0))


def _format_library_label(entry: Dict[str, Any]) -> str:
    """Format a library entry for UI display.

    Args:
        entry: Library entry dictionary.

    Returns:
        Human-readable label string.
    """
    label = entry.get("label") or entry.get("source_key") or "Unknown"
    source_type = entry.get("source_type", "source")
    return f"{label} ({source_type})"


def _get_linked_sources(session: ChatSession) -> set:
    """Return linked source ids for the session.

    Args:
        session: Chat session instance.

    Returns:
        Set of linked source ids.
    """
    return set(session.metadata.get("linked_sources", []))


def _link_source_to_session(session: ChatSession, source_id: str) -> bool:
    """Link a source to the session if not already linked.

    Args:
        session: Chat session instance.
        source_id: Library source identifier.

    Returns:
        True if the source was linked, False if already linked.
    """
    linked_sources = _get_linked_sources(session)
    if source_id in linked_sources:
        return False
    linked_sources.add(source_id)
    session.metadata["linked_sources"] = sorted(linked_sources)
    return True


def upload_to_conversation_store(uploaded_file):
    """Upload a file to the conversation vector store with caching.

    Args:
        uploaded_file: Streamlit uploaded file object.
    """

    with st.spinner(f"Processing {uploaded_file.name}..."):
        try:
            if st.session_state.vector_store_manager is None:
                st.sidebar.error("Vector store not initialized (local LlamaIndex only)")
                return

            file_bytes = uploaded_file.getvalue()
            file_hash = hashlib.sha256(file_bytes).hexdigest()
            source_id = st.session_state.vector_store_manager.get_library_source_id("file", file_hash)

            if source_id:
                linked = add_library_sources_to_conversation([source_id], rerun=False)
                if linked:
                    st.sidebar.info("File already cached. Linked to this chat.")
                else:
                    st.sidebar.info("File already linked to this chat.")
                st.rerun()
                return

            docs, file_type = st.session_state.doc_processor.process_uploaded_file(uploaded_file)

            if not docs:
                st.error("No content extracted")
                return

            source_id, _ = st.session_state.vector_store_manager.register_library_source(
                "file",
                file_hash,
                uploaded_file.name,
                docs,
                {"file_type": file_type},
            )

            add_library_sources_to_conversation([source_id], rerun=False, announce_label=uploaded_file.name)
            st.sidebar.success(f"Added {len(docs)} chunks ({file_type}) to this chat")
            st.rerun()

        except Exception as e:
            st.sidebar.error(f"Upload failed: {e}")
            logger.exception("Conversation upload error")


def fetch_url_to_conversation(url: str):
    """Fetch a URL and add it to the conversation vector store with caching.

    Args:
        url: URL to fetch and index.
    """

    with st.spinner(f"Fetching {url}..."):
        try:
            if st.session_state.vector_store_manager is None:
                st.sidebar.error("Vector store not initialized (local LlamaIndex only)")
                return

            normalized_url = url.strip()
            source_id = st.session_state.vector_store_manager.get_library_source_id("url", normalized_url)

            if source_id:
                linked = add_library_sources_to_conversation([source_id], rerun=False)
                if linked:
                    st.sidebar.info("URL already cached. Linked to this chat.")
                else:
                    st.sidebar.info("URL already linked to this chat.")
                st.rerun()
                return

            docs, url_type = st.session_state.doc_processor.process_url(normalized_url)

            if not docs:
                st.error("No content extracted")
                return

            source_id, _ = st.session_state.vector_store_manager.register_library_source(
                "url",
                normalized_url,
                normalized_url,
                docs,
                {"url_type": url_type},
            )

            add_library_sources_to_conversation([source_id], rerun=False, announce_label=normalized_url)
            st.sidebar.success(f"Added {len(docs)} chunks ({url_type}) to this chat")
            st.rerun()

        except Exception as e:
            st.sidebar.error(f"Fetch failed: {e}")
            logger.exception("URL fetch error")


def add_library_sources_to_conversation(
    source_ids: List[str],
    rerun: bool = True,
    announce_label: Optional[str] = None,
) -> bool:
    """Add cached library sources to the current conversation.

    Args:
        source_ids: Library source identifiers to link.
        rerun: Whether to trigger a Streamlit rerun.
        announce_label: Optional label to announce in the chat history.

    Returns:
        True if any sources were linked, otherwise False.
    """
    session = st.session_state.current_session
    if not session or not st.session_state.vector_store_manager:
        return False

    linked_any = False
    for source_id in source_ids:
        if not _link_source_to_session(session, source_id):
            continue

        conv_index = st.session_state.vector_store_manager.add_library_source_to_conversation(
            source_id,
            session.session_id,
            st.session_state.conversation_index,
        )
        st.session_state.conversation_index = conv_index

        entry = st.session_state.vector_store_manager.library_index.get("sources", {}).get(source_id, {})
        label = announce_label or entry.get("label") or source_id

        session.add_message(
            "user",
            f"Linked: {label}\n\nCached content added to conversation vector store",
            metadata={
                "type": "document_upload",
                "source_id": source_id,
                "storage": "vector_store",
            },
        )
        linked_any = True

    st.session_state.session_manager.save_session(session)
    if rerun:
        st.rerun()
    return linked_any


# ============================================================================
# In-Chat Document Upload
# ============================================================================

def process_chat_document(uploaded_file):
    """Process a document uploaded within a chat.

    Args:
        uploaded_file: Streamlit uploaded file object.
    """

    session = st.session_state.current_session
    agent_config = session.agent_config

    with st.spinner(f"Processing {uploaded_file.name}..."):
        try:
            # Extract content
            docs, file_type = st.session_state.doc_processor.process_uploaded_file(uploaded_file)

            if not docs:
                st.error("Failed to process document")
                return

            full_content = "\n\n".join([doc.text for doc in docs])

            # Strategy depends on framework AND mode
            if agent_config["framework"] == "llamaindex" and agent_config["mode"] == "local":
                # LOCAL LLAMAINDEX: Use conversation-specific vector store
                if st.session_state.vector_store_manager is None:
                    st.error("Vector store not initialized")
                    return

                conv_index = st.session_state.vector_store_manager.add_documents_to_conversation(
                    docs,
                    session.session_id,
                    st.session_state.conversation_index
                )
                st.session_state.conversation_index = conv_index

                # Add message to chat
                session.add_message(
                    "user",
                    f"Uploaded: {uploaded_file.name}\n\nFile indexed in conversation vector store",
                    metadata={
                        "type": "document_upload",
                        "filename": uploaded_file.name,
                        "file_type": file_type,
                        "storage": "vector_store"
                    }
                )

                st.success(f"{uploaded_file.name} indexed with Jina embeddings")

            else:
                # API MODE (any framework): Store full content in message metadata
                session.add_message(
                    "user",
                    f"Uploaded: {uploaded_file.name}\n\nFull content loaded ({len(full_content):,} characters)",
                    metadata={
                        "type": "document_upload",
                        "filename": uploaded_file.name,
                        "full_content": full_content,
                        "file_type": file_type,
                        "storage": "memory"
                    }
                )

                st.success(f"{uploaded_file.name} loaded into chat memory")

            # Save session
            st.session_state.session_manager.save_session(session)
            st.rerun()

        except Exception as e:
            st.error(f"Processing failed: {e}")
            logger.exception("Chat document processing error")


# ============================================================================
# Chat Interface
# ============================================================================

def render_chat_interface():
    """Render the main chat interface."""

    if not st.session_state.current_session:
        st.markdown('<div class="main-header">AI Agent</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Create a new chat to get started</div>', unsafe_allow_html=True)

        # Show example configurations
        st.subheader("Quick Start")

        col1, col2 = st.columns(2)

        with col1:
            st.info("**LlamaIndex + API**\n\nBest for: Long documents, complex RAG\n\nFeatures:\n- Large context windows\n- Full document injection\n- No vector stores needed")

        with col2:
            st.info("**smolagents + API**\n\nBest for: Tool-heavy tasks, MCP integration\n\nFeatures:\n- GitHub, Slack, File system tools\n- Fast code execution\n- Flexible tool calling")

        return

    # Display session info
    session = st.session_state.current_session
    agent_config = session.agent_config

    st.title(f"{session.title}")

    # Show agent configuration
    config_info = f"Framework: {agent_config['framework'].capitalize()} | Mode: {agent_config['mode'].upper()} | Model: {agent_config.get('llm_model', 'N/A').split('/')[-1]}"
    st.caption(config_info)

    st.divider()

    # Display chat history
    for msg in session.messages:
        with st.chat_message(msg.role):
            st.markdown(msg.content)

            # Show document upload indicator
            if msg.metadata.get("type") == "document_upload":
                storage = msg.metadata.get("storage", "unknown")
                if storage == "vector_store":
                    st.caption("Stored in vector store (searchable via RAG)")
                elif storage == "memory":
                    st.caption("Stored in chat memory (full content)")

    render_feedback_section()

    # Document uploads for smolagents
    if agent_config["framework"] == "smolagents":
        with st.sidebar.expander("Document Uploads", expanded=False):
            st.markdown("**Chat files (Docling)**")
            doc_file = st.file_uploader(
                "Add to chat memory",
                type=["pdf", "docx", "doc", "pptx", "ppt", "html", "htm", "csv", "xlsx", "xls", "txt", "md", "json"],
                key="smol_doc_persist",
                help="Processed with Docling and prepended before chat history"
            )
            if doc_file and st.sidebar.button("Process and save", key="smol_doc_persist_btn"):
                process_chat_document(doc_file)

            st.markdown("**Prompt media (multimodal)**")
            prompt_media = st.file_uploader(
                "Attach media to next prompt",
                type=["png", "jpg", "jpeg", "gif", "webp", "mp3", "wav", "m4a", "mp4", "webm"],
                accept_multiple_files=True,
                key="smol_prompt_media",
                help="Attached to the next prompt and processed via the multimodal tool"
            )
            if prompt_media and st.sidebar.button("Attach media to prompt", key="smol_prompt_media_btn"):
                add_prompt_media_files(prompt_media)

    # Prompt media for LlamaIndex (not indexed)
    if agent_config["framework"] == "llamaindex":
        label = "Prompt Media (LlamaIndex Local)" if agent_config["mode"] == "local" else "Prompt Media (LlamaIndex API)"
        allow_audio_video = agent_config["mode"] == "api" or agent_config.get("media_analysis_enabled", False)
        media_types = ["png", "jpg", "jpeg", "gif", "webp"]
        if allow_audio_video:
            media_types.extend(["mp3", "wav", "m4a", "mp4"])
        media_help = "Adds media to the next prompt without indexing in the vector store"
        if agent_config["mode"] == "local" and not allow_audio_video:
            media_help = "Image-only unless Media Analysis (Audio/Video) is enabled"
        media_label = "Attach images/audio/video to next prompt" if allow_audio_video else "Attach images to next prompt"
        with st.sidebar.expander(label, expanded=False):
            prompt_media = st.file_uploader(
                media_label,
                type=media_types,
                accept_multiple_files=True,
                key="llama_prompt_media",
                help=media_help
            )
            if prompt_media and st.sidebar.button("Attach media to prompt", key="llama_prompt_media_btn"):
                add_prompt_media_files(prompt_media)

    prompt = st.chat_input("Ask me anything...", key="chat_input")

    pending_prompt_media = session.metadata.get("pending_prompt_media", [])

    # Handle media files upload with prompt
    if prompt and pending_prompt_media:
        handle_user_message_with_media(prompt, [])
    elif prompt:
        handle_user_message(prompt)


def handle_user_message(prompt: str):
    """Handle a user message and generate a response.

    Args:
        prompt: User prompt text.
    """

    session = st.session_state.current_session

    # Add user message
    session.add_message("user", prompt)
    st.session_state.session_manager.save_session(session)

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response, trace_id = generate_response(prompt)

        st.markdown(response)

    # Add assistant message
    metadata = {"trace_id": trace_id} if trace_id else None
    session.add_message("assistant", response, metadata=metadata)
    if trace_id:
        session.metadata["last_trace_id"] = trace_id
    st.session_state.session_manager.save_session(session)

    st.rerun()


def _infer_media_type(extension: str) -> str:
    ext = extension.lower()
    if not ext.startswith("."):
        ext = f".{ext}"
    if ext in [".png", ".jpg", ".jpeg", ".gif", ".webp"]:
        return "image"
    if ext in [".mp3", ".wav", ".m4a"]:
        return "audio"
    if ext in [".mp4", ".webm"]:
        return "video"
    return "file"


def handle_user_message_with_media(prompt: str, media_files: List):
    """Handle a user message with attached media files.

    Args:
        prompt: User prompt text.
        media_files: List of uploaded media files.
    """

    session = st.session_state.current_session
    agent_config = session.agent_config
    prompt_media = _consume_prompt_media(session)

    # Save media files temporarily and collect paths
    import tempfile
    media_paths = []
    media_info = []

    try:
        for info in prompt_media:
            path = info.get("path")
            if not path:
                continue
            media_paths.append(path)
            media_info.append({
                "filename": info.get("filename") or Path(path).name,
                "path": path,
                "type": info.get("type") or _infer_media_type(Path(path).suffix.lower()),
            })

        for media_file in media_files:
            # Determine media type
            ext = Path(media_file.name).suffix.lower()
            media_type = _infer_media_type(ext)

            # Save to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
                tmp_file.write(media_file.getvalue())
                abs_path = os.path.abspath(tmp_file.name)
                media_paths.append(abs_path)
                media_info.append({
                    'filename': media_file.name,
                    'path': abs_path,
                    'type': media_type
                })

        # Build message content
        if agent_config["framework"] == "smolagents":
            media_summary = ", ".join([f"{m['path']} ({m['type']})" for m in media_info if m["type"] != "document"])
        else:
            media_summary = ", ".join([f"{m['filename']} ({m['type']})" for m in media_info])
        message_content = f"{prompt}\n\n[Attached: {media_summary}]"

        # Add user message with metadata
        session.add_message(
            "user", 
            message_content,
            metadata={
                "type": "multimodal_message",
                "media_files": media_info,
                "media_paths": media_paths
            }
        )
        st.session_state.session_manager.save_session(session)

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
            # Display media previews
            for info in media_info:
                if info['type'] == 'image':
                    st.image(info['path'], caption=info['filename'], width=300)
                elif info['type'] == 'audio':
                    st.audio(info['path'])
                elif info['type'] == 'video':
                    st.video(info['path'])

        # Generate response with media context
        non_pdf_paths = [p for p in media_paths if Path(p).suffix.lower() != ".pdf"]

        with st.chat_message("assistant"):
            with st.spinner("Processing multimodal content..."):
                if non_pdf_paths:
                    response, trace_id = generate_multimodal_response(
                        prompt,
                        non_pdf_paths,
                    )
                else:
                    response, trace_id = generate_response(prompt)

            st.markdown(response)

        # Add assistant message
        metadata = {"trace_id": trace_id} if trace_id else None
        session.add_message("assistant", response, metadata=metadata)
        if trace_id:
            session.metadata["last_trace_id"] = trace_id
        st.session_state.session_manager.save_session(session)

        st.rerun()

    except Exception as e:
        st.error(f"Error processing media files: {e}")
        logger.exception("Multimodal message processing error")
    finally:
        # Cleanup temp files
        import os as os_module
        for path in media_paths:
            try:
                os_module.unlink(path)
            except:
                pass
        _cleanup_prompt_media(prompt_media)


def _parse_agent_response(result) -> tuple[str, Optional[str]]:
    """Normalize agent outputs to (response_text, trace_id).

    Args:
        result: Agent output in tuple/dict/string form.

    Returns:
        Tuple of response text and optional trace id.
    """
    if isinstance(result, tuple) and len(result) == 2:
        response_text, trace_id = result
        return str(response_text), trace_id
    if isinstance(result, dict):
        response_text = result.get("response") or result.get("text") or ""
        trace_id = result.get("trace_id")
        return str(response_text), trace_id
    return str(result), None


def add_prompt_media_files(uploaded_files):
    """Store media files to attach to the next prompt without indexing.

    Args:
        uploaded_files: Streamlit uploaded file(s).
    """
    session = st.session_state.current_session
    if not session:
        st.sidebar.warning("Create or select a chat first.")
        return

    if not uploaded_files:
        st.sidebar.warning("Select at least one media file to attach.")
        return

    if not isinstance(uploaded_files, list):
        uploaded_files = [uploaded_files]

    agent_config = session.agent_config or {}
    restrict_audio_video = (
        agent_config.get("framework") == "llamaindex"
        and agent_config.get("mode") == "local"
        and not agent_config.get("media_analysis_enabled", False)
    )

    upload_dir = Path(".prompt_media") / session.session_id
    upload_dir.mkdir(parents=True, exist_ok=True)

    pending = session.metadata.get("pending_prompt_media", [])
    added = 0
    skipped = []

    for uploaded_file in uploaded_files:
        file_bytes = uploaded_file.getvalue()
        safe_name = Path(uploaded_file.name).name
        ext = Path(safe_name).suffix.lower()
        media_type = _infer_media_type(ext)
        if restrict_audio_video and media_type in ("audio", "video"):
            skipped.append(safe_name)
            continue
        unique_name = f"{uuid.uuid4()}_{safe_name}"
        file_path = upload_dir / unique_name

        with open(file_path, "wb") as handle:
            handle.write(file_bytes)

        pending.append({
            "filename": safe_name,
            "path": os.path.abspath(file_path),
            "extension": ext,
            "type": media_type,
            "size_bytes": len(file_bytes),
        })
        added += 1

    session.metadata["pending_prompt_media"] = pending
    st.session_state.session_manager.save_session(session)
    if added:
        st.sidebar.success(f"Attached {added} media file(s) to the next prompt.")
    if skipped:
        st.sidebar.warning(
            "Skipped audio/video files (enable Media Analysis to attach): "
            + ", ".join(skipped)
        )

def _consume_prompt_media(session: ChatSession) -> List[Dict[str, Any]]:
    """Return and clear prompt media for the next user message.

    Args:
        session: Chat session instance.

    Returns:
        List of pending prompt media entries.
    """
    if not session:
        return []
    return session.metadata.pop("pending_prompt_media", [])


def _cleanup_prompt_media(prompt_media: List[Dict[str, Any]]) -> None:
    """Remove temporary prompt media attachments from disk.

    Args:
        prompt_media: List of prompt media metadata.
    """
    for info in prompt_media or []:
        path = info.get("path")
        if not path:
            continue
        try:
            Path(path).unlink(missing_ok=True)
        except Exception as exc:
            logger.warning("Failed to cleanup prompt media %s: %s", path, exc)


def _build_prompt(
    prompt: str,
    session: ChatSession,
    agent_config: Dict[str, Any],
    max_history: int = 8,
) -> str:
    """Build a prompt with documents, vector store metadata, and history.

    Args:
        prompt: User prompt text.
        session: Chat session instance.
        agent_config: Agent configuration dict.
        max_history: Max number of recent messages to include.

    Returns:
        Full prompt string.
    """
    parts = []

    def _document_block() -> Optional[str]:
        docs = []
        for msg in session.messages:
            meta = msg.metadata or {}
            if meta.get("type") == "document_upload" and meta.get("storage") == "memory":
                filename = meta.get("filename", "document")
                file_type = meta.get("file_type", "unknown")
                content = meta.get("full_content") or ""
                if not content:
                    continue
                docs.append(f"[{filename} | {file_type}]\n{content}")
        if not docs:
            return None
        return "Documents uploaded in this conversation:\n" + "\n\n".join(docs)

    def _vector_store_block() -> Optional[str]:
        sources = []
        for msg in session.messages:
            meta = msg.metadata or {}
            if meta.get("type") == "document_upload" and meta.get("storage") == "vector_store":
                filename = meta.get("filename", "document")
                file_type = meta.get("file_type", "unknown")
                sources.append(f"- {filename} ({file_type})")

        linked = session.metadata.get("linked_sources", [])
        if linked and st.session_state.vector_store_manager:
            lib = st.session_state.vector_store_manager.library_index.get("sources", {})
            for source_id in linked:
                entry = lib.get(source_id, {})
                sources.append(f"- {_format_library_label(entry)}")

        if not sources:
            return None

        seen = set()
        uniq = []
        for item in sources:
            if item in seen:
                continue
            seen.add(item)
            uniq.append(item)
        return "Vector store sources linked to this chat:\n" + "\n".join(uniq)

    def _history_block() -> Optional[str]:
        history = []
        for msg in session.messages:
            meta = msg.metadata or {}
            if meta.get("type") == "document_upload":
                continue
            if msg.role not in ("user", "assistant"):
                continue
            history.append(msg)

        if not history:
            return None

        recent = history[-max_history:]
        lines = []
        for msg in recent:
            role = "User" if msg.role == "user" else "Assistant"
            lines.append(f"{role}: {msg.content}")
        return "Conversation history:\n" + "\n".join(lines)

    if agent_config["framework"] == "llamaindex" and agent_config.get("mode") == "local":
        vector_block = _vector_store_block()
        if vector_block:
            parts.append(vector_block)
        history_block = _history_block()
        if history_block:
            parts.append(history_block)
    elif agent_config["framework"] == "smolagents":
        doc_block = _document_block()
        if doc_block:
            parts.append(doc_block)
        history_block = _history_block()
        if history_block:
            parts.append(history_block)
    else:
        doc_block = _document_block()
        if doc_block:
            parts.append(doc_block)

    parts.append(f"User question:\n{prompt}")
    return "\n\n".join(parts)


def generate_response(prompt: str) -> tuple[str, Optional[str]]:
    """Generate a response using the configured agent.

    Args:
        prompt: User prompt text.

    Returns:
        Tuple of response text and optional trace id.
    """

    session = st.session_state.current_session
    agent_config = session.agent_config

    full_prompt = _build_prompt(prompt, session, agent_config)

    # Generate response using unified run() method
    try:
        agent = st.session_state.agent

        if not agent:
            return "Agent not initialized. Please create a new chat."

        # Both agents now use .run() method
        response = agent.run(full_prompt)
        return _parse_agent_response(response)

    except Exception as e:
        error_msg = f"Error generating response: {e}"
        logger.exception("Response generation error")
        return error_msg, None


def generate_multimodal_response(
    prompt: str,
    media_paths: List[str],
) -> tuple[str, Optional[str]]:
    """Generate a response with multimodal content.

    Args:
        prompt: User prompt text.
        media_paths: Paths to media files to attach.

    Returns:
        Tuple of response text and optional trace id.
    """

    session = st.session_state.current_session
    agent_config = session.agent_config

    try:
        agent = st.session_state.agent

        if not agent:
            return "Agent not initialized. Please create a new chat."

        # For LlamaIndex, process media files using read_and_parse_content
        if agent_config["framework"] == "llamaindex":
            from agent import read_and_parse_content
            
            # Process each media file to extract descriptions
            media_descriptions = []
            for path in media_paths:
                try:
                    docs = read_and_parse_content(path)
                    if docs:
                        description = docs[0].text
                        media_descriptions.append(f"File {path}:\n{description}")
                except Exception as e:
                    logger.warning(f"Failed to process media file {path}: {e}")
                    media_descriptions.append(f"File {path}: [Processing failed]")
            
            # Build context with media descriptions and user prompt
            base_prompt = _build_prompt(prompt, session, agent_config)
            multimodal_context = "\n\n".join(media_descriptions + [base_prompt])
            
            response = agent.run(multimodal_context)
            return _parse_agent_response(response)

        # For smolagents, process with UnifiedMultimodalTool
        elif agent_config["framework"] == "smolagents":
            media_lines = []
            for path in media_paths:
                if Path(path).suffix.lower() == ".pdf":
                    continue
                ext = Path(path).suffix.lower()
                if ext in {".png", ".jpg", ".jpeg", ".gif", ".webp"}:
                    media_type = "image"
                elif ext in {".mp3", ".wav", ".m4a"}:
                    media_type = "audio"
                elif ext in {".mp4", ".avi", ".mov", ".webm"}:
                    media_type = "video"
                else:
                    media_type = "file"
                media_lines.append(f"- {path} ({media_type})")

            attachments = "\n".join(media_lines)
            base_prompt = _build_prompt(prompt, session, agent_config)
            full_context = (
                "Attached media files:\n"
                f"{attachments}\n\n"
                "Use the multimodal_processor tool on each file_path before answering.\n\n"
                f"{base_prompt}"
            )
            response = agent.run(full_context)
            return _parse_agent_response(response)

        else:
            return "Multimodal processing not supported for this agent configuration.", None

    except Exception as e:
        error_msg = f"Error generating multimodal response: {e}"
        logger.exception("Multimodal response generation error")
        return error_msg, None


# ============================================================================
# Feedback
# ============================================================================

def submit_feedback(trace_id: str, score: int, comment: str) -> bool:
    """Send feedback to Langfuse using the agent hook.

    Args:
        trace_id: Trace id to score.
        score: Feedback score (0 or 1).
        comment: Optional feedback comment.

    Returns:
        True if feedback was submitted, otherwise False.
    """
    agent = st.session_state.agent
    if not agent or not hasattr(agent, "add_user_feedback"):
        return False

    try:
        agent.add_user_feedback(trace_id=trace_id, feedback_score=score, comment=comment or None)
        return True
    except Exception as e:
        logger.exception("Agent feedback submission failed: %s", e)
        return False


def render_feedback_section():
    """Render feedback controls for the latest assistant message."""
    session = st.session_state.current_session
    if not session:
        return

    latest_msg = None
    for msg in reversed(session.messages):
        trace_id = msg.metadata.get("trace_id") if msg.metadata else None
        if msg.role == "assistant" and trace_id:
            latest_msg = msg
            break

    if not latest_msg:
        return

    trace_id = latest_msg.metadata.get("trace_id")
    st.subheader("Rate the last response")
    feedback_choice = st.radio(
        "Feedback",
        options=["Helpful", "Not helpful"],
        horizontal=True,
        key=f"feedback_choice_{trace_id}"
    )
    comment = st.text_input(
        "Comment (optional)",
        key=f"feedback_comment_{trace_id}"
    )
    if st.button("Submit feedback", key=f"feedback_submit_{trace_id}"):
        score = 1 if feedback_choice == "Helpful" else 0
        if submit_feedback(trace_id, score, comment):
            st.success("Thanks for the feedback.")
        else:
            st.warning("Feedback is unavailable. Check Langfuse credentials.")


# ============================================================================
# Settings and Info
# ============================================================================

def render_settings():
    """Render the settings sidebar."""

    st.sidebar.header("Settings")

    # API Keys status
    with st.sidebar.expander("API Keys Status"):
        keys = {
            "GOOGLE_API_KEY": "Gemini",
            "OPENAI_API_KEY": "OpenAI",
            "LANGFUSE_SECRET_KEY": "Langfuse",
            "GITHUB_PERSONAL_ACCESS_TOKEN": "GitHub MCP"
        }

        for env_var, name in keys.items():
            if os.environ.get(env_var):
                st.success(f"{name}")
            else:
                st.error(f"{name}")

    # Model info
    with st.sidebar.expander("Model Information"):
        st.markdown("""
        **Embeddings:** Always Jina v3 (4-bit)

        **LlamaIndex:**
        - API: Gemini/OpenAI
        - Local: Qwen-VL, Ministral

        **smolagents:**
        - API only (no local mode)

        **Vector Stores:**
        - LlamaIndex + Local only
        - Global (persistent) + Chat-specific
        """)

    # Export/Import
    st.sidebar.subheader("Data Management")

    if st.sidebar.button("Export All Chats"):
        export_all_sessions()

    if st.sidebar.button("Clear All Data"):
        if st.sidebar.checkbox("I understand this is irreversible"):
            clear_all_data()


def export_all_sessions():
    """Export all chat sessions to JSON."""

    import json
    from datetime import datetime

    sessions = st.session_state.session_manager.list_sessions()

    if not sessions:
        st.sidebar.warning("No sessions to export")
        return

    export_data = {
        "export_date": datetime.now().isoformat(),
        "total_sessions": len(sessions),
        "sessions": []
    }

    for sess_info in sessions:
        session = st.session_state.session_manager.load_session(sess_info["session_id"])
        if session:
            export_data["sessions"].append(session.to_dict())

    # Create download button
    export_json = json.dumps(export_data, indent=2, ensure_ascii=False)

    st.sidebar.download_button(
        label="Download JSON",
        data=export_json,
        file_name=f"ai_agent_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )


def clear_all_data():
    """Clear all application data and reset state."""

    import shutil

    try:
        # Clear sessions
        if Path(".chat_sessions").exists():
            shutil.rmtree(".chat_sessions")

        # Clear vector stores
        if Path("./chroma_db").exists():
            shutil.rmtree("./chroma_db")

        # Reset session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]

        st.sidebar.success("All data cleared")
        st.rerun()

    except Exception as e:
        st.sidebar.error(f"Clear failed: {e}")


# ============================================================================
# Main Application Entry Point
# ============================================================================

def main():
    """Main application entry point."""

    # Initialize session state
    initialize_session_state()

    # Sidebar navigation
    render_session_management()

    st.sidebar.divider()

    # Vector store management (only for LlamaIndex + Local)
    render_vector_store_section()

    st.sidebar.divider()

    # Settings
    render_settings()

    # Main chat interface
    render_chat_interface()

    # Footer
    st.sidebar.divider()
    st.sidebar.caption("Conversational AI Agent v2.0")
    st.sidebar.caption("Powered by LlamaIndex and smolagents")


if __name__ == "__main__":
    main()
