"""ChromaDB vector store management for LlamaIndex local mode."""

import hashlib
import json
import logging
import shutil
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings
from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.core.ingestion import IngestionPipeline, IngestionCache
from llama_index.core.storage.kvstore import SimpleKVStore

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Manage conversation-specific stores and shared cached sources.

    This is only used for LlamaIndex local mode and always uses Jina embeddings.
    """

    def __init__(
        self,
        conversations_dir: str = "./chroma_db/conversations",
        library_dir: str = "./chroma_db/library"
    ):
        """Initialize the vector store manager.

        Args:
            conversations_dir: Directory for conversation-specific stores.
            library_dir: Directory for shared cached sources.
        """
        self.conversations_dir = Path(conversations_dir)
        self.library_dir = Path(library_dir)

        # Create directories
        self.conversations_dir.mkdir(parents=True, exist_ok=True)
        self.library_dir.mkdir(parents=True, exist_ok=True)

        # Import and initialize Jina embeddings
        from custom_models import get_or_create_jina_embedder
        self.embed_model = get_or_create_jina_embedder()

        # Conversation-specific stores (loaded on demand)
        self.conversation_stores: Dict[str, ChromaVectorStore] = {}
        self.conversation_clients: Dict[str, Any] = {}

        # Node parser for hierarchical chunking
        self.node_parser = HierarchicalNodeParser.from_defaults(
            chunk_sizes=[2048, 512, 256],
            chunk_overlap=128
        )

        # Ingestion pipeline with caching
        cache_path = self.library_dir / "ingestion_cache.json"
        if not cache_path.exists():
            # Initialize an empty cache file so SimpleKVStore can load it.
            SimpleKVStore().persist(str(cache_path))
        self.ingestion_pipeline = IngestionPipeline(
            transformations=[
                self.node_parser,
                self.embed_model,
            ],
            cache=IngestionCache(
                cache=SimpleKVStore.from_persist_path(str(cache_path)),
                collection="library_pipeline"
            ),
        )

        # Library index for shared sources
        self.library_index_path = self.library_dir / "library_index.json"
        self.library_index = self._load_library_index()

        logger.info("VectorStoreManager initialized with Jina embeddings")

    def _create_chroma_client(self, path: Path):
        """Create a ChromaDB client with fallback settings.

        Args:
            path: Directory for persistent storage.

        Returns:
            ChromaDB client instance.
        """
        try:
            return chromadb.PersistentClient(path=str(path))
        except Exception:
            try:
                return chromadb.Client(
                    settings=ChromaSettings(
                        chroma_db_impl="duckdb+parquet",
                        persist_directory=str(path)
                    )
                )
            except Exception:
                logger.warning("Could not create persistent client, using in-memory")
                return chromadb.Client()

    def _load_library_index(self) -> Dict[str, Any]:
        """Load the library index from disk.

        Returns:
            Library index dictionary.
        """
        if not self.library_index_path.exists():
            return {"sources": {}, "by_key": {"file": {}, "url": {}}}

        try:
            with open(self.library_index_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
                data.setdefault("sources", {})
                data.setdefault("by_key", {"file": {}, "url": {}})
                data["by_key"].setdefault("file", {})
                data["by_key"].setdefault("url", {})
                return data
        except Exception as exc:
            logger.error("Failed to load library index: %s", exc)
            return {"sources": {}, "by_key": {"file": {}, "url": {}}}

    def _save_library_index(self) -> None:
        """Persist the library index to disk."""
        try:
            with open(self.library_index_path, "w", encoding="utf-8") as handle:
                json.dump(self.library_index, handle, indent=2, ensure_ascii=True)
        except Exception as exc:
            logger.error("Failed to save library index: %s", exc)

    def get_library_source_id(self, source_type: str, source_key: str) -> Optional[str]:
        """Look up a library source id by type and key.

        Args:
            source_type: Source type ("file" or "url").
            source_key: Unique source key.

        Returns:
            Source id if found, otherwise None.
        """
        return self.library_index.get("by_key", {}).get(source_type, {}).get(source_key)

    def register_library_source(
        self,
        source_type: str,
        source_key: str,
        label: str,
        documents: List[Document],
        source_meta: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, bool]:
        """Register a source in the shared library.

        Args:
            source_type: Source type ("file" or "url").
            source_key: Unique source key.
            label: Human-readable label.
            documents: Documents to cache.
            source_meta: Optional metadata to include.

        Returns:
            Tuple of (source_id, created).
        """
        existing_id = self.get_library_source_id(source_type, source_key)
        if existing_id:
            return existing_id, False

        source_id = hashlib.sha256(f"{source_type}:{source_key}".encode("utf-8")).hexdigest()
        payload_meta = source_meta or {}

        serialized_docs = []
        for idx, doc in enumerate(documents):
            doc_id = hashlib.sha256(
                f"{source_id}:{idx}".encode("utf-8") + (doc.text or "").encode("utf-8")
            ).hexdigest()
            doc.metadata = dict(doc.metadata or {})
            doc.metadata.update(
                {
                    "source_id": source_id,
                    "source_type": source_type,
                    "source_key": source_key,
                    "label": label,
                }
            )
            doc.id_ = doc_id
            serialized_docs.append(
                {
                    "id": doc_id,
                    "text": doc.text,
                    "metadata": doc.metadata,
                }
            )

        source_path = self.library_dir / f"{source_id}.json"
        try:
            with open(source_path, "w", encoding="utf-8") as handle:
                json.dump(serialized_docs, handle, indent=2, ensure_ascii=True)
        except Exception as exc:
            logger.error("Failed to persist library source %s: %s", source_id, exc)
            raise

        entry = {
            "source_id": source_id,
            "source_type": source_type,
            "source_key": source_key,
            "label": label,
            "doc_count": len(serialized_docs),
            "created_at": datetime.utcnow().isoformat(),
        }
        entry.update(payload_meta)
        self.library_index["sources"][source_id] = entry
        self.library_index["by_key"].setdefault(source_type, {})[source_key] = source_id
        self._save_library_index()
        return source_id, True

    def list_library_sources(self) -> List[Dict[str, Any]]:
        """Return all known sources in the library.

        Returns:
            Sorted list of library source entries.
        """
        sources = list(self.library_index.get("sources", {}).values())
        return sorted(sources, key=lambda item: item.get("created_at", ""), reverse=True)

    def load_library_documents(self, source_id: str) -> List[Document]:
        """Load cached documents for a library source.

        Args:
            source_id: Library source id.

        Returns:
            List of Documents for the source.
        """
        source_path = self.library_dir / f"{source_id}.json"
        if not source_path.exists():
            logger.warning("Library source not found: %s", source_id)
            return []

        try:
            with open(source_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception as exc:
            logger.error("Failed to load library source %s: %s", source_id, exc)
            return []

        documents = []
        for item in data:
            metadata = item.get("metadata", {}) or {}
            documents.append(
                Document(
                    text=item.get("text", ""),
                    metadata=metadata,
                    id_=item.get("id"),
                )
            )
        return documents

    def get_or_create_conversation_store(self, session_id: str) -> ChromaVectorStore:
        """Get or create a conversation-specific vector store.

        Args:
            session_id: Session identifier.

        Returns:
            ChromaVectorStore for the session.
        """

        if session_id in self.conversation_stores:
            return self.conversation_stores[session_id]

        # Create conversation-specific directory
        conv_dir = self.conversations_dir / session_id
        conv_dir.mkdir(exist_ok=True)

        # Initialize client and collection
        conv_client = self._create_chroma_client(conv_dir)
        conv_collection = conv_client.get_or_create_collection(f"conv_{session_id}")
        conv_vector_store = ChromaVectorStore(chroma_collection=conv_collection)

        # Cache it
        self.conversation_stores[session_id] = conv_vector_store
        self.conversation_clients[session_id] = conv_client

        logger.info("Created conversation vector store: %s", session_id)
        return conv_vector_store

    def add_documents_to_conversation(
        self,
        documents: List[Document],
        session_id: str,
        index: Optional[VectorStoreIndex] = None
    ) -> VectorStoreIndex:
        """Add documents to a conversation-specific vector store.

        Args:
            documents: Documents to add.
            session_id: Session identifier.
            index: Existing conversation index (optional).

        Returns:
            Updated or new VectorStoreIndex.
        """
        if not documents:
            return index

        conv_vector_store = self.get_or_create_conversation_store(session_id)

        logger.info("Adding %d documents to conversation %s", len(documents), session_id)

        # Shared ingestion pipeline with caching
        all_nodes = self.ingestion_pipeline.run(documents=documents)
        leaf_nodes = get_leaf_nodes(all_nodes)

        if index is None:
            storage_context = StorageContext.from_defaults(vector_store=conv_vector_store)
            index = VectorStoreIndex(
                nodes=leaf_nodes,
                storage_context=storage_context,
                embed_model=self.embed_model
            )
        else:
            index.insert_nodes(leaf_nodes)

        logger.info("Added %d nodes to conversation %s", len(leaf_nodes), session_id)
        return index

    def add_library_source_to_conversation(
        self,
        source_id: str,
        session_id: str,
        index: Optional[VectorStoreIndex] = None
    ) -> VectorStoreIndex:
        """Add a cached library source to a conversation store.

        Args:
            source_id: Library source id.
            session_id: Session identifier.
            index: Existing conversation index (optional).

        Returns:
            Updated or new VectorStoreIndex.
        """
        documents = self.load_library_documents(source_id)
        if not documents:
            return index
        return self.add_documents_to_conversation(documents, session_id, index)

    def delete_conversation_store(self, session_id: str):
        """Delete a conversation-specific vector store.

        Args:
            session_id: Session identifier.
        """

        conv_dir = self.conversations_dir / session_id

        if conv_dir.exists():
            try:
                shutil.rmtree(conv_dir)
                logger.info("Deleted conversation store: %s", session_id)
            except Exception as e:
                logger.error("Failed to delete conversation store: %s", e)

        # Remove from cache
        if session_id in self.conversation_stores:
            del self.conversation_stores[session_id]
        if session_id in self.conversation_clients:
            del self.conversation_clients[session_id]

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about vector stores.

        Returns:
            Dictionary of store statistics.
        """
        # Count conversation-specific documents
        conversation_count = 0
        for conv_id, conv_store in self.conversation_stores.items():
            try:
                conversation_count += conv_store.chroma_collection.count()
            except Exception:
                pass

        return {
            "conversation_documents": conversation_count,
            "library_sources": len(self.library_index.get("sources", {})),
            "conversations_dir": str(self.conversations_dir),
            "library_dir": str(self.library_dir),
            "active_conversations": len(self.conversation_stores)
        }

    def get_conversation_document_count(self, session_id: str) -> int:
        """Get document count for a specific conversation.

        Args:
            session_id: Session identifier.

        Returns:
            Document count for the conversation.
        """
        if session_id not in self.conversation_stores:
            return 0
        try:
            return self.conversation_stores[session_id].chroma_collection.count()
        except Exception:
            return 0
