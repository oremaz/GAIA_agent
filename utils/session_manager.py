"""
Chat Session Manager
Handles chat history persistence, loading, and conversation memory
Each session stores its own agent configuration
"""

import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import uuid
import shutil

logger = logging.getLogger(__name__)


class Message:
    """Represents a single chat message"""

    def __init__(
        self,
        role: str,
        content: str,
        timestamp: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.role = role
        self.content = content
        self.timestamp = timestamp or datetime.now().isoformat()
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=data.get("timestamp"),
            metadata=data.get("metadata", {})
        )


class ChatSession:
    """Represents a single chat session with agent configuration"""

    def __init__(
        self,
        session_id: Optional[str] = None,
        title: Optional[str] = None,
        messages: Optional[List[Message]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        agent_config: Optional[Dict[str, Any]] = None
    ):
        self.session_id = session_id or str(uuid.uuid4())
        self.title = title or f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        self.messages = messages or []

        # Store agent configuration
        self.agent_config = agent_config or {}

        self.metadata = metadata or {
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "vector_store_id": f"vs_{self.session_id}",
            "agent_config": self.agent_config
        }

    def add_message(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        """Add a message to the session"""
        message = Message(role=role, content=content, metadata=metadata)
        self.messages.append(message)
        self.metadata["updated_at"] = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "title": self.title,
            "messages": [msg.to_dict() for msg in self.messages],
            "metadata": self.metadata,
            "agent_config": self.agent_config
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChatSession':
        return cls(
            session_id=data["session_id"],
            title=data["title"],
            messages=[Message.from_dict(msg) for msg in data.get("messages", [])],
            metadata=data.get("metadata", {}),
            agent_config=data.get("agent_config", {})
        )


class SessionManager:
    """Manages chat sessions with persistence"""

    def __init__(self, storage_path: str = ".chat_sessions"):
        """
        Initialize session manager

        Args:
            storage_path: Directory to store session files
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)

        # Index file tracks all sessions
        self.index_file = self.storage_path / "sessions_index.json"
        self.sessions_index = self._load_index()

        logger.info("SessionManager initialized: %s", self.storage_path)

    def _load_index(self) -> Dict[str, Dict[str, Any]]:
        """Load sessions index from disk"""
        if not self.index_file.exists():
            return {}

        try:
            with open(self.index_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error("Failed to load sessions index: %s", e)
            return {}

    def _save_index(self):
        """Save sessions index to disk"""
        try:
            with open(self.index_file, 'w', encoding='utf-8') as f:
                json.dump(self.sessions_index, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error("Failed to save sessions index: %s", e)

    def create_session(
        self,
        title: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        agent_config: Optional[Dict[str, Any]] = None
    ) -> ChatSession:
        """Create a new chat session with agent configuration"""
        session = ChatSession(title=title, metadata=metadata, agent_config=agent_config)

        # Add to index
        self.sessions_index[session.session_id] = {
            "title": session.title,
            "created_at": session.metadata["created_at"],
            "updated_at": session.metadata["updated_at"],
            "message_count": 0,
            "agent_config": agent_config or {}
        }

        self._save_index()
        self._save_session(session)

        logger.info("Created session: %s", session.session_id)
        return session

    def load_session(self, session_id: str) -> Optional[ChatSession]:
        """Load a session from disk"""
        session_file = self.storage_path / f"{session_id}.json"

        if not session_file.exists():
            logger.warning("Session not found: %s", session_id)
            return None

        try:
            with open(session_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            session = ChatSession.from_dict(data)
            logger.info("Loaded session: %s", session_id)
            return session

        except Exception as e:
            logger.error("Failed to load session %s: %s", session_id, e)
            return None

    def save_session(self, session: ChatSession):
        """Save a session to disk"""
        self._save_session(session)

        # Update index
        if session.session_id in self.sessions_index:
            self.sessions_index[session.session_id].update({
                "title": session.title,
                "updated_at": session.metadata["updated_at"],
                "message_count": len(session.messages)
            })
            self._save_index()

    def _save_session(self, session: ChatSession):
        """Internal method to save session"""
        session_file = self.storage_path / f"{session.session_id}.json"

        try:
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(session.to_dict(), f, indent=2, ensure_ascii=False)
            logger.debug("Saved session: %s", session.session_id)
        except Exception as e:
            logger.error("Failed to save session %s: %s", session.session_id, e)

    def list_sessions(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        List all sessions sorted by most recent

        Args:
            limit: Maximum number of sessions to return

        Returns:
            List of session metadata dicts
        """
        sessions = [
            {
                "session_id": sid,
                **metadata
            }
            for sid, metadata in self.sessions_index.items()
        ]

        # Sort by updated_at (most recent first)
        sessions.sort(key=lambda x: x.get("updated_at", ""), reverse=True)

        if limit:
            sessions = sessions[:limit]

        return sessions

    def delete_session(self, session_id: str, cleanup_vector_store: bool = True) -> bool:
        """
        Delete a session and optionally its conversation-specific vector store

        Args:
            session_id: Session ID to delete
            cleanup_vector_store: Whether to delete conversation vector store

        Returns:
            True if successful
        """
        session_file = self.storage_path / f"{session_id}.json"

        try:
            # Delete session file
            if session_file.exists():
                session_file.unlink()

            # Remove from index
            if session_id in self.sessions_index:
                del self.sessions_index[session_id]
                self._save_index()

            # Cleanup conversation-specific vector store
            if cleanup_vector_store:
                conv_vector_dir = Path(f"./chroma_db/conversations/{session_id}")
                if conv_vector_dir.exists():
                    shutil.rmtree(conv_vector_dir)
                    logger.info("Deleted conversation vector store: %s", session_id)

            logger.info("Deleted session: %s", session_id)
            return True

        except Exception as e:
            logger.error("Failed to delete session %s: %s", session_id, e)
            return False
