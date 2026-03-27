import os
from typing import List
from datetime import datetime
import chromadb
from chromadb.config import Settings
from .models import MemoryItem
from .crypto import verify_session_item
import logging


class SessionMemory:
    def __init__(self):
        chroma_host = os.getenv("CHROMA_HOST")
        chroma_port = os.getenv("CHROMA_PORT", "8000")
        
        if chroma_host:
            self._client = chromadb.HttpClient(
                host=chroma_host,
                port=int(chroma_port)
            )
        else:
            db_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "chroma_data"
            )
            self._client = chromadb.PersistentClient(
                path=db_path, settings=Settings(allow_reset=True)
            )
        
        self._collection = self._client.get_or_create_collection("session_memory")
        logging.debug("SessionMemory initialized")

    def add(self, item: MemoryItem):
        if item.tier != "SESSION":
            raise ValueError("Only session items allowed")
        if item.expires_at is None or item.expires_at < datetime.utcnow():
            raise ValueError("Item is expired or not set")
        if item.signature is None:
            raise ValueError("Unsigned memory cannot enter session storage")
        if not verify_session_item(item):
            raise ValueError("Checksum verification failed")

        metadata = {
            "tier": item.tier,
            "created_at": item.created_at.isoformat(),
            "signature": item.signature,
            "expires_at": item.expires_at.isoformat(),
            "user_id": item.user_id,
        }

        self._collection.upsert(
            ids=[item.id], documents=[item.content], metadatas=[metadata]
        )

        logging.debug(f"Added session item: {item}")

    def get_active(self) -> List[MemoryItem]:
        now = datetime.utcnow()
        logging.debug("Getting active session items")
        active = []

        results = self._collection.get()
        if results and results.get("ids"):
            for i, m_id in enumerate(results["ids"]):
                metadata = results["metadatas"][i]
                content = results["documents"][i]
                item = MemoryItem.from_dict(metadata, m_id, content)

                if item.expires_at > now and verify_session_item(item):
                    active.append(item)

        return active

    def purge_expired(self):
        now = datetime.utcnow()
        results = self._collection.get()
        if results and results.get("ids"):
            ids_to_delete = []
            for i, m_id in enumerate(results["ids"]):
                metadata = results["metadatas"][i]
                expires_at_str = metadata["expires_at"]
                expires_at = datetime.fromisoformat(expires_at_str)
                if expires_at <= now:
                    ids_to_delete.append(m_id)
            if ids_to_delete:
                self._collection.delete(ids=ids_to_delete)

        logging.debug("Purged expired session items")

    def search(self, query: str, n_results: int = 5) -> List[MemoryItem]:
        logging.debug(f"Searching session items with query: {query}")
        now = datetime.utcnow()
        results = self._collection.query(query_texts=[query], n_results=n_results)

        active = []
        if results and results.get("ids") and results["ids"][0]:
            for i, m_id in enumerate(results["ids"][0]):
                metadata = results["metadatas"][0][i]
                content = results["documents"][0][i]
                item = MemoryItem.from_dict(metadata, m_id, content)

                if item.expires_at > now and verify_session_item(item):
                    active.append(item)

        return active
