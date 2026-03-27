import os
import chromadb
from chromadb.config import Settings
from datetime import datetime
from typing import Dict, List
from .models import MemoryItem
from .crypto import verify_item
import logging


class LongTermMemory:
    def __init__(self, agent_public_key):
        self._agent_public_key = agent_public_key
        
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
        
        self._collection = self._client.get_or_create_collection("longterm_memory")
        logging.debug("LongTermMemory initialized")

    def add(self, item: MemoryItem):
        if item.tier != "LONGTERM":
            raise ValueError("Only long-term items allowed")

        if item.signature is None:
            raise ValueError("Unsigned memory cannot enter long-term storage")

        if not verify_item(item, self._agent_public_key):
            raise ValueError("Signature verification failed")

        metadata = {
            "tier": item.tier,
            "created_at": item.created_at.isoformat(),
            "signature": item.signature,
            "user_id": item.user_id,
        }

        self._collection.upsert(
            ids=[item.id], documents=[item.content], metadatas=[metadata]
        )
        logging.debug(f"Added long-term item: {item.id}")


    def get_all_verified(self) -> List[MemoryItem]:
        verified = []
        logging.debug("Getting all verified long-term items")
        results = self._collection.get()
        if results and results.get("ids"):
            for i, m_id in enumerate(results["ids"]):
                metadata = results["metadatas"][i]
                content = results["documents"][i]
                item = MemoryItem.from_dict(metadata, m_id, content)
                if verify_item(item, self._agent_public_key):
                    verified.append(item)
        return verified

    def search(self, query: str, n_results: int = 5) -> List[MemoryItem]:
        logging.debug(f"Searching long-term items with query: {query}")
        results = self._collection.query(query_texts=[query], n_results=n_results)

        verified = []
        if results and results.get("ids") and results["ids"][0]:
            for i, m_id in enumerate(results["ids"][0]):
                metadata = results["metadatas"][0][i]
                content = results["documents"][0][i]
                item = MemoryItem.from_dict(metadata, m_id, content)

                if verify_item(item, self._agent_public_key):
                    verified.append(item)

        return verified
