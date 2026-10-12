"""
Append-only, hash-chained audit log of memory writes.

Each line is a JSON record whose ``hash`` covers the previous record's hash and
its own content, so editing, deleting or reordering a record breaks the chain.
Truncating the *tail* is only detectable if the latest hash (``head()``) was
recorded somewhere else, so anchor it externally if that matters.

The log stores a SHA-256 of the content, not the content itself.
"""

import hashlib
import json
import os
import threading
import time
from typing import Optional, Tuple

from .models import MemoryItem

GENESIS = "0" * 64


class AuditLog:
    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    @classmethod
    def default(cls) -> "AuditLog":
        path = os.getenv("AUDIT_LOG_PATH") or os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "chroma_data", "audit.jsonl"
        )
        return cls(path)

    @staticmethod
    def _digest(prev: str, record: dict) -> str:
        body = json.dumps(record, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256((prev + body).encode()).hexdigest()

    def head(self) -> str:
        last = GENESIS
        if os.path.exists(self.path):
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        last = json.loads(line)["hash"]
        return last

    def record(self, action: str, item: MemoryItem) -> str:
        event = {
            "ts": time.time(),
            "action": action,
            "item_id": item.id,
            "tier": item.tier,
            "user_id": item.user_id,
            "source": item.source,
            "key_id": item.key_id,
            "content_sha256": hashlib.sha256(item.content.encode()).hexdigest(),
        }
        with self._lock:
            prev = self.head()
            digest = self._digest(prev, event)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps({**event, "prev": prev, "hash": digest}, sort_keys=True) + "\n")
        return digest

    def verify(self) -> Tuple[bool, Optional[int]]:
        """Return (ok, index of the first bad record)."""
        prev = GENESIS
        if not os.path.exists(self.path):
            return True, None
        with open(self.path, encoding="utf-8") as f:
            for i, line in enumerate(l for l in f if l.strip()):
                try:
                    rec = json.loads(line)
                    claimed, rec_prev = rec.pop("hash"), rec.pop("prev")
                except (ValueError, KeyError):
                    return False, i
                if rec_prev != prev or self._digest(prev, rec) != claimed:
                    return False, i
                prev = claimed
        return True, None
