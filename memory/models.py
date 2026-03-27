from dataclasses import dataclass
from datetime import datetime
from typing import Optional
import uuid


@dataclass
class MemoryItem:
    id: str
    content: str
    tier: str  # SCRATCH | SESSION | LONGTERM
    created_at: datetime
    signature: Optional[str] = None
    expires_at: Optional[datetime] = None
    user_id: Optional[str] = None

    @staticmethod
    def create(
        content: str,
        tier: str,
        expires_at: Optional[datetime] = None,
        user_id: Optional[str] = None,
    ):
        return MemoryItem(
            id=str(uuid.uuid4()),
            content=content,
            tier=tier,
            created_at=datetime.utcnow(),
            expires_at=expires_at,
            signature=None,
            user_id=user_id,
        )

    @classmethod
    def from_dict(cls, data: dict, m_id: str, content: str):
        expires_at_raw = data.get("expires_at")
        return cls(
            id=m_id,
            content=content,
            tier=data["tier"],
            created_at=datetime.fromisoformat(data["created_at"]),
            signature=data["signature"],
            expires_at=(
                datetime.fromisoformat(expires_at_raw) if expires_at_raw else None
            ),
            user_id=data.get("user_id"),
        )
