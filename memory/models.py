from datetime import datetime
from typing import Optional, Literal
from pydantic import BaseModel, Field
import uuid


class MemoryItem(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    tier: Literal["SCRATCH", "SESSION", "LONGTERM"]
    created_at: datetime = Field(default_factory=datetime.utcnow)
    signature: Optional[str] = None
    expires_at: Optional[datetime] = None
    user_id: Optional[str] = None

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
