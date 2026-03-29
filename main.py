from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
import os
import logging
from dotenv import load_dotenv

load_dotenv()

# Configure logging to show pipeline and router info messages
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


from core.agent import process_discord_message, get_user_memory_summary

app = FastAPI()


class MessageRequest(BaseModel):
    user_id: str
    message_content: str
    conversation_id: Optional[str] = None


class MessageResponse(BaseModel):
    response: str


class MemorySummaryResponse(BaseModel):
    summary: str


@app.get("/")
def main():
    return {"status": "ok"}


@app.post("/chat", response_model=MessageResponse)
async def chat(request: Dict[str, Any]):
    try:
        user_id = request.get("user_id")
        message_content = request.get("message_content")

        response = process_discord_message(
            user_id=user_id,
            message_content=message_content,
        )

        return MessageResponse(response=response)

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing message: {str(e)}"
        )


@app.get("/memory/{user_id}", response_model=MemorySummaryResponse)
async def get_memory_summary(user_id: str):
    try:
        summary = get_user_memory_summary(user_id)
        return MemorySummaryResponse(summary=summary).__dict__
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error getting memory summary: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
