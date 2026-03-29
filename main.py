from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
import os
import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


from core.agent import process_discord_message

app = FastAPI()


class MessageRequest(BaseModel):
    user_id: str
    message_content: str
    conversation_id: Optional[str] = None


class MessageResponse(BaseModel):
    response: str


@app.get("/")
def main():
    logger.debug("Health check requested")
    return {"status": "ok"}


@app.post("/chat", response_model=MessageResponse)
async def chat(request: Dict[str, Any]):
    user_id = request.get("user_id")
    message_content = request.get("message_content")
    
    logger.info(f"Chat request from user_id={user_id}, content_length={len(message_content) if message_content else 0}")
    
    try:
        response = process_discord_message(
            user_id=user_id,
            message_content=message_content,
        )
        
        logger.info(f"Chat response generated for user_id={user_id}, response_length={len(response)}")
        return MessageResponse(response=response)

    except Exception as e:
        logger.error(f"Error processing chat for user_id={user_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Error processing message: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
