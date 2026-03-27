import os
import logging
from .models import MemoryItem
from classifier.pipeline import run as run_pipeline

logger = logging.getLogger(__name__)

# use the 5-stage classification pipeline
def classifier(item: MemoryItem) -> None:
    metadata = {
        "user_id": item.user_id,
        "source": "user", # Using 'user' to pass Stage 4 policy allowlist
    }
    
    # Run the full pipeline
    result = run_pipeline(
        content=item.content,
        metadata=metadata,
        llm_api_key=os.getenv("GEMINI_API_KEY")
    )
    
    if result.passed and result.tier:
        item.tier = result.tier
        msg = f"Pipeline classified memory {item.id[:8]} as {result.tier} (Confidence: {result.confidence:.2f})"
        print(f"\n[DEBUG] {msg}")  # Guaranteed terminal feedback
        logger.info(msg)
    else:
        item.tier = "SCRATCH" # Default to scratch if rejected by pipeline
        msg = f"Pipeline rejected memory {item.id[:8]}, defaulting to SCRATCH. Reason: {result.reasoning}"
        print(f"\n[DEBUG] {msg}")  # Guaranteed terminal feedback
        logger.info(msg)
