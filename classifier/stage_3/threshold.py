import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLD = 0.65
DEFAULT_MARGIN = 0.15

def check_confidence(
    scores: Dict[str, float],
    threshold: float = DEFAULT_THRESHOLD,
    min_margin: float = DEFAULT_MARGIN,
) -> Tuple[Optional[str], bool]:
    """
    Evaluate whether Stage 2 probability scores exceed the confidence threshold.

    Returns:
        (top_tier, is_confident)

        is_confident=True  → score is decisive; skip Stage 5 and go straight to Final Decision.
        is_confident=False → score is ambiguous; route to Stage 5 (LLM Judge).
    """
    if not scores:
        logger.warning("Stage 3 | received empty scores")
        return None, False

    top_tier = max(scores, key=lambda k: scores[k])
    top_score = scores[top_tier]

    runner_up = 0.0
    for t, s in scores.items():
        if t != top_tier and s > runner_up:
            runner_up = s

    margin = top_score - runner_up
    is_confident = top_score >= threshold and margin >= min_margin

    logger.debug(
        "Stage 3 | top_tier=%s top_score=%.3f runner_up=%.3f margin=%.3f threshold=%.3f min_margin=%.3f confident=%s",
        top_tier,
        top_score,
        runner_up,
        margin,
        threshold,
        min_margin,
        is_confident,
    )

    return top_tier, is_confident
