"""
Memory Classification Pipeline
===============================

Stage 1  – Deterministic heuristics             → Pass / Reject
Stage 2  – Feature-based embedding classifier   → Probability scores
Stage 3  – Confidence threshold check           → Confident / Ambiguous
Stage 4  – Policy & Safety Gate (always runs)   → Pass / Reject / Ambiguous
             ├─ Reject           → Rejected result  (any confidence level)
             ├─ Pass + Confident → Final Decision
             ├─ Pass + Ambiguous → Stage 5
             └─ Ambiguous        → Stage 5
Stage 5  – LLM Judge advisory                   → Final Decision

Final Decision routes to: SCRATCH | SESSION | LONGTERM
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Set

from .normalize import suspicion_flags
from . import poison as poison_detector
from .stage_1.heuristics import run_heuristics
from .stage_2.embedding import classify as embedding_classify
from .stage_3.threshold import check_confidence
from .stage_4.policy import check_policy, PolicyVerdict, DEFAULT_LONGTERM_SOURCES
from .stage_5.llm_judge import judge as llm_judge, JUDGE_BASE_URL, JUDGE_MODEL

logger = logging.getLogger(__name__)

DEFAULT_CONFIDENCE_THRESHOLD = 0.65

@dataclass
class ClassificationResult:
    """Outcome of the full classification pipeline."""

    passed: bool
    """False when the memory was rejected at any stage."""

    tier: Optional[str]
    """SCRATCH | SESSION | LONGTERM, or None when rejected."""

    confidence: float
    """Probability / confidence of the chosen tier (0–1)."""

    stage_reached: int
    """Last stage that executed (1–5)."""

    reasoning: str
    """Human-readable decision explanation."""

    stage2_scores: Dict[str, float] = field(default_factory=dict)
    """Stage 2 probability distribution (populated when passed=True)."""

    capped_from: Optional[str] = None
    """Tier originally chosen, when a safety cap lowered it (else None)."""

    poison_score: Optional[float] = None
    """Poison detector score (0-1): likelihood the text instructs the agent."""


def cap_tier(
    tier: str,
    metadata: Dict[str, Any],
    longterm_sources: Optional[Set[str]],
    content: str = "",
    ambiguous: bool = False,
    harden: bool = True,
    poison_score: Optional[float] = None,
    poison_cap: float = poison_detector.DEFAULT_CAP_THRESHOLD,
):
    """
    Safety caps that lower LONGTERM to SESSION. They run after the LLM judge,
    so the judge can never promote something these rules held back.

    * untrusted source     - only trusted sources may write permanent memory
    * ambiguous content    - policy flagged it (injection-like phrasing)
    * suspicious content   - URL / e-mail / IP / shell / encoded payload
    * poison score         - the text reads like an instruction to the agent

    Returns ``(tier, capped_from, note)``. ``longterm_sources=None`` disables
    the source cap and ``harden=False`` the other two (legacy, for ablation).
    """
    if tier != "LONGTERM":
        return tier, None, ""
    reasons = []
    source = str(metadata.get("source", "")).strip().lower()
    if longterm_sources is not None and source not in longterm_sources:
        reasons.append(f"source {source!r} is not trusted for long-term memory")
    if harden and ambiguous:
        reasons.append("policy flagged the content as ambiguous")
    if harden:
        flags = suspicion_flags(content)
        if flags:
            reasons.append("suspicious content (" + ", ".join(flags) + ")")
    if poison_score is not None and poison_score >= poison_cap:
        reasons.append(f"reads like an instruction to the agent (poison score {poison_score:.2f})")
    if not reasons:
        return tier, None, ""
    return "SESSION", "LONGTERM", "; capped LONGTERM->SESSION: " + "; ".join(reasons)


def run(
    content: str,
    metadata: Dict[str, Any],
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    allowed_sources: Optional[Set[str]] = None,
    min_trust_score: float = 0.0,
    longterm_sources: Optional[Set[str]] = DEFAULT_LONGTERM_SOURCES,
    harden: bool = True,
    poison: bool = True,
    poison_cap: float = poison_detector.DEFAULT_CAP_THRESHOLD,
    poison_reject: float = poison_detector.DEFAULT_REJECT_THRESHOLD,
    llm_api_key: Optional[str] = None,
    llm_base_url: str = JUDGE_BASE_URL,
    llm_model: str = JUDGE_MODEL,
) -> ClassificationResult:
    """
    Run the full 5-stage memory classification pipeline.

    Stage 4 (Policy & Safety Gate) is a hard gate that runs on **all** content
    regardless of Stage 3 confidence — a REJECT there is always final.
    Ambiguous or low-confidence content is forwarded to Stage 5 (LLM Judge).

    Args:
        content:              Raw text of the memory candidate.
        metadata:             Must contain at least ``{"source": str}``.
        confidence_threshold: Minimum Stage 2 top-score to skip Stage 5.
        allowed_sources:      Overrides the Stage 4 default source allowlist.
        min_trust_score:      Stage 4 minimum trust score (0–1).
        longterm_sources:     Sources allowed to reach LONGTERM; others are capped at
                              SESSION. ``None`` disables the cap.
        harden:               Normalise text before the safety patterns and cap
                              ambiguous/suspicious content at SESSION.
        poison:               Run the poison detector: a score >= poison_cap keeps the
                              item out of LONGTERM, >= poison_reject drops it entirely.
        llm_api_key:          OpenAI-compatible key (falls back to OPENROUTER_API_KEY env var).
        llm_base_url:         Chat completions base URL for Stage 5.
        llm_model:            Model name for Stage 5.

    Returns:
        :class:`ClassificationResult`
    """

    if not run_heuristics(content, metadata, harden=harden):
        logger.info("Pipeline | REJECTED at Stage 1")
        return ClassificationResult(
            passed=False,
            tier=None,
            confidence=1.0,
            stage_reached=1,
            reasoning="Rejected by Stage 1 deterministic heuristics",
        )

    scores = embedding_classify(content, metadata)
    logger.info("Pipeline | Stage 2 scores: %s", {k: f"{v:.3f}" for k, v in scores.items()})

    pz = poison_detector.score(content, metadata) if poison else None
    pz_score = pz.score if pz else None
    if pz and pz.score >= poison_reject:
        logger.info("Pipeline | REJECTED by poison detector (score=%.2f, %s)", pz.score, pz.features)
        return ClassificationResult(
            passed=False,
            tier=None,
            confidence=pz.score,
            stage_reached=2,
            reasoning=f"Rejected by poison detector: reads like an instruction to the agent (score={pz.score:.2f}, features={pz.features})",
            stage2_scores=scores,
            poison_score=pz.score,
        )

    top_tier, is_confident = check_confidence(scores, threshold=confidence_threshold)

    candidate_tier = top_tier or "SCRATCH"

    policy = check_policy(
        content=content,
        metadata=metadata,
        tier=candidate_tier,
        allowed_sources=allowed_sources,
        min_trust_score=min_trust_score,
        harden=harden,
    )
    ambiguous = policy.verdict == PolicyVerdict.AMBIGUOUS

    if policy.verdict == PolicyVerdict.REJECT:
        logger.info("Pipeline | REJECTED at Stage 4: %s", policy.reason)
        return ClassificationResult(
            passed=False,
            tier=None,
            confidence=scores.get(candidate_tier, 0.0),
            stage_reached=4,
            reasoning=f"Rejected by Stage 4 policy gate: {policy.reason}",
            stage2_scores=scores,
        )

    if policy.verdict == PolicyVerdict.PASS and is_confident:
        
        tier, capped_from, note = cap_tier(top_tier, metadata, longterm_sources, content, ambiguous, harden, pz_score, poison_cap)
        logger.info("Pipeline | PASSED Stage 4 + confident → Final Decision tier=%s", tier)
        return ClassificationResult(
            passed=True,
            tier=tier,
            confidence=scores[top_tier],
            stage_reached=4,
            reasoning=(
                f"Stage 4 policy gate passed "
                f"(embedding score={scores[top_tier]:.3f}){note}"
            ),
            stage2_scores=scores,
            capped_from=capped_from,
            poison_score=pz_score,
        )

    if policy.verdict == PolicyVerdict.AMBIGUOUS:
        logger.info("Pipeline | Stage 4 AMBIGUOUS → routing to Stage 5 LLM Judge")
    else:
        logger.info(
            "Pipeline | Stage 4 passed but below confidence (top=%.3f) → Stage 5 LLM Judge",
            scores.get(top_tier, 0.0) if top_tier else 0.0,
        )

    tier, confidence, reasoning = llm_judge(
        content,
        metadata,
        scores,
        api_key=llm_api_key,
        base_url=llm_base_url,
        model=llm_model,
    )

    tier, capped_from, note = cap_tier(tier, metadata, longterm_sources, content, ambiguous, harden, pz_score, poison_cap)
    logger.info("Pipeline | Stage 5 Final Decision → tier=%s confidence=%.3f", tier, confidence)
    return ClassificationResult(
        passed=True,
        tier=tier,
        confidence=confidence,
        stage_reached=5,
        reasoning=reasoning + note,
        stage2_scores=scores,
        capped_from=capped_from,
        poison_score=pz_score,
    )
