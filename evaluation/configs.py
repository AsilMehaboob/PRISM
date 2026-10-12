"""
Pipeline configurations used for ablation.

``staged`` re-composes the real stage functions with switches so each
component's contribution can be measured. With every switch on it must agree
with ``classifier.pipeline.run``; ``run_eval`` checks that, so the ablations
cannot silently drift from the production code.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

from classifier.pipeline import run as pipeline_run
from classifier.stage_1.heuristics import run_heuristics
from classifier.stage_2.embedding import classify as embed_classify
from classifier.stage_3.threshold import check_confidence
from classifier.stage_4.policy import PolicyVerdict, check_policy
from classifier.stage_5.llm_judge import judge as llm_judge


@dataclass
class Outcome:
    passed: bool
    tier: Optional[str]      # None when rejected
    stage: int               # last stage reached
    confident: Optional[bool] = None


def flat_baseline(content: str, meta: Dict[str, Any], **_) -> Outcome:
    """Conventional flat memory: everything is stored as long-term."""
    return Outcome(True, "LONGTERM", 0)


def staged(
    content: str,
    meta: Dict[str, Any],
    *,
    s1: bool = True,
    s4: bool = True,
    s5: bool = True,
    threshold: float = 0.65,
    api_key: Optional[str] = None,
) -> Outcome:
    if s1 and not run_heuristics(content, meta):
        return Outcome(False, None, 1)

    scores = embed_classify(content, meta)
    top, confident = check_confidence(scores, threshold=threshold)
    candidate = top or "SCRATCH"

    verdict = PolicyVerdict.PASS
    if s4:
        verdict = check_policy(content=content, metadata=meta, tier=candidate).verdict
        if verdict == PolicyVerdict.REJECT:
            return Outcome(False, None, 4, confident)
        if verdict == PolicyVerdict.PASS and confident:
            return Outcome(True, top, 4, confident)
    elif confident:
        return Outcome(True, top, 3, confident)

    if not s5:
        return Outcome(True, top, 3, confident)

    tier, _, _ = llm_judge(content, meta, scores, api_key=api_key)
    return Outcome(True, tier, 5, confident)


def production(content: str, meta: Dict[str, Any], *, threshold: float = 0.65,
               api_key: Optional[str] = None, **_) -> Outcome:
    r = pipeline_run(content, meta, confidence_threshold=threshold, llm_api_key=api_key)
    return Outcome(r.passed, r.tier if r.passed else None, r.stage_reached)


# name -> callable(content, meta, threshold=..., api_key=...)
CONFIGS = {
    "A flat memory (baseline)": flat_baseline,
    "B stage2 only": lambda c, m, **k: staged(c, m, s1=False, s4=False, s5=False, **k),
    "C stage1+2": lambda c, m, **k: staged(c, m, s1=True, s4=False, s5=False, **k),
    "D stage1+2+4 (no LLM judge)": lambda c, m, **k: staged(c, m, s1=True, s4=True, s5=False, **k),
    "E full pipeline": production,
}
