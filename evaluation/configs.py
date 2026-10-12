"""
Pipeline configurations used for ablation.

``staged`` re-composes the real stage functions with switches so each
component's contribution can be measured. With every switch on it must agree
with ``classifier.pipeline.run``; ``run_eval`` checks that, so the ablations
cannot silently drift from the production code.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

from classifier import poison as poison_detector
from classifier.pipeline import cap_tier, run as pipeline_run
from classifier.stage_4.policy import DEFAULT_LONGTERM_SOURCES
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
    ambiguous: bool = False
    poison: Optional[float] = None


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
    trust_cap: bool = True,
    harden: bool = True,
    poison: bool = False,
    threshold: float = 0.65,
    api_key: Optional[str] = None,
) -> Outcome:
    lt_sources = DEFAULT_LONGTERM_SOURCES if trust_cap else None
    out = _staged(content, meta, s1=s1, s4=s4, s5=s5, harden=harden, poison=poison, threshold=threshold, api_key=api_key)
    if out.passed and out.tier:
        out.tier = cap_tier(out.tier, meta, lt_sources, content, out.ambiguous, harden, out.poison)[0]
    return out


def _staged(
    content: str,
    meta: Dict[str, Any],
    *,
    s1: bool,
    s4: bool,
    s5: bool,
    harden: bool,
    poison: bool,
    threshold: float,
    api_key: Optional[str],
) -> Outcome:
    if s1 and not run_heuristics(content, meta, harden=harden):
        return Outcome(False, None, 1)

    scores = embed_classify(content, meta)
    pz = poison_detector.score(content, meta).score if poison else None
    if pz is not None and pz >= poison_detector.DEFAULT_REJECT_THRESHOLD:
        return Outcome(False, None, 2, poison=pz)

    top, confident = check_confidence(scores, threshold=threshold)
    candidate = top or "SCRATCH"

    verdict = PolicyVerdict.PASS
    ambiguous = False
    if s4:
        verdict = check_policy(content=content, metadata=meta, tier=candidate, harden=harden).verdict
        ambiguous = verdict == PolicyVerdict.AMBIGUOUS
        if verdict == PolicyVerdict.REJECT:
            return Outcome(False, None, 4, confident, poison=pz)
        if verdict == PolicyVerdict.PASS and confident:
            return Outcome(True, top, 4, confident, ambiguous, pz)
    elif confident:
        return Outcome(True, top, 3, confident, poison=pz)

    if not s5:
        return Outcome(True, top, 3, confident, ambiguous, pz)

    tier, _, _ = llm_judge(content, meta, scores, api_key=api_key)
    return Outcome(True, tier, 5, confident, ambiguous, pz)


def production(content: str, meta: Dict[str, Any], *, threshold: float = 0.65,
               api_key: Optional[str] = None, **kw) -> Outcome:
    r = pipeline_run(content, meta, confidence_threshold=threshold, llm_api_key=api_key, **kw)
    return Outcome(r.passed, r.tier if r.passed else None, r.stage_reached, poison=r.poison_score)


# name -> callable(content, meta, threshold=..., api_key=...)
CONFIGS = {
    "A flat memory (baseline)": flat_baseline,
    "B stage2 only": lambda c, m, **k: staged(c, m, s1=False, s4=False, s5=False, trust_cap=False, harden=False, **k),
    "C stage1+2": lambda c, m, **k: staged(c, m, s1=True, s4=False, s5=False, trust_cap=False, harden=False, **k),
    "D stage1+2+4 (no LLM judge)": lambda c, m, **k: staged(c, m, s1=True, s4=True, s5=False, trust_cap=False, harden=False, **k),
    "E1 legacy: no trust cap, no hardening": lambda c, m, **k: production(c, m, longterm_sources=None, harden=False, poison=False, **k),
    "E2 trust cap only": lambda c, m, **k: production(c, m, harden=False, poison=False, **k),
    "E3 + hardening (no poison detector)": lambda c, m, **k: production(c, m, poison=False, **k),
    "F full pipeline": production,
}
