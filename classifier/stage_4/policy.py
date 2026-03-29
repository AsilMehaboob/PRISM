import re
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Set

logger = logging.getLogger(__name__)

class PolicyVerdict(Enum):
    PASS      = "pass"
    REJECT    = "reject"
    AMBIGUOUS = "ambiguous"  

@dataclass
class PolicyResult:
    verdict: PolicyVerdict
    reason: str

DEFAULT_ALLOWED_SOURCES: Set[str] = {"user", "agent", "system", "api", "tool"}

_PROHIBITED: list[re.Pattern] = [
    
    re.compile(
        r'\b(kill|murder|attack|bomb|weapon|explosive|poison)\b.{0,40}'
        r'\b(how to|instruction|guide|tutorial|step.?by.?step|recipe)\b',
        re.IGNORECASE,
    ),
    
    re.compile(
        r'\b(hack|crack|exploit|bypass|inject|brute.?force)\b.{0,40}'
        r'\b(system|password|auth|database|credential|account)\b',
        re.IGNORECASE,
    ),
    
    re.compile(r'\b(suicide|self.?harm|self.?destruct|end my life)\b', re.IGNORECASE),
    
    re.compile(
        r'\b(malware|ransomware|trojan|spyware|keylogger|rootkit)\b.{0,30}'
        r'\b(create|build|deploy|install|spread|distribute)\b',
        re.IGNORECASE,
    ),
]

_AMBIGUOUS: list[re.Pattern] = [
    re.compile(r'\b(confidential|classified|top.?secret|internal only)\b', re.IGNORECASE),
    re.compile(
        r'\b(delete|remove|wipe|purge|destroy)\b.{0,20}'
        r'\b(memory|data|record|log|history|trace)\b',
        re.IGNORECASE,
    ),
    re.compile(
        r'\b(override|bypass|ignore|disable|circumvent)\b.{0,20}'
        r'\b(rule|policy|filter|check|restriction|safety|guard)\b',
        re.IGNORECASE,
    ),
    re.compile(r'\b(jailbreak|prompt.?inject|system.?prompt|ignore previous)\b', re.IGNORECASE),
]

_TIER_MIN_LENGTH: Dict[str, int] = {
    "SCRATCH":  1,
    "SESSION":  5,
    "LONGTERM": 10,
}

def check_policy(
    content: str,
    metadata: Dict[str, Any],
    tier: str,
    allowed_sources: Optional[Set[str]] = None,
    min_trust_score: float = 0.0,
) -> PolicyResult:
    """
    Run the Policy and Safety Gate against a candidate memory.

    Checks (in order):
      1. Source allowlist
      2. Trust score floor
      3. Tier-specific minimum content length
      4. Prohibited content  → REJECT
      5. Ambiguous content   → AMBIGUOUS (route to Stage 5)

    Returns:
        :class:`PolicyResult` with verdict PASS | REJECT | AMBIGUOUS
    """
    sources = allowed_sources or DEFAULT_ALLOWED_SOURCES
    source = str(metadata.get("source", "")).strip().lower()

    if source not in sources:
        logger.debug("Stage 4 | REJECT source=%r not in allowlist", source)
        return PolicyResult(
            verdict=PolicyVerdict.REJECT,
            reason=f"Source {source!r} is not in the allowed sources list",
        )

    try:
        raw_score = float(metadata.get("trust_score", 1.0))
        trust_score = min(1.0, max(0.0, raw_score))
    except (TypeError, ValueError):
        trust_score = 0.0

    if trust_score < min_trust_score:
        logger.debug("Stage 4 | REJECT trust_score=%.2f < min=%.2f", trust_score, min_trust_score)
        return PolicyResult(
            verdict=PolicyVerdict.REJECT,
            reason=f"Trust score {trust_score:.2f} is below the minimum {min_trust_score:.2f}",
        )

    min_len = _TIER_MIN_LENGTH.get(tier, 1)
    if len(content.strip()) < min_len:
        logger.debug("Stage 4 | REJECT content too short for tier=%s", tier)
        return PolicyResult(
            verdict=PolicyVerdict.REJECT,
            reason=f"Content is too short for {tier} tier (minimum {min_len} characters)",
        )

    for pattern in _PROHIBITED:
        if pattern.search(content):
            logger.warning("Stage 4 | REJECT prohibited content matched pattern=%s", pattern.pattern[:60])
            return PolicyResult(
                verdict=PolicyVerdict.REJECT,
                reason="Content matches a prohibited safety pattern",
            )

    for pattern in _AMBIGUOUS:
        if pattern.search(content):
            logger.info("Stage 4 | AMBIGUOUS matched pattern=%s", pattern.pattern[:60])
            return PolicyResult(
                verdict=PolicyVerdict.AMBIGUOUS,
                reason="Content matches an ambiguous policy pattern; routing to LLM judge",
            )

    logger.debug("Stage 4 | PASS tier=%s source=%r", tier, source)
    return PolicyResult(verdict=PolicyVerdict.PASS, reason="All policy checks passed")
