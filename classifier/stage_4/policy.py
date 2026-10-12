import re
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Set

from ..normalize import variants

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

# Only content that originates from the user or the operator may become a
# permanent (LONGTERM) memory. Tool output, web pages and text the agent
# generated itself can carry indirect injections, so they are capped at SESSION.
DEFAULT_LONGTERM_SOURCES: Set[str] = {"user", "system"}

_PROHIBITED: list[re.Pattern] = [
    # Violence instructions (Forward & Reverse)
    re.compile(
        r'\b(kill|murder|attack|bomb|weapon|explosive|poison)\b'
        r'(?:(?!\n\n).){0,60}'
        r'\b(how to|instruction|guide|tutorial|step.?by.?step|recipe)\b',
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r'\b(how to|instruction|guide|tutorial|step.?by.?step|recipe)\b'
        r'(?:(?!\n\n).){0,60}'
        r'\b(kill|murder|attack|bomb|weapon|explosive|poison)\b',
        re.IGNORECASE | re.DOTALL,
    ),
    
    # Hacking (Forward & Reverse)
    re.compile(
        r'\b(hack|crack|exploit|bypass|inject|brute.?force)\b'
        r'(?:(?!\n\n).){0,60}'
        r'\b(system|password|auth|database|credential|account)\b',
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r'\b(system|password|auth|database|credential|account)\b'
        r'(?:(?!\n\n).){0,60}'
        r'\b(hack|crack|exploit|bypass|inject|brute.?force)\b',
        re.IGNORECASE | re.DOTALL,
    ),
    
    re.compile(r'\b(suicide|self.?harm|self.?destruct|end my life)\b', re.IGNORECASE),
    
    # Malware creation (Forward & Reverse)
    re.compile(
        r'\b(malware|ransomware|trojan|spyware|keylogger|rootkit)\b'
        r'(?:(?!\n\n).){0,60}'
        r'\b(create|build|deploy|install|spread|distribute)\b',
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r'\b(create|build|deploy|install|spread|distribute)\b'
        r'(?:(?!\n\n).){0,60}'
        r'\b(malware|ransomware|trojan|spyware|keylogger|rootkit)\b',
        re.IGNORECASE | re.DOTALL,
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
        r'\b(override|bypass|ignore|disregard|forget|disable|circumvent|skip|turn off)\b.{0,30}'
        r'\b(rule|policy|filter|check|restriction|safety|guard|guideline|instruction|confirmation|verification)s?\b',
        re.IGNORECASE,
    ),
    re.compile(
        r'\b(ignore|disregard|forget|override)\b.{0,20}\b(previous|prior|earlier|above|all)\b.{0,20}'
        r'\b(instruction|rule|guideline|prompt)s?\b',
        re.IGNORECASE,
    ),
    re.compile(r'\b(developer|debug|admin|god|dan)\s+mode\b', re.IGNORECASE),
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
    harden: bool = True,
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

    # Hardened mode also matches against de-obfuscated copies of the text.
    texts = variants(content) if harden else [content]

    for pattern in _PROHIBITED:
        if any(pattern.search(t) for t in texts):
            logger.warning("Stage 4 | REJECT prohibited content matched pattern=%s", pattern.pattern[:60])
            return PolicyResult(
                verdict=PolicyVerdict.REJECT,
                reason="Content matches a prohibited safety pattern",
            )

    for pattern in _AMBIGUOUS:
        if any(pattern.search(t) for t in texts):
            logger.info("Stage 4 | AMBIGUOUS matched pattern=%s", pattern.pattern[:60])
            return PolicyResult(
                verdict=PolicyVerdict.AMBIGUOUS,
                reason="Content matches an ambiguous policy pattern; routing to LLM judge",
            )

    logger.debug("Stage 4 | PASS tier=%s source=%r", tier, source)
    return PolicyResult(verdict=PolicyVerdict.PASS, reason="All policy checks passed")
