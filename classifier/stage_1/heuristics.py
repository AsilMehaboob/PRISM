import logging
import re
from typing import Dict, Any

SENSITIVE_PATTERNS = {
    "credit_card": re.compile(r'\b(?:\d[ -]*?){13,16}\b'), 
    "api_key": re.compile(r'(?i)(?:api[_-]?key|secret|token|sk-[a-zA-Z0-9]{32,})'), 
    "ssn":          re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
    "passport":     re.compile(r'\b[A-Z]{1,2}\d{6,9}\b'),
    "email_pass_combo": re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}:[^\s]+'), 
    "numeric_id": re.compile(r'\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b'),
}

TEMPORAL_RULES = {
    "immediate": {
        "weight": 3,
        "patterns": [
            r"\bright now\b",
            r"\bat the moment\b",
            r"\bcurrently\b",
            r"\bfor now\b",
            r"\bfor the time being\b",
        ],
    },
    "short_window": {
        "weight": 2,
        "patterns": [
            r"\btoday\b",
            r"\btonight\b",
            r"\btomorrow\b",
            r"\bthis week\b",
            r"\bthis weekend\b",
            r"\bthis month\b",
            r"\bthis year\b",
        ],
    },
    "future_intent": {
        "weight": 3,
        "patterns": [
            r"\bi will\b",
            r"\bi'll\b",
            r"\bgoing to\b",
            r"\bplanning to\b",
            r"\bintend to\b",
            r"\bmay\b",
            r"\bmight\b",
        ],
    },
    "duration_bound": {
        "weight": 2,
        "patterns": [
            r"\buntil\b",
            r"\btill\b",
            r"\bfor a while\b",
            r"\btemporarily\b",
            r"\bshort term\b",
        ],
    },
    "conditional": {
        "weight": 1,
        "patterns": [
            r"\bif\b",
            r"\bdepending on\b",
            r"\bunless\b",
            r"\bmaybe\b",
            r"\bperhaps\b",
        ],
    },
}

QUESTION_PATTERN = r"\?$"
PERMANENT_FACT_EXCEPTIONS = [
    r"\bi was born\b",
    r"\bi am born\b",
    r"\bi have lived\b",
    r"\bi live in\b",
]

def detect_sensitive_patterns(content: str) -> bool:
    for pattern_name, regex in SENSITIVE_PATTERNS.items():
        if regex.search(content):
            logging.debug(f"Sensitive Pattern Detected: {pattern_name}")
            return True
    return False

def detect_temporal_patterns(content: str) -> bool:
    content_lower = content.lower()
    
    for exception_pattern in PERMANENT_FACT_EXCEPTIONS:
        if re.search(exception_pattern, content_lower):
            return False
            
    if re.search(QUESTION_PATTERN, content_lower):
        logging.debug("Temporal Pattern Detected: Question")
        return True

    total_weight = 0
    
    for category, rule_data in TEMPORAL_RULES.items():
        weight = rule_data["weight"]
        for pattern in rule_data["patterns"]:
            if re.search(pattern, content_lower):
                logging.debug(f"Temporal Rule Triggered: {category} (+{weight})")
                total_weight += weight
                break 
    THRESHOLD = 3
    if total_weight >= THRESHOLD:
        logging.debug(f"Memory flagged as Temporal/Transient (Score: {total_weight})")
        return True
        
    return False

def run_heuristics(content: str, metadata: Dict[str, Any]) -> bool:
    if not content or not content.strip():
        logging.debug("Heuristics: Rejected - Text is empty")
        return False

    if len(content.strip()) < 3:
        logging.debug(f"Heuristics: Rejected - Text too short ({len(content)} chars)")
        return False
        
    if len(content) > 10000:
        logging.debug(f"Heuristics: Rejected - Text too long ({len(content)} chars)")
        return False

    if metadata is None or not isinstance(metadata, dict):
        logging.debug("Heuristics: Rejected - Missing or invalid metadata")
        return False
    
    required_fields = ["source"]
    for field in required_fields:
        if field not in metadata:
            logging.debug(f"Heuristics: Rejected - Missing required field: {field}")
            return False

    try:
        content.encode('utf-8').decode('utf-8')
    except (UnicodeEncodeError,UnicodeDecodeError):
        logging.debug("Heuristics: Rejected - Invalid UTF-8 encoding")
        return False

    if detect_sensitive_patterns(content):
        logging.debug("Heuristics: Rejected - Contains sensitive patterns")
        return False

    if detect_temporal_patterns(content):
        logging.debug("Heuristics: Rejected - Contains temporal patterns")
        return False

    return True
