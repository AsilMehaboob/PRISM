"""
Text normalisation for the safety stages.

Attackers hide trigger phrases from regex filters with invisible characters,
look-alike letters, leetspeak, spaced-out letters or base64. ``variants``
returns de-obfuscated copies of a text so every pattern is also tried against
what a language model would actually read. ``obfuscation_flags`` reports the
tricks that are suspicious in themselves.
"""

import base64
import binascii
import re
import unicodedata
from typing import List

# Invisible / direction-control characters that have no place in a memory.
_HIDDEN = {
    "​", "⁠", "﻿", "­", "᠎",
    "‪", "‫", "‬", "‭", "‮",
    "⁦", "⁧", "⁨", "⁩",
}
# Joiners are legitimate in emoji / Indic scripts: dropped for matching, not flagged.
_JOINERS = {"‌", "‍"}

_HOMOGLYPHS = str.maketrans({
    # Cyrillic
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x",
    "і": "i", "ј": "j", "ѕ": "s", "һ": "h", "ԁ": "d", "ӏ": "l", "ԛ": "q",
    "к": "k", "м": "m", "н": "h", "т": "t", "в": "b",
    # Greek
    "ο": "o", "α": "a", "ν": "v", "ι": "i", "ρ": "p", "κ": "k", "τ": "t",
    "υ": "u", "ε": "e", "β": "b", "η": "n", "χ": "x",
})

_LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s", "!": "i"})

_B64 = re.compile(r"[A-Za-z0-9+/]{16,}={0,2}")


def clean(text: str) -> str:
    """NFKC-normalise and strip invisible characters."""
    text = unicodedata.normalize("NFKC", text)
    return "".join(c for c in text if c not in _HIDDEN and c not in _JOINERS)


def _script(ch: str) -> str:
    if not ch.isalpha():
        return ""
    try:
        name = unicodedata.name(ch)
    except ValueError:
        return ""
    for s in ("LATIN", "CYRILLIC", "GREEK"):
        if name.startswith(s):
            return s
    return "OTHER"


def _fold_leet(text: str) -> str:
    def word(m: re.Match) -> str:
        w = m.group(0)
        has_letter = any(c.isalpha() for c in w)
        has_leet = any(c in "013457@$!" for c in w)
        return w.translate(_LEET) if has_letter and has_leet else w
    return re.sub(r"\S+", word, text)


_SPACED_RUN = re.compile(r"(?<!\S)(?:[A-Za-z] ){3,}[A-Za-z](?!\S)")


def _collapse_spaced(text: str) -> str:
    """'i g n o r e   p r e v i o u s' -> 'ignore previous' (runs of 4+ single letters)."""
    return _SPACED_RUN.sub(lambda m: m.group(0).replace(" ", ""), text)


def decode_base64(text: str) -> List[str]:
    """Printable payloads hidden in base64 blobs."""
    found = []
    for blob in _B64.findall(text):
        try:
            raw = base64.b64decode(blob + "=" * (-len(blob) % 4), validate=True)
            dec = raw.decode("utf-8")
        except (binascii.Error, UnicodeDecodeError, ValueError):
            continue
        if dec and sum(c.isprintable() or c.isspace() for c in dec) / len(dec) > 0.95 and any(c.isalpha() for c in dec):
            found.append(dec)
    return found


def variants(text: str) -> List[str]:
    """The text plus de-obfuscated copies. Always contains the cleaned text first."""
    base = clean(text)
    folded = base.translate(_HOMOGLYPHS)
    layers = [base, folded, _collapse_spaced(_fold_leet(folded))]
    for dec in decode_base64(base):
        d = clean(dec)
        layers += [d, _collapse_spaced(_fold_leet(d.translate(_HOMOGLYPHS)))]
    seen, out = set(), []
    for v in layers:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def obfuscation_flags(text: str) -> List[str]:
    """Tricks that are suspicious on their own (used to reject at Stage 1)."""
    flags = []
    if any(c in _HIDDEN for c in unicodedata.normalize("NFKC", text)):
        flags.append("hidden_characters")
    for word in re.findall(r"\w+", text):
        scripts = {_script(c) for c in word} - {""}
        if len(scripts & {"LATIN", "CYRILLIC", "GREEK"}) >= 2:
            flags.append("mixed_script_word")
            break
    return flags


_URL = re.compile(r"(?i)\b(?:https?://|ftp://|www\.)\S+")
_EMAIL = re.compile(r"(?i)\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
_IPV4 = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_SHELL = re.compile(r"(?i)\b(?:curl|wget|powershell|eval|exec|chmod|base64\s+-d)\b")


def suspicion_flags(text: str) -> List[str]:
    """Content that can carry an exfiltration/execution channel."""
    flags = []
    scan = " ".join(variants(text))
    if _URL.search(scan):
        flags.append("url")
    if _EMAIL.search(scan):
        flags.append("email")
    if _IPV4.search(scan):
        flags.append("ip_address")
    if _SHELL.search(scan):
        flags.append("shell_command")
    if decode_base64(clean(text)):
        flags.append("encoded_payload")
    return flags
