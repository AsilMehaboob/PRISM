import hashlib
import hmac
import json
import logging
import os
from typing import Dict, Iterable, Mapping, Optional

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from .models import MemoryItem

PAYLOAD_VERSION = 2


def load_keypair():
    priv_hex = os.environ.get("AGENT_PRIVATE_KEY")
    pub_hex = os.environ.get("AGENT_PUBLIC_KEY")

    if not priv_hex or not pub_hex:
        raise ValueError(
            "AGENT_PRIVATE_KEY and AGENT_PUBLIC_KEY must be set in the environment"
        )

    private_key = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(priv_hex))
    public_key = Ed25519PublicKey.from_public_bytes(bytes.fromhex(pub_hex))
    logging.debug("Loaded keypair from environment variables")
    return private_key, public_key


def load_retired_public_keys() -> list:
    """Public keys of rotated-out signing keys (comma-separated hex in AGENT_RETIRED_PUBLIC_KEYS).

    Items signed with a retired key keep verifying; new items are signed with the current key.
    """
    raw = os.environ.get("AGENT_RETIRED_PUBLIC_KEYS", "")
    return [Ed25519PublicKey.from_public_bytes(bytes.fromhex(h.strip())) for h in raw.split(",") if h.strip()]


def _raw_public(public_key: Ed25519PublicKey) -> bytes:
    return public_key.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)


def key_id(public_key: Ed25519PublicKey) -> str:
    """Short, stable identifier of a public key (first 16 hex chars of its SHA-256)."""
    return hashlib.sha256(_raw_public(public_key)).hexdigest()[:16]


def build_keyring(keys: Iterable[Ed25519PublicKey]) -> Dict[str, Ed25519PublicKey]:
    return {key_id(k): k for k in keys}


def _payload(item: MemoryItem) -> bytes:
    """Canonical, versioned encoding of everything that must not change after signing.

    Binding tier, user, source and expiry (not just id/content/time) stops a valid
    signature being moved to another user, promoted to another tier, relabelled as
    a more trusted source, or given a longer lifetime.
    """
    return json.dumps(
        {
            "v": PAYLOAD_VERSION,
            "id": item.id,
            "content": item.content,
            "created_at": item.created_at.isoformat(),
            "tier": item.tier,
            "user_id": item.user_id,
            "source": item.source,
            "expires_at": item.expires_at.isoformat() if item.expires_at else None,
            "key_id": item.key_id,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode()


# ---------------------------------------------------------------- long-term (Ed25519)

def sign_item(item: MemoryItem, private_key: Ed25519PrivateKey):
    item.key_id = key_id(private_key.public_key())
    signature = private_key.sign(_payload(item))
    item.signature = signature.hex()
    logging.debug(f"Signed item: {item.id}")
    return item


def verify_item(item: MemoryItem, public_key: Ed25519PublicKey) -> bool:
    if item.signature is None:
        logging.warning(f"Item has no signature: {item.id}")
        return False
    try:
        public_key.verify(bytes.fromhex(item.signature), _payload(item))
        return True
    except Exception:
        logging.error(f"Failed to verify item: {item.id}")
        return False


def verify_item_keyring(item: MemoryItem, keyring: Mapping[str, Ed25519PublicKey]) -> bool:
    """Verify against the key the item names. Unknown key ids are rejected."""
    public_key = keyring.get(item.key_id or "")
    if public_key is None:
        logging.error(f"Item {item.id} names unknown key id {item.key_id!r}")
        return False
    return verify_item(item, public_key)


# ---------------------------------------------------------------- session (HMAC-SHA256)

def session_key(private_key: Optional[Ed25519PrivateKey] = None) -> bytes:
    """MAC key derived from the agent's private key; never stored or transmitted."""
    if private_key is None:
        priv_hex = os.environ.get("AGENT_PRIVATE_KEY")
        if not priv_hex:
            raise ValueError("AGENT_PRIVATE_KEY must be set to MAC session memory")
        raw = bytes.fromhex(priv_hex)
    else:
        raw = private_key.private_bytes(
            serialization.Encoding.Raw,
            serialization.PrivateFormat.Raw,
            serialization.NoEncryption(),
        )
    return hmac.new(raw, b"prism/session-mac/v1", hashlib.sha256).digest()


def session_mac(item: MemoryItem, key: Optional[bytes] = None) -> str:
    return hmac.new(key or session_key(), _payload(item), hashlib.sha256).hexdigest()


def sign_session_item(item: MemoryItem, key: Optional[bytes] = None) -> MemoryItem:
    item.signature = session_mac(item, key)
    logging.debug(f"MAC'd session item: {item.id}")
    return item


def verify_session_item(item: MemoryItem, key: Optional[bytes] = None) -> bool:
    if item.signature is None:
        logging.warning(f"Item has no MAC: {item.id}")
        return False
    ok = hmac.compare_digest(item.signature, session_mac(item, key))
    if not ok:
        logging.error(f"Failed to verify session item MAC: {item.id}")
    return ok
