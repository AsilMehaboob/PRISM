import os
import zlib
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from .models import MemoryItem
import logging


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


def _payload(item: MemoryItem) -> bytes:
    logging.debug(f"Creating payload for item: {item}")
    return (
        f"{item.id}|{item.content}|{item.created_at.isoformat()}".encode()
    )


def sign_item(item: MemoryItem, private_key: Ed25519PrivateKey):
    signature = private_key.sign(_payload(item))
    item.signature = signature.hex()
    logging.debug(f"Signed item: {item}")
    return item


def verify_item(item: MemoryItem, public_key: Ed25519PublicKey) -> bool:
    if item.signature is None:
        logging.warning(f"Item has no signature: {item}")
        return False
    try:
        public_key.verify(bytes.fromhex(item.signature), _payload(item))
        logging.debug(f"Verified item: {item}")
        return True
    except Exception:
        logging.error(f"Failed to verify item: {item}")
        return False


def generate_checksum(item: MemoryItem) -> str:
    logging.debug(f"Generating checksum for item: {item}")
    return f"{zlib.adler32(_payload(item)):x}"


def sign_session_item(item: MemoryItem) -> MemoryItem:
    item.signature = generate_checksum(item)
    logging.debug(f"Checksummed session item: {item}")
    return item


def verify_session_item(item: MemoryItem) -> bool:
    if item.signature is None:
        logging.warning(f"Item has no checksum: {item}")
        return False

    is_valid = item.signature == generate_checksum(item)
    if is_valid:
        logging.debug(f"Verified session item checksum: {item}")
    else:
        logging.error(f"Failed to verify session item checksum: {item}")
    return is_valid
