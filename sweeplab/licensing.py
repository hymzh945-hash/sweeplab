"""Offline Ed25519 license verification (shared scheme across the product suite)."""

from __future__ import annotations

import base64
import json
import os
from datetime import date
from pathlib import Path

PUBLIC_KEY_B64 = (
    "LS0tLS1CRUdJTiBQVUJMSUMgS0VZLS0tLS0KTUNvd0JRWURLMlZ3QXlFQW9md3lFcWZnZkJ3Z0NXMER6TmVO"
    "d25IeTNyRU5GTjAzR1FERGRFNERhWXM9Ci0tLS0tRU5EIFBVQkxJQyBLRVktLS0tLQo="
)
PRODUCT = "sweeplab"
ENV_VAR = "SWEEPLAB_LICENSE"
CANDIDATE_PATHS = [
    Path.home() / "sweeplab-license.json",
    Path.home() / ".sweeplab" / "license.json",
]


def _canonical(payload: dict) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def _load_key() -> dict | None:
    path = os.environ.get(ENV_VAR)
    candidates = ([Path(path)] if path else []) + CANDIDATE_PATHS
    for candidate in candidates:
        if candidate and candidate.exists():
            try:
                return json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return {"_broken": str(candidate)}
    return None


def verify(doc: dict | None) -> tuple[bool, str]:
    if not doc or "_broken" in doc:
        return False, "no license file"
    try:
        payload = doc["payload"]
        sig = base64.b64decode(doc["sig"])
    except (KeyError, TypeError, ValueError):
        return False, "malformed license"
    if payload.get("product") != PRODUCT:
        return False, f"license is for {payload.get('product')}"

    try:
        from cryptography.hazmat.primitives.serialization import load_pem_public_key
        key = load_pem_public_key(base64.b64decode(PUBLIC_KEY_B64))
        key.verify(sig, _canonical(payload))
    except Exception:
        return False, "signature does not verify"

    expires = payload.get("expires") or ""
    if expires:
        try:
            if date.fromisoformat(expires) < date.today():
                return False, f"license expired on {expires}"
        except ValueError:
            return False, "bad expiry date"
    return True, payload.get("email", "licensed")


def status() -> dict:
    doc = _load_key()
    ok, info = verify(doc)
    return {
        "tier": "pro" if ok else "free",
        "valid": ok,
        "info": info,
        "email": (doc or {}).get("payload", {}).get("email", "") if ok else "",
        "expires": (doc or {}).get("payload", {}).get("expires", "") if ok else "",
    }


def license_path() -> str:
    path = os.environ.get(ENV_VAR)
    if path:
        return path
    for candidate in CANDIDATE_PATHS:
        if candidate.exists():
            return str(candidate)
    return str(CANDIDATE_PATHS[0])


def is_pro() -> bool:
    return status()["valid"]
