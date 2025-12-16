"""
API key issuance + verification for hosted usage.

Design:
- Users authenticate with Firebase (AUTH_MODE=firebase or api_key_or_firebase).
- Users can generate long-lived API keys for MCP clients and scripts.
- Keys are stored as SHA-256 digests (never store plaintext).
- We keep a global index collection for fast lookup:
    api_keys/{digest} -> {tenant_id, label, created_at, revoked_at}
  and a tenant-scoped view:
    tenants/{tenant_id}/api_keys/{digest} -> {label, created_at, last_used_at, revoked_at}
"""

from __future__ import annotations

import hashlib
import secrets
import time
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

from .config import get_config

try:
    from google.cloud import firestore
    HAS_FIRESTORE = True
except Exception:
    firestore = None
    HAS_FIRESTORE = False


def _digest_key(api_key: str) -> str:
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


def _new_plaintext_key() -> str:
    # 32 bytes -> 64 hex chars
    return secrets.token_hex(32)


@dataclass(frozen=True)
class ApiKeyRecord:
    digest: str
    tenant_id: str
    label: str
    created_at: float
    last_used_at: Optional[float] = None
    revoked_at: Optional[float] = None


class ApiKeyStore:
    def __init__(self):
        if not HAS_FIRESTORE:
            raise RuntimeError("google-cloud-firestore is required for ApiKeyStore")
        cfg = get_config()
        if cfg.firestore.project_id:
            self._client = firestore.Client(project=cfg.firestore.project_id)
        else:
            self._client = firestore.Client()
        self._tenants = cfg.multi_tenant.tenants_collection

    def _global_ref(self, digest: str):
        return self._client.collection("api_keys").document(digest)

    def _tenant_ref(self, tenant_id: str, digest: str):
        return (
            self._client.collection(self._tenants)
            .document(tenant_id)
            .collection("api_keys")
            .document(digest)
        )

    def create(self, tenant_id: str, label: str) -> Dict[str, str]:
        plain = _new_plaintext_key()
        digest = _digest_key(plain)
        now = time.time()

        global_doc = self._global_ref(digest)
        if global_doc.get().exists:
            # Extremely unlikely, but retry once.
            plain = _new_plaintext_key()
            digest = _digest_key(plain)

        data = {"tenant_id": tenant_id, "label": label, "created_at": now, "revoked_at": None}
        self._global_ref(digest).set(data)
        self._tenant_ref(tenant_id, digest).set({**data, "last_used_at": None})

        return {"api_key": plain, "digest": digest}

    def list_for_tenant(self, tenant_id: str) -> List[ApiKeyRecord]:
        docs = self._client.collection(self._tenants).document(tenant_id).collection("api_keys").stream()
        out: List[ApiKeyRecord] = []
        for d in docs:
            data = d.to_dict() or {}
            out.append(
                ApiKeyRecord(
                    digest=d.id,
                    tenant_id=tenant_id,
                    label=data.get("label", ""),
                    created_at=float(data.get("created_at") or 0),
                    last_used_at=data.get("last_used_at"),
                    revoked_at=data.get("revoked_at"),
                )
            )
        return sorted(out, key=lambda r: r.created_at, reverse=True)

    def revoke(self, tenant_id: str, digest: str) -> bool:
        now = time.time()
        g = self._global_ref(digest).get()
        if not g.exists:
            return False
        data = g.to_dict() or {}
        if data.get("tenant_id") != tenant_id:
            return False
        self._global_ref(digest).update({"revoked_at": now})
        self._tenant_ref(tenant_id, digest).update({"revoked_at": now})
        return True

    def lookup_tenant(self, api_key: str) -> Optional[str]:
        digest = _digest_key(api_key)
        doc = self._global_ref(digest).get()
        if not doc.exists:
            return None
        data = doc.to_dict() or {}
        if data.get("revoked_at"):
            return None
        tenant_id = data.get("tenant_id")
        if not tenant_id:
            return None
        now = time.time()
        try:
            self._tenant_ref(tenant_id, digest).update({"last_used_at": now})
        except Exception:
            pass
        return tenant_id

