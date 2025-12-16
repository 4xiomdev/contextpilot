"""
Hosted billing helpers.

This is intentionally minimal: it reads tenant billing state from Firestore and
optionally enforces it for write-heavy endpoints (crawl/normalize/key issuance).
"""

from __future__ import annotations

import os
from typing import Optional, Dict, Any

from .config import get_config

try:
    from google.cloud import firestore
    HAS_FIRESTORE = True
except Exception:
    firestore = None
    HAS_FIRESTORE = False


def _db():
    if not HAS_FIRESTORE:
        raise RuntimeError("google-cloud-firestore not installed")
    cfg = get_config()
    if cfg.firestore.project_id:
        return firestore.Client(project=cfg.firestore.project_id)
    return firestore.Client()


def get_tenant_billing(tenant_id: str) -> Optional[Dict[str, Any]]:
    cfg = get_config()
    if not cfg.multi_tenant.enabled or not cfg.has_firestore:
        return None
    doc = _db().collection(cfg.multi_tenant.tenants_collection).document(tenant_id).get()
    return doc.to_dict() if doc.exists else None


def is_tenant_active(tenant_id: str) -> bool:
    cfg = get_config()
    if os.getenv("BILLING_REQUIRED", "").lower() != "true":
        return True
    billing = get_tenant_billing(tenant_id) or {}
    status = str(billing.get("subscription_status") or "").lower()
    return status in ("active", "trialing")

