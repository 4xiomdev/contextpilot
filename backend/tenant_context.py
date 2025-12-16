"""
Tenant context helpers.

In hosted mode, ContextPilot is multi-tenant: each authenticated user maps to a tenant.
For local/self-host usage, we default to a single tenant ("default").
"""

from __future__ import annotations

from contextvars import ContextVar, Token

_tenant_id: ContextVar[str] = ContextVar("contextpilot_tenant_id", default="default")


def get_tenant_id() -> str:
    return _tenant_id.get()


def set_tenant_id(tenant_id: str) -> Token:
    return _tenant_id.set(tenant_id or "default")


def reset_tenant_id(token: Token) -> None:
    _tenant_id.reset(token)

