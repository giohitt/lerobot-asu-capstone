"""
Read-only Supabase PostgREST access to ATOMS `items` (no POST/PATCH/DELETE).

Used from sort_controller during DETECTING when ATOMS_REST_POLL is enabled.
On first use, loads ``scripts/cylinder_sorting/.env`` (via ``load_env_file``) so
``ATOMS_REST_POLL`` and Supabase keys apply without exporting them in the shell.
Credentials load once via ``SupabaseRestClient.from_env``.
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable

_client = None
_dotenv_loaded = False


def _bootstrap_env_from_dotenv() -> None:
    """Load scripts/cylinder_sorting/.env (and repo .env) once so ATOMS_REST_POLL etc. exist."""
    global _dotenv_loaded
    if _dotenv_loaded:
        return
    from supabase_atoms_rest import load_env_file

    load_env_file()
    _dotenv_loaded = True


def atoms_rest_poll_enabled() -> bool:
    _bootstrap_env_from_dotenv()
    v = os.environ.get("ATOMS_REST_POLL", "").strip().lower()
    return v in ("1", "true", "yes", "on")


def _client_singleton():
    global _client
    if _client is None:
        from supabase_atoms_rest import SupabaseRestClient

        _client = SupabaseRestClient.from_env(load_dotenv=True)
    return _client


def fetch_items_by_title_prefix(
    title_prefix: str | None = None,
    *,
    select: str = "id,title,type,data",
    limit: int = 10,
) -> tuple[int, Any]:
    """
    GET /rest/v1/items — read only.
    Returns (http_status, parsed JSON list or error body).
    """
    _bootstrap_env_from_dotenv()
    prefix = (title_prefix or os.environ.get("ATOMS_READ_TITLE_PREFIX", "IF-008")).strip()
    from supabase_atoms_rest import DEFAULT_ATOMS_PROJECT_ID, atoms_project_id

    client = _client_singleton()
    project_id = atoms_project_id(fallback=DEFAULT_ATOMS_PROJECT_ID)
    # PostgREST ilike: *wildcards* in filter value
    query: dict[str, str] = {
        "project_id": f"eq.{project_id}",
        "title": f"ilike.*{prefix}*",
        "select": select,
        "limit": str(max(1, limit)),
    }
    return client.get_json("/rest/v1/items", query=query)


def _truncate(s: str, max_chars: int) -> str:
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 20] + "…[truncated]"


def log_atoms_rest_detect_cycle(
    log_state: Callable[[str, str], None],
    *,
    state: str,
    vision_color: str | None,
) -> None:
    """
    One DETECTING pass: GET matching requirement(s), log HTTP status, full JSON snapshot,
    and a single clean vision line.

    vision_color: HSV result for this frame ("green" / "blue" / "yellow") or None.
    """
    if not atoms_rest_poll_enabled():
        return
    try:
        status, payload = fetch_items_by_title_prefix()
    except Exception as e:
        log_state(state, f"ATOMS_REST error: {e}")
        return

    rows = payload if isinstance(payload, list) else []
    log_state(state, f"ATOMS_REST http={status} rows={len(rows)}")

    max_json = int(os.environ.get("ATOMS_REST_LOG_MAX_CHARS", "12000"))
    if status != 200:
        detail = json.dumps(payload, default=str, separators=(",", ":")) if payload is not None else ""
        log_state(state, f"ATOMS_REST body={_truncate(detail, max_json)}")
    else:
        blob = json.dumps(rows, default=str, separators=(",", ":"))
        log_state(state, f"ATOMS_REST body={_truncate(blob, max_json)}")

    vc = "none" if not vision_color else vision_color
    log_state(state, f"vision_color={vc}")
