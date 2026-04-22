"""
Read-only Supabase PostgREST access to ATOMS `items` (no POST/PATCH/DELETE).

Used from sort_controller during DETECTING when ATOMS_REST_POLL is enabled.
On first use, loads ``scripts/cylinder_sorting/.env`` (via ``load_env_file``) so
``ATOMS_REST_POLL`` and Supabase keys apply without exporting them in the shell.
Credentials load once via ``SupabaseRestClient.from_env``.

Between successful or failed GETs, polls are throttled by ``ATOMS_REST_POLL_INTERVAL_SEC``
(default 30s) so we do not hammer PostgREST on every DETECTING pass (~0.3s).

After a successful GET we also run a tiny ATOMS → ``sort_config.json`` bridge: the
requirement body is parsed for desired ``enabled_colors``; model paths come from
``ATOMS_BRIDGE_MODEL_{GREEN,BLUE,YELLOW}`` in .env (never from ATOMS, so robot paths
stay local). ``sort_config.json`` is rewritten atomically only when the result
changes, so ``sort_controller``'s existing mtime-based hot-reload picks it up on
the next detect pass.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Callable

SUPPORTED_COLORS: tuple[str, ...] = ("green", "blue", "yellow")
_CONFIG_FILE = Path(__file__).resolve().parent / "sort_config.json"

_client = None
_dotenv_loaded = False
_last_poll_monotonic: float | None = None


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


def _poll_interval_sec() -> float:
    raw = os.environ.get("ATOMS_REST_POLL_INTERVAL_SEC", "30").strip()
    try:
        v = float(raw)
    except ValueError:
        return 30.0
    return max(1.0, v)  # floor 1s so a typo "0" does not melt the API


def log_atoms_rest_detect_cycle(
    log_state: Callable[[str, str], None],
    *,
    state: str,
    vision_color: str | None,
) -> None:
    """
    Throttled poll of ATOMS and sync of sort_config.json. Quiet by design: emits
    at most one ATOMS: line per poll, and only when something is worth saying.
    """
    if not atoms_rest_poll_enabled():
        return

    global _last_poll_monotonic
    now = time.monotonic()
    interval = _poll_interval_sec()
    if _last_poll_monotonic is not None and (now - _last_poll_monotonic) < interval:
        return

    try:
        status, payload = fetch_items_by_title_prefix()
    except Exception as e:
        _last_poll_monotonic = now
        log_state(state, f"ATOMS: error ({e})")
        return

    _last_poll_monotonic = now

    if status != 200:
        log_state(state, f"ATOMS: error (http {status})")
        return

    rows = payload if isinstance(payload, list) else []
    _sync_sort_config_from_rows(rows, log_state, state)


# ─────────────────────────────────────────────────────────────────────────────
# ATOMS → sort_config.json bridge
# ─────────────────────────────────────────────────────────────────────────────

_ENABLED_COLORS_JSON_RE = re.compile(
    r'"enabled_colors"\s*:\s*(\[[^\]]*\])', re.IGNORECASE
)
_ONLY_COLOR_RE = re.compile(
    r"\bonly\s+(green|blue|yellow)\b", re.IGNORECASE
)


def _parse_enabled_colors_from_body(body: str) -> list[str] | None:
    """
    Extract desired enabled_colors from a requirement body string.
    Order of precedence:
      1. An explicit ``"enabled_colors": [...]`` JSON fragment.
      2. A phrase like ``only green`` / ``only blue``.
      3. Colors mentioned anywhere in the body (intersected with SUPPORTED_COLORS).
    Returns a de-duplicated list in SUPPORTED_COLORS order, or None if nothing found.
    """
    if not body:
        return None

    m = _ENABLED_COLORS_JSON_RE.search(body)
    if m:
        try:
            arr = json.loads(m.group(1))
            if isinstance(arr, list):
                colors = [c.lower() for c in arr if isinstance(c, str)]
                ordered = [c for c in SUPPORTED_COLORS if c in colors]
                if ordered:
                    return ordered
        except Exception:
            pass

    only_matches = {m.group(1).lower() for m in _ONLY_COLOR_RE.finditer(body)}
    if only_matches:
        return [c for c in SUPPORTED_COLORS if c in only_matches]

    lower = body.lower()
    mentioned = [c for c in SUPPORTED_COLORS if re.search(rf"\b{c}\b", lower)]
    return mentioned or None


def _env_model_map() -> dict[str, str]:
    """Read per-color model paths from env; missing/blank values are omitted."""
    out: dict[str, str] = {}
    for color in SUPPORTED_COLORS:
        raw = os.environ.get(f"ATOMS_BRIDGE_MODEL_{color.upper()}", "").strip()
        if raw:
            out[color] = raw
    return out


def _infer_enabled_colors(rows: list[Any]) -> list[str] | None:
    """Search each row's data.body for desired enabled_colors; first hit wins."""
    for row in rows:
        if not isinstance(row, dict):
            continue
        data = row.get("data") or {}
        body = data.get("body") if isinstance(data, dict) else None
        if not isinstance(body, str):
            continue
        colors = _parse_enabled_colors_from_body(body)
        if colors:
            return colors
    return None


def _read_current_sort_config() -> dict | None:
    if not _CONFIG_FILE.exists():
        return None
    try:
        return json.loads(_CONFIG_FILE.read_text())
    except Exception:
        return None


def _write_sort_config_atomic(cfg: dict) -> None:
    tmp = _CONFIG_FILE.with_suffix(_CONFIG_FILE.suffix + ".tmp")
    tmp.write_text(json.dumps(cfg, indent=2) + "\n")
    os.replace(tmp, _CONFIG_FILE)


def _sync_sort_config_from_rows(
    rows: list[Any],
    log_state: Callable[[str, str], None],
    state: str,
) -> None:
    """
    Parse requirement body → enabled_colors, join with env model paths, and
    rewrite sort_config.json atomically when the rendered config changes.
    Safety: never wipe a working config on a bad read.
    """
    desired = _infer_enabled_colors(rows)
    if not desired:
        log_state(state, "ATOMS: could not parse requirement — sort_config.json untouched")
        return

    env_models = _env_model_map()
    models = {c: env_models[c] for c in desired if c in env_models}
    if not models:
        log_state(
            state,
            f"ATOMS: wants {_fmt_colors(desired)} but no ATOMS_BRIDGE_MODEL_* set",
        )
        return

    enabled = list(models.keys())
    new_cfg = {"enabled_colors": enabled, "models": models}

    current = _read_current_sort_config()
    cur_enabled: list[str] = []
    if current is not None:
        cur_enabled = list(current.get("enabled_colors") or [])
        cur_models = current.get("models") or {}
        if cur_enabled == enabled and cur_models == models:
            # Unchanged → quiet
            return

    try:
        _write_sort_config_atomic(new_cfg)
    except Exception as e:
        log_state(state, f"ATOMS: error (write failed: {e})")
        return

    if cur_enabled:
        log_state(
            state,
            f"ATOMS: requirement changed {_fmt_colors(cur_enabled)} → {_fmt_colors(enabled)}",
        )
    else:
        log_state(state, f"ATOMS: now sorting {_fmt_colors(enabled)}")


def _fmt_colors(colors: list[str]) -> str:
    return "[" + ", ".join(colors) + "]"
