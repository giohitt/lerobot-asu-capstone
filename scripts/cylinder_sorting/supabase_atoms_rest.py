"""
Supabase PostgREST client for ATOMS (`items`, etc.).

Auth: publishable (anon) or service_role key — both are static until rotated in the
dashboard. There is no per-request OAuth refresh for these keys; load once from `.env`
and reuse the same headers for every HTTP call (safe for a long-running bridge).

Environment (e.g. `scripts/cylinder_sorting/.env` — already gitignored via repo `.env`):
  SUPABASE_URL              https://<ref>.supabase.co
  SUPABASE_ANON_KEY         publishable / anon key (try this first under RLS)
  Optional alias: SUPABASE_KEY (same as SUPABASE_ANON_KEY if the latter is unset)

  Optional (server-only, bypasses RLS — do not ship to untrusted hosts):
  SUPABASE_SERVICE_ROLE_KEY   if set, used instead of SUPABASE_ANON_KEY

  ATOMS_PROJECT_ID          UUID of the ATOMS project (see atoms_project_id() and DEFAULT_ATOMS_PROJECT_ID)
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

# Default when ATOMS_PROJECT_ID is unset (ACT Color Sorting Arm — override for other projects).
DEFAULT_ATOMS_PROJECT_ID = "95fa5a31-7beb-4629-80d5-bb2a8d07c3e2"


def load_env_file(path: Path | None = None) -> None:
    """Load KEY=value pairs into os.environ if not already set. Skips comments and blanks."""
    candidates = []
    if path is not None:
        candidates.append(path)
    here = Path(__file__).resolve().parent
    candidates.append(here / ".env")
    candidates.append(Path.cwd() / ".env")
    seen: set[Path] = set()
    for p in candidates:
        try:
            rp = p.resolve()
        except OSError:
            continue
        if rp in seen or not rp.is_file():
            continue
        seen.add(rp)
        for raw in rp.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip("'").strip('"')
            if key and key not in os.environ:
                os.environ[key] = val


def _resolve_api_key() -> str:
    if os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "").strip():
        return os.environ["SUPABASE_SERVICE_ROLE_KEY"].strip()
    for name in ("SUPABASE_ANON_KEY", "SUPABASE_KEY"):
        v = os.environ.get(name, "").strip()
        if v:
            return v
    raise ValueError(
        "Set SUPABASE_ANON_KEY or SUPABASE_KEY in the environment or .env "
        "(optional: SUPABASE_SERVICE_ROLE_KEY for server-side RLS bypass)."
    )


def atoms_project_id(*, fallback: str | None = None) -> str:
    """
    UUID of the ATOMS project for REST filters (`project_id=eq....`).

    Reads ``ATOMS_PROJECT_ID`` from the environment (call ``load_env_file()`` first
    in CLI tools so ``scripts/cylinder_sorting/.env`` is applied).

    If unset and ``fallback`` is provided, returns ``fallback`` (smoke tests / local dev).
    If unset and ``fallback`` is None, raises ``ValueError``.
    """
    v = os.environ.get("ATOMS_PROJECT_ID", "").strip()
    if v:
        return v
    if fallback is not None:
        return fallback.strip() if isinstance(fallback, str) else fallback
    raise ValueError(
        "ATOMS_PROJECT_ID is not set. Add it to scripts/cylinder_sorting/.env "
        "(see .env.example) or export ATOMS_PROJECT_ID=<uuid>."
    )


class SupabaseRestClient:
    """One instance per process: same API key on every request (no re-auth churn)."""

    def __init__(self, base_url: str, api_key: str) -> None:
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key

    @classmethod
    def from_env(cls, *, load_dotenv: bool = True, dotenv_path: Path | None = None) -> SupabaseRestClient:
        if load_dotenv:
            load_env_file(dotenv_path)
        url = os.environ.get("SUPABASE_URL", "").strip()
        if not url:
            raise ValueError("SUPABASE_URL is not set")
        return cls(url, _resolve_api_key())

    def _headers(self, *, accept: str = "application/json") -> dict[str, str]:
        return {
            "apikey": self._api_key,
            "Authorization": f"Bearer {self._api_key}",
            "Accept": accept,
        }

    def request(
        self,
        method: str,
        path: str,
        *,
        query: dict[str, str] | None = None,
        accept: str = "application/json",
        timeout: float = 60.0,
    ) -> tuple[int, bytes]:
        """
        `path` is relative to host, e.g. `/rest/v1/items`.
        Returns (status_code, response_body).
        """
        q = urllib.parse.urlencode(query) if query else ""
        url = f"{self.base_url}{path}"
        if q:
            url = f"{url}?{q}"
        req = urllib.request.Request(url, headers=self._headers(accept=accept), method=method.upper())
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.status, resp.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()

    def get_json(self, path: str, *, query: dict[str, str] | None = None) -> tuple[int, Any]:
        status, body = self.request("GET", path, query=query)
        if not body:
            return status, None
        try:
            return status, json.loads(body.decode("utf-8"))
        except json.JSONDecodeError:
            return status, body.decode("utf-8", errors="replace")
