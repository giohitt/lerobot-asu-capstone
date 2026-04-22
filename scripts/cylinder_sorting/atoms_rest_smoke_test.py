#!/usr/bin/env python3
"""
Verify Supabase REST can read ATOMS `items` for a project (expect HTTP 200).

  cd /path/to/lerobot
  # Put SUPABASE_URL + SUPABASE_ANON_KEY in scripts/cylinder_sorting/.env or export them
  python3 scripts/cylinder_sorting/atoms_rest_smoke_test.py

Optional:
  python3 scripts/cylinder_sorting/atoms_rest_smoke_test.py --limit 5 --project-id <uuid>
"""

from __future__ import annotations

import argparse
import json
import sys

from supabase_atoms_rest import (
    DEFAULT_ATOMS_PROJECT_ID,
    SupabaseRestClient,
    atoms_project_id,
    load_env_file,
)


def main() -> int:
    load_env_file()
    parser = argparse.ArgumentParser(description="Smoke test Supabase REST -> ATOMS items")
    parser.add_argument("--limit", type=int, default=3, help="max rows to fetch")
    parser.add_argument(
        "--project-id",
        default=atoms_project_id(fallback=DEFAULT_ATOMS_PROJECT_ID),
        help="ATOMS project UUID (default: ATOMS_PROJECT_ID from .env, else ACT Color Sorting Arm)",
    )
    args = parser.parse_args()

    try:
        client = SupabaseRestClient.from_env(load_dotenv=True)
    except ValueError as e:
        print(f"Config error: {e}", file=sys.stderr)
        return 1

    pid = args.project_id
    query = {
        "project_id": f"eq.{pid}",
        "select": "id,title,type,updated_at",
        "limit": str(max(1, args.limit)),
        "order": "updated_at.desc",
    }
    status, payload = client.get_json("/rest/v1/items", query=query)

    print(f"HTTP {status}")
    if status != 200:
        print(json.dumps(payload, indent=2) if isinstance(payload, (dict, list)) else payload)
        return 1

    rows = payload if isinstance(payload, list) else []
    print(f"OK — read {len(rows)} item(s) for project_id={pid}")
    for row in rows:
        print(f"  - {row.get('id')}: {row.get('type')}: {row.get('title', '')[:80]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
