#!/usr/bin/env python3
"""
Fetch PostgREST OpenAPI from a Supabase project (learn what rest/v1 exposes).

Usage:
  export SUPABASE_ANON_KEY='your-publishable-or-anon-key'
  python scripts/cylinder_sorting/fetch_supabase_openapi.py

Optional:
  export SUPABASE_URL='https://<project-ref>.supabase.co'
  python scripts/cylinder_sorting/fetch_supabase_openapi.py --out /tmp/openapi.json

Do not commit keys. Output file `openapi.supabase.json` is listed in the repo root `.gitignore`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from supabase_atoms_rest import SupabaseRestClient


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch Supabase PostgREST OpenAPI (rest/v1).")
    parser.add_argument(
        "--out",
        default=os.path.join(os.path.dirname(__file__), "openapi.supabase.json"),
        help="Output path for raw OpenAPI JSON",
    )
    parser.add_argument(
        "--url",
        default=None,
        help="Override SUPABASE_URL for this run only",
    )
    args = parser.parse_args()

    if args.url:
        os.environ["SUPABASE_URL"] = args.url.rstrip("/")

    try:
        client = SupabaseRestClient.from_env(load_dotenv=True)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    openapi_url = f"{client.base_url}/rest/v1/"
    print(f"GET {openapi_url}")
    print("Accept: application/openapi+json")

    status, body = client.request("GET", "/rest/v1/", accept="application/openapi+json")
    if status != 200:
        print(f"HTTP {status}", file=sys.stderr)
        err = body.decode("utf-8", errors="replace")[:2000]
        if err:
            print(err, file=sys.stderr)
        return 1

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(body)

    print(f"Wrote {len(body)} bytes -> {out_path}")

    try:
        doc = json.loads(body.decode("utf-8"))
    except json.JSONDecodeError as e:
        print(f"Warning: not valid JSON: {e}", file=sys.stderr)
        return 0

    # Short summary so you can learn without opening the whole file
    print("\n--- OpenAPI summary ---")
    print(f"openapi: {doc.get('openapi', doc.get('swagger', '?'))}")
    info = doc.get("info") or {}
    print(f"title: {info.get('title', '?')}")
    print(f"version: {info.get('version', '?')}")

    paths = doc.get("paths") or {}
    print(f"\npaths count: {len(paths)}")
    for i, p in enumerate(sorted(paths.keys())):
        if i >= 40:
            print(f"... and {len(paths) - 40} more")
            break
        print(f"  {p}")

    components = doc.get("components", {})
    schemas = components.get("schemas") or {}
    if schemas:
        print(f"\ncomponents.schemas count: {len(schemas)}")
        for i, name in enumerate(sorted(schemas.keys())):
            if i >= 25:
                print(f"... and {len(schemas) - 25} more")
                break
            print(f"  {name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
