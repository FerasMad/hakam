"""Check a running Hakam server end to end.

    python scripts/smoke_test.py http://localhost:8000
    python scripts/smoke_test.py https://YOUR_USER-hakam.hf.space --clip clip_0.mp4 --clip clip_1.mp4
    python scripts/smoke_test.py URL --token hf_xxx     # private Space: a read token

Prints one line per check and exits non-zero if any required check fails.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

CONTRACT = {
    "action_id": "smoke-test",
    "offence": {"label": "offence", "confidence": 0.85},
    "card": {"label": "card", "confidence": 0.7},
    "card_colour": None,
    "attributes": {"action_class": {"label": "tackle", "confidence": 0.8},
                   "body_part": {"label": "under_body", "confidence": 0.8}},
    "model_version": "smoke",
    "num_views": 2,
}
SECTIONS = {"decision", "restart", "disciplinary", "law", "why", "confidence"}
OPTIONAL = ("LLM key present",)


def run(base: str, clips: list[str], token: str | None = None, client=None) -> list[tuple[str, bool, str]]:
    import httpx

    headers = {"Authorization": f"Bearer {token}"} if token else {}
    http = client or httpx.Client(base_url=base.rstrip("/"), timeout=300, headers=headers)
    results = []

    def check(name, ok, detail=""):
        results.append((name, bool(ok), detail))

    try:
        r = http.get("/api/health")
        health = r.json() if r.status_code == 200 else {}
        check("health responds", r.status_code == 200, str(health))
        check("model loaded", health.get("model_loaded"),
              str(health.get("model_error") or health.get("device")))
        check("retrieval index built", health.get("index_built"))
        check("LLM key present (live explanations)", health.get("llm_key_present"),
              "offline rulings still work without it")

        start = time.time()
        r = http.post("/api/explain", json=CONTRACT)
        body = r.json() if r.status_code == 200 else {}
        check("explain returns the six-part ruling", SECTIONS <= set(body.get("sections") or {}),
              f"engine={body.get('engine')} in {time.time() - start:.1f}s")
        check("no unsupported claims", (body.get("faithfulness") or {}).get("unsupported") == [])

        r = http.get("/api/cases")
        count = len(r.json()) if r.status_code == 200 else 0
        check("cases endpoint", r.status_code == 200, f"{count} cases")

        if clips:
            files = [("files", (Path(p).name, Path(p).read_bytes(), "video/mp4")) for p in clips]
            r = http.post("/api/predict", files=files)
            body = r.json() if r.status_code == 200 else {}
            check("predict from uploaded clips", "contract" in body,
                  f"{body.get('seconds')}s on {body.get('device')}" if body else r.text[:200])
    except Exception as exc:  # connection refused, DNS, timeout
        check("server reachable", False, f"{type(exc).__name__}: {exc}")
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--clip", action="append", default=[], help="video file for the predict check (repeatable)")
    ap.add_argument("--token", default=None, help="Hugging Face read token for a private Space")
    args = ap.parse_args()

    results = run(args.url, args.clip, args.token)
    for name, ok, detail in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))
    failed = [n for n, ok, _ in results if not ok and not n.startswith(OPTIONAL)]
    print("\nall required checks passed" if not failed else f"\n{len(failed)} required check(s) failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
