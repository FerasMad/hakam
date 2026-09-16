"""Smoke-test a running Hakam deployment without exposing secrets."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def run(
    base: str,
    clip: str | None,
    token: str | None = None,
    frontend_url: str | None = None,
) -> list[tuple[str, bool, str]]:
    import httpx

    headers = {"Authorization": f"Bearer {token}"} if token else {}
    results: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        results.append((name, bool(ok), detail))

    try:
        with httpx.Client(base_url=base.rstrip("/"), timeout=600, headers=headers) as client:
            response = client.get("/api/health")
            check("health responds", response.status_code == 200, response.text[:200])
            check("request tracing enabled", bool(response.headers.get("x-request-id")))

            response = client.get("/api/ready")
            ready = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
            check("model ready", response.status_code == 200 and ready.get("modelLoaded") is True, str(ready))
            check("expected model heads", set(ready.get("tasks", [])) == {"offence", "card", "action_class", "body_part"})
            check("LLM key configured", ready.get("llmConfigured") is True, "offline fallback remains available")

            if clip:
                path = Path(clip)
                started = time.perf_counter()
                with path.open("rb") as stream:
                    response = client.post(
                        "/api/analyze",
                        files={"video": (path.name, stream, "video/mp4")},
                    )
                payload = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
                check("video analysis responds", response.status_code == 200, response.text[:300])
                check("analysis status valid", payload.get("status") in {"completed", "abstained"}, f"{time.perf_counter() - started:.1f}s")
                if payload.get("status") == "completed":
                    explanation = payload.get("explanation", {})
                    check("grounded ruling returned", bool(explanation.get("sections")))
                    check("no unsupported claims", explanation.get("faithfulness", {}).get("unsupported") == [])

        if frontend_url:
            response = httpx.get(frontend_url.rstrip("/") + "/", timeout=30)
            check("frontend served", response.status_code == 200 and "Hakam" in response.text)
    except Exception as exc:
        check("server reachable", False, f"{type(exc).__name__}: {exc}")
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("--clip", default=None)
    parser.add_argument("--token", default=None, help="read token for a private Space")
    parser.add_argument("--frontend-url", default="http://127.0.0.1:3000")
    args = parser.parse_args()

    results = run(args.url, args.clip, args.token, args.frontend_url)
    for name, ok, detail in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))
    failed = [name for name, ok, _ in results if not ok and name != "LLM key configured"]
    print("\nall required checks passed" if not failed else f"\n{len(failed)} required check(s) failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
