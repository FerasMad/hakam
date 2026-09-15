"""Download what the server needs but git does not hold, from private Hugging Face repos.

    python scripts/fetch_assets.py            # uses the environment variables below

| Variable | Downloads |
|---|---|
| `HAKAM_WEIGHTS_REPO` (model repo) | `weights/final.pt`, `weights/thresholds.json` |
| `HAKAM_CASES_REPO` (dataset repo, optional) | `artifacts/contracts/…` and `data/mvfouls/Test/…` (test-set browser) |
| `HF_TOKEN` | a read token for those private repos |

Anything already on disk is skipped, so it is safe to run on every start.
The repos are filled once with scripts/upload_assets.py.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config

ROOT = config.PROJECT_ROOT
WEIGHT_FILES = ("final.pt", "thresholds.json")
CASE_PATTERNS = ["artifacts/contracts/*", "artifacts/contracts/**", "data/mvfouls/Test/**"]


def fetch(weights_repo: str | None = None, cases_repo: str | None = None,
          token: str | None = None, root: Path = ROOT) -> list[str]:
    """Return a line per action taken, for the startup log."""
    from huggingface_hub import hf_hub_download, snapshot_download

    weights_repo = weights_repo or os.getenv("HAKAM_WEIGHTS_REPO")
    cases_repo = cases_repo or os.getenv("HAKAM_CASES_REPO")
    token = token or os.getenv("HF_TOKEN")
    done = []

    if weights_repo:
        target = root / "weights"
        target.mkdir(parents=True, exist_ok=True)
        for name in WEIGHT_FILES:
            if (target / name).exists():
                done.append(f"weights/{name}: present")
                continue
            hf_hub_download(repo_id=weights_repo, filename=name, repo_type="model",
                            token=token, local_dir=target)
            done.append(f"weights/{name}: downloaded from {weights_repo}")

    if cases_repo:
        if (root / "artifacts" / "contracts" / "test_index.json").exists():
            done.append("test-set cases: present")
        else:
            snapshot_download(repo_id=cases_repo, repo_type="dataset", token=token,
                              local_dir=root, allow_patterns=CASE_PATTERNS)
            done.append(f"test-set cases: downloaded from {cases_repo}")
    return done


def main() -> int:
    if not (os.getenv("HAKAM_WEIGHTS_REPO") or os.getenv("HAKAM_CASES_REPO")):
        print("nothing to fetch: set HAKAM_WEIGHTS_REPO (and optionally HAKAM_CASES_REPO)")
        return 0
    for line in fetch():
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
