"""Push the model weights (and optionally the test-set browser data) to private Hugging Face repos.

Run once from a laptop that has the files:

    huggingface-cli login                      # a token with write access
    python scripts/upload_assets.py --weights-repo YOUR_USER/hakam-weights
    python scripts/upload_assets.py --weights-repo YOUR_USER/hakam-weights --cases-repo YOUR_USER/hakam-cases

Both repos are created private. The server downloads them with scripts/fetch_assets.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config

ROOT = config.PROJECT_ROOT


def main() -> int:
    from huggingface_hub import HfApi

    ap = argparse.ArgumentParser()
    ap.add_argument("--weights-repo", required=True, help="e.g. hakam-team/hakam-weights")
    ap.add_argument("--cases-repo", default=None, help="optional dataset repo for the test-set browser")
    args = ap.parse_args()
    api = HfApi()

    weights = ROOT / "weights"
    missing = [n for n in ("final.pt", "thresholds.json") if not (weights / n).exists()]
    if missing:
        sys.exit(f"missing in weights/: {missing} (see weights/README.md)")
    api.create_repo(args.weights_repo, repo_type="model", private=True, exist_ok=True)
    for name in ("final.pt", "thresholds.json"):
        api.upload_file(path_or_fileobj=str(weights / name), path_in_repo=name,
                        repo_id=args.weights_repo, repo_type="model")
        print(f"uploaded weights/{name} -> {args.weights_repo}")

    if args.cases_repo:
        contracts = ROOT / "artifacts" / "contracts"
        clips = ROOT / "data" / "mvfouls" / "Test"
        if not (contracts / "test_index.json").exists():
            sys.exit("artifacts/contracts/test_index.json missing - download contracts_v3 first")
        api.create_repo(args.cases_repo, repo_type="dataset", private=True, exist_ok=True)
        api.upload_folder(folder_path=str(contracts), path_in_repo="artifacts/contracts",
                          repo_id=args.cases_repo, repo_type="dataset")
        print(f"uploaded artifacts/contracts -> {args.cases_repo}")
        if clips.exists():
            api.upload_folder(folder_path=str(clips), path_in_repo="data/mvfouls/Test",
                              repo_id=args.cases_repo, repo_type="dataset",
                              allow_patterns=["action_*/clip_*.mp4"])
            print(f"uploaded data/mvfouls/Test clips -> {args.cases_repo}")
        else:
            print("no data/mvfouls/Test folder: the browser will show contracts without clips")
    return 0


if __name__ == "__main__":
    sys.exit(main())
