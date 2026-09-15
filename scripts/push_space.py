"""Deploy the repository to a Hugging Face Docker Space.

    huggingface-cli login
    python scripts/push_space.py --space YOUR_USER/hakam

Creates the Space (private, Docker) if needed and uploads the code with the
Space-specific README from deploy/space_README.md. Data, weights, secrets and
local caches are never uploaded - the Space downloads weights itself at start
(scripts/fetch_assets.py) using its secrets.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

IGNORE = [
    ".git/*", ".venv/*", "venv/*", "**/__pycache__/*", "*.pyc", ".env",
    "data/*", "artifacts/*", "frame_cache/*", "features_cache/*", "weights/*.pt",
    "weights/thresholds.json", "web/node_modules/*", "web/dist/*", "notebooks/*",
    "README.md", "*.log", "logs/*", "tmp/*",
]


def main() -> int:
    from huggingface_hub import HfApi

    ap = argparse.ArgumentParser()
    ap.add_argument("--space", required=True, help="e.g. hakam-team/hakam")
    ap.add_argument("--hardware", default=None,
                    help="optional: cpu-upgrade or t4-small (can also be set in the Space settings)")
    args = ap.parse_args()
    api = HfApi()

    api.create_repo(args.space, repo_type="space", space_sdk="docker", private=True, exist_ok=True)
    api.upload_folder(folder_path=str(ROOT), repo_id=args.space, repo_type="space",
                      ignore_patterns=IGNORE, commit_message="Deploy Hakam")
    api.upload_file(path_or_fileobj=str(ROOT / "deploy" / "space_README.md"), path_in_repo="README.md",
                    repo_id=args.space, repo_type="space")
    if args.hardware:
        api.request_space_hardware(args.space, args.hardware)
    print(f"pushed -> https://huggingface.co/spaces/{args.space}")
    print("next: set the Space secrets, watch the build logs, then run scripts/smoke_test.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
