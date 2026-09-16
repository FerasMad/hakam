"""Copy the dataset and labels from the mounted Drive package into the repo.

Run in Colab after drive.mount("/content/drive"):

    python scripts/colab_setup.py
"""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

PKG = Path("/content/drive/MyDrive/hakam_colab")
ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    for split in ["Train", "Valid", "Test"]:
        if (ROOT / "data" / "mvfouls" / split).exists():
            print(split, "already there")
            continue
        local = Path("/content") / f"{split}.zip"
        shutil.copy(PKG / "data" / f"{split}.zip", local)
        zipfile.ZipFile(local).extractall(ROOT / "data" / "mvfouls")
        local.unlink()
        print(split, "ready")

    labels = ROOT / "artifacts" / "preprocessing" / "private"
    labels.mkdir(parents=True, exist_ok=True)
    for f in (PKG / "manifests").glob("*.csv"):
        shutil.copy(f, labels)
    print("labels ready")


if __name__ == "__main__":
    main()
