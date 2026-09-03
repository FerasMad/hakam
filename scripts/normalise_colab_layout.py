"""Put the downloaded splits where the loaders expect them.

Two traps, both observed on 3 Sep 2026 during the first real Colab run.

**The downloader nests.** Given ``LocalDirectory=".../data/mvfouls"`` it writes
to ``.../data/mvfouls/mvfouls/train.zip`` - one level deeper than asked. A
non-recursive glob finds nothing, extracts nothing, and says nothing, which
leaves the verification step to fail later with a confusing missing-annotations
error rather than at the point of the actual problem.

**The zips extract flat.** Every split contains ``action_0``, ``action_1``, ...
so two splits extracted into one directory silently overwrite each other and the
counts come out wrong rather than absent. Each split therefore gets its own
directory.

This lives in the repo rather than in a notebook cell so it can be fixed with a
push and re-run with one line, instead of being hand-edited inside Colab.

    python scripts/normalise_colab_layout.py
"""

from __future__ import annotations

import shutil
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config

SPLITS = {"Train": "train", "Valid": "valid", "Test": "test"}


def show_layout(root: Path, limit: int = 40) -> None:
    print("--- what the downloader actually produced ---")
    entries = sorted(root.rglob("*"))
    if not entries:
        print("  (nothing)")
    for p in entries[:limit]:
        detail = f"  ({p.stat().st_size / 1e6:.1f} MB)" if p.is_file() else "/"
        print(f"  {p.relative_to(root)}{detail}")
    if len(entries) > limit:
        print(f"  ... and {len(entries) - limit} more")
    print()


def find_archive(root: Path, lower: str) -> Path | None:
    """Recursive, exact stem first so 'test.zip' is not shadowed by a partial match."""
    return (
        next((z for z in root.rglob("*.zip") if z.stem.lower() == lower), None)
        or next((z for z in root.rglob("*.zip") if lower in z.stem.lower()), None)
    )


def place_split(root: Path, proper: str, lower: str) -> bool:
    target = root / proper
    if (target / "annotations.json").exists():
        print(f"{proper}: already in place")
        return True

    archive = find_archive(root, lower)
    if archive is not None:
        target.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as zf:
            try:
                zf.extractall(target)
            except RuntimeError as exc:
                print(f"{proper}: {archive.name} appears encrypted ({exc})")
                return False
        print(f"{proper}: extracted {archive.name}")

        # Some archives carry their own top-level folder, which would leave
        # annotations.json one level below where the loaders look.
        if not (target / "annotations.json").exists():
            nested = next((a.parent for a in target.rglob("annotations.json")), None)
            if nested is not None and nested != target:
                for item in list(nested.iterdir()):
                    shutil.move(str(item), str(target / item.name))
                print(f"{proper}: flattened {nested.name}/")
        return (target / "annotations.json").exists()

    # Or the downloader already extracted it somewhere else in the tree.
    found = next(
        (a.parent for a in root.rglob("annotations.json")
         if proper.lower() in str(a.parent).lower() and a.parent != target),
        None,
    )
    if found is not None:
        shutil.move(str(found), str(target))
        print(f"{proper}: moved {found} -> {target}")
        return True

    print(f"{proper}: NOT FOUND - check the layout above")
    return False


def main() -> int:
    root = config.DATA_ROOT / "mvfouls"
    if not root.exists():
        print(f"{root} does not exist - has the download run?")
        return 1

    show_layout(root)

    ok = True
    for proper, lower in SPLITS.items():
        ok &= place_split(root, proper, lower)

    print()
    for proper in SPLITS:
        d = root / proper
        print(f"{proper:6} annotations.json={(d / 'annotations.json').exists()}  "
              f"actions={len(list(d.glob('action_*')))}")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
