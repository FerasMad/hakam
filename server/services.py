"""Everything the web app needs, as plain functions.

The FastAPI routes in server/app.py are thin wrappers around these, so the
backend logic is testable without a server and the frontend never depends on
model or LLM internals.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
from pathlib import Path

from src import config
from src.contract import HakamContract

ROOT = config.PROJECT_ROOT
CONTRACTS_DIR = config.ARTIFACTS / "contracts"
CLIPS_DIR = config.DATA_ROOT / "mvfouls" / "Test"
DEMO_CASES = ROOT / "server" / "demo_cases.json"

VERSION = "1.0"
MAX_FILES = 4
MAX_MB = 50
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}

_ARTICLE_FIELDS = ("id", "law", "section", "title_ar", "text_ar", "text_en")

_lock = threading.Lock()
_predictor = None
_predictor_error: str | None = None
_explanations: dict[str, dict] = {}


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

def get_predictor():
    """Load the vision model once. A missing weights folder is reported, not raised at import."""
    global _predictor, _predictor_error
    with _lock:
        if _predictor is None and _predictor_error is None:
            try:
                from src.inference.predict import load_predictor

                _predictor = load_predictor()
            except Exception as exc:  # weights missing, torch problem
                _predictor_error = f"{type(exc).__name__}: {exc}"
    if _predictor is None:
        raise RuntimeError(_predictor_error or "model not loaded")
    return _predictor


def warm_up() -> None:
    """Load the vision model and the embedding model before the first user waits for them.

    Cold, the first explanation takes ~10 s (embedding model load); warm, well under 1 s.
    """
    try:
        get_predictor()
    except RuntimeError:
        pass
    try:
        from src.contract import mock_contract
        from src.llm.retrieve import retrieve

        retrieve(mock_contract(colour=None))
    except Exception:  # retrieval falls back to tags on its own; never block startup
        pass


def health() -> dict:
    from src.llm.retrieve import EMBEDDINGS_PATH

    try:
        device = str(get_predictor().device)
    except RuntimeError:
        device = None
    return {
        "version": VERSION,
        "model_loaded": _predictor is not None,
        "model_error": _predictor_error,
        "device": device,
        "llm_key_present": bool(os.getenv("OPENAI_API_KEY")),
        "index_built": EMBEDDINGS_PATH.exists(),
        "cases_enabled": (CONTRACTS_DIR / "test_index.json").exists(),
        "test_cases": len(list_cases()),
    }


def predict_clips(files: list[tuple[str, bytes]]) -> dict:
    """``files`` is ``[(filename, content)]``: 1-4 camera views of the same foul."""
    if not files:
        raise ValueError("upload at least one clip")
    if len(files) > MAX_FILES:
        raise ValueError(f"at most {MAX_FILES} clips per incident")
    for name, content in files:
        if Path(name).suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError(f"not a video file: {name}")
        if len(content) > MAX_MB * 1024 * 1024:
            raise ValueError(f"{name} is larger than {MAX_MB} MB")

    predictor = get_predictor()
    with tempfile.TemporaryDirectory(prefix="hakam-") as tmp:
        paths = []
        for i, (name, content) in enumerate(files):
            path = Path(tmp) / f"view_{i}{Path(name).suffix.lower()}"
            path.write_bytes(content)
            paths.append(path)
        contract = predictor.predict(paths)
    return {"contract": contract.to_dict(), "seconds": predictor.last_seconds,
            "device": str(predictor.device)}


# --------------------------------------------------------------------------
# Explanation
# --------------------------------------------------------------------------

def _contract_key(payload: dict) -> str:
    fields = {k: payload.get(k) for k in ("offence", "card", "card_colour", "attributes")}
    return hashlib.sha256(json.dumps(fields, sort_keys=True).encode("utf-8")).hexdigest()


def explain_contract(payload: dict) -> dict:
    """Full Arabic ruling for a contract. Uses the LLM when a key is set, else the offline ruling.

    The LLM only ever receives this contract - no video, file names or labels.
    """
    from src.llm.faithfulness import score
    from src.llm.generate import explain, safe_v3

    contract = HakamContract.from_dict(payload)
    key = _contract_key(payload)
    if key in _explanations:
        return _explanations[key]

    engine, detail = "live", "GPT-5 nano + retrieval"
    sections = None
    try:
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY not set")
        result = explain(contract)
        text, articles, abstained, sections = (result.text_ar, result.articles,
                                               result.abstained, result.sections)
    except Exception as exc:  # no key, network down, provider error
        engine, detail = "offline", f"grounded ruling without the LLM ({type(exc).__name__})"
        abstained = contract.should_abstain()
        if abstained:
            text, articles = config.ABSTAIN_MESSAGE_AR, []
        else:
            text, sections, articles = safe_v3(contract)

    response = {
        "abstained": abstained,
        "text_ar": text,
        "sections": sections,
        "articles": [{k: a.get(k) for k in _ARTICLE_FIELDS} for a in articles],
        "faithfulness": None if abstained else score(text, contract, articles),
        "engine": engine,
        "engine_detail": detail,
    }
    if engine == "live":  # never cache a fallback: the next call may reach the LLM
        _explanations[key] = response
    return response


# --------------------------------------------------------------------------
# Test-set browser (local only - clips and labels are covered by the NDA)
# --------------------------------------------------------------------------

def _index() -> list[dict]:
    path = CONTRACTS_DIR / "test_index.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def _check_id(action_id: str) -> str:
    if not str(action_id).isdigit():
        raise ValueError("action id must be a number")
    return str(action_id)


def clip_files(action_id: str) -> list[Path]:
    folder = CLIPS_DIR / f"action_{_check_id(action_id)}"
    if not folder.exists():
        return []
    return sorted(folder.glob("clip_*.mp4"), key=lambda p: int(p.stem.split("_")[1]))


def list_cases() -> list[dict]:
    return [{"action_id": c["action_id"], "offence": c.get("offence"), "card": c.get("card"),
             "abstain": c.get("abstain"), "clips": len(clip_files(c["action_id"]))} for c in _index()]


def get_case(action_id: str, truth: bool = False) -> dict:
    action_id = _check_id(action_id)
    path = CONTRACTS_DIR / "test" / f"{action_id}.json"
    if not path.exists():
        raise KeyError(action_id)
    case = {
        "contract": json.loads(path.read_text(encoding="utf-8")),
        "clips": [f"/api/cases/{action_id}/clips/{i}" for i, _ in enumerate(clip_files(action_id))],
    }
    if truth:
        case["referee_label"] = next((c.get("truth") for c in _index() if c["action_id"] == action_id), None)
    return case


def clip_path(action_id: str, n: int) -> Path:
    files = clip_files(action_id)
    if not 0 <= n < len(files):
        raise KeyError(f"{action_id}/{n}")
    return files[n]


def demo_cases() -> list[dict]:
    return json.loads(DEMO_CASES.read_text(encoding="utf-8")) if DEMO_CASES.exists() else []
