"""Retrieve the IFAB rule chunks that best match a :class:`HakamContract`.

The lexical side is deliberately contract-first: only labels emitted by the
vision contract are allowed into the query. The optional semantic side uses a
multilingual E5 model and caches the 40 corpus embeddings on disk.
"""

from __future__ import annotations

import json
import os
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np

from src.config import ACTION_FAMILIES, LAWS_DIR, RETRIEVAL_TOP_K
from src.contract import HakamContract
from src.llm.lexicon import arabic_forms


CORPUS_PATH = LAWS_DIR / "corpus.json"
EMBEDDINGS_PATH = LAWS_DIR / "embeddings.npy"
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"


def _key(value: str) -> str:
    return str(value).strip().lower().replace("_", " ")


@lru_cache(maxsize=1)
def _load_corpus() -> tuple[dict, ...]:
    if not CORPUS_PATH.exists():
        raise FileNotFoundError(
            f"Laws corpus not found at {CORPUS_PATH}. See laws/README.md."
        )
    with CORPUS_PATH.open(encoding="utf-8") as stream:
        corpus = json.load(stream)
    if not isinstance(corpus, list) or not corpus:
        raise ValueError("laws/corpus.json must contain a non-empty JSON list")
    return tuple(corpus)


def _contract_values(contract: HakamContract) -> set[str]:
    return {_key(value) for value in contract.citable_values()}


def _query(contract: HakamContract) -> str:
    labels = sorted(contract.citable_values())
    arabic = sorted({form for label in labels for form in arabic_forms(label)})
    return "query: " + " ".join(labels + arabic)


def _tag_scores(contract: HakamContract, corpus: tuple[dict, ...]) -> np.ndarray:
    values = _contract_values(contract)
    generic = {"offence", "no offence", "card", "no card"}
    action_labels = {
        "tackling",
        "standing tackling",
        "high leg",
        "holding",
        "pushing",
        "elbowing",
        "challenge",
        "dive",
    }
    action = contract.attributes.get("action_class")
    contact = contract.attributes.get("contact")
    # The model emits a family ("hands"); corpus tags name its members ("holding").
    action_tags: set[str] = set()
    if action is not None:
        action_tags = {_key(action.label)} | {_key(m) for m in ACTION_FAMILIES.get(action.label, [])}
    scores = []
    for chunk in corpus:
        tags = {_key(tag) for tag in chunk.get("tags", [])}
        overlap = values & tags
        # Specific physical attributes carry more retrieval signal than the
        # generic top-level decision labels.
        score = sum(1.0 if tag in generic else 2.0 for tag in overlap)
        if "no offence" in values and "no offence" in tags:
            score += 4.0
        if (
            "no offence" in values
            and chunk.get("id") == "law12-no-offence-fair-challenge"
        ):
            score += 6.0
        if "no offence" not in values and "offence" in tags:
            score += 1.0
        if action is not None and tags & action_tags:
            score += 3.0
        if (
            action is not None
            and _key(action.label) != "dont know"
            and tags & action_labels
            and not tags & action_tags
        ):
            score -= 4.0
        if action is not None and _key(action.label) == "dont know":
            if "dont know" in tags:
                score += 5.0
            # An unknown action must not be turned into a specific visual act
            # just because another attribute happens to match that rule.
            if tags & {
                "tackling",
                "standing tackling",
                "high leg",
                "holding",
                "pushing",
                "elbowing",
                "challenge",
                "dive",
            } and "dont know" not in tags:
                score -= 5.0
        if contact is not None and _key(contact.label) in tags:
            score += 3.0

        # The contract has no field for a promising attack, goal opportunity,
        # penalty-area location or goal direction. Shared labels such as
        # pushing/no-attempt must therefore never make a DOGSO article look
        # incident-specific.
        if chunk.get("id") in {
            "law12-3-stopping-promising-attack",
            "law12-3-dogso-attempt-play-ball",
            "law12-3-dogso-no-attempt",
            "law12-4-thrown-object-reckless",
            "law12-4-thrown-object-excessive",
        }:
            score -= 12.0

        # A missing colour is an explicit contract state. Prefer the chunk
        # explaining conditional sanctions and avoid making a severe-offence
        # article look like evidence that severe force actually occurred.
        if (
            contract.card_colour is None
            and contract.card is not None
            and _key(contract.card.label) == "card"
        ):
            if chunk.get("id") == "law12-card-colour-conditional":
                score += 6.0
            if chunk.get("id") in {
                "law12-1-excessive-force-definition",
                "law12-3-serious-foul-play",
                "law12-3-serious-foul-play-legs",
                "law12-3-violent-conduct",
                "law12-3-face-head-contact",
                "glossary-excessive-force",
            }:
                score -= 8.0

        if contract.card is not None and _key(contract.card.label) == "no card":
            if "no card" in tags:
                score += 4.0
        scores.append(score)
    return np.asarray(scores, dtype=np.float32)


def _passages(corpus: tuple[dict, ...]) -> list[str]:
    return [
        "passage: "
        + " | ".join(
            [
                str(chunk.get("law", "")),
                str(chunk.get("section", "")),
                str(chunk.get("title_ar", "")),
                str(chunk.get("text_ar", "")),
                str(chunk.get("text_en", "")),
                " ".join(map(str, chunk.get("tags", []))),
            ]
        )
        for chunk in corpus
    ]


@lru_cache(maxsize=1)
def _embedding_model():
    from sentence_transformers import SentenceTransformer

    model_name = os.getenv("HAKAM_EMBEDDING_MODEL", EMBEDDING_MODEL)
    return SentenceTransformer(model_name)


def _semantic_scores(query: str, corpus: tuple[dict, ...]) -> np.ndarray:
    model = _embedding_model()
    embeddings: np.ndarray
    if EMBEDDINGS_PATH.exists():
        embeddings = np.load(EMBEDDINGS_PATH)
        if embeddings.ndim != 2 or embeddings.shape[0] != len(corpus):
            embeddings = np.empty((0, 0), dtype=np.float32)
    else:
        embeddings = np.empty((0, 0), dtype=np.float32)

    if embeddings.shape[0] != len(corpus):
        embeddings = np.asarray(
            model.encode(
                _passages(corpus),
                normalize_embeddings=True,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )
        EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.save(EMBEDDINGS_PATH, embeddings)

    query_embedding = np.asarray(
        model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0],
        dtype=np.float32,
    )
    return embeddings @ query_embedding


def _normalise_scores(values: np.ndarray) -> np.ndarray:
    low, high = float(values.min()), float(values.max())
    if high - low < 1e-12:
        return np.zeros_like(values)
    return (values - low) / (high - low)


def retrieve(contract: HakamContract, k: int = RETRIEVAL_TOP_K) -> list[dict]:
    """Return the top ``k`` law chunks for an incident.

    Hybrid retrieval is the default. Set ``HAKAM_RETRIEVAL_MODE=tags`` for a
    deterministic, dependency-light tag-only run (used by unit tests). If the
    embedding runtime is unavailable, retrieval degrades explicitly to the
    acceptable tag-only v1 and emits a warning.
    """

    if k < 1:
        raise ValueError(f"k must be positive, got {k}")
    corpus = _load_corpus()
    tag_scores = _tag_scores(contract, corpus)
    mode = os.getenv("HAKAM_RETRIEVAL_MODE", "hybrid").strip().lower()

    if mode == "tags":
        combined = _normalise_scores(tag_scores)
        semantic = np.zeros_like(combined)
    elif mode == "hybrid":
        try:
            semantic = _semantic_scores(_query(contract), corpus)
            combined = 0.65 * _normalise_scores(tag_scores) + 0.35 * _normalise_scores(semantic)
        except (ImportError, OSError, RuntimeError) as exc:
            warnings.warn(
                f"Embedding retrieval unavailable ({type(exc).__name__}: {exc}); "
                "using tag-only retrieval.",
                RuntimeWarning,
                stacklevel=2,
            )
            semantic = np.zeros_like(tag_scores)
            combined = _normalise_scores(tag_scores)
    else:
        raise ValueError("HAKAM_RETRIEVAL_MODE must be 'hybrid' or 'tags'")

    # Stable secondary ordering makes tied tag-only tests reproducible.
    order = sorted(range(len(corpus)), key=lambda i: (-float(combined[i]), corpus[i]["id"]))
    results = []
    for index in order[: min(k, len(corpus))]:
        chunk = dict(corpus[index])
        chunk["retrieval"] = {
            "combined": round(float(combined[index]), 6),
            "tag": round(float(tag_scores[index]), 6),
            "semantic": round(float(semantic[index]), 6),
            "mode": mode,
        }
        results.append(chunk)
    return results
