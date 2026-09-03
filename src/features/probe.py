"""Linear probes over frozen features, with a permutation null.

A probe answers one question: is there any linearly decodable signal about this
label in these embeddings? It is deliberately the weakest reasonable model. If a
logistic regression on frozen features finds nothing, a deeper head on the same
features is unlikely to rescue it, and the honest move is to change the
features rather than the head.

Balanced accuracy is the metric throughout because every target here is skewed -
stage 1 is 88/12, and plain accuracy would reward a constant predictor.

The nominal chance level of 0.5 is not the right thing to compare against on a
few hundred samples: cross-validated balanced accuracy has real variance, and a
probe can clear 0.5 on noise alone. ``permutation_null`` estimates the actual
null by refitting on shuffled labels, so the reported p-value reflects this
sample size and this CV split rather than an idealised coin flip.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src import config


def build_estimator(C: float = 1.0, max_iter: int = 2000):
    """Standardise then fit a class-balanced logistic regression."""
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=C, max_iter=max_iter, class_weight="balanced"),
    )


def drop_rare_classes(X: np.ndarray, y: np.ndarray, min_count: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """Remove classes too small to appear in every CV fold.

    Reported rather than silently applied - a probe run on a quietly truncated
    label set is a misleading probe.
    """
    labels, counts = np.unique(y, return_counts=True)
    keep = set(labels[counts >= min_count])
    mask = np.array([v in keep for v in y])
    dropped = {str(l): int(c) for l, c in zip(labels, counts) if l not in keep}
    return X[mask], y[mask], dropped


def linear_probe(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    seed: int = config.SEED,
    C: float = 1.0,
) -> dict:
    """Cross-validated balanced accuracy, plus the confusion matrix."""
    X, y, dropped = drop_rare_classes(np.asarray(X), np.asarray(y), min_count=n_splits)
    n_classes = len(np.unique(y))
    if n_classes < 2:
        return {"error": "fewer than two classes survive", "dropped_classes": dropped}

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    est = build_estimator(C=C)

    scores = cross_val_score(est, X, y, cv=cv, scoring="balanced_accuracy")
    preds = cross_val_predict(est, X, y, cv=cv)
    labels = sorted(np.unique(y).tolist())

    return {
        "n": int(len(y)),
        "n_classes": n_classes,
        "chance": 1.0 / n_classes,
        "balanced_accuracy": float(scores.mean()),
        "std": float(scores.std()),
        "per_fold": [float(s) for s in scores],
        "labels": labels,
        "confusion": confusion_matrix(y, preds, labels=labels).tolist(),
        "class_counts": {str(l): int((y == l).sum()) for l in labels},
        "dropped_classes": dropped,
    }


def permutation_null(
    X: np.ndarray,
    y: np.ndarray,
    n: int = 100,
    n_splits: int = 5,
    seed: int = config.SEED,
    C: float = 1.0,
) -> dict:
    """Null distribution of CV balanced accuracy under shuffled labels.

    The same estimator and fold count as ``linear_probe``, so the observed score
    and the null are directly comparable.
    """
    X, y, _ = drop_rare_classes(np.asarray(X), np.asarray(y), min_count=n_splits)
    if len(np.unique(y)) < 2:
        return {"error": "fewer than two classes survive"}

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    est = build_estimator(C=C)
    observed = float(cross_val_score(est, X, y, cv=cv, scoring="balanced_accuracy").mean())

    rng = np.random.default_rng(seed)
    null = np.empty(n, dtype=float)
    for i in range(n):
        null[i] = cross_val_score(
            est, X, rng.permutation(y), cv=cv, scoring="balanced_accuracy"
        ).mean()

    # Add-one correction: with n shuffles the smallest reportable p is 1/(n+1),
    # never 0.
    p = float((np.sum(null >= observed) + 1) / (n + 1))
    return {
        "observed": observed,
        "null_mean": float(null.mean()),
        "null_std": float(null.std()),
        "null_p95": float(np.percentile(null, 95)),
        "p_value": p,
        "n_permutations": int(n),
        "null": null.tolist(),
    }
