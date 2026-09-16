from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src import config

from backend.errors import ArtifactValidationError


REQUIRED_THRESHOLDS = {"offence", "card", "body_part"}
EXPECTED_TASKS = {"offence", "card", "action_class", "body_part"}


@dataclass(frozen=True)
class LoadedPredictor:
    predictor: Any
    thresholds: dict[str, float]
    tasks: list[str]
    model_version: str
    device: str


@dataclass
class RuntimeState:
    predictor: Any | None = None
    thresholds: dict[str, float] = field(default_factory=dict)
    tasks: list[str] = field(default_factory=list)
    model_version: str | None = None
    device: str | None = None
    reason_code: str | None = "MODEL_LOADING"
    final_pt_present: bool = False
    thresholds_present: bool = False

    @property
    def ready(self) -> bool:
        return self.predictor is not None and self.reason_code is None


def weights_directory() -> Path:
    return Path(os.getenv("HAKAM_WEIGHTS_DIR", config.PROJECT_ROOT / "weights")).resolve()


def read_thresholds(path: Path) -> dict[str, float]:
    if not path.is_file():
        raise ArtifactValidationError("THRESHOLDS_MISSING", "thresholds.json is missing")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactValidationError("THRESHOLDS_INVALID", "thresholds.json is invalid") from exc

    raw = payload.get("thresholds", payload) if isinstance(payload, dict) else None
    if not isinstance(raw, dict):
        raise ArtifactValidationError("THRESHOLDS_INVALID", "thresholds.json has no threshold map")

    missing = REQUIRED_THRESHOLDS - set(raw)
    if missing:
        raise ArtifactValidationError(
            "THRESHOLDS_INVALID",
            "thresholds.json is missing required validation thresholds",
        )
    if "action_class" in raw or "card_colour" in raw:
        raise ArtifactValidationError(
            "THRESHOLDS_INVALID",
            "thresholds.json contains a threshold for a non-binary production head",
        )

    thresholds: dict[str, float] = {}
    for name in REQUIRED_THRESHOLDS:
        value = raw[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ArtifactValidationError("THRESHOLDS_INVALID", "threshold values must be numeric")
        numeric = float(value)
        if not 0.0 <= numeric <= 1.0:
            raise ArtifactValidationError("THRESHOLDS_INVALID", "threshold values must be in [0, 1]")
        thresholds[name] = numeric
    return thresholds


def load_validated_predictor(
    weights_dir: Path | None = None,
    device: str | None = None,
    thresholds: dict[str, float] | None = None,
) -> LoadedPredictor:
    directory = (weights_dir or weights_directory()).resolve()
    checkpoint = directory / "final.pt"
    threshold_path = directory / "thresholds.json"
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        raise ArtifactValidationError("WEIGHTS_MISSING", "final.pt is missing or empty")

    expected_thresholds = thresholds or read_thresholds(threshold_path)

    try:
        from src.inference.predict import load_predictor

        predictor = load_predictor(directory, device=device)
    except ArtifactValidationError:
        raise
    except Exception as exc:
        raise ArtifactValidationError("MODEL_LOAD_FAILED", "final.pt could not be loaded") from exc

    tasks = sorted(predictor.tasks)
    if set(tasks) != EXPECTED_TASKS:
        raise ArtifactValidationError(
            "MODEL_HEADS_INVALID",
            "final.pt does not contain the expected production heads",
        )
    if "card_colour" in tasks:
        raise ArtifactValidationError(
            "MODEL_HEADS_INVALID",
            "card colour is not a deployed production head",
        )
    if predictor.thresholds != expected_thresholds:
        raise ArtifactValidationError(
            "THRESHOLDS_MISMATCH",
            "the predictor did not load the supplied validation thresholds exactly",
        )

    return LoadedPredictor(
        predictor=predictor,
        thresholds=expected_thresholds,
        tasks=tasks,
        model_version=predictor.model_version,
        device=str(predictor.device),
    )
