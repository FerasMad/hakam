from __future__ import annotations

import json

import pytest

from backend.errors import ArtifactValidationError
from backend.runtime import read_thresholds


def test_reads_supplied_validation_thresholds_exactly(tmp_path):
    path = tmp_path / "thresholds.json"
    path.write_text(
        json.dumps({"offence": 0.60, "card": 0.50, "body_part": 0.49}),
        encoding="utf-8",
    )

    assert read_thresholds(path) == {
        "offence": 0.60,
        "card": 0.50,
        "body_part": 0.49,
    }


@pytest.mark.parametrize(
    "payload",
    [
        {"offence": 0.60, "card": 0.50},
        {"offence": 0.60, "card": 0.50, "body_part": 0.49, "action_class": 0.5},
        {"offence": 0.60, "card": 0.50, "body_part": 0.49, "card_colour": 0.5},
    ],
)
def test_rejects_missing_or_invented_thresholds(tmp_path, payload):
    path = tmp_path / "thresholds.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ArtifactValidationError, match="threshold"):
        read_thresholds(path)
