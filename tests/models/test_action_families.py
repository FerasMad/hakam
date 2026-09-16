import numpy as np
import pandas as pd

from src.config import ACTION_FAMILIES
from src.llm import lexicon
from src.models.data import IGNORE, LEGACY_CLASSES, TASKS, _task_label, to_current_classes


def _rec(action):
    return pd.Series({"Action class": action, "Bodypart": "Under body",
                      "target_offence": "offence", "supervise_offence": True,
                      "target_card": "card", "supervise_card": True})


def test_raw_classes_map_to_families():
    fam = TASKS["action_class"]
    assert _task_label(_rec("Challenge"), "action_class") == fam.index("tackle")
    assert _task_label(_rec("Standing tackling"), "action_class") == fam.index("tackle")
    assert _task_label(_rec("Pushing"), "action_class") == fam.index("hands")
    assert _task_label(_rec("Holding"), "action_class") == fam.index("hands")
    assert _task_label(_rec("High leg"), "action_class") == fam.index("high leg")


def test_dive_is_ignored_only_by_the_action_head():
    rec = _rec("Dive")
    assert _task_label(rec, "action_class") == IGNORE
    assert _task_label(rec, "card") != IGNORE


def test_legacy_scores_sum_into_families():
    legacy = LEGACY_CLASSES["action_class"]
    p = np.zeros((1, len(legacy)))
    p[0, legacy.index("challenge")] = 0.3
    p[0, legacy.index("standing tackling")] = 0.2
    p[0, legacy.index("pushing")] = 0.4
    p[0, legacy.index("dive")] = 0.1
    probs, y = to_current_classes("action_class", p, np.array([legacy.index("pushing")]))
    fam = TASKS["action_class"]
    assert np.allclose(probs[0, fam.index("tackle")], 0.5 / 0.9)
    assert np.allclose(probs[0, fam.index("hands")], 0.4 / 0.9)
    assert y[0] == fam.index("hands")
    _, y_dive = to_current_classes("action_class", p, np.array([legacy.index("dive")]))
    assert y_dive[0] == IGNORE


def test_every_family_has_arabic_forms():
    assert not lexicon.missing_labels(set(ACTION_FAMILIES))


def test_current_class_scores_pass_through():
    p = np.full((2, len(TASKS["action_class"])), 0.25)
    y = np.array([0, 1])
    out_p, out_y = to_current_classes("action_class", p, y)
    assert out_p is p and out_y is y
