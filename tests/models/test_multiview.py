import numpy as np
import torch

from src.models.data import IGNORE
from src.models.train_mt import average_by_action, best_threshold, task_loss


def test_task_loss_ignores_missing_labels():
    logits = torch.tensor([[2.0, -1.0], [0.0, 0.0], [-3.0, 3.0]])
    weights = torch.ones(2)
    with_missing = task_loss(logits, torch.tensor([0, IGNORE, 1]), weights, 0.0)
    only_labelled = task_loss(logits[[0, 2]], torch.tensor([0, 1]), weights, 0.0)
    assert torch.isclose(with_missing, only_labelled)


def test_task_loss_matches_weighted_cross_entropy():
    logits = torch.tensor([[1.0, 0.5], [0.2, 0.9], [0.3, 0.1]])
    labels = torch.tensor([0, 1, 1])
    weights = torch.tensor([0.7, 1.8])
    expected = torch.nn.functional.cross_entropy(logits, labels, weight=weights)
    assert torch.isclose(task_loss(logits, labels, weights, 0.0), expected)


def test_soft_target_used_where_label_missing():
    logits = torch.tensor([[0.0, 0.0]])
    loss = task_loss(logits, torch.tensor([IGNORE]), torch.ones(2), 0.0, soft=torch.tensor([0.5]))
    assert torch.isclose(loss, torch.log(torch.tensor(2.0)))


def test_all_missing_gives_zero_loss():
    logits = torch.zeros(2, 2, requires_grad=True)
    loss = task_loss(logits, torch.tensor([IGNORE, IGNORE]), torch.ones(2), 0.1)
    assert loss.item() == 0.0


def test_average_by_action_groups_views():
    ids = np.array(["7", "3", "7", "3", "3"])
    probs = np.array([[1, 0], [0, 1], [0, 1], [0, 1], [1, 0]], dtype=float)
    out_ids, avg, first = average_by_action(ids, probs)
    assert list(out_ids) == ["3", "7"]
    assert np.allclose(avg[0], [1 / 3, 2 / 3]) and np.allclose(avg[1], [0.5, 0.5])
    assert list(ids[first]) == ["3", "7"]


def test_best_threshold_separates_shifted_scores():
    labels = np.array([0, 0, 0, 1, 1, 1])
    p1 = np.array([0.10, 0.20, 0.30, 0.35, 0.40, 0.45])
    t = best_threshold(labels, p1)
    assert 0.30 < t <= 0.35
