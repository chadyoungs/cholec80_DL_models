"""Metric correctness: phase confusion, average precision, video Jaccard."""

from __future__ import annotations

import numpy as np

from cholec80_dl.metrics import PhaseMetrics, ToolMetrics, average_precision, video_phase_metrics


def test_phase_metrics_perfect():
    m = PhaseMetrics()
    m.update([0, 1, 2], [0, 1, 2])
    out = m.compute()
    assert out["phase_acc"] == 1.0
    assert out["phase_f1_macro"] == 1.0
    assert m.confusion().trace() == 3


def test_phase_metrics_macro_ignores_absent_classes():
    m = PhaseMetrics()
    m.update([1, 1, 1], [1, 1, 2])       # only phases 1 and 2 present
    out = m.compute()
    assert abs(out["phase_acc"] - 2 / 3) < 1e-6
    # f1 for class 1: precision 1, recall 2/3 -> 0.8 ; class 2: 0/0 -> 0
    assert abs(out["phase_f1_macro"] - (0.8 + 0.0) / 2) < 1e-6


def test_phase_metrics_ignores_unknown_labels():
    m = PhaseMetrics()
    m.update([0, 3], [-1, 3])
    assert m.confusion().sum() == 1


def test_average_precision_known_values():
    assert average_precision(np.array([1, 1, 0]), np.array([0.9, 0.8, 0.5])) == 1.0
    ap = average_precision(np.array([1, 0, 1]), np.array([0.9, 0.8, 0.7]))
    assert abs(ap - (0.5 + 0.5 * 2 / 3)) < 1e-6
    assert np.isnan(average_precision(np.zeros(3), np.array([0.2, 0.5, 0.9])))


def test_tool_metrics_map():
    m = ToolMetrics()
    probs = np.zeros((4, 7), dtype=np.float32)
    probs[:, 0] = [0.9, 0.8, 0.2, 0.1]     # perfectly ranked for tool 0
    probs[:, 1] = [0.1, 0.9, 0.8, 0.2]     # inverse ranking for tool 1
    targets = np.zeros((4, 7), dtype=np.float32)
    targets[:2, 0] = 1
    targets[1:3, 1] = 1
    m.update(probs, targets)
    out = m.compute()
    assert out["tool_ap/Grasper"] == 1.0
    assert 0.0 <= out["tool_map"] <= 1.0
    assert out["tool_f1_micro"] > 0


def test_video_phase_metrics_jaccard():
    pred = np.array([0, 0, 1, 1])
    target = np.array([0, 1, 1, 1])
    out = video_phase_metrics(pred, target)
    assert out["video_acc"] == 0.75
    # phase 0: |intersection| = 1, |union| = 2 -> 0.5
    assert abs(out["video_jaccard/Preparation"] - 0.5) < 1e-6
    # phase 1: intersection 2, union 3 -> 2/3
    assert abs(out["video_jaccard/CalotTriangleDissection"] - 2 / 3) < 1e-6
