"""Tests for the evaluation metrics.

The metric is the one component that cannot be allowed to be subtly wrong: if
it is, every downstream number and every optimizer decision is meaningless, and
the failure is silent. So the cases here are ones whose answers can be computed
by hand.
"""

import numpy as np
import pytest

from nucleiseg import metrics as M


def box(shape, y0, y1, x0, x1, label=1, base=None):
    a = np.zeros(shape, dtype=np.int32) if base is None else base
    a[y0:y1, x0:x1] = label
    return a


class TestRelabel:
    def test_gaps_are_closed(self):
        labels = np.array([[0, 5, 5], [0, 9, 9]], dtype=np.int32)
        out, n = M.relabel_sequential(labels)
        assert n == 2
        assert set(np.unique(out)) == {0, 1, 2}

    def test_empty(self):
        out, n = M.relabel_sequential(np.zeros((4, 4), dtype=np.int32))
        assert n == 0 and out.max() == 0


class TestIoU:
    def test_identical_is_one(self):
        gt = box((10, 10), 2, 6, 2, 6)
        assert M.iou_matrix(gt, gt.copy())[0, 0] == pytest.approx(1.0)

    def test_disjoint_is_zero(self):
        gt = box((10, 20), 0, 4, 0, 4)
        pred = box((10, 20), 0, 4, 10, 14)
        assert M.iou_matrix(gt, pred)[0, 0] == pytest.approx(0.0)

    def test_known_half_overlap(self):
        # GT is 4x4=16 px at x=0..4; pred is 4x4=16 px at x=2..6.
        # intersection 4x2=8, union 16+16-8=24 -> IoU = 1/3.
        gt = box((10, 10), 0, 4, 0, 4)
        pred = box((10, 10), 0, 4, 2, 6)
        assert M.iou_matrix(gt, pred)[0, 0] == pytest.approx(8 / 24)

    def test_shape_and_noncontiguous_ids(self):
        gt = np.zeros((10, 10), dtype=np.int32)
        gt[0:3, 0:3] = 7
        gt[5:8, 5:8] = 42
        pred = gt.copy()
        m = M.iou_matrix(gt, pred)
        assert m.shape == (2, 2)
        assert np.allclose(np.diag(m), 1.0)


class TestMatching:
    def test_perfect_match(self):
        gt = np.zeros((20, 20), dtype=np.int32)
        gt[0:5, 0:5] = 1
        gt[10:15, 10:15] = 2
        r = M.match_instances(M.iou_matrix(gt, gt.copy()), 0.5)
        assert r.tp == 2 and r.fp == 0 and r.fn == 0
        assert r.dsb_score == pytest.approx(1.0)
        assert r.f1 == pytest.approx(1.0)

    def test_missed_object_is_fn(self):
        gt = np.zeros((20, 20), dtype=np.int32)
        gt[0:5, 0:5] = 1
        gt[10:15, 10:15] = 2
        pred = np.zeros_like(gt)
        pred[0:5, 0:5] = 1
        r = M.match_instances(M.iou_matrix(gt, pred), 0.5)
        assert (r.tp, r.fn, r.fp) == (1, 1, 0)
        assert r.dsb_score == pytest.approx(1 / 2)

    def test_spurious_object_is_fp(self):
        gt = box((20, 20), 0, 5, 0, 5)
        pred = gt.copy()
        pred[10:15, 10:15] = 2
        r = M.match_instances(M.iou_matrix(gt, pred), 0.5)
        assert (r.tp, r.fn, r.fp) == (1, 0, 1)
        assert r.dsb_score == pytest.approx(1 / 2)

    def test_greedy_equals_hungarian_above_half(self):
        # The uniqueness argument: no prediction can exceed IoU 0.5 with two
        # disjoint GT objects, so at thresholds >= 0.5 greedy is optimal.
        rng = np.random.default_rng(0)
        for _ in range(15):
            gt = np.zeros((60, 60), dtype=np.int32)
            pred = np.zeros((60, 60), dtype=np.int32)
            for i in range(6):
                y, x = rng.integers(0, 50, 2)
                gt[y : y + 8, x : x + 8] = i + 1
                dy, dx = rng.integers(-2, 3, 2)
                ys, xs = max(0, y + dy), max(0, x + dx)
                pred[ys : ys + 8, xs : xs + 8] = i + 1
            ious = M.iou_matrix(gt, pred)
            for t in M.IOU_THRESHOLDS:
                g = M.match_instances(ious, float(t), "greedy")
                h = M.match_instances(ious, float(t), "hungarian")
                assert g.tp == h.tp, f"greedy != hungarian at IoU {t}"


class TestSplitsAndMerges:
    def test_split_detected(self):
        # One GT object; two predictions each covering ~half of it.
        gt = box((20, 20), 0, 10, 0, 10)
        pred = np.zeros((20, 20), dtype=np.int32)
        pred[0:5, 0:10] = 1
        pred[5:10, 0:10] = 2
        splits, merges = M.count_splits_and_merges(gt, pred)
        assert splits == 1 and merges == 0

    def test_merge_detected(self):
        # Two GT objects; one prediction covering both.
        gt = np.zeros((20, 20), dtype=np.int32)
        gt[0:5, 0:10] = 1
        gt[5:10, 0:10] = 2
        pred = box((20, 20), 0, 10, 0, 10)
        splits, merges = M.count_splits_and_merges(gt, pred)
        assert splits == 0 and merges == 1

    def test_clean_match_has_neither(self):
        gt = np.zeros((20, 20), dtype=np.int32)
        gt[0:5, 0:5] = 1
        gt[10:15, 10:15] = 2
        s, m = M.count_splits_and_merges(gt, gt.copy())
        assert (s, m) == (0, 0)


class TestScoreImage:
    def test_perfect_is_one(self):
        gt = np.zeros((40, 40), dtype=np.int32)
        gt[0:8, 0:8] = 1
        gt[20:28, 20:28] = 2
        s = M.score_image(gt, gt.copy(), name="perfect")
        assert s.ap == pytest.approx(1.0)
        assert s.f1_50 == pytest.approx(1.0)
        assert s.count_error == 0
        assert s.mean_iou_matched == pytest.approx(1.0)

    def test_empty_gt_empty_pred_is_one(self):
        z = np.zeros((10, 10), dtype=np.int32)
        assert M.score_image(z, z.copy()).ap == pytest.approx(1.0)

    def test_empty_gt_with_prediction_is_zero(self):
        z = np.zeros((10, 10), dtype=np.int32)
        assert M.score_image(z, box((10, 10), 0, 3, 0, 3)).ap == pytest.approx(0.0)

    def test_missing_all_predictions_is_zero(self):
        gt = box((10, 10), 0, 3, 0, 3)
        assert M.score_image(gt, np.zeros((10, 10), dtype=np.int32)).ap == pytest.approx(0.0)

    def test_ap_decreases_as_boundary_degrades(self):
        # Shifting a prediction reduces IoU, which should cost more at strict
        # thresholds than at lenient ones.
        gt = box((40, 40), 10, 20, 10, 20)
        shifted = box((40, 40), 10, 20, 12, 22)
        s = M.score_image(gt, shifted)
        assert s.per_threshold[0.5] >= s.per_threshold[0.9]
        assert 0.0 < s.ap < 1.0


class TestAggregate:
    def _mk(self, gt, pred, name):
        return M.score_image(gt, pred, name=name)

    def test_count_bias_sign(self):
        gt = np.zeros((30, 30), dtype=np.int32)
        gt[0:5, 0:5] = 1
        gt[10:15, 10:15] = 2
        under = np.zeros_like(gt)
        under[0:5, 0:5] = 1
        agg = M.aggregate([self._mk(gt, under, "a")])
        assert agg["count_bias_pct"] < 0  # predicted fewer than GT
        assert agg["n_gt_total"] == 2 and agg["n_pred_total"] == 1

    def test_ci_brackets_mean_and_perfect_is_tight(self):
        gt = box((20, 20), 0, 6, 0, 6)
        scores = [self._mk(gt, gt.copy(), f"i{i}") for i in range(8)]
        agg = M.aggregate(scores)
        lo, hi = agg["ap_macro_ci95"]
        assert lo <= agg["ap_macro"] <= hi
        assert agg["ap_macro"] == pytest.approx(1.0)

    def test_empty_fields_counted_and_excluded(self):
        gt = box((20, 20), 0, 6, 0, 6)
        z = np.zeros((20, 20), dtype=np.int32)
        scores = [self._mk(gt, gt.copy(), "good"), self._mk(z, box((20, 20), 0, 2, 0, 2), "empty")]
        agg = M.aggregate(scores)
        assert agg["n_empty_gt"] == 1
        assert agg["ap_macro"] == pytest.approx(0.5)          # (1.0 + 0.0) / 2
        assert agg["ap_macro_nonempty"] == pytest.approx(1.0)  # empty field excluded
