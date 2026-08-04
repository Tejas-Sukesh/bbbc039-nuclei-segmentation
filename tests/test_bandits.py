"""Tests for the bandit optimizers.

A bandit that silently fails to find a clearly best arm looks exactly like a
bandit that correctly found no difference, so these use synthetic problems with
a known answer.
"""

import numpy as np
import pytest

from nucleiseg import bandits as B
from nucleiseg.features import FeatureScaler, raw_features


class TestUCB1:
    def test_finds_best_arm(self):
        true = np.array([0.30, 0.50, 0.80, 0.45])
        rng = np.random.default_rng(0)
        algo = B.UCB1(len(true), c=1.0, seed=0)
        for _ in range(2000):
            a = algo.select()
            algo.update(a, float(np.clip(true[a] + rng.normal(0, 0.08), 0, 1)))
        assert int(np.argmax(algo.means)) == 2

    def test_concentrates_pulls_on_winner(self):
        true = np.array([0.2, 0.9, 0.25])
        rng = np.random.default_rng(1)
        algo = B.UCB1(3, seed=1)
        for _ in range(1500):
            a = algo.select()
            algo.update(a, float(np.clip(true[a] + rng.normal(0, 0.05), 0, 1)))
        # The best arm should absorb the clear majority of the budget.
        assert algo.counts[1] > 0.7 * algo.counts.sum()

    def test_every_arm_tried_once_first(self):
        algo = B.UCB1(5, seed=0)
        seen = set()
        for _ in range(5):
            a = algo.select()
            seen.add(a)
            algo.update(a, 0.5)
        assert seen == set(range(5))


class TestThompson:
    def test_finds_best_arm(self):
        true = np.array([0.25, 0.45, 0.85, 0.40])
        rng = np.random.default_rng(2)
        algo = B.ThompsonSampling(len(true), seed=2)
        for _ in range(2000):
            a = algo.select()
            algo.update(a, float(np.clip(true[a] + rng.normal(0, 0.08), 0, 1)))
        assert int(np.argmax(algo.means)) == 2

    def test_posterior_tracks_reward_level(self):
        algo = B.ThompsonSampling(2, seed=0)
        for _ in range(200):
            algo.update(0, 0.9)
            algo.update(1, 0.1)
        assert algo.means[0] > 0.8 and algo.means[1] < 0.2


class TestRunBandit:
    def test_history_and_counts_consistent(self):
        true = np.array([0.4, 0.7])

        def reward(arm, ctx):
            return float(true[arm])

        algo = B.UCB1(2, seed=0)
        res = B.run_bandit(algo, ["a", "b"], reward, n_rounds=100, n_contexts=5, seed=0)
        assert len(res.history) == 100
        assert res.counts.sum() == 100
        assert res.arm_names[res.best_arm] == "b"

    def test_summary_is_sorted_best_first(self):
        true = np.array([0.1, 0.9, 0.5])
        algo = B.ThompsonSampling(3, seed=0)
        res = B.run_bandit(
            algo, ["lo", "hi", "mid"], lambda a, c: float(true[a]), 300, 4, seed=0
        )
        rows = res.summary()
        assert rows[0]["arm"] == "hi"


class TestLinUCB:
    def test_learns_context_dependent_optimum(self):
        """Arm 0 is best when the feature is negative, arm 1 when positive.

        No single arm wins globally, so a non-contextual bandit cannot do better
        than ~0.5 here while a working contextual one should approach 1.0.
        """
        rng = np.random.default_rng(0)
        algo = B.LinUCB(n_arms=2, n_features=2, alpha=0.5, seed=0)
        for _ in range(600):
            x = np.array([rng.normal(), 1.0])
            a = algo.select(x)
            best = 0 if x[0] < 0 else 1
            algo.update(a, x, 1.0 if a == best else 0.0)

        correct = 0
        for _ in range(300):
            x = np.array([rng.normal(), 1.0])
            best = 0 if x[0] < 0 else 1
            correct += int(algo.greedy_policy(x) == best)
        assert correct / 300 > 0.85

    def test_greedy_policy_has_no_exploration_bonus(self):
        algo = B.LinUCB(n_arms=3, n_features=2, alpha=5.0, seed=0)
        x = np.array([0.5, 1.0])
        for _ in range(60):
            algo.update(0, x, 1.0)
            algo.update(1, x, 0.0)
            algo.update(2, x, 0.0)
        # With a large alpha, select() may still explore; greedy must not.
        assert algo.greedy_policy(x) == 0

    def test_untried_arms_are_tried_first(self):
        algo = B.LinUCB(n_arms=4, n_features=2, seed=0)
        x = np.array([0.1, 1.0])
        seen = set()
        for _ in range(4):
            a = algo.select(x)
            seen.add(a)
            algo.update(a, x, 0.5)
        assert seen == set(range(4))


class TestFeatures:
    def test_raw_features_shape_and_finite(self):
        rng = np.random.default_rng(0)
        img = (rng.random((64, 64)) * 500 + 120).astype(np.uint16)
        img[20:30, 20:30] = 3000
        f = raw_features(img)
        assert f.shape == (3,)
        assert np.all(np.isfinite(f))

    def test_empty_field_does_not_crash(self):
        flat = np.full((32, 32), 120, dtype=np.uint16)
        f = raw_features(flat)
        assert np.all(np.isfinite(f))

    def test_foreground_fraction_tracks_density(self):
        base = np.full((64, 64), 120, dtype=np.uint16)
        sparse = base.copy()
        sparse[0:8, 0:8] = 3000
        dense = base.copy()
        dense[0:40, 0:40] = 3000
        assert raw_features(dense)[0] > raw_features(sparse)[0]

    def test_scaler_standardises_and_appends_bias(self):
        feats = np.array([[1.0, 10.0, 0.5], [3.0, 20.0, 1.5], [2.0, 15.0, 1.0]])
        sc = FeatureScaler.fit(feats)
        out = sc.transform(feats)
        assert out.shape == (3, 4)
        assert np.allclose(out[:, -1], 1.0)          # bias term
        assert np.allclose(out[:, :3].mean(axis=0), 0.0, atol=1e-9)
        assert sc.n_features == 4

    def test_scaler_handles_constant_feature(self):
        feats = np.array([[1.0, 5.0], [1.0, 7.0]])
        sc = FeatureScaler.fit(feats)
        assert np.all(np.isfinite(sc.transform(feats)))

    def test_scaler_roundtrip(self):
        sc = FeatureScaler.fit(np.array([[1.0, 2.0], [3.0, 5.0]]))
        sc2 = FeatureScaler.from_dict(sc.to_dict())
        x = np.array([2.0, 3.0])
        assert np.allclose(sc.transform(x), sc2.transform(x))
