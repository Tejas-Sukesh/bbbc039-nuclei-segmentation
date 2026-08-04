"""Bandit algorithms for tuning segmentation parameters.

Why a bandit rather than gradient descent
-----------------------------------------
Average precision is **non-differentiable**. Computing it involves thresholding
IoU, matching objects one-to-one, and counting -- none of which has a useful
derivative. So there is no gradient to follow, and any method that needs one is
unavailable. What remains is search, and the useful framing for search under a
noisy, expensive objective is the multi-armed bandit.

This is also the honest boundary of the approach: the *segmentation* is done by a
supervised model, because dense pixel labels exist and a per-pixel gradient is
strictly more information than a scalar reward. The bandit sits on top, choosing
parameters. Using RL for the segmentation itself would be throwing away
supervision.

The vocabulary, once
--------------------
* **Arm** -- one thing you could choose. Here, one parameter configuration.
* **Pull** -- one trial. Here: evaluate a configuration on one image and observe
  its AP. Noisy, because images differ enormously (measured range 0.70 to 0.87).
* **Regret** -- the score lost by not having played the best arm all along. The
  quantity these algorithms are designed to bound.
* **Exploration vs exploitation** -- the whole problem. Exploit and you may be
  stuck on an arm that only looked good from three lucky images; explore and you
  waste budget on arms already known to be bad.

`UCB1` and `ThompsonSampling` solve the plain version (one best arm overall).
`LinUCB` solves the **contextual** version: the best arm may depend on features
of the image in front of you, so the policy learns a mapping from context to arm
rather than a single winner.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np


@dataclass
class BanditResult:
    """Trace of a bandit run, for reporting and for plotting the response curve."""

    arm_names: list[str]
    counts: np.ndarray  # pulls per arm
    means: np.ndarray  # empirical mean reward per arm
    history: list[tuple[int, float]] = field(default_factory=list)  # (arm, reward)

    @property
    def best_arm(self) -> int:
        """Arm with the highest empirical mean among those actually pulled."""
        masked = np.where(self.counts > 0, self.means, -np.inf)
        return int(np.argmax(masked))

    def summary(self) -> list[dict]:
        return sorted(
            (
                {
                    "arm": self.arm_names[i],
                    "pulls": int(self.counts[i]),
                    "mean_reward": float(self.means[i]) if self.counts[i] else float("nan"),
                }
                for i in range(len(self.arm_names))
            ),
            key=lambda r: (-(r["mean_reward"] if r["mean_reward"] == r["mean_reward"] else -9), ),
        )


# --------------------------------------------------------------------------- #
# Non-contextual: find one best configuration overall
# --------------------------------------------------------------------------- #


class UCB1:
    """Upper Confidence Bound.

    The idea in one sentence: **be optimistic in the face of uncertainty.** For
    each arm keep a running mean plus a bonus that grows when the arm has been
    tried rarely, then always play the arm with the highest mean-plus-bonus.

        score(a) = mean(a) + c * sqrt( ln(t) / n(a) )

    The bonus is the width of a confidence interval on the mean. An arm is played
    either because it looks good (high mean) or because we are unsure about it
    (few pulls, wide interval), and the bonus shrinks as evidence accumulates --
    so exploration happens automatically and then tapers off, with no schedule to
    tune. It is deterministic given the reward sequence.

    `c` scales how much optimism to apply; 1.0 is a reasonable default for
    rewards already in [0, 1], as AP is.
    """

    def __init__(self, n_arms: int, c: float = 1.0, seed: int = 0):
        self.n_arms = n_arms
        self.c = c
        self.counts = np.zeros(n_arms, dtype=np.int64)
        self.sums = np.zeros(n_arms, dtype=np.float64)
        self.rng = np.random.default_rng(seed)

    @property
    def means(self) -> np.ndarray:
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(self.counts > 0, self.sums / np.maximum(self.counts, 1), 0.0)

    def select(self) -> int:
        # Play each arm once before trusting any comparison.
        untried = np.flatnonzero(self.counts == 0)
        if untried.size:
            return int(self.rng.choice(untried))
        t = self.counts.sum()
        bonus = self.c * np.sqrt(np.log(t) / self.counts)
        return int(np.argmax(self.means + bonus))

    def update(self, arm: int, reward: float) -> None:
        self.counts[arm] += 1
        self.sums[arm] += reward


class ThompsonSampling:
    """Thompson sampling with a Beta posterior.

    The idea in one sentence: **keep a probability distribution over how good
    each arm is, draw one sample from each, and play the winner of that draw.**

    Because an arm with little data has a wide posterior, its sample sometimes
    comes out high and it gets tried -- exploration falls out of the randomness
    rather than being added as a bonus term. As evidence accumulates the
    posteriors narrow and the draws concentrate on the genuinely best arm.

    Rewards here are AP in [0, 1], not successes and failures, so this uses the
    standard Beta-Bernoulli trick: treat a reward r as r units of "success" and
    (1 - r) units of "failure". The posterior for arm a is then
    Beta(1 + sum(r), 1 + sum(1 - r)), starting from a uniform prior.

    Compared to UCB it is randomised (so repeated runs differ, and the seed must
    be reported) and usually a little more sample-efficient in practice.
    """

    def __init__(self, n_arms: int, seed: int = 0):
        self.n_arms = n_arms
        self.alpha = np.ones(n_arms, dtype=np.float64)
        self.beta = np.ones(n_arms, dtype=np.float64)
        self.counts = np.zeros(n_arms, dtype=np.int64)
        self.sums = np.zeros(n_arms, dtype=np.float64)
        self.rng = np.random.default_rng(seed)

    @property
    def means(self) -> np.ndarray:
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(self.counts > 0, self.sums / np.maximum(self.counts, 1), 0.0)

    def select(self) -> int:
        return int(np.argmax(self.rng.beta(self.alpha, self.beta)))

    def update(self, arm: int, reward: float) -> None:
        r = float(np.clip(reward, 0.0, 1.0))
        self.alpha[arm] += r
        self.beta[arm] += 1.0 - r
        self.counts[arm] += 1
        self.sums[arm] += r


def run_bandit(
    algo,
    arm_names: Sequence[str],
    reward_fn: Callable[[int, int], float],
    n_rounds: int,
    n_contexts: int,
    seed: int = 0,
) -> BanditResult:
    """Drive a non-contextual bandit for `n_rounds` pulls.

    `reward_fn(arm, context_index) -> reward`. The context index selects which
    image to evaluate on; it is drawn uniformly at random so that no arm is
    advantaged by being tested on easier fields.
    """
    rng = np.random.default_rng(seed + 1)
    result = BanditResult(
        arm_names=list(arm_names),
        counts=np.zeros(len(arm_names), dtype=np.int64),
        means=np.zeros(len(arm_names), dtype=np.float64),
    )
    for _ in range(n_rounds):
        arm = algo.select()
        reward = reward_fn(arm, int(rng.integers(0, n_contexts)))
        algo.update(arm, reward)
        result.history.append((arm, float(reward)))
    result.counts = algo.counts.copy()
    result.means = algo.means.copy()
    return result


# --------------------------------------------------------------------------- #
# Contextual: let the best configuration depend on the image
# --------------------------------------------------------------------------- #


class LinUCB:
    """Disjoint LinUCB -- a contextual bandit with one linear model per arm.

    The motivation is a measured fact: performance varies a lot across fields
    (0.70 to 0.87 in a 12-image sample), and the worst field was a sparse one
    with 43 nuclei while dense fields behave completely differently. A single
    global parameter set is a compromise across images that do not want the same
    compromise.

    So instead of asking "which configuration is best on average," ask "which
    configuration is best *for an image that looks like this*." Each arm keeps a
    ridge-regression model predicting its reward from a context vector x, and the
    arm played is the one with the highest optimistic prediction:

        score(a) = theta_a . x  +  alpha * sqrt( x^T A_a^-1 x )

    The first term is the predicted reward, the second is the uncertainty of that
    prediction for *this* x -- the same optimism-under-uncertainty idea as UCB1,
    but now uncertainty is direction-dependent: an arm well tested on dense
    fields is still uncertain about sparse ones.

    Two guardrails matter more than the algorithm here:

    1. **Capacity.** With ~100 training images, a rich context vector will
       memorise. Keep the feature count small (2-4) and standardise it.
    2. **No label leakage.** Contexts must be computable from the image alone.
       Using ground truth to pick per-image parameters at test time would be
       cheating; the policy has to map image features -> arm and then run blind.
    """

    def __init__(self, n_arms: int, n_features: int, alpha: float = 0.5, seed: int = 0):
        self.n_arms = n_arms
        self.d = n_features
        self.alpha = alpha
        # A = X^T X + I  (ridge), b = X^T y, per arm.
        self.A = np.stack([np.eye(n_features) for _ in range(n_arms)])
        self.b = np.zeros((n_arms, n_features), dtype=np.float64)
        self.counts = np.zeros(n_arms, dtype=np.int64)
        self.sums = np.zeros(n_arms, dtype=np.float64)
        self.rng = np.random.default_rng(seed)

    @property
    def means(self) -> np.ndarray:
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(self.counts > 0, self.sums / np.maximum(self.counts, 1), 0.0)

    def theta(self, arm: int) -> np.ndarray:
        return np.linalg.solve(self.A[arm], self.b[arm])

    def scores(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64).ravel()
        out = np.empty(self.n_arms)
        for a in range(self.n_arms):
            A_inv_x = np.linalg.solve(self.A[a], x)
            out[a] = self.theta(a) @ x + self.alpha * np.sqrt(max(x @ A_inv_x, 0.0))
        return out

    def select(self, x: np.ndarray) -> int:
        untried = np.flatnonzero(self.counts == 0)
        if untried.size:
            return int(self.rng.choice(untried))
        return int(np.argmax(self.scores(x)))

    def update(self, arm: int, x: np.ndarray, reward: float) -> None:
        x = np.asarray(x, dtype=np.float64).ravel()
        self.A[arm] += np.outer(x, x)
        self.b[arm] += reward * x
        self.counts[arm] += 1
        self.sums[arm] += reward

    def greedy_policy(self, x: np.ndarray) -> int:
        """Arm with the highest *predicted* reward -- no exploration bonus.

        This is what gets deployed: exploration is for training, and at test time
        the policy should commit to its best guess.
        """
        x = np.asarray(x, dtype=np.float64).ravel()
        return int(np.argmax([self.theta(a) @ x for a in range(self.n_arms)]))
