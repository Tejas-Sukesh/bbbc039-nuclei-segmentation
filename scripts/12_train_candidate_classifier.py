#!/usr/bin/env python
"""Train a classifier to decide which residual-pass candidates are real nuclei.

The residual detector in `smallobj.py` proposes small objects the first pass
missed, and hand-tuning its three gates tops out at 28% precision -- sweeping them
further loses recall without gaining precision, because no single threshold on any
one cue separates the classes. This trains the second stage of that detector:
propose generously, then let a model weigh eleven cues jointly.

**Design decisions that matter for honesty.**

* **Trained on the training split, evaluated on validation.** The 100 training
  images have never been used for anything else in this project, so they are
  genuinely held out from every earlier decision.
* **Proposals are generated with loose gates.** Proposal recall bounds the whole
  two-stage system, so the gates are opened and precision is left entirely to the
  classifier. The hand-tuned gates become the baseline to beat.
* **Labels are detection-style** -- a candidate is positive if its centroid falls
  inside a ground-truth nucleus the first pass missed. Requiring IoU >= 0.5 at
  labelling time would mark real detections negative whenever the half-height
  footprint disagrees with the annotation, which is a segmentation question, not a
  detection one. The *evaluation* still uses the project's usual IoU >= 0.5
  matching, so nothing is being graded on the easier criterion.
* **Gradient-boosted trees on eleven interpretable features**, not a CNN on
  patches. A few hundred positives cannot train a convolutional model that would
  not memorise, and a fitted tree can be interrogated afterwards -- if
  signal-to-noise dominates the importances, that is a statement about the imaging
  rather than about the classifier.

    python scripts/12_train_candidate_classifier.py
"""

from __future__ import annotations

import argparse
import json
import pickle

import numpy as np
from scipy import ndimage as ndi

from nucleiseg import failures as F
from nucleiseg import metrics as M
from nucleiseg.candidates import FEATURES, describe, to_matrix
from nucleiseg.data import REPO_ROOT, load_sample, split_names
from nucleiseg.evaluate import RESULTS, evaluate_named, print_summary, save_results
from nucleiseg.segmenters import CellposeParams, FlowCache
from nucleiseg.smallobj import ResidualParams, augment_labels, propose

MODEL_PATH = REPO_ROOT / "results" / "candidate_classifier.pkl"
# Deliberately looser than the hand-tuned defaults: proposal recall is the
# ceiling on the whole system, and precision is now the classifier's job.
LOOSE = ResidualParams(threshold=0.04, min_rel_contrast=0.15, min_edge_ratio=1.0)


def build_dataset(names, cache, params, rp, progress_every=20):
    """Features + labels for every proposal on `names`."""
    X, y, meta = [], [], []
    for i, name in enumerate(names, 1):
        s = load_sample(name)
        first = cache.masks(name, params)
        gt, ng = M.relabel_sequential(s.labels)
        missed = {m.gt_id for m in F.missed_objects(s.image, s.labels, first, name=name)}
        for mask, resp in propose(s.image, first, rp):
            X.append(describe(s.image, first, mask, resp))
            cy, cx = ndi.center_of_mass(mask)
            gid = int(gt[int(round(cy)), int(round(cx))])
            y.append(1 if (gid > 0 and gid in missed) else 0)
            meta.append(name)
        if i % progress_every == 0 or i == len(names):
            print(f"  {i}/{len(names)}  candidates={len(X)}  positives={sum(y)}",
                  flush=True)
    return to_matrix(X), np.array(y), meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-split", default="training")
    ap.add_argument("--eval-split", default="validation")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import average_precision_score

    cache, params = FlowCache(), CellposeParams()
    tr = split_names(args.train_split)
    ev = split_names(args.eval_split)
    if args.limit:
        tr, ev = tr[: args.limit], ev[: args.limit]

    missing = [n for n in tr + ev if not cache.has(n)]
    if missing:
        raise SystemExit(f"{len(missing)} images not cached; run scripts/01_cache_flows.py")

    print(f"[1/4] proposals on {len(tr)} {args.train_split} images (loose gates)")
    Xtr, ytr, _ = build_dataset(tr, cache, params, LOOSE)
    print(f"      {len(ytr)} candidates, {ytr.sum()} real "
          f"({ytr.mean():.1%} — this is the proposal-stage precision)")

    print(f"\n[2/4] proposals on {len(ev)} {args.eval_split} images")
    Xev, yev, _ = build_dataset(ev, cache, params, LOOSE)
    print(f"      {len(yev)} candidates, {yev.sum()} real ({yev.mean():.1%})")

    print("\n[3/4] training gradient-boosted trees")
    clf = HistGradientBoostingClassifier(
        max_depth=3, max_iter=200, learning_rate=0.06,
        min_samples_leaf=15, l2_regularization=1.0, random_state=0,
    )
    clf.fit(Xtr, ytr)
    p_ev = clf.predict_proba(Xev)[:, 1]
    print(f"      average precision on {args.eval_split}: "
          f"{average_precision_score(yev, p_ev):.3f}"
          f"   (a random ranker would score {yev.mean():.3f})")

    # Operating points: what precision is available at each recall?
    print(f"\n{'cut':>6}{'kept':>7}{'correct':>9}{'precision':>11}{'recall':>9}")
    rows = []
    for cut in (0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90):
        sel = p_ev >= cut
        kept, corr = int(sel.sum()), int(yev[sel].sum())
        prec = corr / max(kept, 1)
        rec = corr / max(int(yev.sum()), 1)
        rows.append({"cut": cut, "kept": kept, "correct": corr,
                     "precision": prec, "recall": rec})
        print(f"{cut:6.2f}{kept:7d}{corr:9d}{prec:11.1%}{rec:9.1%}")

    imp = sorted(zip(FEATURES, _permutation_importance(clf, Xev, yev)),
                 key=lambda kv: -kv[1])
    print("\n      feature importance (permutation, drop in average precision):")
    for k, v in imp:
        print(f"        {k:16s} {v:+.4f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MODEL_PATH.open("wb") as fh:
        pickle.dump({"model": clf, "features": FEATURES,
                     "params": LOOSE.__dict__}, fh)

    # ----- end-to-end: does it actually help the segmentation metric? --------
    best = max(rows, key=lambda r: r["precision"] * r["recall"])
    cut = best["cut"]
    print(f"\n[4/4] end-to-end on {args.eval_split}, classifier cut = {cut:.2f}")

    def keep(mask, resp, _img=None):
        return True  # replaced below per-image

    base = lambda n: cache.masks(n, params)  # noqa: E731

    def augmented(name):
        s = load_sample(name)
        first = base(name)

        def _keep(mask, resp):
            x = to_matrix([describe(s.image, first, mask, resp)])
            return float(clf.predict_proba(x)[0, 1]) >= cut

        lab, _ = augment_labels(s.image, first, LOOSE, keep=_keep)
        return lab

    a_before = save_results(f"cellpose_default_{args.eval_split}",
                            evaluate_named(base, ev), params=params)
    a_after = save_results(f"classified_residual_{args.eval_split}",
                           evaluate_named(augmented, ev), params=params,
                           extra={"cut": cut, "proposal_params": LOOSE.__dict__})
    print_summary("before", a_before)
    print_summary("after", a_after)

    print(f"\n  {'':26}{'before':>10}{'after':>10}")
    for label, b, a, f in [
        ("F1@0.5", a_before["f1_50_macro"], a_after["f1_50_macro"], "{:.4f}"),
        ("AP@[.5:.95]", a_before["ap_macro"], a_after["ap_macro"], "{:.4f}"),
        ("objects predicted", a_before["n_pred_total"], a_after["n_pred_total"], "{:d}"),
    ]:
        print(f"  {label:26}{f.format(b):>10}{f.format(a):>10}")

    out = RESULTS / f"candidate_classifier_{args.eval_split}.json"
    out.write_text(json.dumps({
        "n_train_candidates": int(len(ytr)), "n_train_positive": int(ytr.sum()),
        "proposal_precision_train": float(ytr.mean()),
        "n_eval_candidates": int(len(yev)), "n_eval_positive": int(yev.sum()),
        "average_precision": float(average_precision_score(yev, p_ev)),
        "operating_points": rows, "chosen_cut": cut,
        "feature_importance": {k: float(v) for k, v in imp},
        "end_to_end": {"f1_before": a_before["f1_50_macro"],
                       "f1_after": a_after["f1_50_macro"],
                       "ap_before": a_before["ap_macro"],
                       "ap_after": a_after["ap_macro"],
                       "ap_ci_after": a_after["ap_macro_ci95"]},
    }, indent=2, default=float))
    print(f"\nwrote {out.name} and {MODEL_PATH.name}")


def _permutation_importance(clf, X, y, n_repeat=3, seed=0):
    """Drop in average precision when each feature is shuffled."""
    from sklearn.metrics import average_precision_score

    rng = np.random.default_rng(seed)
    base = average_precision_score(y, clf.predict_proba(X)[:, 1])
    out = []
    for j in range(X.shape[1]):
        drops = []
        for _ in range(n_repeat):
            Xp = X.copy()
            rng.shuffle(Xp[:, j])
            drops.append(base - average_precision_score(y, clf.predict_proba(Xp)[:, 1]))
        out.append(float(np.mean(drops)))
    return out


if __name__ == "__main__":
    main()
