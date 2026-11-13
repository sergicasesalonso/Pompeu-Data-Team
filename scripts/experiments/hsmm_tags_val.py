"""HSMM with synthetic Smart Tagging on external VAL: fit w_tagging, report spellF1 at 2/5/8 s.

    python scripts/experiments/hsmm_tags_val.py                    # hsmm_v2_fit2, seeds 0-4
    python scripts/experiments/hsmm_tags_val.py --weights 0 0.5 1 2 --seeds 0 1

On external matches hsmm.prepare finds no Smart Tagging (build_features' "auto" returns None),
so st_vote is NaN -> 0 and w_tagging was never fitted. Here each (val match, seed) is degraded
like robust_val.py (seed 1000*seed + index, the ATD's no-ball run lengths), cleaned (A1
defaults), and prepared twice:

    notags   as the bench does (no tags anywhere): the reference
    tags     with synthetic tags (pompeu.external.synthetic_tags, configs/synthetic_tags_v1.json,
             seeded 7919 + 1000*seed + index from the 2 s labels) injected into build_features,
             so every consumer of the context columns (the HSMM tagging expert, and A3/A4 if
             they read set pieces) sees them, then decoded at each w_tagging in --weights

Optionally (--gap-models) the gap-team model is swapped too, e.g. for the tag-aware model of
gap_team_tags.py. Base HSMM parameters: --config (default hsmm_v2_fit2). Scoring is
robust_val's (gap_team_weight.score: seeds joined, harness scores, bootstrap intervals).
Only `val` (never test). w_tagging is chosen AND reported on val, so its score is optimistic.
"""

import argparse
import copy
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "experiments"))
from pompeu import external, paths  # noqa: E402
from pompeu.clean import clean_tracking  # noqa: E402
from pompeu.external import degrade, synthetic_tags as st  # noqa: E402
from pompeu.io.contract import load_tracking  # noqa: E402
from pompeu.models import hsmm  # noqa: E402
from pompeu.validate import harness  # noqa: E402
from pompeu.validate.stress import missing_run_lengths  # noqa: E402
import possession_metrics as pm  # noqa: E402
from gap_team_weight import present, score  # noqa: E402
from robust_val import fmt  # noqa: E402

VALUES = (2.0, 5.0, 8.0)
TAG_SEED = 7919


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="hsmm_v2_fit2")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--weights", type=float, nargs="+", default=[0, 0.25, 0.5, 1, 2, 4])
    parser.add_argument("--gap-models", nargs="*", default=[])
    parser.add_argument("--draws", type=int, default=1000)
    args = parser.parse_args()
    started = time.time()

    cfg = harness.load_config(os.path.join(paths.CONFIGS_DIR, args.config + ".yaml"))
    params = cfg["pipeline"]["params"]
    base = params.get("hsmm") or {}
    candidates = {"notags": ("notags", dict(base))}
    for gm in [None] + list(args.gap_models):
        for w in args.weights:
            p = dict(base, w_tagging=w)
            name = "tags w={:g}".format(w)
            if gm:
                p["gap_team_model"] = gm
                name += " " + os.path.splitext(os.path.basename(gm))[0]
            candidates[name] = ("tags", p)
    model_st = st.load_error_model()
    names = [external.canonical(m) for m in external.SPLITS["val"]]          # never "test"
    matches = [m for m in names if present(m)]
    run_lengths = missing_run_lengths(load_tracking("atd"))
    print("val {}, seeds {}, base {}, {} candidates".format(matches, args.seeds, args.config,
                                                             len(candidates)))
    truths = {v: [] for v in VALUES}
    preds = {c: {} for c in candidates}
    tag_cov = []
    for index, name in enumerate(matches):
        tracking, labels = external.load(name)
        fps = tracking.meta.get("fps", 25)
        half = tracking.half()
        graded = np.where(half > 0, labels, pm.X).astype(np.int8)
        for v in VALUES:
            truths[v].append(external.align_labels(graded, v, fps))
        for seed in args.seeds:
            t0 = time.time()
            degraded = degrade.degrade(tracking, labels, 1000 * seed + index, run_lengths, None)
            clean, _ = clean_tracking(degraded, {})
            tags = st.generate(external.align_labels(graded, 2.0, fps), half, model_st,
                               seed=TAG_SEED + 1000 * seed + index, fps=fps)
            preps = {"notags": hsmm.prepare(clean, params.get("deadball") or None,
                                            params.get("touches") or None)}
            with st.injected({clean.meta["game"]: tags}):
                preps["tags"] = hsmm.prepare(clean, params.get("deadball") or None,
                                             params.get("touches") or None)
            tag_cov.append(float(np.mean(preps["tags"]["F"]["st_vote_n"].to_numpy()[half > 0] > 0)))
            for cname, (kind, p) in candidates.items():
                out = hsmm.decode_prepared(preps[kind], copy.deepcopy(p))
                preds[cname].setdefault(seed, []).append(pm.label_codes(out["labels"]))
            print("  {} seed {} ({:.0f} s)".format(name, seed, time.time() - t0), flush=True)
    print("synthetic tag coverage of in-half frames: mean {:.3f}".format(np.mean(tag_cov)))

    subsets = [("all val", None)] + [(m, [i]) for i, m in enumerate(matches)]
    means = {}
    for v in VALUES:
        for sname, keep in subsets:
            print("\nmin_possession_s {:g}, {}:  spellF1 [5%, 95%]   macroF1 [5%, 95%]   "
                  "poss/gt".format(v, sname))
            for c in candidates:
                r = score(truths[v], preds[c], args.draws, keep)
                if keep is None:
                    means.setdefault(v, {})[c] = r["spell"]
                print("  {:<34} {}".format(c, fmt(r)), flush=True)
    print("\nranking (all val, spellF1)")
    for v in VALUES:
        top = sorted(means[v], key=lambda c: -means[v][c])
        print("  {:g} s: {}".format(v, " > ".join("{} ({:.4f})".format(c, means[v][c]) for c in top[:5])))
    avg = {c: np.mean([means[v][c] for v in VALUES]) for c in candidates}
    best = max(avg, key=avg.get)
    print("best by mean spellF1 over 2/5/8 s: {} ({:.4f}; notags {:.4f})".format(
        best, avg[best], avg["notags"]))
    print("runtime {:.0f} s".format(time.time() - started))


if __name__ == "__main__":
    main()
