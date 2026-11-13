"""Gap-team expert with synthetic Smart Tagging features: train with / without tags, compare on val.

    python scripts/experiments/gap_team_tags.py                       # DFL train, seeds 0-4
    python scripts/experiments/gap_team_tags.py --skillcorner         # + SkillCorner extra_train

QUESTION. On BAR-ATH the gap-team expert could also read the Smart Tagging vote, but an expert
fitted on external data cannot learn its weight because external matches have no tags. Here
every external match gets synthetic tags corrupted like the real ones
(pompeu.external.synthetic_tags, error model configs/synthetic_tags_v1.json measured on BAR-ATH
without ground truth), and two models are fitted on EXACTLY the same rows:

    notags   gap_team_v2's recipe (collective columns + 3 s means, y-mirror and home/away swap
             augmentation, labels at 2/5/8 s merged, weighted ridge logistic)
    tags     the same + TAG_COLUMNS (st_vote, st_vote_n, st_prev_vote, sp_team) and their 3 s
             means; under the home/away swap the signed tag columns change sign

Training: external `train` (DFL only, like v2) degraded with seeds --seeds (1000*seed + index),
cleaned with A1 defaults; synthetic tags seeded 7919 + 1000*seed + index from the 2 s labels.
--skillcorner adds the SkillCorner EXTRA_TRAIN matches (labels from phases of play; seed 0 only).
Evaluation: external `val` (never test), seeds --val-seeds, gap_team_compare's metrics
(accuracy on gap frames, by gap length, changed gaps recovered), plus configs/gap_team_v2.json
for reference. Both fitted models are written to configs/gap_team_tags_exp_{notags,tags}.json
(experiment only, never in a submission). Only aggregates are printed.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "experiments"))
from pompeu import external, paths  # noqa: E402
from pompeu.experts import gap_team  # noqa: E402

# Params
TAG_COLUMNS = ("st_vote", "st_vote_n", "st_prev_vote", "sp_team")
SIGNED_TAGS = ("st_vote", "st_prev_vote", "sp_team")
MIN_POSSESSION_S = (2.0, 5.0, 8.0)
VARIANTS = ("identity", "mirror_y", "swap", "swap_mirror_y")
TAG_SEED = 7919
OUT = os.path.join(paths.CONFIGS_DIR, "gap_team_tags_exp_{}.json")


def augment(F, width, variant):
    import train_gap_team_aug as aug
    Fv, swapped = aug.augment(F, width, variant)
    if swapped:
        for c in SIGNED_TAGS:
            if c in Fv:
                Fv[c] = -Fv[c]
    return Fv, swapped


def features(name, index, seed, run_lengths, model):
    """(cleaned Tracking, F with collective + tag columns, labels, graded) for one degraded match."""
    from pompeu.clean import clean_tracking
    from pompeu.external import degrade, synthetic_tags as st
    from pompeu.features.build import build_features
    import possession_metrics as pm

    tracking, labels = external.load(name)
    fps = tracking.meta.get("fps", 25)
    half = tracking.half()
    graded = np.where(half > 0, labels, pm.X).astype(np.int8)
    degraded = degrade.degrade(tracking, labels, 1000 * seed + index, run_lengths, None)
    cleaned, _ = clean_tracking(degraded, {})
    F = build_features(cleaned, tagging=None, with_windows=False, groups=("collective",))
    tags = st.generate(external.align_labels(graded, 2.0, fps), half, model,
                       seed=TAG_SEED + 1000 * seed + index, fps=fps)
    ctx = st.context_features(cleaned, tags)
    for c in TAG_COLUMNS:
        F[c] = np.asarray(ctx[c], np.float64)
    return cleaned, F, labels, graded


def match_rows(job):
    name, index, seed, run_lengths, stride, model = job
    import train_gap_team_aug as aug
    import possession_metrics as pm
    cleaned, F, labels, graded = features(name, index, seed, run_lengths, model)
    fps = cleaned.meta.get("fps", 25)
    width = float(cleaned.meta["pitch"][1])
    half = cleaned.half()
    gts = [external.align_labels(graded, v, fps) for v in MIN_POSSESSION_S]
    live, ybar, weight = aug.merge_conventions(gts, pm.H, pm.A)
    keep = (half > 0) & ~cleaned.ball_present & live
    rows = np.flatnonzero(keep)[::stride]
    base = list(gap_team.BASE_COLUMNS) + list(TAG_COLUMNS)
    Xs, ys, ws = [], [], []
    for variant in VARIANTS:
        Fv, swapped = augment(F, width, variant)
        Xs.append(gap_team.design_matrix(Fv, fps, base)[rows])
        ys.append(1.0 - ybar[rows] if swapped else ybar[rows])
        ws.append(weight[rows])
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(ws), int(keep.sum())


def fit(X, y, weight, base, name, training):
    import train_gap_team_aug as aug
    Z, mean, std = gap_team.standardise(X)
    w, b = aug.fit_logistic_weighted(Z, y, weight)
    return {"name": name, "kind": "logistic",
            "target": "P(away has the ball) on in-half frames without ball",
            "base_columns": list(base), "window_s": gap_team.WINDOW_S,
            "features": gap_team.feature_names(base), "mean": [round(float(v), 8) for v in mean],
            "std": [round(float(v), 8) for v in std], "coef": [round(float(v), 8) for v in w],
            "intercept": round(float(b), 8), "ridge": gap_team.RIDGE, "training": training}


def collect_val(matches, seeds, models, model_st, run_lengths):
    """{value: data dict as gap_team_compare.collect} with one prediction per model."""
    import gap_team as gt_exp
    parts = {v: [] for v in MIN_POSSESSION_S}
    for index, name in enumerate(matches):
        for seed in seeds:
            started = time.time()
            cleaned, F, labels, graded = features(name, index, seed, run_lengths, model_st)
            fps = cleaned.meta.get("fps", 25)
            half = cleaned.half()
            logits = {k: gap_team.away_logit(F, fps, m) for k, m in models.items()}
            near_all = gt_exp.nearest_team(cleaned)
            for v in MIN_POSSESSION_S:
                gt = external.align_labels(graded, v, fps)
                gid, age, length, carry, near = gt_exp.gap_table(cleaned.ball_present, half, gt,
                                                                  near_all)
                keep = (gid >= 0) & np.isin(gt, (gt_exp.H, gt_exp.A)) & (carry >= 0)
                changed, post = gt_exp.change_masks(gid, gt, carry)
                uid = np.where(gid >= 0, gid + 1_000_000 * (1000 * seed + index), -1)
                p = dict(y=(gt[keep] == gt_exp.A).astype(float), gid=uid[keep],
                         length=length[keep] / fps, carry=carry[keep], near=near[keep],
                         changed=changed[keep], post=post[keep],
                         source=np.full(int(keep.sum()), name.split(":")[0]),
                         tag_cov=(F["st_vote_n"].to_numpy()[keep] > 0))
                for k in models:
                    p["pred:" + k] = (np.nan_to_num(logits[k][keep]) > 0).astype(float)
                parts[v].append(p)
            print("  val {} seed {} ({:.0f} s)".format(name, seed, time.time() - started), flush=True)
    return {v: {k: np.concatenate([p[k] for p in ps]) for k in ps[0]} for v, ps in parts.items()}


def report(data, models):
    import gap_team as gt_exp
    for v in MIN_POSSESSION_S:
        d = data[v]
        preds = {"carry": (d["carry"] == gt_exp.A).astype(float),
                 "nearest": (d["near"] == gt_exp.A).astype(float)}
        for k in models:
            preds[k] = d["pred:" + k]
        y, dfl_m, met_m = d["y"], d["source"] == "dfl", d["source"] == "metrica"
        print("\n==== ground truth at min_possession_s = {:g} s ====".format(v))
        print(gt_exp.table("accuracy on gap frames (H/A truth, carry defined)", preds, y,
                           [dfl_m, met_m, dfl_m & d["changed"], dfl_m & d["post"],
                            dfl_m & d["tag_cov"], dfl_m & ~d["tag_cov"]],
                           ["dfl", "metrica", "dfl chg gap", "dfl post-chg", "dfl tagged",
                            "dfl untagged"]))
        gb = gt_exp.bucket(d["length"], np.array(gt_exp.GAP_BUCKETS_S, float))
        nb = len(gt_exp.GAP_BUCKETS_S) - 1
        print(gt_exp.table("dfl, by gap length", preds, y, [dfl_m & (gb == i) for i in range(nb)],
                           [gt_exp.edge_label(gt_exp.GAP_BUCKETS_S, i) for i in range(nb)]))
        sub = {k: val[dfl_m] for k, val in d.items()}
        print("dfl changed gaps recovered (post-change team on > half of post-change frames)")
        for name, pred in preds.items():
            share, n = gt_exp.per_gap_recovery(pred[dfl_m], sub)
            print("  {:<22}{:>8.3f}   (of {} changed gaps)".format(name, share, n))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--val-seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--skillcorner", action="store_true")
    parser.add_argument("--tag", default="", help="suffix for the output JSON names")
    args = parser.parse_args()
    started = time.time()

    import coverage_sweep
    import gap_team as gt_exp
    from pompeu.external import synthetic_tags as st
    model_st = st.load_error_model()
    run_lengths = coverage_sweep.synthetic_run_lengths()
    train = [m for m in (external.canonical(m) for m in external.SPLITS["train"])
             if m.startswith("dfl:") and gt_exp.present(m)]                # never val/test
    jobs = [(name, index, seed, run_lengths, args.stride, model_st)
            for index, name in enumerate(train) for seed in args.seeds]
    if args.skillcorner:
        extra = external.resolve("extra_train")
        jobs += [(name, 100 + index, 0, run_lengths, args.stride, model_st)
                 for index, name in enumerate(extra)]
    print("train jobs {} ({} DFL matches x seeds {}{})".format(
        len(jobs), len(train), args.seeds, " + SkillCorner" if args.skillcorner else ""))
    if args.workers > 1:
        with ProcessPoolExecutor(args.workers) as pool:
            results = list(pool.map(match_rows, jobs))
    else:
        results = [match_rows(job) for job in jobs]
    X = np.concatenate([r[0] for r in results])
    y = np.concatenate([r[1] for r in results])
    weight = np.concatenate([r[2] for r in results])
    print("rows {}  (live gap frames {})".format(len(y), sum(r[3] for r in results)))
    del results

    nb = len(gap_team.BASE_COLUMNS)
    base_all = list(gap_team.BASE_COLUMNS) + list(TAG_COLUMNS)
    nt = len(TAG_COLUMNS)
    # design_matrix layout: [raw base_all | rolled base_all]
    idx_notags = list(range(nb)) + list(range(nb + nt, 2 * nb + nt))
    training = {"matches": train, "seeds": args.seeds, "skillcorner": args.skillcorner,
                "stride": args.stride, "synthetic_tags": "configs/synthetic_tags_v1.json",
                "tag_seed": "7919 + 1000*seed + index", "variants": list(VARIANTS),
                "min_possession_s": list(MIN_POSSESSION_S)}
    suffix = args.tag
    models = {}
    for key, cols, base in (("notags", idx_notags, list(gap_team.BASE_COLUMNS)),
                            ("tags", list(range(X.shape[1])), base_all)):
        model = fit(X[:, cols], y, weight, base, "gap_team_tags_exp_" + key + suffix, training)
        blob = (json.dumps(model, indent=1) + "\n").encode("utf-8")
        path = OUT.format(key + suffix)
        with open(path, "wb") as handle:
            handle.write(blob)
        print("wrote {} sha256 {}".format(os.path.relpath(path, ROOT), hashlib.sha256(blob).hexdigest()[:16]))
        models[key + suffix] = model
    tag_coef = dict(zip(models["tags" + suffix]["features"], models["tags" + suffix]["coef"]))
    print("tag coefficients (standardised): " + ", ".join(
        "{} {:+.3f}".format(k, v) for k, v in tag_coef.items() if k.startswith(("st_", "sp_"))))
    models["gap_team_v2"] = gap_team.load_model("configs/gap_team_v2.json")

    val = [m for m in (external.canonical(m) for m in external.SPLITS["val"]) if gt_exp.present(m)]
    print("val {}  seeds {}".format(val, args.val_seeds))
    data = collect_val(val, args.val_seeds, models, model_st, run_lengths)
    report(data, models)
    print("\nruntime {:.0f} s".format(time.time() - started))


if __name__ == "__main__":
    main()
