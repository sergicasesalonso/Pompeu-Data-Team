"""Experiment: self-train the gap-team expert on algorithmic pseudo-labels of BAR-ATH.

    python scripts/experiments/self_train_gap.py                    # val seeds 0-2
    python scripts/experiments/self_train_gap.py --seeds 0 --no-downstream

QUESTION. gap_team_v1 (pompeu.experts.gap_team) is fitted on external DFL train only. BAR-ATH has
~2 h of tracking without labels. Does adding BAR-ATH frames labelled by an ALGORITHM (never by
hand) help the expert on external VAL?

PSEUDO-LABELS (BAR-ATH, A1-cleaned ATD, nothing else):
  (a) in-half frames with a (cleaned) ball where rules_v0 and hsmm_v1 (default params) give the
      same team H/A, and the frame lies inside a run of that team of >= N s in BOTH outputs;
  (b) optional: Smart Tagging possession vote (attacking code -> that team has it, defending code
      -> the other; pompeu.features.context.possession_vote), on in-half frames that both models
      call live (H/A), only where (a) is absent or agrees. Conflicts (a) vs (b) are dropped.
  The public GT window (evaluation/gt, only its graded span is read, never its labels' content
  beyond pm.valid_runs) plus a MARGIN_S margin is excluded, so nothing is learned there.
  Features are ball-free (A2's collective group), so every pseudo-labelled frame is used whether
  the ball is visible or not.

TRAINING. Same features and ridge logistic as scripts/train_gap_team.py: external DFL train,
degraded with seeds 0-2, stride 5 (this script refits v1 from them and checks it against
configs/gap_team_v1.json), plus the BAR-ATH pseudo-labelled frames (stride 5) weighted so that
they carry `share` of the training mass. Weighted standardisation and weighted Newton.

EVALUATION. External VAL only (dfl:J03WN1; metrica:2 reported separately), degraded like the
bench, seeds 0-2. Never test, never the public GT. Gap metrics as scripts/experiments/gap_team.py
(gap frames with H/A truth and a carried team), computed with the logit WITHOUT intercept (as the
HSMM uses it). Also: the teacher's pseudo-label precision measured on degraded val (same rule
(a), no tagging there), and optionally the downstream hsmm spellF1 at gap_team_weight 0.25.

Everything heavy is cached in POMPEU_CACHE_DIR/selftrain (gitignored). Only aggregates printed.
"""

import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "experiments"))
from pompeu import external, paths  # noqa: E402
from pompeu.experts import gap_team  # noqa: E402
import possession_metrics as pm  # noqa: E402

# Params
H, A = pm.H, pm.A
MIN_POSSESSION_S = 5.0
MARGIN_S = 5.0              # extra exclusion around the public window (3 s rolling features)
STRIDE = 5
GAP_WEIGHT = 0.25           # hsmm_v2's weight, for the downstream check
VARIANTS = (                # (name, N seconds, use tagging, match share of training mass)
    ("a_N4_s50", 4.0, False, 0.5),
    ("ab_N4_s50", 4.0, True, 0.5),
    ("a_N4_s25", 4.0, False, 0.25),
    ("a_N8_s50", 8.0, False, 0.5),
    ("ab_N8_s50", 8.0, True, 0.5),
)
CACHE = os.path.join(paths.CACHE_DIR, "selftrain")


# -- pure logic (tests/test_self_train_gap.py) -------------------------------------------------

def long_runs(codes, half, min_len):
    """(N,) bool: frame inside a maximal run of one H/A code, within one half, >= min_len frames."""
    codes = np.asarray(codes)
    half = np.asarray(half)
    out = np.zeros(len(codes), bool)
    key = np.where(np.isin(codes, (H, A)) & (half > 0), codes.astype(int) + 10 * half, -1)
    change = np.flatnonzero(np.diff(key) != 0) + 1
    starts = np.concatenate(([0], change))
    stops = np.concatenate((change, [len(codes)]))
    for s, e in zip(starts, stops):
        if key[s] >= 0 and e - s >= min_len:
            out[s:e] = True
    return out


def pseudo_labels(rules, hmm, ball_present, half, min_len, tag_vote=None, exclude=None):
    """((N,) int8 pseudo-label H/A or -1, stats). rules/hmm: pm codes. tag_vote: +1 home, -1 away,
    0 none. exclude: bool mask of frames that may never be labelled."""
    rules, hmm, half = np.asarray(rules), np.asarray(hmm), np.asarray(half)
    n = len(rules)
    exclude = np.zeros(n, bool) if exclude is None else np.asarray(exclude, bool)
    agree = (rules == hmm) & np.isin(rules, (H, A)) & (half > 0)
    long_both = long_runs(rules, half, min_len) & long_runs(hmm, half, min_len)
    a = agree & long_both & np.asarray(ball_present, bool) & ~exclude
    out = np.where(a, rules, -1).astype(np.int8)
    stats = {"a": int(a.sum()), "agree_live": int(agree.sum())}
    if tag_vote is not None:
        vote = np.asarray(tag_vote)
        tag = np.where(vote > 0, H, np.where(vote < 0, A, -1))
        live_both = np.isin(rules, (H, A)) & np.isin(hmm, (H, A)) & (half > 0)
        b = (tag >= 0) & live_both & ~exclude
        both = a & b
        conflict = both & (tag != out)
        stats.update(b=int(b.sum()), overlap=int(both.sum()), conflict=int(conflict.sum()),
                     ab_agreement=float(1 - conflict.sum() / max(both.sum(), 1)))
        out = np.where(b & ~a, tag, out).astype(np.int8)
        out[conflict] = -1
    stats["labelled"] = int((out >= 0).sum())
    stats["home_share"] = float(np.mean(out[out >= 0] == H)) if stats["labelled"] else float("nan")
    return out, stats


def mass_weights(n_ext, n_match, share):
    """Per-row weights: match rows carry `share` of the total mass, total mass = n_ext + n_match."""
    total = n_ext + n_match
    if n_match == 0 or share <= 0:
        return np.ones(n_ext), np.zeros(n_match)
    if n_ext == 0 or share >= 1:
        return np.zeros(n_ext), np.ones(n_match)
    return (np.full(n_ext, (1 - share) * total / n_ext),
            np.full(n_match, share * total / n_match))


def weighted_standardise(X, w):
    """(Z, mean, std) as gap_team.standardise, with row weights w (NaN ignored)."""
    ok = ~np.isnan(X)
    ww = w[:, None] * ok
    mean = np.nansum(np.where(ok, X, 0) * w[:, None], axis=0) / np.maximum(ww.sum(axis=0), 1e-12)
    var = np.sum(np.where(ok, (X - mean) ** 2, 0) * w[:, None], axis=0) / np.maximum(
        ww.sum(axis=0), 1e-12)
    std = np.sqrt(var)
    mean = np.where(ww.sum(axis=0) > 0, mean, 0.0)
    std = np.where(~(std > 1e-9), 1.0, std)
    return gap_team.standardise(X, mean, std)


def fit_weighted_logistic(Z, y, w, ridge=gap_team.RIDGE, iters=50, tol=1e-8):
    """(coef, intercept) of a row-weighted ridge logistic regression (Newton). w = 1 -> as
    gap_team.fit_logistic."""
    n, d = Z.shape
    Xb = np.column_stack([Z, np.ones(n)])
    beta = np.zeros(d + 1)
    penalty = np.full(d + 1, ridge)
    penalty[-1] = 0.0
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(Xb @ beta, -30, 30)))
        grad = Xb.T @ (w * (p - y)) + penalty * beta
        hess = (Xb * (w * p * (1 - p))[:, None]).T @ Xb + np.diag(penalty) + 1e-9 * np.eye(d + 1)
        step = np.linalg.solve(hess, grad)
        beta -= step
        if np.max(np.abs(step)) < tol:
            break
    return beta[:-1], beta[-1]


def model_dict(name, mean, std, coef, intercept):
    return {"name": name, "kind": "logistic", "base_columns": list(gap_team.BASE_COLUMNS),
            "window_s": gap_team.WINDOW_S, "features": gap_team.feature_names(),
            "mean": [float(v) for v in mean], "std": [float(v) for v in std],
            "coef": [float(v) for v in coef], "intercept": float(intercept)}


# -- data (cached) -----------------------------------------------------------------------------

def _cached(name, build):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, name + ".npz")
    if os.path.isfile(path):
        with np.load(path, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    out = build()
    np.savez_compressed(path, **out)
    return out


def barath_arrays():
    """Teacher outputs, ball presence, halves, tagging vote, exclusion mask and features."""
    def build():
        from pompeu.clean import clean_tracking
        from pompeu.io import gt as gt_io
        from pompeu.io.contract import load_tracking
        from pompeu.models import hsmm
        tracking = load_tracking("atd")
        cleaned, _ = clean_tracking(tracking, {})
        fps = cleaned.meta.get("fps", 25)
        prep = hsmm.prepare(cleaned)
        rules = pm.label_codes(np.asarray(prep["rules_labels"]))
        hmm = pm.label_codes(hsmm.decode_prepared(prep, {})["labels"])
        F = prep["F"]
        vote = np.where(F["st_vote_n"].to_numpy() > 0, np.sign(F["st_vote"].to_numpy()), 0.0)
        exclude = np.zeros(cleaned.n_frames, bool)
        margin = int(MARGIN_S * fps)
        for s, e in gt_io.graded_window(gt_io.load_gt_codes()):     # span only
            lo, hi = s - cleaned.frame0 - margin, e - cleaned.frame0 + margin
            exclude[max(lo, 0):max(min(hi, cleaned.n_frames), 0)] = True
        return {"rules": rules.astype(np.int8), "hmm": hmm.astype(np.int8),
                "ball": np.asarray(cleaned.ball_present, bool),
                "half": cleaned.half().astype(np.int8), "vote": vote.astype(np.int8),
                "exclude": exclude, "X": gap_team.design_matrix(F, fps).astype(np.float32),
                "fps": np.array(fps)}
    return _cached("barath_teacher", build)


def ext_train_arrays(seeds, stride):
    """DFL train rows exactly as scripts/train_gap_team.py builds them."""
    def build():
        import coverage_sweep
        import train_gap_team as tgt
        run_lengths = coverage_sweep.synthetic_run_lengths()
        names = [external.canonical(m) for m in external.SPLITS["train"]]
        names = [m for m in names if m.startswith("dfl:") and tgt.present(m)]
        Xs, ys = [], []
        for index, name in enumerate(names):
            for seed in seeds:
                X, y, _ = tgt.match_rows(name, index, seed, run_lengths, stride)
                Xs.append(X)
                ys.append(y)
        return {"X": np.concatenate(Xs), "y": np.concatenate(ys)}
    return _cached("ext_train_s{}_st{}".format("".join(map(str, seeds)), stride), build)


def val_prepared(seeds):
    """Yield (name, seed, dict) per val match and seed: gap info, features, teacher, prep."""
    import coverage_sweep
    import gap_team as exp
    from pompeu.clean import clean_tracking
    from pompeu.external import degrade
    from pompeu.models import hsmm
    run_lengths = coverage_sweep.synthetic_run_lengths()
    names = [external.canonical(m) for m in external.SPLITS["val"]]          # never "test"
    names = [m for m in names if exp.present(m)]
    for index, name in enumerate(names):
        tracking, labels = external.load(name)
        fps = tracking.meta.get("fps", 25)
        half = tracking.half()
        graded = np.where(half > 0, labels, pm.X).astype(np.int8)
        gt = external.align_labels(graded, MIN_POSSESSION_S, fps)
        for seed in seeds:
            t0 = time.time()
            degraded = degrade.degrade(tracking, labels, 1000 * seed + index, run_lengths, None)
            cleaned, _ = clean_tracking(degraded, {})
            prep = hsmm.prepare(cleaned)
            gid, age, length, carry, near = exp.gap_table(cleaned.ball_present, half, gt,
                                                          exp.nearest_team(cleaned))
            changed, post = exp.change_masks(gid, gt, carry)
            yield name, seed, {
                "prep": prep, "fps": fps, "gt": gt, "half": half,
                "ball": np.asarray(cleaned.ball_present, bool),
                "rules": pm.label_codes(np.asarray(prep["rules_labels"])),
                "X": gap_team.design_matrix(prep["F"], fps),
                "gid": np.where(gid >= 0, gid + 1_000_000 * (1000 * seed + index), -1),
                "length": length / fps, "carry": carry, "changed": changed, "post": post,
                "t_prep": time.time() - t0}


# -- report ------------------------------------------------------------------------------------

def gap_metrics(pred_away, d):
    """Accuracy dict on one val (match, seed) block; pred_away bool per frame."""
    keep = (d["gid"] >= 0) & np.isin(d["gt"], (H, A)) & (d["carry"] >= 0)
    y = d["gt"] == A
    hit = (pred_away == y)
    out = {"n": int(keep.sum()), "hit": int(np.sum(hit & keep))}
    for k, m in (("chg", keep & d["changed"]), ("post", keep & d["post"]),
                 ("nochg", keep & ~d["changed"]),
                 ("short", keep & (d["length"] < 5)), ("long", keep & (d["length"] >= 10))):
        out["n_" + k], out["hit_" + k] = int(m.sum()), int(np.sum(hit & m))
    ids = np.unique(d["gid"][keep & d["post"]])
    rec = 0
    for g in ids:
        m = keep & d["post"] & (d["gid"] == g)
        rec += np.mean(hit[m]) > 0.5
    out["n_gaps"], out["rec_gaps"] = len(ids), int(rec)
    return out


def add(acc, m):
    for k, v in m.items():
        acc[k] = acc.get(k, 0) + v


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="val seeds")
    parser.add_argument("--train-seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--no-downstream", action="store_true")
    parser.add_argument("--draws", type=int, default=500)
    args = parser.parse_args()
    started = time.time()

    t0 = time.time()
    bar = barath_arrays()
    fps = int(bar["fps"])
    print("BAR-ATH teacher arrays ready ({:.0f} s)".format(time.time() - t0), flush=True)
    t0 = time.time()
    ext = ext_train_arrays(args.train_seeds, STRIDE)
    Xe, ye = ext["X"].astype(np.float64), ext["y"].astype(np.float64)
    print("external train rows {} ({:.0f} s), away share {:.3f}".format(
        len(ye), time.time() - t0, ye.mean()), flush=True)

    # v1 refit sanity check against the committed JSON.
    v1 = gap_team.load_model("configs/gap_team_v1.json")
    Z, mean, std = gap_team.standardise(Xe)
    w1, b1 = gap_team.fit_logistic(Z, ye)
    print("v1 refit vs configs/gap_team_v1.json: max |coef diff| {:.2e}".format(
        float(np.max(np.abs(w1 - np.asarray(v1["coef"]))))))

    # Pseudo-labels.
    half, live = bar["half"], bar["half"] > 0
    print("\nBAR-ATH (in-half frames {}, ball after cleaning {:.1%}, excluded public window + "
          "margin {})".format(int(live.sum()), float(np.mean(bar["ball"][live])),
                              int(np.sum(bar["exclude"] & live))))
    print("  rules_v0 vs hsmm_v1 same team on live frames: {:.1%}".format(
        float(np.mean((bar["rules"] == bar["hmm"])[live & np.isin(bar["rules"], (H, A))]))))
    print("  {:<11}{:>9}{:>9}{:>9}{:>9}{:>9}{:>11}{:>8}{:>10}{:>10}".format(
        "variant", "(a)", "(b)", "overlap", "conflict", "a~b agr", "labelled", "home",
        "v1 agree", "min"))
    models = {"v1": v1}
    for name, N, use_tag, share in VARIANTS:
        lab, st = pseudo_labels(bar["rules"], bar["hmm"], bar["ball"], half, int(N * fps),
                                bar["vote"] if use_tag else None, bar["exclude"])
        rows = np.flatnonzero(lab >= 0)[::STRIDE]
        Xm = bar["X"][rows].astype(np.float64)
        ym = (lab[rows] == A).astype(np.float64)
        v1_logit = gap_team.standardise(Xm, np.asarray(v1["mean"]), np.asarray(v1["std"]))[0] \
            @ np.asarray(v1["coef"])
        v1_agree = float(np.mean((v1_logit > 0) == (ym > 0)))
        print("  {:<11}{:>9}{:>9}{:>9}{:>9}{:>9}{:>11}{:>8.3f}{:>10.3f}{:>10.1f}".format(
            name, st["a"], st.get("b", "-"), st.get("overlap", "-"), st.get("conflict", "-"),
            "{:.3f}".format(st["ab_agreement"]) if "ab_agreement" in st else "-",
            st["labelled"], st["home_share"], v1_agree, st["labelled"] / fps / 60))
        we, wm = mass_weights(len(ye), len(ym), share)
        X = np.vstack([Xe, Xm])
        y = np.concatenate([ye, ym])
        w = np.concatenate([we, wm])
        Zw, mu, sd = weighted_standardise(X, w)
        coef, b = fit_weighted_logistic(Zw, y, w)
        models[name] = model_dict(name, mu, sd, coef, b)
    print("  ((a) per frame; 'v1 agree' = v1 without intercept agrees with the pseudo-label on "
          "the used frames, in-sample for the self-trained models)")

    # Evaluation on val.
    print("\nVAL (never test), seeds {}".format(args.seeds), flush=True)
    import gap_team_weight as gtw
    from pompeu.models import hsmm
    gm = {}                    # (source, model) -> accumulated counts
    teacher = {}               # (source, N) -> [labelled, correct, ball frames]
    truths, preds = {}, {}
    for name, seed, d in val_prepared(args.seeds):
        src = name.split(":")[0]
        for mname, model in models.items():
            logit = gap_team.away_logit(d["prep"]["F"], d["fps"], model)
            m = gap_metrics(np.nan_to_num(logit) > 0, d)
            add(gm.setdefault((src, mname), {}), m)
            add(gm.setdefault(("all", mname), {}), m)
        hmm_codes = pm.label_codes(hsmm.decode_prepared(d["prep"], {})["labels"])
        for N in sorted({v[1] for v in VARIANTS}):
            lab, _ = pseudo_labels(d["rules"], hmm_codes, d["ball"], d["half"], int(N * d["fps"]))
            ok = lab >= 0
            t = teacher.setdefault((src, N), [0, 0, 0, 0])
            t[0] += int(ok.sum())
            t[1] += int(np.sum(lab[ok] == d["gt"][ok]))
            t[2] += int(np.sum(d["ball"] & np.isin(d["gt"], (H, A))))
            t[3] += int(np.sum(d["gt"][ok] == pm.D))
        if not args.no_downstream:
            truths.setdefault(name, d["gt"])
            preds.setdefault("hsmm_v1", {}).setdefault(seed, {})[name] = hmm_codes
            for mname, model in models.items():
                lab = hsmm.decode_prepared(d["prep"], {"gap_team_weight": GAP_WEIGHT,
                                                       "gap_team_model": model})["labels"]
                preds.setdefault(mname, {}).setdefault(seed, {})[name] = pm.label_codes(lab)
        print("  {} seed {} done (prep {:.0f} s)".format(name, seed, d["t_prep"]), flush=True)

    print("\nteacher pseudo-label rule (a) on degraded val (precision vs GT at {} s)".format(
        MIN_POSSESSION_S))
    for (src, N), (n, c, nb, nd) in sorted(teacher.items()):
        print("  {:<8} N={:g}s  labelled {:>7} ({:.0%} of live ball frames)  precision {:.3f}  "
              "of which GT=D {:.3f}".format(src, N, n, n / max(nb, 1), c / max(n, 1), nd / max(n, 1)))

    def r(a, b):
        return a / b if b else float("nan")
    for src in ("dfl", "metrica", "all"):
        if not any(k[0] == src for k in gm):
            continue
        print("\nVAL gap metrics, {} (logit without intercept, as in the HSMM)".format(src))
        print("  {:<11}{:>8}{:>9}{:>9}{:>9}{:>9}{:>9}{:>12}".format(
            "model", "all", "no-chg", "chg gap", "post", "<5s", ">=10s", "gaps rec"))
        for mname in models:
            m = gm[(src, mname)]
            print("  {:<11}{:>8.3f}{:>9.3f}{:>9.3f}{:>9.3f}{:>9.3f}{:>9.3f}{:>8.3f}/{}".format(
                mname, r(m["hit"], m["n"]), r(m["hit_nochg"], m["n_nochg"]),
                r(m["hit_chg"], m["n_chg"]), r(m["hit_post"], m["n_post"]),
                r(m["hit_short"], m["n_short"]), r(m["hit_long"], m["n_long"]),
                r(m["rec_gaps"], m["n_gaps"]), m["n_gaps"]))
        print("  frames {}".format(gm[(src, "v1")]["n"]))

    if not args.no_downstream:
        names = list(truths)
        truth_parts = [truths[n] for n in names]
        print("\nVAL downstream hsmm (gap_team_weight {:g}): spellF1 [5%, 95%]  macroF1 [5%, 95%]  "
              "poss/gt".format(GAP_WEIGHT))
        subsets = [("all val", None)] + [(n + " only", [i]) for i, n in enumerate(names)]
        for sname, keep in subsets:
            print("  {}".format(sname))
            for c in ["hsmm_v1"] + list(models):
                by_seed = {s: [parts[n] for n in names] for s, parts in preds[c].items()}
                res = gtw.score(truth_parts, by_seed, args.draws, keep)
                print("    {:<12} {}".format("hsmm_v1 w=0" if c == "hsmm_v1" else c,
                                             gtw.fmt(res)), flush=True)
    print("\nruntime {:.0f} s".format(time.time() - started))


if __name__ == "__main__":
    main()
