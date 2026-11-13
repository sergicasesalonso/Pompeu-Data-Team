"""Real BAR-ATH Smart Tagging vs our synthetic tags: GT-free aggregates, and how much tags move the TCN.

    python scripts/experiments/tcn_tags_real.py --model cache/tcn/models/tcn_r2.pt --seeds 0 1 2 3 4

WHY. The TCN's gain over hsmm_v1_fit_mbr on external val leans on synthetic tags generated from
the true labels. If BAR-ATH's real tags carry less information, the gain will not transfer.

SIDES, measured with the SAME code and the SAME GT-free reference on both:
  * real:  BAR-ATH, the provided ATD (A1 cleaning), its real Smart Tagging;
  * synth: external `val` (never `test`), degraded exactly as tcn_val.py / robust_val.py (seed
           1000*seed + index, the ATD's no-ball run lengths), synthetic tags with tag seed
           TAG_SEED + degradation seed.
  The reference ("proxy") is rules_v0's labels on that same tracking, WITHOUT any tagging input
  (touches + control state machine from the ATD ball). Nothing compares to the public 5 minutes.
  On val the same statistics are also given against the true labels, to read the proxy numbers.

PART 1 tag statistics: tags per minute by code; per-tag team agreement with the proxy where the
ball exists; tag-start offsets to the nearest proxy possession start; frames within X s of a tag
boundary; gaps between tag phases and uncovered runs; coverage of ball-present / ball-missing
frames; the 18 tag input columns (mean, share non-zero, share NaN).

PART 2 the decisive check: the TCN with its tag inputs vs the same TCN with NO tags (an empty
tagging: every tag column as on an untagged stretch). Rates of change of its per-frame posterior
and of HSMM+TCN's labels (hsmm_v1_fit_mbr prepared without tags, TCN expert wt/wd), on BAR-ATH
(real tags) and on val (synthetic tags).

PART 3 (--sensitivity) on val only: the synthetic tags degraded -- phases dropped (25 / 50 %),
shifted in time (sigma 3 / 6 s), team flipped (10 / 20 %), all tags removed -- and HSMM+TCN /
TCN-alone spellF1 at 2/5/8 s against rules_v0 and hsmm_v1_fit_mbr.

Only aggregate numbers are printed. Nothing per frame is written.
"""

import argparse
import json
import os
import sys
import time
from multiprocessing import get_context

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "experiments"))
from pompeu import external, paths  # noqa: E402
from pompeu.experimental import tcn  # noqa: E402
from pompeu.validate import harness  # noqa: E402
import possession_metrics as pm  # noqa: E402

VALUES = (2.0, 5.0, 8.0)
NEAR_S = (2.0, 5.0, 10.0)
PERTURB = (("none", None), ("drop25", 0.25), ("drop50", 0.50), ("jit3", 3.0), ("jit6", 6.0),
           ("flip10", 0.10), ("flip20", 0.20), ("notags", None))
_STATE = {}


# -- tags ----------------------------------------------------------------------------------------

def empty_tagging():
    return pd.DataFrame({"code": pd.Series([], dtype=object),
                         "start_frame": pd.Series([], dtype=np.int64),
                         "end_frame": pd.Series([], dtype=np.int64),
                         "team": pd.Series([], dtype=np.int64),
                         "type": pd.Series([], dtype=object),
                         "half": pd.Series([], dtype=np.int64),
                         "tags": pd.Series([], dtype=object)})


def like_synthetic(tags):
    """Real tags restricted to what the generator produces: possession phases, and set pieces of
    the generator's four types. Zone / goal / kick-off codes never occur in training (their input
    columns are constant 0 there), so on BAR-ATH they would be out-of-distribution inputs."""
    from pompeu.external.synthetic_tags import ATTACK_CODES, DEFEND_CODES, load_error_model
    types = set(load_error_model()["set_piece_types"])
    keep = (tags["code"].isin([c for c in ATTACK_CODES + DEFEND_CODES if c != "SET PIECES"])
            | ((tags["code"] == "SET PIECES") & tags["type"].isin(types)))
    return tags[keep].reset_index(drop=True)


def perturb(tags, kind, value, seed, n, fps):
    """Synthetic tags degraded by phase (an attacking phase and its paired defending phase share
    the span, so they are dropped / shifted / flipped together)."""
    if kind == "none":
        return tags
    if kind == "notags":
        return empty_tagging()
    rng = np.random.default_rng(seed)
    out = tags.copy().reset_index(drop=True)
    key = list(zip(out["start_frame"], out["end_frame"], out["half"]))
    groups = sorted(set(key))
    gid = np.array([groups.index(k) for k in key]) if len(key) else np.zeros(0, int)
    u = rng.random(len(groups))
    if kind.startswith("drop"):
        out = out[u[gid] >= value]
    elif kind.startswith("flip"):
        flip = (u[gid] < value) & (out["team"].to_numpy() >= 0)
        out.loc[flip, "team"] = 1 - out.loc[flip, "team"]
    elif kind.startswith("jit"):
        shift = np.round(rng.normal(0.0, value, len(groups)) * fps).astype(int)[gid]
        out["start_frame"] = np.clip(out["start_frame"].to_numpy() + shift, 0, n - 1)
        out["end_frame"] = np.clip(out["end_frame"].to_numpy() + shift, 1, n)
        out = out[out["end_frame"] > out["start_frame"]]
    return out.sort_values(["start_frame", "end_frame", "code"], kind="stable").reset_index(drop=True)


def runs(values):
    values = np.asarray(values)
    if not len(values):
        return []
    breaks = np.flatnonzero(values[1:] != values[:-1]) + 1
    bounds = np.concatenate(([0], breaks, [len(values)]))
    return [(int(a), int(b), values[a].item()) for a, b in zip(bounds[:-1], bounds[1:])]


def tag_stats(tags, ref, ball, graded, frame0, fps, ctx):
    """Summable aggregates of one tagging against a reference labelling (codes 0..3).

    ref: (N,) H=0/A=1/D=2/X=3; ball: (N,) bool ball in the tracking; graded: (N,) bool in a half.
    """
    from pompeu.external.synthetic_tags import ATTACK_CODES, DEFEND_CODES
    n = len(ref)
    codes = tags["code"].to_numpy()
    team = tags["team"].to_numpy()
    a = np.clip(tags["start_frame"].to_numpy() - frame0, 0, n)
    b = np.clip(tags["end_frame"].to_numpy() - frame0, 0, n)
    inside = (b > a) & graded[np.clip(a, 0, n - 1)]
    voting = np.isin(codes, ATTACK_CODES + DEFEND_CODES) & (team >= 0) & inside
    poss = np.where(np.isin(codes, ATTACK_CODES), team, 1 - team)
    out = {"minutes": float(graded.sum()) / fps / 60.0,
           "per_code": {}, "team_agree": 0, "team_checked": 0, "team_unverifiable": 0,
           "start_off": [], "start_off_any": [], "phase_gaps": [], "uncovered_runs": []}
    for c in np.unique(codes[inside]) if inside.any() else []:
        out["per_code"][str(c)] = int(np.sum((codes == c) & inside))
    # Team of each voting tag vs the reference where the ball is in the tracking and the
    # reference says H or A.
    live_ball = ball & (ref <= 1) & graded
    for i in np.flatnonzero(voting):
        sl = slice(a[i], b[i])
        m = live_ball[sl]
        if m.sum() < 0.2 * fps:
            out["team_unverifiable"] += 1
            continue
        major = 0 if np.mean(ref[sl][m] == 0) >= 0.5 else 1
        out["team_checked"] += 1
        out["team_agree"] += int(major == poss[i])
    # Tag starts (attacking phases, one per distinct start) vs the reference's possession starts.
    spell_starts = {0: [], 1: []}
    for s, e, v in runs(np.where(graded, ref, 3)):
        if v in (0, 1) and e - s >= fps:                      # spells of at least 1 s
            spell_starts[v].append(s)
    allst = np.array(sorted(spell_starts[0] + spell_starts[1]), float)
    st = {t: np.array(v, float) for t, v in spell_starts.items()}
    att = voting & np.isin(codes, ATTACK_CODES)
    seen = set()
    for i in np.flatnonzero(att):
        if (a[i], poss[i]) in seen:
            continue
        seen.add((a[i], poss[i]))
        cand = st[int(poss[i])]
        if len(cand):
            d = (a[i] - cand) / fps
            j = np.argmin(np.abs(d))
            if abs(d[j]) <= 30.0:
                out["start_off"].append(float(d[j]))
        if len(allst):
            d = (a[i] - allst) / fps
            out["start_off_any"].append(float(d[np.argmin(np.abs(d))]))
    # Phase starts (distinct, voting) and their gaps; uncovered runs inside halves.
    starts = np.unique(a[voting])
    out["phase_gaps"] = list(np.diff(starts) / fps) if len(starts) > 1 else []
    covered = np.asarray(ctx["st_vote_n"], float) > 0
    for s, e, v in runs(np.where(graded, covered.astype(int), 2)):
        if v == 0:
            out["uncovered_runs"].append((e - s) / fps)
    # Frame-level quantities.
    bounds = np.zeros(n + 1)
    for i in np.flatnonzero(voting):
        bounds[a[i]] = 1
        bounds[min(b[i], n)] = 1
    pos = np.flatnonzero(bounds[:n] > 0)
    frames = np.arange(n)
    if len(pos):
        j = np.clip(np.searchsorted(pos, frames), 1, len(pos)) - 1
        k = np.clip(j + 1, 0, len(pos) - 1)
        dist = np.minimum(np.abs(frames - pos[j]), np.abs(frames - pos[k])) / fps
    else:
        dist = np.full(n, np.inf)
    vote = np.nan_to_num(np.asarray(ctx["st_vote"], float))
    live_ref = graded & (ref <= 1)
    sign_ok = np.sign(vote) == np.where(ref == 0, 1.0, -1.0)
    out["frames"] = {
        "graded": int(graded.sum()),
        "covered": int((covered & graded).sum()),
        "ball": int((ball & graded).sum()), "ball_cov": int((ball & graded & covered).sum()),
        "noball": int((~ball & graded).sum()), "noball_cov": int((~ball & graded & covered).sum()),
        "near": {str(x): int((graded & (dist <= x)).sum()) for x in NEAR_S},
        "sign_live_ball": int((live_ref & ball & covered).sum()),
        "sign_ok_ball": int((live_ref & ball & covered & sign_ok).sum()),
        "sign_live_noball": int((live_ref & ~ball & covered).sum()),
        "sign_ok_noball": int((live_ref & ~ball & covered & sign_ok).sum()),
    }
    out["columns"] = {}
    for name, col in ctx.items():
        v = np.asarray(col, float)[graded]
        fin = np.isfinite(v)
        out["columns"][name] = [float(np.nansum(v)), int(fin.sum()), int((fin & (v != 0)).sum()),
                                int((~fin).sum())]
    return out


def merge_stats(parts):
    out = {"minutes": 0.0, "per_code": {}, "team_agree": 0, "team_checked": 0,
           "team_unverifiable": 0, "start_off": [], "start_off_any": [], "phase_gaps": [],
           "uncovered_runs": [], "frames": {}, "columns": {}}
    for p in parts:
        out["minutes"] += p["minutes"]
        for c, k in p["per_code"].items():
            out["per_code"][c] = out["per_code"].get(c, 0) + k
        for key in ("team_agree", "team_checked", "team_unverifiable"):
            out[key] += p[key]
        for key in ("start_off", "start_off_any", "phase_gaps", "uncovered_runs"):
            out[key] += list(p[key])
        for key, v in p["frames"].items():
            if isinstance(v, dict):
                d = out["frames"].setdefault(key, {})
                for kk, vv in v.items():
                    d[kk] = d.get(kk, 0) + vv
            else:
                out["frames"][key] = out["frames"].get(key, 0) + v
        for c, v in p["columns"].items():
            acc = out["columns"].setdefault(c, [0.0, 0, 0, 0])
            for i in range(4):
                acc[i] += v[i]
    return out


def q(values, ps=(25, 50, 75, 90)):
    v = np.asarray(values, float)
    if not len(v):
        return "n=0"
    return "n={} ".format(len(v)) + " ".join("p{}={:.1f}".format(p, np.percentile(v, p)) for p in ps)


def summarise(s):
    f = s["frames"]
    r = {"tags_per_min": {c: k / s["minutes"] for c, k in sorted(s["per_code"].items())},
         "team_agree": s["team_agree"] / max(1, s["team_checked"]),
         "team_checked": s["team_checked"], "team_unverifiable": s["team_unverifiable"],
         "start_off_same_team": q(s["start_off"]),
         "start_off_within5s": float(np.mean(np.abs(s["start_off"]) <= 5)) if s["start_off"] else 0,
         "start_off_any": q(s["start_off_any"]),
         "phase_gaps_s": q(s["phase_gaps"]), "uncovered_runs_s": q(s["uncovered_runs"]),
         "uncovered_share_in_runs_over_30s": float(np.sum([x for x in s["uncovered_runs"] if x > 30])
                                                   / max(1e-9, np.sum(s["uncovered_runs"]))),
         "coverage": f["covered"] / f["graded"],
         "coverage_ball": f["ball_cov"] / max(1, f["ball"]),
         "coverage_noball": f["noball_cov"] / max(1, f["noball"]),
         "noball_share": f["noball"] / f["graded"],
         "near": {k: v / f["graded"] for k, v in f["near"].items()},
         "sign_agree_ball": f["sign_ok_ball"] / max(1, f["sign_live_ball"]),
         "sign_agree_noball": f["sign_ok_noball"] / max(1, f["sign_live_noball"])}
    r["columns"] = {c: {"mean": v[0] / max(1, v[1]), "nonzero": v[2] / max(1, v[1] + v[3]),
                        "nan": v[3] / max(1, v[1] + v[3])} for c, v in s["columns"].items()}
    return r


# -- the TCN with and without tags --------------------------------------------------------------

def tcn_logp(clean, tags, F):
    X, names = tcn.inputs_from(clean, tags, F)
    return tcn.predict_frames(_STATE["model"], _STATE["norm"], X, names, clean.n_frames), \
        dict(zip(names, X.T))


def change_stats(lp_with, lp_without, lab_with, lab_without, graded, ball, ref):
    """Summable counts of how much the tag inputs move the TCN and HSMM+TCN, and whether the move
    goes towards the reference (ref codes; live frames where ref says H or A)."""
    lo_w = lp_with[:, 0] - lp_with[:, 1]
    lo_o = lp_without[:, 0] - lp_without[:, 1]
    am_w, am_o = np.argmax(lp_with, 1), np.argmax(lp_without, 1)
    sm_w, sm_o = tcn.smooth_labels(lp_with, 100), tcn.smooth_labels(lp_without, 100)
    out = {}
    for tag, m in (("all", graded), ("ball", graded & ball), ("noball", graded & ~ball)):
        live = m & (am_w <= 1) & (am_o <= 1)
        out[tag] = {
            "n": int(m.sum()),
            "abs_dlogodds": float(np.abs(lo_w - lo_o)[m].sum()),
            "abs_logodds_with": float(np.abs(lo_w)[m].sum()),
            "abs_logodds_without": float(np.abs(lo_o)[m].sum()),
            "argmax_changed": int((am_w != am_o)[m].sum()),
            "live_both": int(live.sum()),
            "team_flipped": int((np.sign(lo_w) != np.sign(lo_o))[live].sum()),
            "smooth100_changed": int((sm_w != sm_o)[m].sum()),
            "hsmm_tcn_changed": int((lab_with != lab_without)[m].sum()),
            "ref_live": int((m & (ref <= 1)).sum()),
            "tcn_agree_with": int((m & (ref <= 1) & (am_w == ref)).sum()),
            "tcn_agree_without": int((m & (ref <= 1) & (am_o == ref)).sum()),
            "hsmm_tcn_agree_with": int((m & (ref <= 1) & (lab_with == ref)).sum()),
            "hsmm_tcn_agree_without": int((m & (ref <= 1) & (lab_without == ref)).sum()),
        }
    return out


def merge_change(parts):
    out = {}
    for p in parts:
        for tag, d in p.items():
            acc = out.setdefault(tag, {})
            for k, v in d.items():
                acc[k] = acc.get(k, 0) + v
    return out


def summarise_change(c):
    out = {}
    for tag, d in c.items():
        n = max(1, d["n"])
        out[tag] = {"mean_abs_dlogodds": d["abs_dlogodds"] / n,
                    "mean_abs_logodds_with": d["abs_logodds_with"] / n,
                    "mean_abs_logodds_without": d["abs_logodds_without"] / n,
                    "argmax_changed": d["argmax_changed"] / n,
                    "team_flipped_live": d["team_flipped"] / max(1, d["live_both"]),
                    "smooth100_changed": d["smooth100_changed"] / n,
                    "hsmm_tcn_changed": d["hsmm_tcn_changed"] / n}
        r = max(1, d["ref_live"])
        for k in ("tcn_agree_with", "tcn_agree_without", "hsmm_tcn_agree_with",
                  "hsmm_tcn_agree_without"):
            out[tag][k] = d[k] / r
    return out


# -- workers -------------------------------------------------------------------------------------

def _init(model_path, threads, wt, wd):
    import torch
    torch.set_num_threads(threads)
    from pompeu.models import deadball_gbm
    if not os.path.isfile(deadball_gbm.ARTIFACT):
        art = os.path.join(paths.ROOT, "artifacts", "deadball_gbm_v1.txt")
        deadball_gbm.load.__defaults__ = (art,)
        deadball_gbm.ARTIFACT = art
    from pompeu.io.contract import load_tracking
    from pompeu.validate.stress import missing_run_lengths
    _STATE["run_lengths"] = missing_run_lengths(load_tracking("atd"))
    _STATE["model"], _STATE["norm"] = tcn.load(model_path, "cpu")
    _STATE["rules"] = harness.load_config(os.path.join(paths.CONFIGS_DIR, "rules_v0.yaml"))
    _STATE["hsmm"] = harness.load_config(os.path.join(paths.CONFIGS_DIR, "hsmm_v1_fit_mbr.yaml"))
    _STATE["w"] = (wt, wd)


def _decode_hsmm_tcn(prep, base, log_p):
    from pompeu.models import hsmm
    prep["learned"] = {"team": log_p[:, 0] - log_p[:, 1],
                       "dead": log_p[:, 2] - np.logaddexp(log_p[:, 0], log_p[:, 1])}
    wt, wd = _STATE["w"]
    out = hsmm.decode_prepared(prep, dict(base, w_learned_team=wt, w_learned_dead=wd))["labels"]
    prep["learned"] = {}
    return pm.label_codes(out)


def _common(raw, tags_for_proxy_off):
    """(proxy codes, clean, prep without tags, A2 features) for one raw tracking."""
    from pompeu import pipeline
    from pompeu.clean import clean_tracking
    from pompeu.external import synthetic_tags as st
    from pompeu.features.build import build_features
    from pompeu.models import hsmm
    hcfg = _STATE["hsmm"]["pipeline"]
    params = hcfg["params"]
    with st.injected(tags_for_proxy_off):          # no tagging reaches rules_v0 / the HSMM
        proxy = pm.label_codes(pipeline.predict(raw, _STATE["rules"]["pipeline"]))
        clean, _ = clean_tracking(raw, hcfg.get("clean") or {})
        prep = hsmm.prepare(clean, params.get("deadball") or None, params.get("touches") or None)
    F = build_features(clean, tagging=None)
    return proxy, clean, prep, F


def _job_real(_):
    import games
    from pompeu.io.contract import load_tracking
    from pompeu.io.tagging import load_tagging
    started = time.time()
    raw = load_tracking("atd")
    game = raw.meta.get("game", games.GAME)
    fps = raw.meta.get("fps", 25)
    proxy, clean, prep, F = _common(raw, {game: None})
    base = dict(_STATE["hsmm"]["pipeline"]["params"].get("hsmm") or {}, w_tagging=0.0)
    graded = clean.half() > 0
    ball = raw.ball_present
    from pompeu.features import context
    lp_o, _ = tcn_logp(clean, empty_tagging(), F)
    lab_o = _decode_hsmm_tcn(prep, base, lp_o)
    out = {"side": "real"}
    raw_tags = load_tagging(raw.meta["team_names"], game)
    for key, tags in (("raw", raw_tags), ("filtered", like_synthetic(raw_tags))):
        lp_w, _ = tcn_logp(clean, tags, F)
        ctx = context.context_features(clean, tags)
        ctx = {k: ctx[k] for k in context.TAGGING_COLUMNS if k in ctx}
        out["stats_" + key] = tag_stats(tags, proxy, ball, graded, raw.frame0, fps, ctx)
        lab_w = _decode_hsmm_tcn(prep, base, lp_w)
        out["change_" + key] = change_stats(lp_w, lp_o, lab_w, lab_o, graded, ball, proxy)
    out["secs"] = time.time() - started
    return out


def _job_val(job):
    index, name, seed, sensitivity = job
    from pompeu.external import degrade, synthetic_tags as st
    started = time.time()
    tracking, labels = external.load(name)
    fps = tracking.meta.get("fps", 25)
    half = tracking.half()
    graded_codes = np.where(half > 0, labels, pm.X).astype(np.int8)
    dseed = 1000 * seed + index
    raw = degrade.degrade(tracking, labels, dseed, _STATE["run_lengths"], None)
    game = raw.meta.get("game")
    proxy, clean, prep, F = _common(raw, {game: None})
    base = dict(_STATE["hsmm"]["pipeline"]["params"].get("hsmm") or {})
    graded = half > 0
    ball = raw.ball_present
    tags = st.generate(external.align_labels(graded_codes, 2.0, fps), half, st.load_error_model(),
                       seed=tcn.TAG_SEED + dseed, fps=fps)
    lp_w, _ = tcn_logp(clean, tags, F)
    lp_o, _ = tcn_logp(clean, empty_tagging(), F)
    from pompeu.features import context
    ctx = st.context_features(clean, tags)
    ctx = {k: ctx[k] for k in context.TAGGING_COLUMNS if k in ctx}
    gt2 = external.align_labels(graded_codes, 2.0, fps)
    stats = tag_stats(tags, proxy, ball, graded, raw.frame0, fps, ctx)
    stats_gt = tag_stats(tags, gt2, ball, graded, raw.frame0, fps, ctx)
    lab_w = _decode_hsmm_tcn(prep, base, lp_w)
    lab_o = _decode_hsmm_tcn(prep, base, lp_o)
    change = change_stats(lp_w, lp_o, lab_w, lab_o, graded, ball, proxy)
    change_gt = change_stats(lp_w, lp_o, lab_w, lab_o, graded, ball, gt2)
    preds = {}
    if sensitivity:
        from pompeu.decode.policy import finalize
        from pompeu.models import hsmm
        from pompeu import pipeline
        preds["rules_v0"] = pm.label_codes(pipeline.predict(raw, _STATE["rules"]["pipeline"]))
        preds["hsmm_v1_fit_mbr"] = pm.label_codes(hsmm.decode_prepared(prep, dict(base))["labels"])
        policy = _STATE["hsmm"]["pipeline"].get("policy") or {}
        for k, (kind, value) in enumerate(PERTURB):
            if kind == "none":
                lp = lp_w
            elif kind == "notags":
                lp = lp_o
            else:
                t2 = perturb(tags, kind, value, 7 * dseed + 1000003 * k + 11, clean.n_frames, fps)
                lp, _ = tcn_logp(clean, t2, F)
            preds["hsmm+tcn " + kind] = _decode_hsmm_tcn(prep, base, lp)
            lab = np.array(["H", "A", "D"])[tcn.smooth_labels(lp, 100)]
            preds["tcn s=100 " + kind] = pm.label_codes(finalize(lab, clean, policy))
    return {"side": "val", "index": index, "seed": seed, "stats": stats, "stats_gt": stats_gt,
            "change": change, "change_gt": change_gt, "preds": preds, "secs": time.time() - started}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--wt", type=float, default=1.0)
    parser.add_argument("--wd", type=float, default=2.0)
    parser.add_argument("--sensitivity", action="store_true")
    parser.add_argument("--json", default=None)
    args = parser.parse_args()
    from gap_team_weight import present, score
    from robust_val import fmt
    started = time.time()
    names = [external.canonical(m) for m in external.SPLITS["val"]]          # never "test"
    matches = [m for m in names if present(m)]
    print("val {}, seeds {}, model {}, HSMM+TCN {:g}/{:g} MBR".format(
        matches, args.seeds, os.path.basename(args.model), args.wt, args.wd), flush=True)
    jobs = [("real",)] + [(i, m, s, args.sensitivity) for i, m in enumerate(matches)
                          for s in args.seeds]
    results = []
    with get_context("spawn").Pool(args.workers, initializer=_init,
                                   initargs=(args.model, args.threads, args.wt, args.wd)) as pool:
        handles = [pool.apply_async(_job_real, (None,))] + \
                  [pool.apply_async(_job_val, (j,)) for j in jobs[1:]]
        for h in handles:
            r = h.get()
            results.append(r)
            print("  {} {} ({:.0f} s)".format(r["side"], "" if r["side"] == "real" else
                                               "{} seed {}".format(matches[r["index"]], r["seed"]),
                                               r["secs"]), flush=True)
    real = [r for r in results if r["side"] == "real"][0]
    vals = [r for r in results if r["side"] == "val"]
    report = {
        "real_vs_proxy": summarise(merge_stats([real["stats_raw"]])),
        "real_filtered_vs_proxy": summarise(merge_stats([real["stats_filtered"]])),
        "synth_vs_proxy": summarise(merge_stats([r["stats"] for r in vals])),
        "synth_vs_gt": summarise(merge_stats([r["stats_gt"] for r in vals])),
        "change_real": summarise_change(merge_change([real["change_raw"]])),
        "change_real_filtered": summarise_change(merge_change([real["change_filtered"]])),
        "change_synth": summarise_change(merge_change([r["change"] for r in vals])),
        "change_synth_vs_gt": summarise_change(merge_change([r["change_gt"] for r in vals])),
    }
    print(json.dumps({k: v for k, v in report.items()}, indent=1, default=float))
    if args.sensitivity:
        truths = {v: [] for v in VALUES}
        for name in matches:
            tracking, labels = external.load(name)
            graded = np.where(tracking.half() > 0, labels, pm.X).astype(np.int8)
            for v in VALUES:
                truths[v].append(external.align_labels(graded, v, tracking.meta.get("fps", 25)))
        preds = {}
        for r in vals:
            for c, codes in r["preds"].items():
                preds.setdefault(c, {}).setdefault(r["seed"], {})[r["index"]] = codes
        preds = {c: {s: [d[i] for i in range(len(matches))] for s, d in by.items()}
                 for c, by in preds.items()}
        res = {c: {v: score(truths[v], preds[c], args.draws) for v in VALUES} for c in preds}
        order = ["rules_v0", "hsmm_v1_fit_mbr"] + ["hsmm+tcn " + k for k, _ in PERTURB] + \
                ["tcn s=100 " + k for k, _ in PERTURB]
        print("\nsensitivity (val, seeds {}): spellF1 2 / 5 / 8 s, mean, worst loss vs rules_v0, "
              "mean macroF1".format(args.seeds))
        sens = {}
        for c in order:
            spell = [res[c][v]["spell"] for v in VALUES]
            loss = max(res["rules_v0"][v]["spell"] - res[c][v]["spell"] for v in VALUES)
            macro = float(np.mean([res[c][v]["macro"] for v in VALUES]))
            sens[c] = {"spell": spell, "mean": float(np.mean(spell)), "worst_loss": float(loss),
                       "macro": macro, "by_value": {str(v): res[c][v] for v in VALUES}}
            print("  {:<22} {:.3f} / {:.3f} / {:.3f}   {:.4f}  {:+.4f}  {:.4f}".format(
                c, *spell, np.mean(spell), loss, macro))
            for v in VALUES:
                print("      {:g} s  {}".format(v, fmt(res[c][v])))
        report["sensitivity"] = sens
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=1, default=float)
    print("runtime {:.0f} s".format(time.time() - started))


if __name__ == "__main__":
    main()
