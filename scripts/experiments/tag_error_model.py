"""Smart Tagging error model on BAR-ATH, measured WITHOUT any ground truth.

    python scripts/experiments/tag_error_model.py            # writes configs/synthetic_tags_v1.json

Reference = our own algorithms where they are confident: rules_v0 and hsmm_v1 WITHOUT the
tagging expert (w_tagging 0, so the reference never sees the tags it is compared with) agree on
H or A, the (cleaned) ATD ball is visible, inside a half. The public 5-minute window (span of the
public ground-truth file's graded frames, only its span is read, never a label) is excluded with
a 60 s margin on each side, from every statistic.

Possession spells of the reference: maximal runs of the same team in hsmm_v1 (no tags), kept
when >= MIN_SPELL_S long, >= 80% of their frames agree with rules_v0 and >= 50% have the ball.
Tag spells: the team-voting phases (attacking codes -> their team, defending codes -> the other
team), merged per possessing team when they overlap or touch within MERGE_S.

Everything printed and written is aggregate. The two predictions are cached under
POMPEU_CACHE_DIR/syntags/ (derived per-frame data, never committed).
"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from pompeu import paths  # noqa: E402
from pompeu.clean import clean_tracking  # noqa: E402
from pompeu.external import synthetic_tags  # noqa: E402
from pompeu.features import context  # noqa: E402
from pompeu.io import gt as gt_io  # noqa: E402
from pompeu.io.contract import load_tracking  # noqa: E402
from pompeu.io.tagging import load_tagging  # noqa: E402
from pompeu.models import hsmm, rules  # noqa: E402
import games  # noqa: E402

# Params
MARGIN_S = 60.0
MIN_SPELL_S = 2.0
MERGE_S = 1.0
OUT = os.path.join(paths.CONFIGS_DIR, "synthetic_tags_v1.json")


def runs(values):
    """[(start, stop, value)] maximal runs of equal values."""
    values = np.asarray(values)
    if not len(values):
        return []
    breaks = np.flatnonzero(values[1:] != values[:-1]) + 1
    bounds = np.concatenate(([0], breaks, [len(values)]))
    return [(int(a), int(b), values[a]) for a, b in zip(bounds[:-1], bounds[1:])]


def merge(intervals, gap):
    out = []
    for a, b in sorted(intervals):
        if out and a <= out[-1][1] + gap:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [tuple(x) for x in out]


def q(values, scale=1.0):
    v = np.asarray(values, float) * scale
    if not len(v):
        return {"n": 0}
    return {"n": int(len(v)), "median": round(float(np.median(v)), 3),
            "q25": round(float(np.percentile(v, 25)), 3),
            "q75": round(float(np.percentile(v, 75)), 3),
            "mean": round(float(np.mean(v)), 3), "std": round(float(np.std(v)), 3)}


def predictions(tracking):
    cache = os.path.join(paths.CACHE_DIR, "syntags")
    os.makedirs(cache, exist_ok=True)
    path = os.path.join(cache, "reference_predictions_v1.npz")
    if os.path.isfile(path):
        with np.load(path) as z:
            return z["rules"], z["hsmm"]
    clean, _ = clean_tracking(tracking, {})
    r = np.asarray(rules.predict(clean)).astype("<U1")
    h = np.asarray(hsmm.predict(clean, {"w_tagging": 0.0})).astype("<U1")
    np.savez_compressed(path, rules=r, hsmm=h, ball=clean.ball_present)
    return r, h


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=OUT)
    args = parser.parse_args()

    tracking = load_tracking("atd")
    fps = tracking.meta.get("fps", 25)
    n = tracking.n_frames
    r_lab, h_lab = predictions(tracking)
    with np.load(os.path.join(paths.CACHE_DIR, "syntags", "reference_predictions_v1.npz")) as z:
        ball = z["ball"].astype(bool)
    half = tracking.half()

    # Public window: span only.
    spans = gt_io.graded_window(gt_io.load_gt_codes())
    excluded = np.zeros(n, bool)
    for a, b in spans:
        excluded[max(0, a - int(MARGIN_S * fps)):min(n, b + int(MARGIN_S * fps))] = True
    usable = (half > 0) & ~excluded

    team = np.where(h_lab == "H", 0, np.where(h_lab == "A", 1, -1))
    agree = (r_lab == h_lab) & usable
    conf_live = agree & ball & (team >= 0)
    conf_dead = agree & (h_lab == "D")

    tagging = load_tagging(tracking.meta["team_names"])
    codes = tagging["code"].to_numpy()
    tteam = tagging["team"].to_numpy()
    starts = np.clip(tagging["start_frame"].to_numpy(), 0, n)
    stops = np.clip(tagging["end_frame"].to_numpy(), 0, n)
    votes = np.array([context.possession_vote(c, t) for c, t in zip(codes, tteam)])
    is_attack = np.isin(codes, context.ATTACK_CODES)
    is_defend = np.isin(codes, context.DEFEND_CODES)
    voting = (is_attack | is_defend) & (tteam >= 0)
    # Keep tags whose midpoint lies in a usable frame (outside the public window, in a half).
    mid = np.clip((starts + stops) // 2, 0, n - 1)
    keep_tag = usable[mid]

    F = context.context_features(tracking, tagging)
    st_vote = F["st_vote"]
    st_n = F["st_vote_n"]
    sp_active = F["st_active"]

    out = {"source": "BAR-ATH Smart Tagging vs rules_v0 & hsmm_v1(w_tagging 0) consensus, "
                     "ATD ball visible, public window +-{:g} s excluded".format(MARGIN_S),
           "fps": fps}
    usable_live = usable & (team >= 0)
    out["frames"] = {"usable": int(usable.sum()), "usable_hsmm_live": int(usable_live.sum()),
                     "confident_live": int(conf_live.sum()), "confident_dead": int(conf_dead.sum())}

    # -- coverage and agreement (frame level) --
    cov = st_n > 0
    any_tag = sp_active > 0
    out["coverage"] = {
        "confident_live_under_voting_tag": round(float(cov[conf_live].mean()), 4),
        "hsmm_live_under_voting_tag": round(float(cov[usable_live].mean()), 4),
        "confident_live_under_any_tag": round(float(any_tag[conf_live].mean()), 4),
        "confident_dead_under_voting_tag": round(float(cov[conf_dead].mean()), 4),
        "usable_under_voting_tag": round(float(cov[usable].mean()), 4),
    }
    ref_sign = np.where(team == 0, 1.0, -1.0)
    m = conf_live & cov
    out["agreement_frames"] = {
        "sign_agrees": round(float((np.sign(st_vote[m]) == ref_sign[m]).mean()), 4),
        "sign_disagrees": round(float((np.sign(st_vote[m]) == -ref_sign[m]).mean()), 4),
        "tied_zero": round(float((st_vote[m] == 0).mean()), 4),
        "mean_votes_when_covered": round(float(st_n[m].mean()), 3),
        "two_or_more_votes": round(float((st_n[m] >= 2).mean()), 4),
    }

    # -- per-tag statistics --
    per_code = {}
    for code in sorted(set(codes[keep_tag])):
        sel = keep_tag & (codes == code)
        dur = (stops[sel] - starts[sel]) / fps
        rec = {"n": int(sel.sum()), "duration_s": q(dur),
               "team_na": round(float((tteam[sel] < 0).mean()), 4)}
        if code in context.ATTACK_CODES + context.DEFEND_CODES:
            agr, live_share = [], []
            for a, b, v in zip(starts[sel], stops[sel], votes[sel]):
                cl = conf_live[a:b]
                cd = conf_dead[a:b]
                if cl.sum() + cd.sum():
                    live_share.append(cl.sum() / (cl.sum() + cd.sum()))
                if cl.sum() and v != 0:
                    agr.append(float(np.mean(ref_sign[a:b][cl] == np.sign(v))))
            rec["frame_agreement"] = round(float(np.mean(agr)), 4) if agr else None
            rec["tags_majority_wrong"] = (round(float(np.mean(np.array(agr) < 0.5)), 4)
                                          if agr else None)
            rec["mean_live_share"] = round(float(np.mean(live_share)), 4) if live_share else None
        per_code[code] = rec
    out["per_code"] = per_code

    # Pairing: attacking tag with a simultaneous defending tag of the other team.
    att = np.flatnonzero(keep_tag & is_attack & (tteam >= 0) & (codes != "SET PIECES"))
    dfd = np.flatnonzero(keep_tag & is_defend & (tteam >= 0))
    paired, offs = 0, []
    for i in att:
        cand = dfd[(tteam[dfd] == 1 - tteam[i]) & (starts[dfd] < stops[i]) & (stops[dfd] > starts[i])]
        if len(cand):
            paired += 1
            j = cand[np.argmin(np.abs(starts[cand] - starts[i]))]
            offs.append((starts[j] - starts[i]) / fps)
    out["pairing"] = {"attack_tags": int(len(att)), "with_opposite_defend": round(paired / max(1, len(att)), 4),
                      "defend_start_minus_attack_start_s": q(offs),
                      "defend_tags": int(len(dfd))}
    # Code mix (attacking, defending), of the voting tags.
    out["code_mix"] = {c: int(np.sum(keep_tag & (codes == c))) for c in
                       context.ATTACK_CODES + context.DEFEND_CODES}
    # Set-piece type mix.
    sp = keep_tag & (codes == "SET PIECES")
    types = tagging["type"].to_numpy()
    out["set_piece_types"] = {str(t): int(np.sum(sp & (types == t))) for t in sorted(set(types[sp]), key=str)}

    # -- spells: reference vs tag spells (the moments the generator is calibrated on) --
    spells = []
    for a, b, v in runs(np.where(usable, team, -2)):
        if v < 0 or (b - a) < MIN_SPELL_S * fps:
            continue
        if np.mean(r_lab[a:b] == h_lab[a:b]) < 0.8 or np.mean(ball[a:b]) < 0.5:
            continue
        spells.append((a, b, int(v)))
    out["spells"] = synthetic_tags.tag_statistics(tagging, team, conf_live, conf_dead, usable,
                                                  spells, fps, MERGE_S)
    out["spells"]["confident_spell_duration_s"] = q([(b - a) / fps for a, b, _ in spells])
    # Missing rate by spell length.
    bounds = (2, 5, 10, 20, 1e9)
    by_len = {}
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        sub = [s for s in spells if lo * fps <= s[1] - s[0] < hi * fps]
        if sub:
            st = synthetic_tags.tag_statistics(tagging, team, conf_live, conf_dead, usable, sub,
                                               fps, MERGE_S)
            by_len["{:g}-{:g}s".format(lo, hi)] = {"n": len(sub),
                                                  "missing": round(st["missing_rate"], 4)}
    out["spells"]["missing_by_length"] = by_len
    # Overlap of opposite-possession votes (st_vote sign conflicts) on usable frames.
    pos = np.zeros(n + 1)
    neg = np.zeros(n + 1)
    for a, b, v, ok in zip(starts, stops, votes, voting):
        if ok and v > 0:
            pos[a] += 1; pos[b] -= 1
        elif ok and v < 0:
            neg[a] += 1; neg[b] -= 1
    pos, neg = np.cumsum(pos)[:n] > 0, np.cumsum(neg)[:n] > 0
    out["conflict_share_of_covered"] = round(float((pos & neg)[usable & cov].mean()), 4)

    # Set pieces against reference dead runs (>= 2 s).
    sp_idx = np.flatnonzero(sp)
    d_runs = [(a, b) for a, b, v in runs(np.where(usable, h_lab, "X")) if v == "D" and b - a >= 2 * fps]
    d_a = np.array([a for a, _ in d_runs])
    d_b = np.array([b for _, b in d_runs])
    rel, matched = [], set()
    for i in sp_idx:
        s0 = starts[i]
        cand = np.flatnonzero((d_a - 5 * fps <= s0) & (s0 <= d_b + 5 * fps))
        if len(cand):
            j = int(cand[np.argmin(np.abs(d_b[cand] - s0))])
            matched.add(j)
            rel.append((s0 - d_b[j]) / fps)
    sp_team_ok, overlap_other = [], []
    nonsp_voting = voting & (codes != "SET PIECES")
    for i in sp_idx:
        a, b = starts[i], stops[i]
        cl = conf_live[a:b]
        if tteam[i] >= 0 and cl.sum():
            sp_team_ok.append(np.mean(team[a:b][cl] == tteam[i]))
        overlap_other.append(bool(np.any(nonsp_voting & (starts < b) & (stops > a))))
    out["set_pieces"] = {
        "n": int(len(sp_idx)), "dead_runs_ge_2s": len(d_runs),
        "matched_to_dead_run": len(rel), "set_piece_rate": round(len(matched) / max(1, len(d_runs)), 4),
        "start_minus_dead_end_s": q(rel),
        "team_agreement": round(float(np.mean(sp_team_ok)), 4) if sp_team_ok else None,
        "overlaps_possession_tag": round(float(np.mean(overlap_other)), 4) if overlap_other else None}

    doc = {}
    if os.path.isfile(args.out):
        with open(args.out, encoding="utf-8") as handle:
            doc = json.load(handle)
    # The calibrated generator block (calibrate_synthetic_tags.py) is kept.
    doc = {"measured": out, **{k: v for k, v in doc.items() if k in ("generator", "calibration")}}
    blob = (json.dumps(doc, indent=1, default=float) + "\n").encode("utf-8")
    with open(args.out, "wb") as handle:
        handle.write(blob)
    print(json.dumps(out, indent=1, default=float))
    print("wrote {} sha256 {}".format(os.path.relpath(args.out, ROOT), hashlib.sha256(blob).hexdigest()))


if __name__ == "__main__":
    main()
