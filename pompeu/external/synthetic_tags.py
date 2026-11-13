"""Synthetic Smart Tagging for external labelled matches, corrupted like BAR-ATH's real tags.

    from pompeu.external import synthetic_tags as st
    model = st.load_error_model()                        # configs/synthetic_tags_v1.json
    tagging = st.generate(labels, tracking.half(), model, seed=0, fps=25)
    F_ctx = st.context_features(tracking, tagging)       # st_vote & co. on the external match
    with st.injected({tracking.meta["game"]: tagging}):  # build_features("auto") / hsmm.prepare
        prep = hsmm.prepare(clean)

WHY. Only BAR-ATH has Smart Tagging, so every model fitted on external data ignores it. Synthetic
tags on the external matches let a model learn how much to trust them -- as long as they are as
wrong as the real ones. The error model is measured on BAR-ATH without any ground truth
(scripts/experiments/tag_error_model.py: Smart Tagging versus our own algorithms where they agree
and the ball is visible, public window excluded) and the generator's free rates are calibrated so
that tag_statistics() of the synthetic tags against the external labels matches tag_statistics()
of the real tags against that reference (scripts/experiments/calibrate_synthetic_tags.py).

WHAT THE REAL TAGS LOOK LIKE (BAR-ATH, see the JSON): phases are long (median 12-30 s, several
codes have a FIXED length: CREATING CHANCES 30 s, SET PIECES 35 s), so a tag usually outlives the
possession that started it; an attacking phase comes with a defending phase of the other team over
exactly the same span (86%); phases of the two teams rarely overlap, i.e. the tagger is sequential.

GENERATION, from an external match's labels (H=0, A=1, D=2, X=3; X / half 0 never tagged),
half by half, in time order -- a sequential tagger:
  1. possession spells: maximal runs of H or A inside a half (D and X break them).
  2. a spell is a candidate tag start at spell start + a start offset (normal with the measured
     median and IQR / 1.349, clipped to +-max_offset_s). If a previous tag is still running
     there, the spell gets no tag of its own (the "missed, other team tagged" case); otherwise
     it is dropped when shorter than min_spell_s, or with probability missing_rate (the
     calibrated random misses).
  3. wrong team with wrong_team_rate.
  4. phases: from the start, attacking phases (code from the measured mix, duration from that
     code's measured median/IQR) are laid back to back until they pass the spell end + an end
     offset; the last is cut there when truncate_last, else keeps its full length.
     Each phase gets a defending phase of the other team over the same span with pairing_rate.
  5. set pieces: each D run >= set_piece_min_dead_s gets, with set_piece_rate, a SET PIECES tag
     starting at the run END + a set-piece offset (measured, negative: before the restart), for the team of the following
     spell (N/A with set_piece_team_na), type from the measured mix, measured duration. It does
     not block the possession tags (on BAR-ATH they overlap them).
  6. false positives: Binomial(n tag spells, false_rate) extra attacking phases started inside
     D runs >= 2 s, for a random team.
Every draw comes from one numpy Generator seeded with `seed`: same inputs, same DataFrame.

OUTPUT has the columns of pompeu.io.tagging.load_tagging (code, start_frame, end_frame
exclusive, team 0/1/-1, type, half, tags), in the frame space of the external Tracking.
"""

import contextlib
import json
import os

import numpy as np
import pandas as pd

from pompeu import paths

# Params
H, A, D, X = 0, 1, 2, 3
DEFAULT_MODEL = os.path.join(paths.CONFIGS_DIR, "synthetic_tags_v1.json")
TEAM_NAMES = ("home", "away")
ATTACK_CODES = ("BUILD UP", "PROGRESSION", "CREATING CHANCES", "LONG BALL",
                "ATTACKING TRANSITION", "SET PIECES")
DEFEND_CODES = ("DEFENDING IN MIDDLE THIRD", "DEFENDING IN DEFENSIVE THIRD",
                "DEFENDING IN ATTACKING THIRD", "DEFENSIVE TRANSITION")
DEFEND_FOR = {"BUILD UP": "DEFENDING IN ATTACKING THIRD", "PROGRESSION": "DEFENDING IN MIDDLE THIRD",
              "CREATING CHANCES": "DEFENDING IN DEFENSIVE THIRD",
              "LONG BALL": "DEFENDING IN MIDDLE THIRD",
              "ATTACKING TRANSITION": "DEFENSIVE TRANSITION"}
# Used when the measured file lacks a key (and by the unit tests).
DEFAULTS = {
    "missing_rate": 0.15,
    "min_spell_s": 0.0,
    "wrong_team_rate": 0.1,
    "false_rate": 0.03,
    "start_offset_s": {"median": -1.0, "iqr": 10.0},
    "end_offset_s": {"median": 5.0, "iqr": 15.0},
    "max_offset_s": 20.0,
    "min_phase_s": 1.0,
    "attack_mix": {"BUILD UP": 21, "PROGRESSION": 61, "CREATING CHANCES": 29, "LONG BALL": 27,
                   "ATTACKING TRANSITION": 23},
    "phase_duration_s": {"BUILD UP": {"median": 20.4, "iqr": 5.2},
                         "PROGRESSION": {"median": 17.5, "iqr": 11.0},
                         "CREATING CHANCES": {"median": 30.0, "iqr": 0.0},
                         "LONG BALL": {"median": 12.4, "iqr": 5.1},
                         "ATTACKING TRANSITION": {"median": 15.6, "iqr": 6.6}},
    "pairing_rate": 0.86,
    "truncate_last": True,
    "set_piece_min_dead_s": 2.0,
    "set_piece_rate": 0.5,
    "set_piece_offset_s": {"median": -6.3, "iqr": 5.6},
    "set_piece_team_na": 0.55,
    "set_piece_duration_s": {"median": 35.0, "iqr": 0.0},
    "set_piece_types": {"throw-in": 25, "corner kick": 12, "goal kick": 9, "free kick": 4},
}


def load_error_model(path=DEFAULT_MODEL):
    """DEFAULTS overridden by the "generator" block of the measured JSON (if the file exists)."""
    model = json.loads(json.dumps(DEFAULTS))
    if path and os.path.isfile(path):
        with open(path, encoding="utf-8") as handle:
            model.update(json.load(handle).get("generator", {}))
    return model


# -- pieces ------------------------------------------------------------------------------------

def runs(values):
    """[(start, stop, value)] maximal runs of equal values."""
    values = np.asarray(values)
    if not len(values):
        return []
    breaks = np.flatnonzero(values[1:] != values[:-1]) + 1
    bounds = np.concatenate(([0], breaks, [len(values)]))
    return [(int(a), int(b), values[a].item()) for a, b in zip(bounds[:-1], bounds[1:])]


def spells(labels, half, min_s=0.0, fps=25):
    """[(start, stop, team)] maximal H or A runs inside halves, at least min_s long."""
    marked = np.where(np.asarray(half) > 0, np.asarray(labels), X)
    return [(a, b, int(v)) for a, b, v in runs(marked)
            if v in (H, A) and b - a >= min_s * fps]


def dead_runs(labels, half, min_s=0.0, fps=25):
    """[(start, stop)] maximal D runs inside halves, at least min_s long."""
    marked = np.where(np.asarray(half) > 0, np.asarray(labels), X)
    return [(a, b) for a, b, v in runs(marked) if v == D and b - a >= min_s * fps]


def _normal(rng, spec, lo, hi):
    sigma = max(float(spec.get("iqr", 0.0)), 0.0) / 1.349
    return float(np.clip(float(spec.get("median", 0.0)) + sigma * rng.standard_normal(), lo, hi))


def _choice(rng, mix):
    keys = sorted(mix)
    p = np.array([float(mix[k]) for k in keys])
    return keys[int(rng.choice(len(keys), p=p / p.sum()))]


# -- generator ---------------------------------------------------------------------------------

def generate(labels, half, model=None, seed=0, fps=25):
    """DataFrame like io.tagging.load_tagging for one external match (see the module doc)."""
    model = model or load_error_model()
    rng = np.random.default_rng(int(seed))
    labels = np.asarray(labels)
    half = np.asarray(half)
    n = len(labels)
    limit = float(model["max_offset_s"])
    min_len = max(1, int(round(model["min_phase_s"] * fps)))
    rows = []

    def add(code, a, b, team, kind=None):
        a, b = int(np.clip(a, 0, n)), int(np.clip(b, 0, n))
        if b - a < 1:
            return
        h = int(half[a])
        if h == 0:  # clip to the half the tag was meant for
            return
        last = int(np.flatnonzero(half == h)[-1]) + 1
        b = min(b, last)
        rows.append({"code": code, "start_frame": a, "end_frame": b, "team": int(team),
                     "type": kind, "half": h,
                     "tags": {"Team": TEAM_NAMES[team] if team >= 0 else "N/A",
                              "Half": {1: "1st Half", 2: "2nd Half"}.get(h, ""),
                              **({"Type": kind} if kind else {})}})

    possession = spells(labels, half)
    n_tag_spells = 0
    active_until, active_half = -1, 0
    for a, b, team in possession:
        # The same number of draws per spell whatever happens, so one rate does not reshuffle
        # the draws of the other steps.
        u_miss, u_wrong = rng.random(), rng.random()
        so = _normal(rng, model["start_offset_s"], -limit, limit)
        eo = _normal(rng, model["end_offset_s"], -limit, limit)
        h = int(half[a])
        lo_half = int(np.flatnonzero(half == h)[0])
        start = max(lo_half, a + int(round(so * fps)))
        if h == active_half and start < active_until:
            continue
        if b - a < model.get("min_spell_s", 0.0) * fps:
            continue
        if u_miss < model["missing_rate"]:
            continue
        if u_wrong < model["wrong_team_rate"]:
            team = 1 - team
        target = max(b + int(round(eo * fps)), start + min_len)
        t = start
        while t < target:
            code = _choice(rng, model["attack_mix"])
            spec = model["phase_duration_s"].get(code, {"median": 15.0, "iqr": 5.0})
            length = max(min_len, int(round(_normal(rng, spec, model["min_phase_s"], 600.0) * fps)))
            if model.get("truncate_last", True) and t + length > target:
                length = max(1, target - t)
            pair = rng.random() < model["pairing_rate"]
            add(code, t, t + length, team)
            if pair:
                add(DEFEND_FOR.get(code, "DEFENDING IN MIDDLE THIRD"), t, t + length, 1 - team)
            t += length
        n_tag_spells += 1
        active_until, active_half = t, h

    # Set pieces at dead runs, for the team of the spell that follows.
    starts = np.array([s[0] for s in possession], int)
    for a, b in dead_runs(labels, half):
        u, u_na = rng.random(), rng.random()
        kind = _choice(rng, model["set_piece_types"])
        dur = _normal(rng, model["set_piece_duration_s"], model["min_phase_s"], 600.0)
        so = _normal(rng, model["set_piece_offset_s"], -limit, limit)
        if b - a < model["set_piece_min_dead_s"] * fps or u >= model["set_piece_rate"]:
            continue
        nxt = np.flatnonzero((starts >= b) & (starts <= b + int(fps)))
        if not len(nxt):
            continue
        team = possession[int(nxt[0])][2] if u_na >= model["set_piece_team_na"] else -1
        s0 = b + int(round(so * fps))
        add("SET PIECES", s0, s0 + int(round(dur * fps)), team, kind)

    # False positives inside dead runs.
    long_dead = dead_runs(labels, half, 2.0, fps)
    n_false = int(rng.binomial(n_tag_spells, min(1.0, model["false_rate"]))) if long_dead else 0
    for _ in range(n_false):
        a, b = long_dead[int(rng.integers(len(long_dead)))]
        code = _choice(rng, model["attack_mix"])
        spec = model["phase_duration_s"].get(code, {"median": 15.0, "iqr": 5.0})
        dur = int(round(_normal(rng, spec, model["min_phase_s"], 600.0) * fps))
        s0 = int(rng.integers(a, b))
        team = int(rng.integers(2))
        add(code, s0, s0 + dur, team)
        if rng.random() < model["pairing_rate"]:
            add(DEFEND_FOR.get(code, "DEFENDING IN MIDDLE THIRD"), s0, s0 + dur, 1 - team)

    columns = ["code", "start_frame", "end_frame", "team", "type", "half", "tags"]
    frame = pd.DataFrame(rows, columns=columns)
    return frame.sort_values(["start_frame", "end_frame", "code"], kind="stable").reset_index(drop=True)


# -- statistics (the same on real and synthetic tags) ------------------------------------------

def _q(values):
    v = np.asarray(values, float)
    if not len(v):
        return {"n": 0}
    return {"n": int(len(v)), "median": round(float(np.median(v)), 3),
            "q25": round(float(np.percentile(v, 25)), 3),
            "q75": round(float(np.percentile(v, 75)), 3)}


def _merge(intervals, gap):
    out = []
    for a, b in sorted(intervals):
        if out and a <= out[-1][1] + gap:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [tuple(x) for x in out]


def vote_arrays(tagging, n):
    """(st_vote, st_vote_n) over n frames, as context.context_features computes them."""
    vote = np.zeros(n + 1)
    count = np.zeros(n + 1)
    for code, a, b, team in zip(tagging["code"], tagging["start_frame"], tagging["end_frame"],
                                tagging["team"]):
        if team < 0 or code not in ATTACK_CODES + DEFEND_CODES:
            continue
        sign = 1.0 if team == 0 else -1.0
        v = sign if code in ATTACK_CODES else -sign
        a, b = int(np.clip(a, 0, n)), int(np.clip(b, 0, n))
        vote[a] += v
        vote[b] -= v
        count[a] += 1
        count[b] -= 1
    return np.cumsum(vote)[:n], np.cumsum(count)[:n]


def tag_statistics(tagging, team, live, dead, usable, spell_list, fps=25, merge_s=1.0):
    """Aggregate agreement of a tagging with a reference possession, the error-model moments.

    team (N,) reference possessor 0/1 (-1 none); live / dead (N,) bool confident live / dead
    frames; usable (N,) bool frames that count (tags count when their midpoint is usable);
    spell_list [(start, stop, team)] the reference possession spells to look for.
    """
    n = len(team)
    team = np.asarray(team)
    codes = tagging["code"].to_numpy()
    tteam = tagging["team"].to_numpy()
    starts = np.clip(tagging["start_frame"].to_numpy(), 0, n)
    stops = np.clip(tagging["end_frame"].to_numpy(), 0, n)
    mid = np.clip((starts + stops) // 2, 0, n - 1)
    keep = usable[mid] if len(mid) else np.zeros(0, bool)
    voting = np.isin(codes, ATTACK_CODES + DEFEND_CODES) & (tteam >= 0) & keep
    poss = np.where(np.isin(codes, ATTACK_CODES), tteam, 1 - tteam)
    st_vote, st_n = vote_arrays(tagging, n)
    cov = st_n > 0
    ref_sign = np.where(team == 0, 1.0, -1.0)
    m = live & cov & usable
    out = {
        "coverage_live": float(cov[live & usable].mean()),
        "coverage_dead": float(cov[dead & usable].mean()) if (dead & usable).any() else 0.0,
        "frame_agreement": float((np.sign(st_vote[m]) == ref_sign[m]).mean()) if m.any() else 0.0,
        "frame_disagreement": float((np.sign(st_vote[m]) == -ref_sign[m]).mean()) if m.any() else 0.0,
    }
    tag_spells = []
    for t in (0, 1):
        sel = voting & (poss == t)
        tag_spells += [(a, b, t) for a, b in _merge(list(zip(starts[sel], stops[sel])),
                                                   int(merge_s * fps))]
    tag_spells.sort()
    tsa = np.array([s[0] for s in tag_spells], int)
    tsb = np.array([s[1] for s in tag_spells], int)
    tst = np.array([s[2] for s in tag_spells], int)
    start_off, end_off, missing, other = [], [], 0, 0
    for a, b, v in spell_list:
        ov = np.minimum(tsb, b) - np.maximum(tsa, a)
        same = (ov > 0) & (tst == v)
        if not same.any():
            missing += 1
            other += int(((ov > 0) & (tst != v)).any())
            continue
        j = np.flatnonzero(same)[np.argmax(ov[same])]
        start_off.append((tsa[j] - a) / fps)
        end_off.append((tsb[j] - b) / fps)
    verdict = {"correct": 0, "wrong_team": 0, "false_dead": 0, "unverifiable": 0}
    for a, b, t in tag_spells:
        cl, cd = live[a:b], dead[a:b]
        if cl.sum() >= 0.2 * fps:
            verdict["correct" if np.mean(team[a:b][cl] == t) >= 0.5 else "wrong_team"] += 1
        elif cd.sum() >= 0.5 * (b - a):
            verdict["false_dead"] += 1
        else:
            verdict["unverifiable"] += 1
    verifiable = max(1, verdict["correct"] + verdict["wrong_team"] + verdict["false_dead"])
    n_spells = max(1, len(spell_list))
    out.update({
        "n_spells": len(spell_list), "n_tag_spells": len(tag_spells),
        "tag_spells_per_spell": len(tag_spells) / n_spells,
        "missing_rate": missing / n_spells, "missing_other_tagged": other / n_spells,
        "start_offset_s": _q(start_off), "end_offset_s": _q(end_off),
        "tag_spell_duration_s": _q([(b - a) / fps for a, b, _ in tag_spells]),
        "verdicts": verdict,
        "wrong_team_rate": verdict["wrong_team"] / verifiable,
        "false_dead_rate": verdict["false_dead"] / verifiable,
    })
    return out


# -- using them --------------------------------------------------------------------------------

@contextlib.contextmanager
def _registered(tracking):
    """context.context_features reads the tag frame space from games.GAMES[meta game]; an external
    match is not there, so register its own length for the duration of the call."""
    import games
    game = tracking.meta.get("game", games.GAME)
    added = game not in games.GAMES
    if added:
        games.GAMES[game] = {"n_video_frames": int(tracking.frame0 + tracking.n_frames),
                             "fps": int(tracking.meta.get("fps", 25))}
    try:
        yield
    finally:
        if added:
            games.GAMES.pop(game, None)


def context_features(tracking, tagging):
    """pompeu.features.context.context_features on an external match with synthetic tags."""
    from pompeu.features import context
    with _registered(tracking):
        return context.context_features(tracking, tagging)


@contextlib.contextmanager
def injected(tagging_by_game):
    """Within the block, build_features(tagging="auto") -- and so hsmm.prepare, rules.predict --
    uses tagging_by_game[tracking.meta["game"]] for the matches listed (others unchanged)."""
    from pompeu.features import build
    original_load, original_ctx = build.load_tagging_for, build.context.context_features

    def load(tracking):
        game = tracking.meta.get("game")
        return tagging_by_game[game] if game in tagging_by_game else original_load(tracking)

    def ctx(tracking, tagging):
        if tagging is not None and tracking.meta.get("game") in tagging_by_game:
            with _registered(tracking):
                return original_ctx(tracking, tagging)
        return original_ctx(tracking, tagging)

    build.load_tagging_for = load
    build.context.context_features = ctx
    try:
        yield
    finally:
        build.load_tagging_for = original_load
        build.context.context_features = original_ctx
