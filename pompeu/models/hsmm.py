"""A5: hsmm_v1 -- a hidden semi-Markov model of possession with explicit segment durations.

rules_v0 makes hard decisions and never changes possession without a ball event. This model
weighs every piece of evidence, with or without a ball, and scores whole segments -- which is
what spellF1 scores.

FOUR HIDDEN STATES, THREE LABELS. A segment is (team, alive):

    HL  home has the ball, ball alive        -> label H
    AL  away has the ball, ball alive        -> label A
    HD  ball dead, home will restart          -> label D
    AD  ball dead, away will restart          -> label D

A dead ball still has a team: the one that restarts (CHALLENGE.md: the tracking names it, the
ground truth throws it away). Making it part of the state is what keeps team evidence from being
mistaken for liveness -- a goalkeeper placing the ball for a goal kick is strong "home" evidence
during a stoppage, and with a single D state the model could only explain it by calling the ball
alive. With HD it is simply HD. And a restart is structural: HD is followed by HL.

DURATIONS are log-normal in seconds, truncated: possessions median 8 s (>= 0.2 s), dead balls
median 20 s (>= 1 s) -- the plan's 0.5-120 s and 3-90 s ranges. A segment longer than
max_segment_s is two segments of the same state joined by a tiny self-transition; labels are
unaffected.

TRANSITIONS at the frame a segment starts:
    HL -> AL       turnover                      1 - p_poss_to_dead
    HL -> HD / AD  stoppage, restarted by either p_poss_to_dead, split p_same_restarts / rest
    HD -> HL       the restart                   1 - p_restart_mismatch
    HD -> AL       restart team wrong in the tracking, kept possible, unlikely
    (and the mirror images)

EMISSIONS: a product of experts, factorised over the two halves of the state:

    log e(team, alive) = +/- L_team / 2  +/-  L_dead / 2

    L_dead   A3's per-frame dead-ball evidence (pompeu.events.deadball.evidence: slow field, ball
             out, restart kick, set piece, pre-kickoff...), scaled for autocorrelation as in A3,
             plus A3's final segmentation as one more expert (+/- a3_segment_llr per look): it
             carries what the raw cues do not -- a stoppage starts where the ball crossed the
             line, ends at the restart kick, and a ball handled during it is not a restart (so
             kicks inside A3's dead stretches are dropped from the kick cue).
    L_team   proximity + events + tagging + restart:
      proximity  (d1_A - d1_H) / proximity_scale_m, clipped, faded out when nobody is within
                 reach of the ball (a phantom ball far from everyone says nothing) and when both
                 teams are within contest_m of it (a contested ball stays with whoever had it, so
                 a challenger's nearness is not control). Scaled by 1 / (evidence_tau_s * fps):
                 consecutive frames are one look per ~0.5 s.
      events     A4 controls add control_llr per frame, touches touch_llr on their frame, times the
                 team confidence |2 p_home - 1|; fly-by touches skipped. Unscaled: already sparse.
      tagging    Smart Tagging possession vote (attacking codes name the team with the ball,
                 defending codes the one without), tag_vote_llr per vote, scaled by
                 1 / (tag_tau_s * fps): one tag is ONE decision however long it lasts, so a 20 s
                 phase counts as ~4 looks, not 500.
      restart    A3's restart team, as one observation of log-odds logit(0.5 + confidence / 2)
                 spread over the restart_window_s before the restart: it names the team of the
                 dead segment that ends there.
      rules      A4's own output: rules_v0's team on its live frames, +/- rules_llr per look. It
                 carries the convention the per-frame cues cannot -- wait for control, then date
                 the change back to the challenger's first touch -- the way A3's segmentation
                 carries its conventions into L_dead.

INFERENCE per half (pompeu.decode.viterbi, numba): semi-Markov Viterbi for the labels;
forward-backward for P(H), P(A), P(D) per frame when posteriors=True (A9's MBR needs them).
Frames outside the halves are labelled D, as rules_v0 does (A10 decides their final label).

No weight is fitted: they are physical or plan values. The plan fits them on `external` (A6).
"""

import copy

import numpy as np
from scipy.stats import lognorm

from pompeu.decode import viterbi as sm
from pompeu.events import deadball, touches
from pompeu.features.build import build_features
import games

# Params
STATES = ("HL", "AL", "HD", "AD")
HL, AL, HD, AD = 0, 1, 2, 3
LABEL_OF_STATE = np.array(["H", "A", "D", "D"])
TEAM_SIGN = np.array([1.0, -1.0, 1.0, -1.0])    # +: home
DEAD_SIGN = np.array([-1.0, -1.0, 1.0, 1.0])    # +: dead
PARAMS = {
    "evidence_tau_s": 0.5,
    "poss_median_s": 8.0,
    "poss_sigma": 1.1,
    "poss_min_s": 0.2,
    "dead_median_s": 20.0,
    "dead_sigma": 0.8,
    "dead_min_s": 1.0,
    "max_segment_s": 120.0,
    "p_poss_to_dead": 0.3,
    "p_same_restarts": 0.5,      # the team that had the ball also restarts (free kick for, ...)
    "p_restart_mismatch": 0.02,  # dead ball 'owned' by one team, next possession the other's
    "self_transition": 1e-6,
    "w_dead": 1.0,
    "a3_segment_llr": 3.0,       # per look, for A3's final dead/live segmentation
    "proximity_scale_m": 1.0,
    "proximity_clip": 3.0,
    "proximity_reach_m": 4.0,    # nearest player beyond this: the frame fades out ...
    "proximity_fade_m": 0.7,     # ... over this many metres (logistic)
    "contest_m": 2.5,            # both teams this close: a contested ball, no evidence ...
    "contest_fade_m": 0.5,       # ... fading in over this many metres (logistic)
    "w_proximity": 1.0,
    "control_llr": 0.5,
    "touch_llr": 2.0,
    "fast_ball_speed": 15.0,
    "w_events": 1.0,
    "tag_vote_llr": 1.0,
    "tag_tau_s": 5.0,
    "w_tagging": 1.0,
    "restart_window_s": 3.0,
    "w_restart": 1.0,
    "rules_llr": 0.0,            # per look, for rules_v0's team on its live frames (A4's output)
    "posteriors": False,
}


def _settings(params):
    out = copy.deepcopy(PARAMS)
    unknown = set(params or {}) - set(out)
    if unknown:
        raise KeyError("unknown hsmm params {}".format(sorted(unknown)))
    out.update(params or {})
    return out


# -- model pieces ------------------------------------------------------------------------------

def log_durations(p, fps):
    """(4, Dmax+1) log pmf over segment lengths in frames: log-normal in seconds, truncated."""
    d_max = int(round(p["max_segment_s"] * fps))
    lengths = np.arange(d_max + 1) / fps
    out = np.full((4, d_max + 1), -np.inf)
    shapes = {HL: ("poss", ), AL: ("poss", ), HD: ("dead", ), AD: ("dead", )}
    for state, (kind,) in shapes.items():
        median, sigma, minimum = (p[kind + "_median_s"], p[kind + "_sigma"], p[kind + "_min_s"])
        allowed = lengths >= minimum - 1e-9
        allowed[0] = False
        logpdf = lognorm.logpdf(lengths[allowed], s=sigma, scale=median)
        out[state, allowed] = logpdf - np.logaddexp.reduce(logpdf)
    return out


def transition_matrix(p):
    """(4, 4) log transition probabilities between consecutive segments."""
    T = np.zeros((4, 4))
    stop, same = p["p_poss_to_dead"], p["p_same_restarts"]
    T[HL] = [0.0, 1 - stop, stop * same, stop * (1 - same)]
    T[AL] = [1 - stop, 0.0, stop * (1 - same), stop * same]
    miss = p["p_restart_mismatch"]
    T[HD] = [1 - miss, miss, 0.0, 0.0]
    T[AD] = [miss, 1 - miss, 0.0, 0.0]
    T[np.arange(4), np.arange(4)] = p["self_transition"]
    T /= T.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore"):
        return np.log(T)


def team_evidence(tracking, F, events, dead_result, p, rules_labels=None):
    """{expert: (N,) log-odds home vs away}, before weights."""
    fps = tracking.meta.get("fps", 25)
    n = tracking.n_frames
    scale = 1.0 / (p["evidence_tau_s"] * fps)

    d_home = F["d1_H"].to_numpy(dtype=np.float64)
    d_away = F["d1_A"].to_numpy(dtype=np.float64)
    nearest = np.fmin(d_home, d_away)
    with np.errstate(invalid="ignore", over="ignore"):
        lean = np.clip((d_away - d_home) / p["proximity_scale_m"],
                       -p["proximity_clip"], p["proximity_clip"])
        reach = 1.0 / (1.0 + np.exp((nearest - p["proximity_reach_m"]) / p["proximity_fade_m"]))
        # A contested ball -- both teams close -- stays with whoever had it (the challenge's
        # rule): nearness of a challenger is not control, so it says nothing.
        farther = np.fmax(d_home, d_away)
        uncontested = 1.0 / (1.0 + np.exp(-(farther - p["contest_m"]) / p["contest_fade_m"]))
    proximity = np.nan_to_num(lean * reach * uncontested) * scale

    event_llr = np.zeros(n)
    for ev in events.itertuples(index=False):
        confidence = abs(2 * ev.p_home - 1)
        sign = 1.0 if ev.p_home >= 0.5 else -1.0
        lo = ev.frame - tracking.frame0
        if ev.kind == "control":
            hi = min(n, ev.end - tracking.frame0)
            event_llr[max(0, lo):hi] += sign * confidence * p["control_llr"]
        elif 0 <= lo < n and min(ev.speed_in, ev.speed_out) <= p["fast_ball_speed"]:
            event_llr[lo] += sign * confidence * p["touch_llr"]

    votes = np.nan_to_num(F["st_vote"].to_numpy(dtype=np.float64))
    tagging = votes * p["tag_vote_llr"] / (p["tag_tau_s"] * fps)

    restart = np.zeros(n)
    window = max(1, int(round(p["restart_window_s"] * fps)))
    for row in dead_result.restarts.itertuples(index=False):
        if row.team < 0 or row.confidence <= 0:
            continue
        prob = min(0.999, 0.5 + 0.5 * row.confidence)
        llr = np.log(prob / (1 - prob)) * (1.0 if row.team == 0 else -1.0)
        hi = row.frame - tracking.frame0
        lo = max(0, hi - window)
        if hi > lo:
            restart[lo:min(hi, n)] += llr / (hi - lo)
    rules_lean = np.zeros(n)
    if rules_labels is not None:
        rules_lean = (np.where(rules_labels == "H", 1.0, np.where(rules_labels == "A", -1.0, 0.0))
                      * p["rules_llr"] * scale)
    return {"proximity": proximity, "events": event_llr, "tagging": tagging, "restart": restart,
            "rules": rules_lean}


def emissions(dead_llr, team, p):
    """(N, 4) per-frame log emissions: team and liveness evidence, factorised."""
    L_team = (p["w_proximity"] * team["proximity"] + p["w_events"] * team["events"]
              + p["w_tagging"] * team["tagging"] + p["w_restart"] * team["restart"]
              + team.get("rules", 0.0))
    L_dead = p["w_dead"] * dead_llr
    return 0.5 * (L_team[:, None] * TEAM_SIGN[None] + L_dead[:, None] * DEAD_SIGN[None])


# -- decoding ----------------------------------------------------------------------------------

def decode(tracking, hsmm=None, deadball_params=None, touches_params=None):
    """{'labels' (N,) H/A/D, 'segments' [(start, stop, state name)] absolute, 'posterior'
    (N, 3) P(H), P(A), P(D) or None, 'evidence', 'deadball', 'events'} for a cleaned Tracking."""
    p = _settings(hsmm)
    fps = tracking.meta.get("fps", 25)
    n = tracking.n_frames
    F = build_features(tracking, with_windows=False,
                       groups=("ball", "proximity", "pitch_control", "collective", "context"))
    dead = deadball.detect(tracking, F=F, params=deadball_params)
    dead_cues = deadball.evidence(F, fps, deadball_params)
    # A kick inside one of A3's dead stretches is a ball handled during the stoppage, which A3
    # already ruled out as a restart: it is not live evidence here either.
    dead_cues["kick"] = np.where(dead.dead, 0.0, dead_cues["kick"])
    tau = deadball._settings(deadball_params)["evidence_tau_s"]
    dead_llr = np.sum(list(dead_cues.values()), axis=0) / (tau * fps)
    dead_llr += np.where(dead.dead, 1.0, -1.0) * p["a3_segment_llr"] / (p["evidence_tau_s"] * fps)
    events = touches.detect_events(tracking, F, params=touches_params)
    rules_labels = None
    if p["rules_llr"]:
        from pompeu.models.rules import labels_from
        rules_labels = labels_from(tracking, dead, events)
    team = team_evidence(tracking, F, events, dead, p, rules_labels)
    emission = emissions(dead_llr, team, p)
    log_dur = log_durations(p, fps)
    T = transition_matrix(p)
    start = np.log(np.full(4, 0.25))

    labels = np.full(n, "D", dtype="<U1")
    posterior = None
    if p["posteriors"]:
        posterior = np.zeros((n, 3))
        posterior[:, 2] = 1.0
    segments = []
    frames = tracking.frames
    for _, first, last in games.half_video_blocks(tracking.meta.get("game", games.GAME)):
        rows = np.flatnonzero((frames >= first) & (frames <= last))
        if not len(rows):
            continue
        lo, hi = int(rows[0]), int(rows[-1]) + 1
        E = sm.cumulative(emission[lo:hi])
        trans = np.broadcast_to(T, (hi - lo + 1, 4, 4))
        states, segs, _ = sm.viterbi(E, log_dur, trans, start)
        labels[lo:hi] = LABEL_OF_STATE[states]
        segments += [(tracking.frame0 + lo + a, tracking.frame0 + lo + b, STATES[s])
                     for a, b, s in segs]
        if posterior is not None:
            occupancy, _ = sm.posteriors(E, log_dur, trans, start)
            posterior[lo:hi] = np.column_stack((occupancy[:, HL], occupancy[:, AL],
                                                occupancy[:, HD] + occupancy[:, AD]))
    evidence = dict(team, dead=dead_llr)
    return {"labels": labels, "segments": segments, "posterior": posterior,
            "evidence": evidence, "deadball": dead, "events": events}


def predict(tracking, hsmm=None, deadball=None, touches=None):
    """(N,) H/A/D labels: the pipeline entry point (method hsmm_v1)."""
    return decode(tracking, hsmm, deadball, touches)["labels"]
