"""A9: minimum-Bayes-risk decoding -- the labels with the best EXPECTED spellF1, not the likeliest.

Viterbi returns the most probable segmentation. The board ranks on spellF1, which is not a
per-frame loss: a short possession the model is 50/50 about can be worth keeping or dropping, and
a boundary placed between two likely positions can match both. MBR asks the question directly:

    1. draw n_samples segmentations from the HSMM posterior (viterbi.sample, exact FFBS);
    2. start from the Viterbi labels;
    3. greedy local search: shift each boundary by +/- shift_frames, or hand a short possession
       (< short_s) to its neighbour; keep a move when it raises the mean spellF1 of the candidate
       against the samples (each sample playing the ground truth); repeat while anything improves.

The score is the challenge's own: spell_f1() reproduces possession_metrics.spell_scores (greedy
one-to-one matching by temporal IoU, F1 averaged over IoU 0.3..0.7) and tests/test_mbr.py checks
it against possession_metrics on random label arrays. It is only fast -- numba, a two-pointer
sweep instead of all pairs -- so the search can try thousands of moves.

Not done from the plan: k-best candidates, the samples' medoid, and the macroF1 tie-break.
"""

import numpy as np
from numba import njit

from pompeu.decode import viterbi as sm

# Params
PARAMS = {
    "n_samples": 200,
    "seed": 0,
    "shift_frames": (5, 15),
    "short_s": 3.0,
    "max_passes": 3,
}
IOUS = np.array([0.3, 0.4, 0.5, 0.6, 0.7])
H, A, D = 0, 1, 2


def spells(labels):
    """(starts, stops, teams) of the maximal H or A runs of a label-code array (H=0 A=1 D=2 X=3)."""
    labels = np.asarray(labels)
    change = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    starts = np.concatenate(([0], change))
    stops = np.concatenate((change, [len(labels)]))
    team = labels[starts]
    keep = (team == H) | (team == A)
    return (starts[keep].astype(np.int64), stops[keep].astype(np.int64),
            team[keep].astype(np.int64))


@njit(cache=True)
def _spell_f1(gs, ge, gk, ps, pe, pk, ious):
    """Mean over IoU thresholds of the spell F1 of prediction (ps, pe, pk) against truth (gs...)."""
    n_g, n_p = len(gs), len(ps)
    if n_g == 0 and n_p == 0:
        return 1.0
    # Every same-team overlapping pair, in (gt, pred) order, by a two-pointer sweep.
    cap = 4 * (n_g + n_p) + 16
    pi = np.empty(cap, np.int64)
    pj = np.empty(cap, np.int64)
    pv = np.empty(cap, np.float64)
    m = 0
    j0 = 0
    for i in range(n_g):
        while j0 < n_p and pe[j0] <= gs[i]:
            j0 += 1
        j = j0
        while j < n_p and ps[j] < ge[i]:
            if pk[j] == gk[i]:
                inter = min(ge[i], pe[j]) - max(gs[i], ps[j])
                union = max(ge[i], pe[j]) - min(gs[i], ps[j])
                if m == cap:
                    cap *= 2
                    pi2 = np.empty(cap, np.int64)
                    pj2 = np.empty(cap, np.int64)
                    pv2 = np.empty(cap, np.float64)
                    pi2[:m] = pi[:m]
                    pj2[:m] = pj[:m]
                    pv2[:m] = pv[:m]
                    pi, pj, pv = pi2, pj2, pv2
                pi[m] = i
                pj[m] = j
                pv[m] = inter / union
                m += 1
            j += 1
    # IoU descending; ties keep (gt, pred) order, like possession_metrics' sort key.
    order = np.argsort(-pv[:m], kind="mergesort")
    total = 0.0
    used_g = np.zeros(n_g, np.bool_)
    used_p = np.zeros(n_p, np.bool_)
    for threshold in ious:
        used_g[:] = False
        used_p[:] = False
        tp = 0
        for o in order:
            if pv[o] < threshold:
                break
            if used_g[pi[o]] or used_p[pj[o]]:
                continue
            used_g[pi[o]] = True
            used_p[pj[o]] = True
            tp += 1
        fp = n_p - tp
        fn = n_g - tp
        total += 2.0 * tp / (2 * tp + fp + fn) if tp else 0.0
    return total / len(ious)


def spell_f1(truth, pred):
    """spellF1 of label-code arrays over one contiguous graded stretch (no X inside)."""
    return float(_spell_f1(*spells(truth), *spells(pred), IOUS))


@njit(cache=True)
def _expected(sample_spells, offsets, ps, pe, pk, ious):
    total = 0.0
    n = len(offsets) - 1
    for k in range(n):
        a, b = offsets[k], offsets[k + 1]
        total += _spell_f1(sample_spells[0, a:b], sample_spells[1, a:b], sample_spells[2, a:b],
                           ps, pe, pk, ious)
    return total / n


def _pack(samples_labels):
    """All samples' spells in one (3, total) array plus offsets, for the numba scorer."""
    parts, offsets = [], [0]
    for labels in samples_labels:
        s, e, k = spells(labels)
        parts.append(np.vstack((s, e, k)))
        offsets.append(offsets[-1] + len(s))
    return np.concatenate(parts, axis=1), np.array(offsets, dtype=np.int64)


def expected_spell_f1(candidate, packed):
    return float(_expected(packed[0], packed[1], *spells(candidate), IOUS))


def runs_of(labels):
    change = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    return np.concatenate(([0], change)), np.concatenate((change, [len(labels)]))


def local_search(labels, packed, p, fps=25):
    """Greedy boundary shifts and short-possession removals that raise the expected spellF1."""
    labels = labels.copy()
    best = expected_spell_f1(labels, packed)
    short = int(round(p["short_s"] * fps))
    for _ in range(p["max_passes"]):
        improved = False
        # Boundary shifts.
        starts, _ = runs_of(labels)
        for boundary in starts[1:]:
            for delta in [d for s in p["shift_frames"] for d in (-s, s)]:
                new = boundary + delta
                left, right = labels[boundary - 1], labels[boundary]
                lo, hi = (new, boundary) if delta < 0 else (boundary, new)
                if lo <= 0 or hi >= len(labels):
                    continue
                trial = labels.copy()
                trial[lo:hi] = right if delta < 0 else left
                # A shift must not swallow a whole neighbouring run.
                if (trial[lo - 1] != trial[lo] and trial[hi] != trial[hi - 1]
                        and len(np.unique(trial[lo - 1:hi + 1])) > 2):
                    continue
                score = expected_spell_f1(trial, packed)
                if score > best + 1e-6:
                    best, labels, improved = score, trial, True
                    break
        # Short possessions handed to a neighbour.
        starts, stops = runs_of(labels)
        for a, b in zip(starts, stops):
            if labels[a] not in (H, A) or b - a >= short or a == 0 or b == len(labels):
                continue
            for neighbour in {labels[a - 1], labels[b]}:
                trial = labels.copy()
                trial[a:b] = neighbour
                score = expected_spell_f1(trial, packed)
                if score > best + 1e-6:
                    best, labels, improved = score, trial, True
                    break
        if not improved:
            break
    return labels, best


def decode_half(E, log_dur, trans, start, label_of_state, params=None, fps=25):
    """(MBR labels as codes H/A/D, expected spellF1 of Viterbi, of the result) for one half."""
    p = dict(PARAMS, **(params or {}))
    states, _, _ = sm.viterbi(E, log_dur, trans, start)
    viterbi_labels = label_of_state[states]
    draws = sm.sample(E, log_dur, trans, start, p["n_samples"], p["seed"])
    packed = _pack([label_of_state[d] for d in draws])
    before = expected_spell_f1(viterbi_labels, packed)
    labels, after = local_search(viterbi_labels, packed, p, fps)
    return labels, before, after
