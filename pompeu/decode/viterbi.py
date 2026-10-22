"""Explicit-duration (semi-Markov) inference: Viterbi segmentation and posterior state marginals.

A segmentation of frames [0, N) is a sequence of segments (start, stop, state). Its log score is

    start[s_1] + sum_k ( E[stop_k, s_k] - E[start_k, s_k] + log_dur[s_k, stop_k - start_k] )
               + sum_{k>1} trans[start_k, s_{k-1}, s_k]

where E is the cumulative per-frame log emission (N+1, S) -- so a segment's evidence is one
subtraction -- log_dur (S, Dmax+1) is the log duration pmf (-inf where a length is not allowed),
and trans (N+1, S, S) is the log transition probability for a segment that STARTS at frame u, so
transitions may depend on time (a restart prior, for instance).

    viterbi(E, log_dur, trans, start)      -> (states (N,), segments [(start, stop, state)], score)
    posteriors(E, log_dur, trans, start)   -> (occupancy (N, S), log Z)

Both are O(N * Dmax * S) and run in numba. Segments longer than Dmax are represented as
consecutive segments of the same state, which the caller allows with a (small) self-transition
probability; the labels they produce are identical, so this only matters to the duration model.

tests/test_hsmm.py checks both against exhaustive enumeration of every segmentation.
"""

import numpy as np
from numba import njit

NEG_INF = -np.inf


@njit(cache=True)
def _logaddexp(a, b):
    if a == NEG_INF:
        return b
    if b == NEG_INF:
        return a
    if a > b:
        return a + np.log1p(np.exp(b - a))
    return b + np.log1p(np.exp(a - b))


@njit(cache=True)
def _viterbi_kernel(E, log_dur, trans, start):
    N = E.shape[0] - 1
    S = E.shape[1]
    D = log_dur.shape[1] - 1
    V = np.full((N + 1, S), NEG_INF)
    best_d = np.zeros((N + 1, S), dtype=np.int64)
    incoming = np.full((N + 1, S), NEG_INF)
    incoming_from = np.full((N + 1, S), -1, dtype=np.int64)
    for s in range(S):
        incoming[0, s] = start[s]
    for t in range(1, N + 1):
        for s in range(S):
            best = NEG_INF
            arg = 0
            for d in range(1, min(D, t) + 1):
                ld = log_dur[s, d]
                if ld == NEG_INF:
                    continue
                u = t - d
                inc = incoming[u, s]
                if inc == NEG_INF:
                    continue
                score = inc + ld + E[t, s] - E[u, s]
                if score > best:
                    best = score
                    arg = d
            V[t, s] = best
            best_d[t, s] = arg
        if t < N:
            for s2 in range(S):
                best = NEG_INF
                arg = -1
                for s in range(S):
                    score = V[t, s] + trans[t, s, s2]
                    if score > best:
                        best = score
                        arg = s
                incoming[t, s2] = best
                incoming_from[t, s2] = arg
    return V, best_d, incoming_from


def viterbi(E, log_dur, trans, start):
    """Best segmentation. Returns (states (N,) int, [(start, stop, state)], log score)."""
    E, log_dur, trans, start = _as_arrays(E, log_dur, trans, start)
    N = E.shape[0] - 1
    V, best_d, incoming_from = _viterbi_kernel(E, log_dur, trans, start)
    state = int(np.argmax(V[N]))
    score = float(V[N, state])
    if not np.isfinite(score):
        raise ValueError("no segmentation is possible with these durations and transitions")
    segments = []
    t = N
    while t > 0:
        d = int(best_d[t, state])
        u = t - d
        segments.append((u, t, state))
        state_before = int(incoming_from[u, state]) if u > 0 else -1
        t, state = u, state_before
    segments.reverse()
    states = np.empty(N, dtype=np.int64)
    for a, b, s in segments:
        states[a:b] = s
    return states, segments, score


@njit(cache=True)
def _forward_backward_kernel(E, log_dur, trans, start):
    N = E.shape[0] - 1
    S = E.shape[1]
    D = log_dur.shape[1] - 1
    alpha = np.full((N + 1, S), NEG_INF)       # a segment of s ends at t
    incoming = np.full((N + 1, S), NEG_INF)    # a segment of s starts at u (forward mass)
    for s in range(S):
        incoming[0, s] = start[s]
    terms = np.empty(D + 1)
    for t in range(1, N + 1):
        for s in range(S):
            # log-sum-exp over durations as max + log(sum(exp(x - max))): one exp per term.
            top = NEG_INF
            last = min(D, t)
            for d in range(1, last + 1):
                x = incoming[t - d, s] + log_dur[s, d] - E[t - d, s]
                terms[d] = x
                if x > top:
                    top = x
            if top == NEG_INF:
                continue
            acc = 0.0
            for d in range(1, last + 1):
                if terms[d] != NEG_INF:
                    acc += np.exp(terms[d] - top)
            alpha[t, s] = top + np.log(acc) + E[t, s]
        if t < N:
            for s2 in range(S):
                acc = NEG_INF
                for s in range(S):
                    acc = _logaddexp(acc, alpha[t, s] + trans[t, s, s2])
                incoming[t, s2] = acc

    beta = np.full((N + 1, S), NEG_INF)       # observations [u, N) given a segment of s starts at u
    outgoing = np.full((N + 1, S), NEG_INF)   # observations [v, N) given a segment of s ended at v
    for s in range(S):
        outgoing[N, s] = 0.0
    for u in range(N - 1, -1, -1):
        for s in range(S):
            top = NEG_INF
            last = min(D, N - u)
            for d in range(1, last + 1):
                x = log_dur[s, d] + E[u + d, s] + outgoing[u + d, s]
                terms[d] = x
                if x > top:
                    top = x
            if top == NEG_INF:
                continue
            acc = 0.0
            for d in range(1, last + 1):
                if terms[d] != NEG_INF:
                    acc += np.exp(terms[d] - top)
            beta[u, s] = top + np.log(acc) - E[u, s]
        if u > 0:
            for s in range(S):
                acc = NEG_INF
                for s2 in range(S):
                    acc = _logaddexp(acc, trans[u, s, s2] + beta[u, s2])
                outgoing[u, s] = acc

    log_z = NEG_INF
    for s in range(S):
        log_z = _logaddexp(log_z, alpha[N, s])
    # P(a segment of s starts at u) and P(one ends at v); occupancy is their running difference.
    occupancy = np.zeros((N, S))
    for s in range(S):
        running = 0.0
        for t in range(N):
            running += np.exp(incoming[t, s] + beta[t, s] - log_z)
            if t > 0:
                running -= np.exp(alpha[t, s] + outgoing[t, s] - log_z)
            occupancy[t, s] = running
    return occupancy, log_z


def posteriors(E, log_dur, trans, start):
    """(occupancy (N, S): P(frame t is in state s), log Z)."""
    E, log_dur, trans, start = _as_arrays(E, log_dur, trans, start)
    occupancy, log_z = _forward_backward_kernel(E, log_dur, trans, start)
    if not np.isfinite(log_z):
        raise ValueError("no segmentation is possible with these durations and transitions")
    np.clip(occupancy, 0.0, 1.0, out=occupancy)
    return occupancy, float(log_z)


def _as_arrays(E, log_dur, trans, start):
    E = np.ascontiguousarray(E, dtype=np.float64)
    log_dur = np.ascontiguousarray(log_dur, dtype=np.float64)
    trans = np.ascontiguousarray(trans, dtype=np.float64)
    start = np.ascontiguousarray(start, dtype=np.float64)
    N, S = E.shape[0] - 1, E.shape[1]
    if log_dur.shape[0] != S or trans.shape != (N + 1, S, S) or start.shape != (S,):
        raise ValueError("shapes: E {}, log_dur {}, trans {}, start {}".format(
            E.shape, log_dur.shape, trans.shape, start.shape))
    return E, log_dur, trans, start


def cumulative(emissions):
    """(N, S) per-frame log emissions -> (N+1, S) cumulative, the E the kernels take."""
    out = np.zeros((emissions.shape[0] + 1, emissions.shape[1]))
    np.cumsum(emissions, axis=0, out=out[1:])
    return out
