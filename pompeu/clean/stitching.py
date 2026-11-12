"""A1 stitching: join tracklets that are the same player, then repack them into fewer slots.

A slot is a tracklet, not a person: the tracker drops someone and picks them up again under another
slot. If tracklet i ends at frame e and tracklet j of the SAME team starts at s, with
0 < s - e <= max_gap (2 s) and the two positions within vmax * (s - e) / fps + margin metres, j
can be the continuation of i. Each end can take at most one start and vice versa, and the choice
is a minimum-cost assignment (Hungarian, scipy.optimize.linear_sum_assignment) with cost =
distance / reachable distance. The candidate graph is sparse, so the assignment is solved per
connected component, which keeps every matrix tiny.

Linked tracklets form chains. Chains of one team are packed into slots by interval partitioning
(earliest start first, into the slot that became free earliest), which uses exactly as many slots
as there are chains alive at once. The result is a Tracking where a slot is much closer to a
person: fewer, longer tracks, same observations. Gaps inside a chain shorter than `fill_max_s` are
bridged linearly at confidence `fill_conf`; longer ones stay NaN.
"""

import heapq

import numpy as np
from scipy.optimize import linear_sum_assignment

from pompeu.features.kinematics import runs

# Params
MAX_GAP_S = 2.0
VMAX = 9.0       # m/s, a sprinting footballer
MARGIN_M = 1.0
FILL_MAX_S = 1.0
FILL_CONF = 0.5
NO_LINK = 1e6


def tracklets(present):
    """[(slot, start, stop)] for every observed run, stop exclusive."""
    out = []
    for slot in range(present.shape[1]):
        starts, stops = runs(present[:, slot])
        out.extend((slot, int(a), int(b)) for a, b in zip(starts, stops))
    return out


def link(xy, team, pieces, fps, max_gap_s=MAX_GAP_S, vmax=VMAX, margin=MARGIN_M):
    """{i: j} tracklet i continues as tracklet j (indices into `pieces`)."""
    max_gap = int(round(max_gap_s * fps))
    starts = np.array([start for _, start, _ in pieces])
    order = np.argsort(starts, kind="stable")
    sorted_starts = starts[order]

    candidates = []  # (i, j, cost)
    for i, (slot_i, _, stop_i) in enumerate(pieces):
        end_frame = stop_i - 1
        lo = np.searchsorted(sorted_starts, end_frame + 1)
        hi = np.searchsorted(sorted_starts, end_frame + max_gap, side="right")
        for j in order[lo:hi]:
            slot_j, start_j, _ = pieces[j]
            if team[slot_j] != team[slot_i] or team[slot_i] < 0:
                continue
            gap = start_j - end_frame
            reach = vmax * gap / fps + margin
            distance = float(np.hypot(*(xy[start_j, slot_j] - xy[end_frame, slot_i])))
            if distance <= reach:
                candidates.append((i, int(j), distance / reach))

    # Connected components of the bipartite graph ends -> starts, via union-find.
    parent = {}

    def find(node):
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for i, j, _ in candidates:
        parent[find(("end", i))] = find(("start", j))
    groups = {}
    for i, j, cost in candidates:
        groups.setdefault(find(("end", i)), []).append((i, j, cost))

    links = {}
    for edges in groups.values():
        ends = sorted({i for i, _, _ in edges})
        begins = sorted({j for _, j, _ in edges})
        matrix = np.full((len(ends), len(begins)), NO_LINK)
        for i, j, cost in edges:
            matrix[ends.index(i), begins.index(j)] = cost
        rows, cols = linear_sum_assignment(matrix)
        for row, col in zip(rows, cols):
            if matrix[row, col] < NO_LINK:
                links[ends[row]] = begins[col]
    return links


def chains(pieces, links):
    """[[tracklet indices in time order]] -- every tracklet in exactly one chain."""
    continued = set(links.values())
    out = []
    for i in range(len(pieces)):
        if i in continued:
            continue
        chain = [i]
        while chain[-1] in links:
            chain.append(links[chain[-1]])
        out.append(chain)
    return out


def pack(pieces, groups, team):
    """[(chain, new slot)] and the team of every new slot: interval partitioning per team."""
    assignment, slot_team = [], []
    for side in sorted({int(team[pieces[c[0]][0]]) for c in groups}):
        mine = [c for c in groups if team[pieces[c[0]][0]] == side]
        mine.sort(key=lambda c: (pieces[c[0]][1], pieces[c[0]][0]))
        free = []  # heap of (frame the slot becomes free, slot)
        for chain in mine:
            start, stop = pieces[chain[0]][1], pieces[chain[-1]][2]
            if free and free[0][0] <= start:
                _, slot = heapq.heappop(free)
            else:
                slot = len(slot_team)
                slot_team.append(side)
            heapq.heappush(free, (stop, slot))
            assignment.append((chain, slot))
    return assignment, np.array(slot_team, dtype=np.int8)


def stitch(xy, conf, team, fps, max_gap_s=MAX_GAP_S, vmax=VMAX, margin=MARGIN_M,
           fill_max_s=FILL_MAX_S, fill_conf=FILL_CONF):
    """(new xy (N, K', 2), new conf (N, K'), new team (K',), report dict)."""
    present = ~np.isnan(xy[:, :, 0])
    pieces = tracklets(present)
    links = link(xy, team, pieces, fps, max_gap_s, vmax, margin)
    groups = chains(pieces, links)
    assignment, slot_team = pack(pieces, groups, team)

    n = xy.shape[0]
    new_xy = np.full((n, len(slot_team), 2), np.nan)
    new_conf = np.zeros((n, len(slot_team)), dtype=np.float32)
    fill_max = int(round(fill_max_s * fps))
    filled = 0
    for chain, slot in assignment:
        for index, piece in enumerate(chain):
            old, start, stop = pieces[piece]
            new_xy[start:stop, slot] = xy[start:stop, old]
            new_conf[start:stop, slot] = conf[start:stop, old]
            if index == 0:
                continue
            _, _, previous_stop = pieces[chain[index - 1]]
            gap = np.arange(previous_stop, start)
            if 0 < len(gap) <= fill_max:
                known = np.array([previous_stop - 1, start])
                for axis in (0, 1):
                    new_xy[gap, slot, axis] = np.interp(gap, known, new_xy[known, slot, axis])
                new_conf[gap, slot] = fill_conf
                filled += len(gap)

    lengths = np.array([stop - start for _, start, stop in pieces])
    chain_lengths = np.array([pieces[c[-1]][2] - pieces[c[0]][1] for c in groups])
    report = {
        "tracklets": len(pieces),
        "links": len(links),
        "chains": len(groups),
        "slots_before": int(present.any(axis=0).sum()),
        "slots_after": len(slot_team),
        "median_tracklet_s": float(np.median(lengths)) / fps if len(lengths) else 0.0,
        "median_chain_s": float(np.median(chain_lengths)) / fps if len(chain_lengths) else 0.0,
        "filled_frames": filled,
    }
    return new_xy, new_conf, slot_team, report
