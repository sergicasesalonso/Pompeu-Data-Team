"""B3 + B8: build ATD+ from the tactical ATD and the broadcast ball aligned to it.

    python scripts/build_atd_plus.py                      # -> cache/atd_plus_broadcast_v2.npz
    python scripts/build_atd_plus.py --out cache/atd_plus_broadcast_v1.npz         --max-fit-m 2 --max-speed-ms 0 --ref-max-m 0      # v1: no candidate filters

Loads the tactical ATD and cleans it exactly as the pipeline does (clean_tracking with {}),
aligns the broadcast ATD to it on player positions (pompeu.io.broadcast.align), turns every
well-fitted broadcast ball into a candidate (broadcast.ball_candidates) and fuses them with
pompeu.vision.fuse.augment_ball. v2 filters the candidates first (see ARQUITECTURA.md, B8): player
fit <= 1 m, broadcast ball speed <= 25 m/s, and no candidate in a stretch where the broadcast ball
disagrees with the tactical ATD ball (median > 1.5 m over the +/- 125 rows of the same shot). The result is saved with Tracking.save at the path a config's
`tracking:` resolves to (pompeu.io.contract.load_tracking joins relative paths to the repository
root), so configs/*_atdplus.yaml read it as-is. The saved Tracking is ALREADY CLEANED: those
configs carry no `clean:` key.

Prints counts of the fusion report and a sha256 of the saved arrays (determinism check) only.
"""

import argparse
import hashlib
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pompeu import paths  # noqa: E402
from pompeu.clean import clean_tracking  # noqa: E402
from pompeu.io import broadcast  # noqa: E402
from pompeu.io.atd import load_atd  # noqa: E402
from pompeu.io.contract import ARRAY_FIELDS, Tracking, load_tracking  # noqa: E402
from pompeu.vision import fuse  # noqa: E402
from starter import load_homography  # noqa: E402

# Params
OUT = os.path.join("cache", "atd_plus_broadcast_v2.npz")   # relative to the repository root
MAX_FIT_M = 1.0        # per-frame player fit for a broadcast ball to become a candidate
MAX_SPEED_MS = 25.0    # broadcast ball speed above which it is likely in the air; 0 = off
REF_MAX_M = 1.5        # local median disagreement with the tactical ball; 0 = off
REF_WINDOW = 125       # +/- tactical rows of the same shot for that median (5 s)


def tracking_hash(tracking):
    """sha256 over frame0, meta and every array of the contract (dtype, shape and bytes)."""
    h = hashlib.sha256()
    h.update(repr(int(tracking.frame0)).encode())
    h.update(repr(tracking.meta).encode())
    for name in ARRAY_FIELDS:
        a = np.asarray(getattr(tracking, name))
        if a.dtype.kind == "f":
            # One NaN and one zero, whatever bits the operation that made them left behind.
            a = np.where(np.isnan(a), np.nan, a + 0.0).astype(a.dtype)
        a = np.ascontiguousarray(a)
        h.update(name.encode())
        h.update(str(a.dtype).encode())
        h.update(repr(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def build(max_fit_m=MAX_FIT_M, max_speed_ms=MAX_SPEED_MS, ref_max_m=REF_MAX_M):
    raw = load_tracking("atd")
    clean, _ = clean_tracking(raw, {})
    bc = Tracking.from_atd(load_atd(source="broadcast"))
    _, _, valid = load_homography.load(source="broadcast")
    # Aligned on the raw tactical players, as scripts/broadcast_coverage.py validated; rows are
    # the same in the cleaned Tracking.
    al = broadcast.align(raw, bc, valid)
    # The reference is the raw tactical ball (observed frames only), the alignment's own rows.
    ref = raw.ball_xy if ref_max_m > 0 else None
    cands, cstats = broadcast.ball_candidates(
        al, bc, max_fit_m=max_fit_m, frame0=clean.frame0,
        max_speed_ms=max_speed_ms if max_speed_ms > 0 else None,
        reference_xy=ref, ref_window=REF_WINDOW, ref_max_m=ref_max_m, return_stats=True)
    plus, report = fuse.augment_ball(clean, [cands])
    report["candidate_filter"] = cstats
    return plus, report


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=OUT, help="relative to the repository root, or absolute")
    parser.add_argument("--max-fit-m", type=float, default=MAX_FIT_M)
    parser.add_argument("--max-speed-ms", type=float, default=MAX_SPEED_MS, help="0 = off")
    parser.add_argument("--ref-max-m", type=float, default=REF_MAX_M, help="0 = off")
    args = parser.parse_args()

    plus, r = build(args.max_fit_m, args.max_speed_ms, args.ref_max_m)
    path = args.out if os.path.isabs(args.out) else os.path.join(paths.ROOT, args.out)
    plus.save(path)

    c = r["candidate_filter"]
    print("filter      {} broadcast balls with fit <= {} m; dropped speed {}, reference {} "
          "(no evidence {}); kept {}".format(c["candidates"], args.max_fit_m, c["dropped_speed"],
                                            c["dropped_reference"], c["reference_no_evidence"],
                                            c["kept"]))
    print("candidates  {} broadcast ball candidates in".format(r["candidates_in"]))
    print("coverage    graded frames with ball: {} -> {} of {} ({:.1%} -> {:.1%})".format(
        r["graded_with_ball_before"], r["graded_with_ball_after"], r["graded_frames"],
        r["graded_coverage_before"], r["graded_coverage_after"]))
    print("accepted    {} {}".format(r["accepted"], r["accepted_per_source"]))
    print("on ATD      {} agree, {} conflict (ATD kept)".format(r["on_atd_agree"],
                                                             r["on_atd_conflict"]))
    print("rejected    gate {}, unsupported {}, same frame {}, spike {}, fragment {}".format(
        r["rejected_gate_atd"], r["rejected_unsupported"], r["rejected_same_frame"],
        r["rejected_spike"], r["rejected_fragment"]))
    print("interpolated {} (old A1 fills cleared {})".format(r["interpolated"],
                                                         r["old_interp_cleared"]))
    print("saved       {}  sha256(arrays) {}".format(args.out, tracking_hash(Tracking.load(path))))


if __name__ == "__main__":
    main()
