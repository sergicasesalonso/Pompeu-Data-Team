"""Experiment: does a TEMPORAL model name the team in ball gaps better than gap_team's logistic?

    python scripts/experiments/gap_team_seq.py                       # full run (GPU if present)
    python scripts/experiments/gap_team_seq.py --smoke               # tiny CPU check
    python scripts/experiments/gap_team_seq.py --check-determinism   # train the first variant twice

QUESTION. pompeu.experts.gap_team (logistic regression on A2's collective group + 3 s means)
names the post-change team in ~75% of external-val gaps where possession changes. It sees one
frame at a time. A sequence model sees the whole context (+-60 s windows): the shape of the play
before and after the gap, and the team near the ball on the visible frames around it.

DATA. Train: the DFL matches of external `train` (Metrica rebuilds its ball from events, its gaps
hold no change of possession). Report: dfl:J03WN1 (`val`). `test` is never touched, nor is any
BAR-ATH file. Each match is degraded like the bench (degrade.PARAMS, seed 1000*seed + index,
coverage_sweep's log-normal gap lengths), cleaned with A1 defaults, and its features built exactly
as scripts/train_gap_team.py does (build_features collective group -> gap_team.design_matrix).

INPUTS at 5 Hz (one step per 5 frames of a half): the design_matrix columns at the step's first
frame, standardised with train in-half statistics (NaN -> 0), plus the share of the step's frames
with a (cleaned) ball and the mean nearest-team sign (+1 away, -1 home) over those frames.

MODEL. Linear -> 2-layer bidirectional GRU (hidden 64) -> logit of P(away). Random 120 s windows,
BCE on live frames (ground truth H/A) only; dead frames and frames outside the halves are masked.
Variants: "seq_all" (every live step weight 1) and "seq_gap" (live steps without ball weight 1,
with ball GAP_VISIBLE_WEIGHT). Fixed number of updates, no early stopping on val. Inference:
overlapping windows (stride WINDOW/4) blended with a Hann weight, then each 25 Hz frame takes the
nearest step.

METRICS. Exactly those of scripts/experiments/gap_team.py (its code is called): gap frames with
H/A ground truth whose gap has a last visible frame; overall, by gap length, by gap age, changed /
no-change gaps, post-change frames, changed gaps recovered. Side by side: carry, nearest, the
logistic fitted here on the same train frames (with and without intercept) and the shipped
configs/gap_team_v1.json (no intercept, as the HSMM uses it). Calibration: mean p vs empirical in
10 bins on the val gap frames. Only aggregates are printed. Weights and prepared arrays go to
POMPEU_CACHE_DIR/gap_team_seq and are never committed.
"""

import argparse
import hashlib
import importlib.util
import os
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from pompeu import external, paths  # noqa: E402
from pompeu.clean import clean_tracking  # noqa: E402
from pompeu.experts import gap_team  # noqa: E402
from pompeu.external import degrade  # noqa: E402
from pompeu.features.build import build_features  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "gap_team_experiment", os.path.join(ROOT, "scripts", "experiments", "gap_team.py"))
E = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E)

# Params
H, A, X = E.H, E.A, E.X
SUB = 5                         # 25 Hz -> 5 Hz
WINDOW_STEPS = 600              # 120 s at 5 Hz
HIDDEN = 64
LAYERS = 2
DROPOUT = 0.2
LR = 2e-3
WEIGHT_DECAY = 1e-4
BATCH = 32
ITERS = 3000
GAP_VISIBLE_WEIGHT = 0.25
LOGISTIC_STRIDE = 5
TRAIN = ["dfl:J03WOH", "dfl:J03WQQ", "dfl:J03WOY", "dfl:J03WR9"]
VAL = ["dfl:J03WN1"]
CACHE = os.path.join(paths.CACHE_DIR, "gap_team_seq")
PREP_VERSION = 1


# -- data --------------------------------------------------------------------------------------

def split_index(name):
    """The match's index inside its split's list (the degrade seed is 1000*seed + index)."""
    for split in ("train", "val"):
        names = [external.canonical(m) for m in external.SPLITS[split]]
        if name in names:
            return names.index(name)
    raise ValueError("{} is not a train/val match".format(name))


def prepare(name, seed, run_lengths, json_model):
    """Per-frame arrays of one degraded match, cached as npz (derived from public data only)."""
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, "{}_s{}_v{}.npz".format(name.replace(":", "_"), seed, PREP_VERSION))
    if os.path.isfile(path):
        with np.load(path) as z:
            return {k: z[k] for k in z.files}
    tracking, labels = external.load(name)
    fps = tracking.meta.get("fps", 25)
    if fps != 25:
        raise ValueError("expected 25 fps")
    half = tracking.half()
    gt = external.align_labels(np.where(half > 0, labels, X).astype(np.int8),
                               E.MIN_POSSESSION_S, fps)
    degraded = degrade.degrade(tracking, labels, 1000 * seed + split_index(name), run_lengths, None)
    cleaned, _ = clean_tracking(degraded, {})
    F = build_features(cleaned, tagging=None, with_windows=False, groups=("collective",))
    Xm = gap_team.design_matrix(F, fps)
    ball = np.asarray(cleaned.ball_present, bool)
    near = E.nearest_team(cleaned)
    gid, age, length, carry, nearc = E.gap_table(ball, half, gt, near)
    changed, post = E.change_masks(gid, gt, carry)
    out = dict(X=Xm.astype(np.float32), ball=ball, near=near.astype(np.int8), gt=gt, half=half,
               gid=gid, age=age / fps, length=length / fps, carry=carry, nearc=nearc,
               changed=changed, post=post,
               v1_logit=gap_team.away_logit(F, fps, json_model).astype(np.float64))
    np.savez(path, **out)
    return out


def sequences(d, mean, std):
    """[(inputs (T, d+2) float32, y (T,), live (T,), gap (T,), frames (T,))] one per half."""
    Z, _, _ = gap_team.standardise(d["X"].astype(np.float64), mean, std)
    sign = np.where(d["near"] == A, 1.0, np.where(d["near"] == H, -1.0, 0.0))
    sign = np.where(d["ball"], sign, 0.0)
    out = []
    for h in (1, 2):
        rows = np.flatnonzero(d["half"] == h)
        if len(rows) == 0:
            continue
        starts = rows[::SUB]
        T = len(starts)
        pad = T * SUB - len(rows)
        blk = np.concatenate([rows, np.full(pad, rows[-1])]).reshape(T, SUB)
        valid = np.concatenate([np.ones(len(rows), bool), np.zeros(pad, bool)]).reshape(T, SUB)
        bfrac = (d["ball"][blk] & valid).sum(1) / valid.sum(1)
        nsum = (sign[blk] * valid).sum(1)
        nvis = np.maximum((d["ball"][blk] & valid).sum(1), 1)
        inputs = np.column_stack([Z[starts], bfrac, nsum / nvis]).astype(np.float32)
        g = d["gt"][starts]
        out.append(dict(x=inputs, y=(g == A).astype(np.float32),
                        live=np.isin(g, (H, A)), gap=bfrac == 0, rows=rows, starts=starts))
    return out


def eval_rows(d):
    """The rows scripts/experiments/gap_team.py scores: gap frames, H/A truth, carry defined."""
    return (d["gid"] >= 0) & np.isin(d["gt"], (H, A)) & (d["carry"] >= 0)


# -- model -------------------------------------------------------------------------------------

def build_model(torch, n_in):
    nn = torch.nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Sequential(nn.Linear(n_in, HIDDEN), nn.GELU())
            self.gru = nn.GRU(HIDDEN, HIDDEN, num_layers=LAYERS, batch_first=True,
                              bidirectional=True, dropout=DROPOUT)
            self.out = nn.Linear(2 * HIDDEN, 1)

        def forward(self, x):
            h, _ = self.gru(self.inp(x))
            return self.out(h).squeeze(-1)

    return Net()


def train_model(torch, seqs, variant, seed, iters, device, monitor=None):
    """(model, seconds, chosen iteration). With `monitor` (model -> score, higher is better,
    computed on a held-out TRAIN match), the checkpoint with the best score is returned."""
    import copy
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = build_model(torch, seqs[0]["x"].shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=iters, pct_start=0.1)
    lengths = np.array([len(s["y"]) for s in seqs])
    prob = lengths / lengths.sum()
    L = WINDOW_STEPS
    started = time.time()
    best = (-np.inf, iters, None)
    model.train()
    for it in range(iters):
        xs, ys, ws = [], [], []
        for k in rng.choice(len(seqs), BATCH, p=prob):
            s = seqs[k]
            a = int(rng.integers(0, max(1, len(s["y"]) - L + 1)))
            sl = slice(a, a + L)
            w = s["live"][sl].astype(np.float32)
            if variant == "seq_gap":
                w = w * np.where(s["gap"][sl], 1.0, GAP_VISIBLE_WEIGHT).astype(np.float32)
            xs.append(s["x"][sl])
            ys.append(s["y"][sl])
            ws.append(w)
        x = torch.from_numpy(np.stack(xs)).to(device)
        y = torch.from_numpy(np.stack(ys)).to(device)
        w = torch.from_numpy(np.stack(ws)).to(device)
        logit = model(x)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, y, weight=w,
                                                                    reduction="sum")
        loss = loss / w.sum().clamp_min(1.0)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if (it + 1) % max(1, iters // 10) == 0:
            score = np.nan
            if monitor is not None:
                score = monitor(model)
                model.train()
                if score > best[0]:
                    best = (score, it + 1, copy.deepcopy(model.state_dict()))
            print("    {} it {:>5}/{}: loss {:.4f}  holdout {:.4f}  ({:.0f} s)".format(
                variant, it + 1, iters, loss.item(), score, time.time() - started), flush=True)
    if best[2] is not None:
        model.load_state_dict(best[2])
    return model, time.time() - started, best[1]


def predict(torch, model, seq, device):
    """(T,) logits over a whole half: Hann-blended overlapping windows."""
    model.eval()
    T, L = len(seq["y"]), WINDOW_STEPS
    stride = L // 4
    starts = list(range(0, max(1, T - L + 1), stride))
    if starts[-1] + L < T:
        starts.append(T - L)
    acc = np.zeros(T)
    wsum = np.zeros(T)
    hann = np.hanning(min(L, T) + 2)[1:-1]
    with torch.no_grad():
        for a in starts:
            x = torch.from_numpy(seq["x"][a:a + L][None]).to(device)
            out = model(x)[0].double().cpu().numpy()
            acc[a:a + len(out)] += out * hann[:len(out)]
            wsum[a:a + len(out)] += hann[:len(out)]
    return acc / np.maximum(wsum, 1e-12)


def upsample(d, seqs, step_logits):
    """(N,) logit per 25 Hz frame from the nearest 5 Hz step of its half; NaN outside halves."""
    out = np.full(len(d["gt"]), np.nan)
    for s, lg in zip(seqs, step_logits):
        pos = np.arange(len(s["rows"]))
        k = np.clip(np.round(pos / SUB).astype(int), 0, len(lg) - 1)
        out[s["rows"]] = lg[k]
    return out


# -- report ------------------------------------------------------------------------------------

def calibration(p, y, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    b = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, bins - 1)
    lines = ["  {:<12}{:>10}{:>10}{:>10}".format("bin", "n", "mean p", "empirical")]
    ece = 0.0
    for i in range(bins):
        m = b == i
        if m.sum():
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
            lines.append("  {:<12}{:>10}{:>10.3f}{:>10.3f}".format(
                "{:.1f}-{:.1f}".format(edges[i], edges[i + 1]), int(m.sum()), p[m].mean(),
                y[m].mean()))
    lines.append("  ECE {:.4f}   log-loss {:.4f}".format(
        ece, float(-np.mean(y * np.log(np.clip(p, 1e-6, 1)) +
                            (1 - y) * np.log(np.clip(1 - p, 1e-6, 1))))))
    return "\n".join(lines)


def report(preds, logits, val):
    y = val["y"]
    every = np.ones(len(y), bool)
    print()
    print(E.table("VAL accuracy on gap frames (live, H/A ground truth, carry defined)",
                  preds, y, [every], ["all"]))
    print()
    gb = E.bucket(val["length"], np.array(E.GAP_BUCKETS_S, float))
    print(E.table("by gap length", preds, y, [gb == i for i in range(len(E.GAP_BUCKETS_S) - 1)],
                  [E.edge_label(E.GAP_BUCKETS_S, i) for i in range(len(E.GAP_BUCKETS_S) - 1)]))
    print()
    ab = E.bucket(val["age"], np.array(E.AGE_BUCKETS_S, float))
    print(E.table("by time since gap start", preds, y,
                  [ab == i for i in range(len(E.AGE_BUCKETS_S) - 1)],
                  [E.edge_label(E.AGE_BUCKETS_S, i) for i in range(len(E.AGE_BUCKETS_S) - 1)]))
    print()
    changed, post = val["changed"], val["post"]
    print(E.table("possession changes inside the gap", preds, y,
                  [~changed, changed, post, changed & ~post],
                  ["no-chg gap", "chg gap", "post-chg", "chg pre"]))
    print()
    print("changed gaps where the method names the post-change team on > half of its "
          "post-change frames")
    for name, pred in preds.items():
        share, n = E.per_gap_recovery(pred, val)
        print("  {:<16}{:>8.3f}   (of {} changed gaps)".format(name, share, n))
    print()
    nochg_ids = np.unique(val["gid"][~changed])
    print("no-change gaps where the method keeps the right team on > half of their frames")
    for name, pred in preds.items():
        hits = sum(np.mean(pred[(val["gid"] == g)] == y[val["gid"] == g]) > 0.5
                   for g in nochg_ids)
        print("  {:<16}{:>8.3f}   (of {} no-change gaps)".format(name, hits / max(len(nochg_ids), 1),
                                                               len(nochg_ids)))
    print()
    print("calibration on val gap frames (p = P(away))")
    for name, lg in logits.items():
        print(" ", name)
        print(calibration(1.0 / (1.0 + np.exp(-np.clip(lg, -30, 30))), y))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--iters", type=int, default=ITERS)
    parser.add_argument("--variants", nargs="+", default=["seq_all", "seq_gap"])
    parser.add_argument("--train-seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true", help="1 train match, seed 0, 20 updates")
    parser.add_argument("--check-determinism", action="store_true",
                        help="train the first variant a second time and compare val logits")
    parser.add_argument("--device", default=None)
    parser.add_argument("--holdout", default=None,
                        help="a TRAIN match left out of training and used to pick the checkpoint "
                             "(gap-frame accuracy, as eval_rows); val is never used for it")
    args = parser.parse_args()
    started = time.time()

    import torch
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    train_names, val_names, seeds = TRAIN, VAL, args.seeds
    if args.smoke:
        train_names, seeds, args.iters = TRAIN[:1], [0], 20
    print("torch {}  device {}  train {}  val {}  seeds {}  iters {}".format(
        torch.__version__, device, train_names, val_names, seeds, args.iters))

    import coverage_sweep
    run_lengths = coverage_sweep.synthetic_run_lengths()
    json_model = gap_team.load_model("configs/gap_team_v1.json")
    tr = [prepare(m, s, run_lengths, json_model) for m in train_names for s in seeds]
    va = [prepare(m, s, run_lengths, json_model) for m in val_names for s in seeds]
    print("prepared {} train and {} val degraded matches ({:.0f} s)".format(
        len(tr), len(va), time.time() - started), flush=True)

    # Logistic baseline, fitted as scripts/train_gap_team.py does (train gap frames, stride 5).
    Xl = np.concatenate([d["X"][(d["half"] > 0) & ~d["ball"] & np.isin(d["gt"], (H, A))]
                         [::LOGISTIC_STRIDE] for d in tr]).astype(np.float64)
    yl = np.concatenate([(d["gt"][(d["half"] > 0) & ~d["ball"] & np.isin(d["gt"], (H, A))]
                          [::LOGISTIC_STRIDE] == A) for d in tr]).astype(float)
    Zl, lmean, lstd = gap_team.standardise(Xl)
    w, b = gap_team.fit_logistic(Zl, yl)
    print("logistic fitted on {} train gap frames".format(len(yl)))

    # Sequence inputs: standardised with train in-half statistics.
    Xin = np.concatenate([d["X"][d["half"] > 0] for d in tr]).astype(np.float64)
    _, smean, sstd = gap_team.standardise(Xin[::SUB])
    held = [d for d, (m, _) in zip(tr, [(m, s) for m in train_names for s in seeds])
            if m == args.holdout]
    tr = [d for d, (m, _) in zip(tr, [(m, s) for m in train_names for s in seeds])
          if m != args.holdout]
    tr_seqs = [s for d in tr for s in sequences(d, smean, sstd)]
    monitor = None
    if held:
        held_seqs = [sequences(d, smean, sstd) for d in held]
        held_rows = [eval_rows(d) for d in held]
        held_y = np.concatenate([(d["gt"][r] == A) for d, r in zip(held, held_rows)])

        def monitor(model):
            out = [upsample(d, sq, [predict(torch, model, s, device) for s in sq])[r]
                   for d, sq, r in zip(held, held_seqs, held_rows)]
            return float(np.mean((np.concatenate(out) > 0) == held_y))
        print("holdout {} ({} degraded copies) picks the checkpoint".format(args.holdout,
                                                                           len(held)))
    va_seqs = [sequences(d, smean, sstd) for d in va]
    print("train sequences {}, steps {}, live {:.3f}, gap {:.3f}".format(
        len(tr_seqs), sum(len(s["y"]) for s in tr_seqs),
        np.mean(np.concatenate([s["live"] for s in tr_seqs])),
        np.mean(np.concatenate([s["gap"] for s in tr_seqs]))), flush=True)

    # Val rows, as the logistic experiment.
    rows = [eval_rows(d) for d in va]
    val = {k: np.concatenate([d[k][r] for d, r in zip(va, rows)])
           for k in ("gid", "age", "length", "carry", "nearc", "changed", "post")}
    val["gid"] = np.concatenate([np.where(r, d["gid"] + 1_000_000 * j, -1)[r]
                                 for j, (d, r) in enumerate(zip(va, rows))])
    val["y"] = np.concatenate([(d["gt"][r] == A) for d, r in zip(va, rows)]).astype(float)
    logits = {}
    lg = [gap_team.standardise(d["X"][r].astype(np.float64), lmean, lstd)[0] @ w for d, r in
          zip(va, rows)]
    logits["logistic"] = np.concatenate(lg) + b
    logits["logistic_noint"] = np.concatenate(lg)
    logits["gap_team_v1"] = np.concatenate([d["v1_logit"][r] for d, r in zip(va, rows)])

    timings, hashes = {}, {}
    runs = list(args.variants) + ([args.variants[0] + "#2"] if args.check_determinism else [])
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for run in runs:
        variant = run.split("#")[0]
        print("training {}".format(run), flush=True)
        model, secs, chosen = train_model(torch, tr_seqs, variant, args.train_seed, args.iters,
                                          device, monitor)
        timings[run] = secs
        print("  {} checkpoint: iteration {}".format(run, chosen), flush=True)
        if not args.smoke:
            torch.save(model.state_dict(), os.path.join(CACHE, "{}.pt".format(run.replace("#", "_"))))
        full = []
        for d, seqs, r in zip(va, va_seqs, rows):
            step = [predict(torch, model, s, device) for s in seqs]
            full.append(upsample(d, seqs, step)[r])
        logits[run] = np.concatenate(full)
        hashes[run] = hashlib.sha256(np.round(logits[run], 6).tobytes()).hexdigest()[:16]

    preds = {"carry": (val["carry"] == A).astype(float),
             "nearest": (val["nearc"] == A).astype(float)}
    for name, lg in logits.items():
        if "#" not in name:
            preds[name] = (lg > 0).astype(float)
    report(preds, {k: v for k, v in logits.items() if "#" not in k}, val)

    print()
    for run, secs in timings.items():
        print("train time {:<10} {:.0f} s   val logits sha256[:16] {}".format(run, secs, hashes[run]))
    if args.check_determinism:
        a, b2 = args.variants[0], args.variants[0] + "#2"
        print("determinism: {} vs {}: {} (max |diff| {:.2e})".format(
            a, b2, "IDENTICAL" if hashes[a] == hashes[b2] else "DIFFERENT",
            float(np.max(np.abs(logits[a] - logits[b2])))))
    if device == "cuda":
        print("peak GPU memory {:.0f} MiB".format(torch.cuda.max_memory_allocated() / 2 ** 20))
    print("runtime {:.0f} s".format(time.time() - started))


if __name__ == "__main__":
    main()
