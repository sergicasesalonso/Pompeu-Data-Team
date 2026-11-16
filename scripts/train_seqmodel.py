"""Train A7's sequence model on the external `train` matches; early stopping and calibration on `val`.

    python scripts/train_seqmodel.py                           # artifacts/seqmodel_v1.pt
    python scripts/train_seqmodel.py --seed 3 --out artifacts/seqmodel_v1_s3.pt   # ensemble member

Data: scripts/build_sequences.py must have built the train (8 seeds x team swap) and val
sequences. Each epoch samples random 120-s windows over all train sequences; the degradations
differ between seeds, so the model rarely sees the same corruption twice (the plan's "new
degradation every epoch", from a pool of 8 per match).

Loss: focal cross-entropy (gamma 2) on H/A/D with inverse-frequency class weights, X ignored,
plus 0.5 x binary cross-entropy of the boundary head against a Gaussian (+/- 1 step) around every
label change. Validation: the whole val sequences, predicted exactly as at inference; the best
epoch by val log-loss is kept (patience 4). Then the A8 calibration: one temperature fitted on
val by log-loss (bounded search), stored with the model. Deterministic given the seed.
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pompeu import external, paths  # noqa: E402
from pompeu.external import examples  # noqa: E402
from pompeu.models import seqmodel  # noqa: E402

TRAIN_SEEDS = range(8)
VAL_SEEDS = range(2)


def sequences(names, seeds, swaps, norm=None):
    """[(players f16, mask, ball f16, features f16, y int8)] and the normalisation used."""
    tables = []
    for name in names:
        for seed in seeds:
            for swap in swaps:
                F, y = examples.load(name, seed, swap=swap)
                tokens = examples.tokens(name, seed, swap=swap)
                tables.append((F, tokens))
    if norm is None:
        norm = seqmodel.fit_normalisation([F for F, _ in tables[:20]])
    out = []
    k = seqmodel.CONFIG["max_players"]
    for F, tokens in tables:
        players, mask = seqmodel.pad_players(tokens, k)
        features = seqmodel.feature_inputs(F, norm)
        n = min(len(players), len(features), len(tokens["y"]))
        out.append((players[:n].astype(np.float16), mask[:n], tokens["ball"][:n].astype(np.float16),
                    features[:n].astype(np.float16), tokens["y"][:n].astype(np.int8)))
    return out, norm


def boundary_target(y):
    """Gaussian (sigma 1 step) around every change of label between graded steps."""
    change = np.zeros(len(y), dtype=np.float32)
    idx = np.flatnonzero((y[1:] != y[:-1]) & (y[1:] != 3) & (y[:-1] != 3)) + 1
    out = np.zeros(len(y), dtype=np.float32)
    for offset, value in ((-2, 0.135), (-1, 0.607), (0, 1.0), (1, 0.607), (2, 0.135)):
        where = np.clip(idx + offset, 0, len(y) - 1)
        out[where] = np.maximum(out[where], value)
    return np.maximum(out, change)


def sample_batch(data, batch, window, rng):
    picks = []
    while len(picks) < batch:
        s = int(rng.integers(len(data)))
        t = len(data[s][4])
        if t <= window:
            continue
        start = int(rng.integers(0, t - window))
        if np.mean(data[s][4][start:start + window] == 3) > 0.5:
            continue
        picks.append((s, start))
    arrays = [np.stack([data[s][i][start:start + window] for s, start in picks]) for i in range(5)]
    arrays.append(np.stack([boundary_target(data[s][4][start:start + window]) for s, start in picks]))
    return arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--windows", type=int, default=1200, help="windows per epoch")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--out", default=seqmodel.ARTIFACT)
    args = parser.parse_args()

    import torch
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)

    started = time.time()
    train, norm = sequences(external.resolve("train"), TRAIN_SEEDS, (False, True))
    val, _ = sequences(external.resolve("val"), VAL_SEEDS, (False,), norm)
    print("train {} sequences ({} steps), val {} ({:.0f} s)".format(
        len(train), sum(len(d[4]) for d in train), len(val), time.time() - started), flush=True)

    labels = np.concatenate([d[4] for d in train])
    counts = np.array([np.sum(labels == c) for c in range(3)], dtype=np.float64)
    class_weight = torch.tensor(counts.sum() / (3 * counts), dtype=torch.float32, device=device)
    model = seqmodel.build_model(train[0][3].shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps_per_epoch = args.windows // args.batch
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr,
                                                    total_steps=args.epochs * steps_per_epoch,
                                                    pct_start=0.05, anneal_strategy="cos")
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    window = seqmodel.CONFIG["window"]

    def loss_fn(logits, boundary_logits, y, boundary):
        valid = y != 3
        target = torch.where(valid, y, torch.zeros_like(y)).long()
        log_p = torch.log_softmax(logits.float(), dim=-1)
        picked = log_p.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        focal = -((1 - picked.exp()) ** 2) * picked * class_weight[target]
        focal = (focal * valid).sum() / valid.sum().clamp(min=1)
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            boundary_logits.float(), boundary, pos_weight=torch.tensor(5.0, device=device),
            reduction="none")
        return focal + 0.5 * (bce * valid).sum() / valid.sum().clamp(min=1)

    def validate(temperature=1.0):
        model.eval()
        nll, correct, total = 0.0, 0, 0
        outs = []
        for players, mask, ball, features, y in val:
            lp = seqmodel.predict_steps(model, players.astype(np.float32), mask,
                                        ball.astype(np.float32), features.astype(np.float32),
                                        temperature)
            graded = y != 3
            outs.append((lp, y))
            nll -= lp[graded, y[graded]].sum()
            correct += int(np.sum(lp[graded].argmax(1) == y[graded]))
            total += int(graded.sum())
        model.train()
        return nll / total, correct / total, outs

    best, best_epoch, waited = np.inf, -1, 0
    for epoch in range(args.epochs):
        model.train()
        epoch_loss, t0 = 0.0, time.time()
        for _ in range(steps_per_epoch):
            players, mask, ball, features, y, boundary = sample_batch(train, args.batch, window, rng)
            batch = [torch.as_tensor(a, device=device) for a in (players, mask, ball, features)]
            batch = [b.float() if b.dtype == torch.float16 else b for b in batch]
            y_t = torch.as_tensor(y, device=device).long()
            boundary_t = torch.as_tensor(boundary, device=device)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits, boundary_logits = model(*batch)
                loss = loss_fn(logits, boundary_logits, y_t, boundary_t)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            epoch_loss += float(loss)
        val_nll, val_acc, _ = validate()
        print("epoch {:>2}  train {:.4f}  val nll {:.4f}  acc {:.4f}  ({:.0f} s)".format(
            epoch, epoch_loss / steps_per_epoch, val_nll, val_acc, time.time() - t0), flush=True)
        if val_nll < best - 1e-4:
            best, best_epoch, waited = val_nll, epoch, 0
            seqmodel.save(model, norm, args.out)
        else:
            waited += 1
            if waited >= args.patience:
                break

    model, norm = seqmodel.load(args.out, device=device)
    _, _, outs = validate()
    # A8: one temperature by val log-loss (golden-section search over [0.25, 4]).
    def nll_at(temperature):
        total, count = 0.0, 0
        for lp, y in outs:
            scaled = lp / temperature
            scaled = scaled - np.logaddexp.reduce(scaled, axis=1, keepdims=True)
            graded = y != 3
            total -= scaled[graded, y[graded]].sum()
            count += graded.sum()
        return total / count
    lo, hi = 0.25, 4.0
    for _ in range(40):
        a, b = lo + (hi - lo) * 0.382, lo + (hi - lo) * 0.618
        if nll_at(a) < nll_at(b):
            hi = b
        else:
            lo = a
    temperature = (lo + hi) / 2
    seqmodel.save(model, norm, args.out, extra={
        "temperature": temperature, "best_epoch": best_epoch, "val_nll": best,
        "val_nll_calibrated": nll_at(temperature), "seed": args.seed})
    print("best epoch {}  val nll {:.4f}  temperature {:.3f} -> nll {:.4f}".format(
        best_epoch, best, temperature, nll_at(temperature)))


if __name__ == "__main__":
    main()
