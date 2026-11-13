"""Train the experimental TCN (pompeu.experimental.tcn) on cached external TRAIN items.

    python scripts/experiments/tcn_train.py --out cache/tcn/tcn_v1.pt
    python scripts/experiments/tcn_train.py --out cache/tcn/tcn_v1b.pt    # determinism re-run

Items come from scripts/experiments/tcn_build.py (cache/tcn/). Train: external `train` (+ the
SkillCorner extra_train items with --extra). Early stopping (best checkpoint by cross-entropy) on
the `val` items (degradation seeds distinct from the ones tcn_val.py scores). Never `test`.
Prints only aggregate losses and the sha256 of the final weights.
"""

import argparse
import glob
import hashlib
import io
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from pompeu import external  # noqa: E402
from pompeu.experimental import tcn  # noqa: E402


def items_for(names, tag):
    paths = []
    for name in names:
        safe = name.replace(":", "_")
        paths += sorted(p for p in glob.glob(os.path.join(tcn.CACHE, safe + "*_{}.npz".format(tag)))
                        if os.path.basename(p).startswith(safe + "_"))
    return paths


def drop_tags(item):
    keep = [i for i, n in enumerate(item["names"]) if not n.startswith("tag:")]
    return dict(item, X=item["X"][:, keep], names=[item["names"][i] for i in keep])


def state_hash(model):
    import torch
    buffer = io.BytesIO()
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()[:16]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("--tag", default="v1")
    parser.add_argument("--extra", action="store_true")
    parser.add_argument("--no-tags", action="store_true",
                        help="ablation: drop the synthetic Smart Tagging channels")
    parser.add_argument("--extra-seeds", type=int, nargs="+", default=None,
                        help="only these degradation seeds of the extra_train items")
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--window", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--channels", type=int, default=tcn.CONFIG["channels"])
    parser.add_argument("--dropout", type=float, default=tcn.CONFIG["dropout"])
    parser.add_argument("--input-dropout", type=float, default=tcn.CONFIG["input_dropout"])
    args = parser.parse_args()
    tcn.set_determinism(args.seed)
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = time.time()

    val_names = list(external.SPLITS["val"])                               # never "test"
    train_paths = items_for(list(external.SPLITS["train"]), args.tag)
    if args.extra:
        extra = items_for(list(external.EXTRA_TRAIN), args.tag)
        if args.extra_seeds is not None:
            extra = [p for p in extra if any("_s{}_".format(s) in os.path.basename(p)
                                             for s in args.extra_seeds)]
        train_paths += extra
    val_paths = items_for(val_names, args.tag)
    # Normalisation from the train split's own items (the extra matches only add examples);
    # inputs kept as float16 in memory, cast per batch.
    load = (lambda p: drop_tags(tcn.load_item(p))) if args.no_tags else tcn.load_item
    train = [load(p) for p in train_paths[:len(items_for(list(external.SPLITS["train"]),
                                                                 args.tag))]]
    norm = tcn.fit_normalisation(train)
    Z, Y = [], []
    for p in train_paths:
        it = load(p)
        Z.append(tcn.normalise(it["X"], it["names"], norm).astype(np.float16))
        Y.append(it["y"].astype(np.int64))
    val = [load(p) for p in val_paths]
    Zv = [tcn.normalise(it["X"], it["names"], norm) for it in val]
    Yv = [it["y"].astype(np.int64) for it in val]
    del train, val                                   # raw inputs no longer needed
    n_in = Z[0].shape[1]
    print("train items {}, val items {}, inputs {}, device {}".format(len(Z), len(Zv), n_in,
                                                                    device.type), flush=True)

    config = dict(tcn.CONFIG, channels=args.channels, dropout=args.dropout,
                  input_dropout=args.input_dropout)
    model = tcn.build_model(n_in, config).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps,
                                                pct_start=0.1)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tcn.X_CODE)
    rng = np.random.default_rng(args.seed)
    lengths = np.array([len(z) for z in Z], dtype=np.float64)
    probs = lengths / lengths.sum()
    L = args.window

    def evaluate():
        model.eval()
        total, count, correct, graded = 0.0, 0, 0, 0
        with torch.no_grad():
            for z, y in zip(Zv, Yv):
                out = model(torch.as_tensor(z[None], device=device))[0]       # (heads, T, 3)
                yt = torch.as_tensor(y, device=device)
                for h in range(out.shape[0]):
                    keep = yt[h] != tcn.X_CODE
                    total += float(torch.nn.functional.cross_entropy(
                        out[h][keep], yt[h][keep], reduction="sum"))
                    count += int(keep.sum())
                keep = yt[1] != tcn.X_CODE
                correct += int((out[1].argmax(-1)[keep] == yt[1][keep]).sum())
                graded += int(keep.sum())
        model.train()
        return total / count, correct / graded

    best = (float("inf"), None, -1)
    model.train()
    running = []
    for step in range(1, args.steps + 1):
        picks = rng.choice(len(Z), size=args.batch, p=probs)
        xb, yb = [], []
        for i in picks:
            start = int(rng.integers(0, max(1, len(Z[i]) - L)))
            xb.append(Z[i][start:start + L].astype(np.float32))
            yb.append(Y[i][:, start:start + L])
        xb = torch.as_tensor(np.stack(xb), device=device)
        yb = torch.as_tensor(np.stack(yb), device=device)                     # (B, heads, L)
        out = model(xb)                                                       # (B, heads, L, 3)
        loss = loss_fn(out.reshape(-1, 3), yb.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        running.append(float(loss))
        if step % args.eval_every == 0 or step == args.steps:
            vloss, vacc = evaluate()
            if vloss < best[0]:
                best = (vloss, {k: v.detach().clone() for k, v in model.state_dict().items()}, step)
            print("step {} train_ce {:.4f} val_ce {:.4f} val_acc5 {:.4f} ({:.0f} s)".format(
                step, np.mean(running), vloss, vacc, time.time() - started), flush=True)
            running = []
    model.load_state_dict(best[1])
    weights_hash = state_hash(model)
    tcn.save(model.cpu(), norm, args.out,
             extra={"best_step": best[2], "best_val_ce": best[0], "train_items": len(Z),
                    "val_items": len(Zv), "args": vars(args), "weights_sha": weights_hash})
    print("best step {} val_ce {:.4f}; weights sha256[:16] {}; {:.0f} s".format(
        best[2], best[0], weights_hash, time.time() - started))


if __name__ == "__main__":
    main()
