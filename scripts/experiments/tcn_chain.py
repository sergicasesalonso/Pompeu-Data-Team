"""One chained job for the tower: build items -> train twice -> compare weights -> sha256 twice.

    python scripts/experiments/tcn_chain.py

Runs the steps one after another (never more than one heavy process at a time), stops at the
first failure and records it in TORRE_METRICS. Only aggregated numbers are written: step
status, weights_sha of both trainings, sha256 of both submissions. The tower stays busy until it
finishes, so it does not power itself off in between.
"""

import json
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from pompeu import paths  # noqa: E402

PY = sys.executable
EXP = os.path.join(ROOT, "scripts", "experiments")
METRICS = {"step": "start"}
TRAIN = ["--extra", "--extra-seeds", "0", "--steps", "2500", "--eval-every", "100",
         "--dropout", "0.3", "--input-dropout", "0.2", "--weight-decay", "0.05"]


def save():
    path = os.environ.get("TORRE_METRICS")
    if path:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(METRICS, handle)


def run(name, argv, env_extra=None):
    METRICS["step"] = name
    save()
    sub = os.path.join(paths.CACHE_DIR, "chain_{}.json".format(name))
    env = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", TORRE_METRICS=sub)
    env.update(env_extra or {})
    started = time.time()
    code = subprocess.run([PY] + argv, cwd=ROOT, env=env, stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL).returncode
    METRICS[name + "_s"] = round(time.time() - started)
    METRICS[name + "_exit"] = code
    result = {}
    if os.path.isfile(sub):
        with open(sub, encoding="utf-8") as handle:
            result = json.load(handle)
    if code:
        METRICS["failed"] = name
        save()
        sys.exit(1)
    save()
    return result


def main():
    run("build_train", [os.path.join(EXP, "tcn_build.py"), "--split", "train", "--seeds",
                        "0", "1", "2", "3", "--workers", "4"])
    run("build_val", [os.path.join(EXP, "tcn_build.py"), "--split", "val", "--seeds", "10", "11"])
    run("build_extra", [os.path.join(EXP, "tcn_build.py"), "--split", "extra_train", "--seeds",
                        "0", "--workers", "4"])
    shas = []
    for tag in ("a", "b"):
        out = os.path.join(paths.CACHE_DIR, "tcn", "models", "chain_{}.pt".format(tag))
        result = run("train_" + tag, [os.path.join(EXP, "tcn_train.py"), "--out", out] + TRAIN)
        shas.append(result.get("weights_sha"))
        METRICS["weights_sha_" + tag] = shas[-1]
    METRICS["weights_equal"] = bool(shas[0]) and shas[0] == shas[1]
    save()
    if not METRICS["weights_equal"]:
        METRICS["failed"] = "weights_differ"
        save()
        sys.exit(1)
    for ext in (".pt", ".json"):
        src = os.path.join(paths.CACHE_DIR, "tcn", "models", "chain_a" + ext)
        if os.path.isfile(src):
            shutil.copyfile(src, os.path.join(ROOT, "artifacts", "tcn_v1" + ext))
    home = os.path.expanduser("~/mtah/agent-cache")
    for tag in ("1", "2"):
        result = run("sha_" + tag, [os.path.join(EXP, "hsmm_tcn_sha.py"), "--config",
                                    "configs/hsmm_tcn.yaml"],
                     {"POMPEU_CACHE_DIR": os.path.join(home, "chain_sha" + tag)})
        METRICS["sha256_" + tag] = result.get("sha256")
    METRICS["sha_equal"] = (METRICS["sha256_1"] is not None
                            and METRICS["sha256_1"] == METRICS["sha256_2"])
    METRICS["step"] = "done"
    save()


if __name__ == "__main__":
    main()
