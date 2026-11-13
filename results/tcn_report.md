# Experimental TCN (marti-tcn): val results

Aggregate numbers only. External **val** (`dfl:J03WN1`, `metrica:2`), never `test`, never the public 5 minutes.

## Protocol

- **Bench:** `scripts/experiments/tcn_val.py` repeats `robust_val.py` exactly, with the same simulator as origin/sergi 959d0c4 (`degrade.py` is identical, with ATD-like ball blur): degradation seed `1000*seed + index`, seeds 0-4, the ATD's no-ball run lengths, A1 cleaning, and `gap_team_weight.score` (seeds joined, bootstrap 5-95 %) at `min_possession_s` 2/5/8. The HSMM candidates get no A10 policy, as in robust_val. Every candidate sees the same degraded match. The rules_v0 numbers below (0.4400/0.4706/0.4128) match the team's robust_val run of 2026-11-14T0115.
- **Model:** `pompeu/experimental/tcn.py`, a non-causal dilated TCN at 5 Hz. It has 96 channels, kernel 3 and dilations 1..128 (about ±51 s of context), with three heads, one per convention 2/5/8 s, averaged at inference. Inputs: A2 features (`build_features(tagging=None)`, `examples.model_columns`) plus the 18 Smart Tagging context columns computed from **synthetic tags** (marti-syntags error model, generated from the 2 s labels). The inputs are standardised and every column that can be NaN gets a missing flag. About 0.5 M parameters.
- **Training:** `scripts/experiments/tcn_train.py`. Data: external `train` (4 DFL + metrica:1), seeds 0-3, each in 4 augmentations (team swap and y-mirror, each recomputed from the clean match). Degradation uses the defaults (ball_missing 0.41, dead_weight 5). Settings: AdamW, one-cycle, 1500 steps of 16 × 1024-step windows, dropout 0.3, input dropout 0.2, weight decay 0.05. The checkpoint is chosen by cross-entropy on val items built with degradation seeds 10 and 11, which are not the seeds that are scored. **r1-gpu** is that configuration trained on the RTX 3060. **r2**, trained on CPU, also adds the 20 SkillCorner extra_train matches (seed 0, 4 augmentations), with 2500 steps.
- **HSMM+TCN:** a product of experts. The TCN's team log-odds `log P(H) − log P(A)` (weight `wt`) and dead log-odds `log P(D) − log(P(H)+P(A))` (weight `wd`) enter `hsmm_v1_fit_mbr` through the HSMM's learned-expert slot (`prep["learned"]`, scaled per look like every cue). The rows `avg+tcn` use `hsmm_v1_fit_avg` as the base.
- **Optimism:** `wt/wd`, the smoothing and the checkpoint step were picked on this same val. The grid is small (6 Viterbi + 4 MBR pairs, 3 smoothings), but the numbers are still in-sample. The baselines were fitted on val too.

## Results: val, 5 seeds, spellF1 (bootstrap 5-95 %)

| candidate | 2 s | 5 s | 8 s | mean | worst Δ vs rules_v0 | macroF1 (mean) | segments (true 609/429/343) |
|---|---|---|---|---|---|---|---|
| rules_v0 | 0.440 [0.411, 0.471] | 0.471 [0.438, 0.503] | 0.413 [0.378, 0.449] | 0.441 | 0 | 0.773 | 395 |
| hsmm_v1_fit_mbr | 0.470 [0.435, 0.507] | 0.561 [0.526, 0.595] | 0.511 [0.474, 0.546] | 0.514 | +0.030 | 0.804 | 343 |
| hsmm_v1_fit_avg | 0.478 [0.444, 0.514] | 0.554 [0.521, 0.588] | 0.504 [0.468, 0.536] | 0.512 | +0.038 | 0.804 | 357 |
| TCN alone, r1-gpu (train only), s=100 | 0.532 [0.499, 0.568] | 0.620 [0.589, 0.651] | 0.554 [0.522, 0.588] | 0.569 | +0.092 | 0.844 | 409 |
| HSMM+TCN r1-gpu, 1/2 MBR | 0.529 [0.495, 0.567] | 0.642 [0.607, 0.675] | 0.591 [0.556, 0.626] | 0.587 | +0.089 | 0.842 | 360 |
| TCN alone, r2 (+SkillCorner), s=100 | 0.563 [0.531, 0.600] | 0.655 [0.622, 0.688] | 0.567 [0.535, 0.600] | 0.595 | +0.123 | 0.851 | 413 |
| **HSMM+TCN r2, 1/2 MBR** | 0.546 [0.511, 0.584] | 0.659 [0.622, 0.693] | **0.605** [0.569, 0.639] | **0.603** | +0.106 | 0.845 | 360 |
| ablation: TCN without tag channels (r1-gpu settings), s=100 | 0.482 [0.449, 0.519] | 0.567 [0.533, 0.602] | 0.505 [0.470, 0.542] | 0.518 | +0.042 | 0.813 | 372 |
| ablation: HSMM+TCN without tags, 2/1 MBR (with tags, r1-gpu 2/1 MBR: 0.538 / 0.636 / 0.574, mean 0.583) | 0.513 [0.479, 0.550] | 0.603 [0.570, 0.639] | 0.547 [0.513, 0.583] | 0.554 | +0.073 | 0.824 | 365 |

"worst Δ vs rules_v0" is the smallest gain over the three conventions (positive = better everywhere). The decision-rule-B condition (loss ≤ 0.02) holds for every TCN row.

- **Gain over hsmm_v1_fit_mbr (r2, 1/2 MBR):** +0.076 at 2 s, +0.098 at 5 s, +0.094 at 8 s. The 5-95 % intervals do not overlap at 5 and 8 s and overlap only slightly at 2 s ([0.511, 0.584] vs [0.435, 0.507]).
- **Under-segmentation:** it eases but does not go away. The HSMM predicts 343 segments, HSMM+TCN 360 and the TCN alone 409-463 against 609 true segments at 2 s. The TCN alone with light smoothing (s=50: 0.578 at 2 s, 463 segments) is the best at 2 s and the worst at 8 s.
- **Tags matter, but they are not the whole gain.** Without the tag channels the TCN alone falls from 0.569 to 0.518 (mean). In the HSMM, the TCN without tags still adds +0.040 over hsmm_v1_fit_mbr (0.554 vs 0.514), and the synthetic tags add another ~+0.03.

## Determinism

- **Training:** two GPU runs with the same seed (`torch.use_deterministic_algorithms(True)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, seeded numpy/torch) give the same weights hash, `2c5aef8be2be8ff6`. The `.pt` files differ only in the archive name inside the zip.
- **Evaluation:** the D2 model (the twin of r1-gpu), scored in a separate run, gives exactly the same spellF1, macroF1, intervals and segment counts as r1-gpu on every candidate they share (rules_v0, both HSMM baselines, TCN s=100, HSMM+TCN 1/2 Viterbi, 2/1 MBR, avg+TCN 2/1).
- A CPU-trained run of the same configuration (r1-cpu) differs slightly from the GPU one (different device and float32 vs float16 storage), so the hash holds per device.

Compute: 3 GPU training runs (D1, D2, no-tags), 2072 s ≈ 35 min on the 3060, data-loader bound, at about 25 % utilisation. r2 trained on a laptop CPU in 14 min. The val scoring ran on the tower's CPU, about 17 min per full grid.

## Risks before trusting it

1. **Synthetic tags.** The tag channels are generated from the true labels with an error model measured on BAR-ATH without GT. If BAR-ATH's real Smart Tagging is less informative than our synthetic tags, the gain on the real match will be smaller. The ablation measures how much of the gain depends on tags. Next check: compare the distribution of the tag features on BAR-ATH, real tags, with the external matches, synthetic tags (no GT needed).
2. **In-sample selection on val** (weights, smoothing, early stopping). It needs the single team-decided `test` pass.
3. **Not reproducible in the pipeline yet:** no `METHODS` entry and no committed weights. Training is about 12 min on the 3060, or about 10 min on a laptop CPU, from cached inputs that `tcn_build.py` regenerates.

## Reproduce

```
python scripts/experiments/tcn_build.py --split train --seeds 0 1 2 3 --workers 5
python scripts/experiments/tcn_build.py --split val --seeds 10 11
python scripts/experiments/tcn_build.py --split extra_train --seeds 0 1 --workers 5     # r2 only
python scripts/experiments/tcn_train.py --out cache/tcn/models/r1.pt --steps 1500 --eval-every 100 --dropout 0.3 --input-dropout 0.2 --weight-decay 0.05
python scripts/experiments/tcn_train.py --out cache/tcn/models/r2.pt --extra --extra-seeds 0 --steps 2500 --eval-every 100 --dropout 0.3 --input-dropout 0.2 --weight-decay 0.05
python scripts/experiments/tcn_val.py --model cache/tcn/models/r2.pt --seeds 0 1 2 3 4 --workers 4 --smooth 50 100 150 --w-team 0.5 1 2 --w-dead 1 2 --mbr 1/0 2/1 1/2 4/1
```
