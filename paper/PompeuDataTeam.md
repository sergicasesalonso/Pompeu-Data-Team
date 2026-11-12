# Pompeu Data Team — Frame-level ball possession

More Than A Hack 2027, Phase 1 · Martí, Sergi, Ander · Repository: https://github.com/sergicasesalonso/Pompeu-Data-Team

> DRAFT. Sections marked **TODO** are filled in as the work lands. Final numbers come from
> `results/leaderboard.csv` (external bench, `test` split) once the configuration is frozen on 16-11.

## 1. Summary

- **Task:** one label per frame (`H`, `A`, `D`, `X`) for the 199,319 frames of the tactical video.
- **Submission:** `configs/final.yaml` → **TODO: final method name**. External-bench spellF1 **TODO** (macroF1 **TODO**).
- **In one paragraph:** **TODO**. Clean the automatic tracking, detect dead-ball periods with an HMM, detect touches and controls, and turn them into possessions with a state machine or a semi-Markov model. Validate everything on degraded external matches with real labels, because the public five minutes may not be used for tuning. **TODO, Line B:** add more ball observations from the video and the broadcast feed.

## 2. Approach and why

Two parallel lines connected by one data contract (`Tracking`, `pompeu/io/contract.py`):

- **Line A, scoring with the provided assets.** A1 cleaning → A2 per-frame features → A3 dead-ball detector → A4 touches and a possession state machine (`rules_v0`) → A5 hidden semi-Markov model (`hsmm_v1`) → **TODO** A7 sequence model, A8 calibration, A9 MBR decoding for spellF1, A10 final label policy.
- **Line B, better data from the video.** **TODO** B0 video checks, B1 pseudo-labels, B2 ball detector, B3 ball tracker and fusion (ATD+), B7 referee whistle, B8 broadcast alignment.

Why this design:
- spellF1 scores whole possessions, so the output has to be a segmentation, not per-frame votes. The baseline has 594 possessions and our models about 250 (**TODO** final).
- The ball is missing on 41 % of graded frames, and that is where possession changes are lost. Both lines target this.
- Every module reads and writes the same `Tracking`, so the Line B tracking (ATD+) drops into any Line A method unchanged.

## 3. Assets used and dropped

| Asset | Used for | Status |
|---|---|---|
| Tactical ATD (players and ball) | Everything | Used |
| Tactical homography | Metres ↔ pixels (B1 pseudo-labels, B3) | **TODO** |
| Smart Tagging | Team evidence (attacking phases = team in possession; defending = team without), set-piece restarts | Used |
| Tactical video | **TODO** B0–B2, B7 | **TODO** |
| Broadcast ATD and homography | **TODO** B8: extra ball observations after alignment | **TODO** |
| Broadcast video | Not used | Dropped (**TODO** reason) |
| External data (DFL, Metrica) | Validation bench and fitting the few model weights | Used, see §6 |

Tried and dropped: **TODO**. For example, A1 cleaning alone did not change the baseline score: the majority vote already absorbed the ball spikes. Its value shows up in A2–A4.

## 4. Handling missing and noisy tracking

- **Ball:** remove spikes (> 40 m/s), drop fragments shorter than 3 frames, spline-fill gaps under 0.5 s. Some ATD balls are "phantom" (still, with no player within 7–10 m); every event needs a player within 2 m, so phantom balls never cause a change.
- **Players:** frozen slots are flagged (24 and 46 are goalkeepers frozen off-screen), duplicates removed, tracklets stitched with the Hungarian algorithm.
- **Velocities** are recomputed, because the ATD velocity column is empty on 89 % of frames.
- **No ball:** features that need the ball are NaN, never imputed. Collective features (centroids, lines, pressing) and Smart Tagging work without the ball.
- **TODO, Line B:** how much ball the detector and the broadcast add (% of graded frames with a ball, before and after).

## 5. Possession rules implemented

- **Restarts:** a possession starts with the restarting team: throw-in for the opponent of the last player near the ball, corner or goal kick from the exit side and the last touch, kick-off after a goal for the team that conceded.
- **Change of possession:** needs control by the opponent: two touches in a row within 3 s, or the ball within 1.5 m of the same player for at least 0.6 s. A single opponent touch is a deflection and does not count.
- **Aerial balls:** a touch with the ball above 15 m/s before and after is ignored, because the ball flew past.
- **Loose or contested ball:** possession stays with the last team in control.
- **Timing:** once confirmed, a change is dated back to the opponent's first touch.
- **Dead ball:** `D` from the ball leaving the pitch, or play stopping, to the restart kick. **TODO** whistle (B7).
- **Final label policy (A10):** no `X` on graded frames. **TODO**.

## 6. Training and validation data

**No human labels on the provided match are used, in any amount** (RULES.md). The five public minutes are used only to check that the pipeline is correct, never to train, tune or choose.

| Dataset | Source and licence | Labels: who and how | How much | Used for |
|---|---|---|---|---|
| DFL / IDSSE open data (Bassek et al., 2025) | **TODO** link and licence | Per-frame `BallPossession` and `BallStatus` from the provider | 7 Bundesliga matches | Splits `train` / `val` / `test` |
| Metrica Sports sample data | **TODO** link and licence | Derived from the provider's event data | 3 matches | Splits `train` / `val` / `test` |
| Pseudo-labels (B1) | Our own code | ATD ball projected through the homography; no person involved | **TODO** frames | Ball detector training |

- **Splits** are by whole match: `test` (DFL J03WMX, J03WPY, Metrica 3) decides; `val` (DFL J03WN1, Metrica 2) is used to fit weights; `train` is for everything that learns.
- **Degradation:** the external matches have the ball erased with the gap-length distribution measured on our ATD, up to about 41 % missing. **TODO** other noise.
- **Metrics** are computed with the organisers' `evaluation/possession_metrics.py`, never reimplemented, with block-bootstrap 5–95 % intervals.

## 7. Results

**TODO: final table from `results/leaderboard.csv`, external `test` split.**

| Method | external spellF1 | external macroF1 | Possessions in the match |
|---|---|---|---|
| baseline_smooth50 | **TODO** | **TODO** | 594 |
| rules_v0 | 0.513 (provisional) | **TODO** | 250 |
| hsmm_v1 | 0.496 (provisional) | **TODO** | 238 |
| hsmm_v1_rules | 0.526 (provisional) | **TODO** | 244 |
| **final** | **TODO** | **TODO** | **TODO** |

## 8. Limitations

- **TODO.** Changes of possession inside long ball gaps are still missed.
- Few touches are detected (about 8 per minute), so dating changes depends on proximity streaks.
- The external matches come from other cameras and trackers; the degradation only imitates our gaps.
- **TODO:** what we would do with more time (B4 ball height, B5 player re-detection, a larger sequence model).

## 9. Reproduction

Python 3.11, `pip install -r requirements.lock`, match data in `data/` (see `data/README.md`), then:

```bash
python scripts/make_submission.py --config configs/final.yaml --out PompeuDataTeam.csv
python evaluation/validate_submission.py PompeuDataTeam.csv
```

Expected sha256: **TODO** (currently `bfd5ff16006b1e5c50371abeb528195cdc7a60397ec8bed90a8fccc468269f41` for `rules_v0`). **TODO:** GPU steps (detector weights, determinism settings) if Line B enters the final configuration.
