# Pompeu Data Team: frame-level ball possession

More Than A Hack 2027, Phase 1 · Martí, Sergi, Ander · Code: https://github.com/sergicasesalonso/Pompeu-Data-Team

Implementation detail for every module (all parameters, file by file) is in `ARQUITECTURA.md` in the repository.

## 1. Summary

- **Task:** label each of the 199,319 frames of the tactical video `H`, `A`, `D` or `X`; ranked on spellF1.
- **Submission:** `configs/final.yaml`, which is `rules_v0`: A1 tracking cleaning, the A3 dead-ball HMM and the A4 possession state machine. **TODO:** final method name and its external `test` spellF1 and macroF1.
- **Approach.** Clean the tracking, segment live and dead ball with an HMM, detect touches and controls with their team, and turn them into possessions with a state machine encoding the challenge's control rule. A hidden semi-Markov model (HSMM) using the same cues plus ball-free evidence is the alternative. Methods are compared only on **other** matches with provider labels (DFL/IDSSE, Metrica), degraded to look like our tracking; conventions that touch the public window are in §6.
- **Ball from the video (Line B)** was explored and is not submitted: the broadcast adds about 3 points of ball coverage, worth about +0.01 spellF1, within the noise (§4).

## 2. Approach and why

Every module reads and writes one data contract, `Tracking` (`pompeu/io/contract.py`): metres on a 105.3 × 68.0 pitch, one row per video frame, a confidence and source per ball observation. **Line A** labels from the provided assets: A1 cleaning → A2 per-frame features (about 200 columns) → A3 dead-ball detector → A4 touches, controls and the `rules_v0` state machine → A5 `hsmm_v1`. **Line B** adds ball from the video and broadcast into an augmented tracking, "ATD+"; switching to it only changes `tracking:` in the YAML.

- **spellF1 scores possessions as objects** (runs matched by temporal IoU). Per-frame voting fragments them: on the external `test` bench `baseline_smooth50_clean` predicts 1,367 possessions against 692 true ones (§7). Hence a segmentation with explicit durations and transitions.
- **The ball is missing on 41% of graded frames** (CHALLENGE.md), where possession changes are lost. Line A uses ball-free evidence (collective shape, Smart Tagging, dead-ball structure); Line B tried to fill the gaps.

## 3. Assets used and dropped

| Asset | Used for | Status |
|---|---|---|
| Tactical ATD (players and ball) | Everything in Line A; source of B1 pseudo-labels | Used |
| Tactical homography | Metres to pixels for B0, B1, B3 | Used in Line B only |
| Smart Tagging | Team evidence (attacking phases vote for the team, defending phases against; zone codes do not vote), set-piece restarts, the `GOALS` phase | Used |
| Tactical video | B0: decoded frame count equals the ATD's (199,319); B1 pseudo-labels for a ball detector | Explored, not in the submission |
| Tactical audio (AAC, 48 kHz) | Referee whistle | Present; not used |
| Broadcast ATD and homography | Extra ball observations after alignment to the tactical clock on player positions | Explored, not in the submission (§4) |
| Broadcast video | Not needed: alignment uses only the broadcast ATD | Dropped |
| External open data (DFL/IDSSE, Metrica) | Validation bench; fitting the few model weights | Used, see §6 |

**Tried and dropped** (figures from `results/leaderboard.csv` and `scripts/min_possession_sensitivity.py`):

- **Smoothed per-frame voting** (`baseline_smooth25/50/125`): 746, 594 and 398 possessions on the full match (counts only) against 250 for `rules_v0`; the 50-frame version scores 0.253 on external `test`.
- **Voting with dead-ball segmentation** (`deadball_vote50` / `125`): 0.318 [0.291, 0.346] against 0.390 [0.355, 0.426] on an earlier two-match DFL bench (684 and 496 possessions for 474 true); both below the possession models.
- **`rules_v0` as an HSMM expert** (`hsmm_v1_rules`): on external `val` it scores 0.401 against 0.400 for `rules_v0` at the 5 s convention, i.e. it mostly copies `rules_v0`.
- **HSMM with a single `D` state:** team evidence during stoppages (for example a goalkeeper placing the ball) overrode dead ball; replaced by team-owned dead states (§5).
- **ATD+ with the broadcast ball:** +3.7 coverage points (v1), +2.9 with filters (v2), about +0.01 spellF1 (§4).

## 4. Handling missing and noisy tracking

Cleaning is done entirely by code (`pompeu/clean/`, A1); nothing is corrected by hand.

- **Ball:** points implying more than 40 m/s to both neighbours are removed, fragments under 3 frames dropped, and gaps under 0.5 s spline-filled (marked `interp`, confidence 0.5). This removes 1,911 of the 2,043 jumps above 40 m/s (`results/cleaning_report.md`). Phantom balls (still, nearest player 7–10 m away) never change possession, because every event needs a player within 2 m.
- **Players:** frozen slots are flagged (confidence 0.3), not deleted; same-team duplicates are removed; tracklets are stitched with the Hungarian algorithm under a speed bound. Velocities are recomputed from positions (the ATD column is empty on 89% of frames).
- **No ball:** ball features stay NaN, never imputed; gap length and age are features. Collective features, Smart Tagging and dead-ball structure work without the ball. The state machine keeps the current possession through a gap; the HSMM still receives ball-free evidence there.

**Line B (explored, not submitted).** The ATD ball projects inside the video image on 99.91% of observed frames, so video, tracking and homography are aligned. The broadcast ATD is aligned to the tactical clock from player positions alone (median 0.29 m), replays are discarded, and its ball enters a physically constrained fusion as a candidate: coverage of graded frames goes from 59.2% to 62.9% (ATD+ v1) or 62.1% with filters (v2). The trackers disagree by about 0.7 m on the ball, plus a tail of fast, airborne or false balls; it is not a clock offset (lag 0 is best). ATD+ regenerates with the same hash.

**Value of coverage** (`scripts/coverage_sweep.py`, external `val` match DFL J03WN1, 5 seeds; spellF1 against the share of in-half frames without ball):

| Ball absent | `rules_v0` | `hsmm_v1` |
|---|---|---|
| 50% | 0.387 | 0.377 |
| 41% (our ATD) | 0.396 | 0.390 |
| 37% | 0.409 | 0.400 |
| 30% | 0.423 | 0.410 |
| 20% | 0.438 | 0.421 |

Levels are nested, so only differences within this table are comparable. One point of coverage is worth about +0.0025 spellF1; the broadcast's 3 points, about +0.01. A video ball detector would have to fill many gaps to matter.

## 5. Possession rules implemented

**Dead ball (A3, `pompeu/events/deadball.py`).** A sticky two-state live/dead HMM (mean live period 40 s, dead 20 s). Cues add per-frame log-odds: slow field, ball out or not visible, ball still at a restart spot, Smart Tagging set pieces and `GOALS`, pre-kick-off, and a restart kick as strong live evidence. After Viterbi, live stretches under 5 s between dead periods are closed. A dead period runs from the ball crossing the line (or the field slowing) to the restart kick. All parameters are physical values, listed in `ARQUITECTURA.md`.

**Restarts.** The restarting team owns the next possession: throw-in for the opponent of the last nearby player; after a goal, the conceding team; over the goal line, corner or goal kick depending on who touched last; otherwise classified from where the ball is placed (free kick by default). The team comes from Smart Tagging first, then geometry, then the kicker.

**Events (A4, `pompeu/events/touches.py`).** A *touch* is a ball speed change above 3 m/s or direction change above 30° with a player within 2 m; its team probability is 80% distance difference and 20% pitch control [4]. A *control* is the ball within 1.5 m of the same player for at least 0.6 s.

**State machine (`rules_v0`, within each live stretch).**

- **Change of possession** needs control by the opponent: a second opponent touch within 3 s, or an opponent control.
- **Deflections and challenges:** a single opponent touch stays pending and is cancelled by a later event of the team in possession.
- **Aerial or fly-by balls:** a touch with the ball above 15 m/s before and after is ignored.
- **Uncertain team:** events with P(home) between 0.35 and 0.65 are ignored.
- **Loose, contested or invisible ball:** possession stays with the last team in control.
- **Timing:** a confirmed change is dated back to the start of the opponent's nearest-to-ball streak, never before the last event of the team in possession (the challenge reference dates a change at the new team's first touch).

**HSMM (`hsmm_v1`, `pompeu/models/hsmm.py`).** States `HL`/`AL` (live, team in possession; median 8 s) and `HD`/`AD` (dead, team to restart; median 20 s) with log-normal durations; team-owned dead states make the restart structural (`HD → HL`, same team with probability 0.98). Experts: A3 cues and segmentation, uncontested ball proximity, A4 events, Smart Tagging and the A3 restart team. Exact semi-Markov Viterbi, tested against exhaustive enumeration.

**Weights.** The HSMM expert weights, including `a3_segment_llr`, are set on the external `val` split only (`scripts/fit_hsmm.py`, coordinate descent on spellF1). **TODO:** fitted values and their `test` score, or drop this paragraph if the fit is not used.

**Final label policy.** No `X` on graded frames (`full_x_on_graded` = 0 in `results/leaderboard.csv`). **TODO:** describe the final policy if it differs from `rules_v0`.

## 6. Training and validation data

**Rules we follow.** No human labels on the provided match, for any purpose. The five public minutes only check the pipeline (alignment, format, label meaning), never training, tuning or selection; an early selection rule using a stressed copy of them was retired after the 22–23 October clarification. External data is about other games only.

| Dataset | Source and licence | Labels: who and how | How much | Used for |
|---|---|---|---|---|
| DFL / IDSSE open data [1, 2] | figshare, doi:10.6084/m9.figshare.28196177 (mirror `pysport/idsse-data`). **CC BY 4.0**; owner Deutsche Fußball Liga | Provider: per-frame `BallStatus` and `BallPossession` (TRACAB, 25 fps), mapped by `align_labels` | 7 full matches, 2022/23 (2 Bundesliga, 5 2. Bundesliga) | `train` / `val` / `test` |
| Metrica Sports sample data [3] | github.com/metrica-sports/sample-data. **No licence file**; the repository asks users to acknowledge the source | Provider events, collected manually; per-frame labels derived by code (team of the last control event; `D` from stoppage to restart) | 3 anonymised full matches, 25 fps | `train` / `val` / `test` |
| Ball pseudo-labels (B1) | Our code (`pompeu/vision/pseudolabels.py`) on the provided match | No person: observed ATD ball positions projected through the homography, filtered | 88,937 frames, kept in `cache/`, never published | **Not used by the submitted file** (ball detector, explored) |

**Splits** are by whole match and fixed (`pompeu/external/__init__.py`): `test` = DFL J03WMX, J03WPY and Metrica 3, never trained or tuned on; `val` = DFL J03WN1 (Metrica 2 could not be downloaded, so every `val` figure here is from J03WN1 alone); `train` = DFL J03WOH, J03WQQ, J03WOY, J03WR9 and Metrica 1. Metrica 3 is the test match because games 1 and 2 have the ball glued to the carrier (median distance to nearest player 0.28 m, against 1.4 m on DFL and 1.9 m on our ATD).

**Label convention: `min_possession_s` = 5 s.** DFL's `BallPossession` changes on almost every touch (8.4 possessions per live minute). `align_labels` gives a possession shorter than 5 s, sandwiched between two of the other team with no stoppage, to the surrounding team, which brings DFL to 4.26 per live minute. This is a label-convention choice: 5 s matches the possession rate of the public reference and the challenge's description of control (a player keeps the ball for a few seconds); no method was tuned on the public window. Because the threshold moves the ranking, we report the range (`scripts/min_possession_sensitivity.py`, external `val`, 5 seeds):

| Method | 0 s | 2 s | 5 s | 8 s |
|---|---|---|---|---|
| `rules_v0` | 0.315 | 0.393 | 0.400 | 0.351 |
| `hsmm_v1` | 0.266 | 0.344 | 0.386 | 0.354 |
| `hsmm_v1_rules` | 0.298 | 0.380 | 0.401 | 0.352 |

Across 2–8 s `rules_v0` is first or within 0.003 of first, so the submitted configuration is chosen to hold across that range, not at 5 s alone.

**Degradation** (`pompeu/external/degrade.py`, seeded). External tracking is made to look like our ATD: the ball is erased in runs drawn from the ATD's gap lengths until 41% of in-half frames have no ball, runs being `dead_weight` = 5.0 times likelier on dead frames; 0.12% of ball frames are displaced 10–40 m; players get identity swaps, freezes, gaps, dropouts, ghosts and slot re-packing. `dead_weight` is backed by a measurement without any label: with our own A3 segmentation of the full match (`scripts/ball_missing_by_state.py`) the ball is absent on 74.7% of dead frames and 12.0% of live ones, a ratio of 6.2. Smart Tagging is not simulated, so tagging cues are silent on external matches.

**Metrics** come from the organisers' `evaluation/possession_metrics.py`, never reimplemented, with block-bootstrap 5–95% intervals.

## 7. Results

External bench, `test` split (DFL J03WMX, J03WPY, Metrica 3), 2 degradation seeds per match, from `results/leaderboard.csv`. "Possessions" is the predicted count over the test matches (692 in the aligned truth); "Full match" is the count on the provided match (no ground truth).

| Method (config) | spellF1 [5–95%] | macroF1 | Possessions | Full match |
|---|---|---|---|---|
| `baseline_smooth50_clean` | 0.253 [0.237, 0.270] | 0.626 | 1367 | 594 |
| `deadball_vote125` | 0.412 [0.386, 0.439] | 0.802 | 800.5 | 284 |
| `rules_v0` (submitted) | **TODO** | **TODO** | **TODO** | 250 |
| `hsmm_v1` | **TODO** | **TODO** | **TODO** | 238 |

A change replaces `rules_v0` only if it improves spellF1 on this bench with non-overlapping intervals (or ties and improves macroF1); overlapping differences are not treated as real.

## 8. Limitations

- **Changes inside long ball gaps** are missed: no ball, no touch events. On external `val`, `rules_v0` goes from 0.438 with 20% of the ball missing to 0.387 with 50% (§4).
- **Touches are sparse** (about 8 per minute), so changes are dated from nearest-team streaks rather than the touch itself.
- **The external bench is imperfect:** other stadiums, trackers and label conventions; Smart Tagging is not simulated; `val` is a single match.
- **The 5 s label mapping** is one global threshold and changes the ranking between the top methods (§6).
- **Metrica labels** are only as precise as the hand-collected events that open each possession.
- **With more time:** a video ball detector good enough to fill a large share of gaps, ball height (aerial balls), player re-detection and a learned sequence model. **TODO:** final failure analysis.

## 9. Reproduction

Python 3.11, `pip install -r requirements.lock` (on Windows, use a short virtual-environment path so numba's DLL loads; `container/pompeu.def` builds the same environment with Apptainer). Unpack the match bundle into `data/` (see `data/README.md`); the repository contains no match data. Then:

```bash
python scripts/make_submission.py --config configs/final.yaml --out PompeuDataTeam.csv
python evaluation/validate_submission.py PompeuDataTeam.csv
```

`make_submission.py` prints the sha256, verified from a fresh clone with an empty cache: `bfd5ff16006b1e5c50371abeb528195cdc7a60397ec8bed90a8fccc468269f41` (**TODO:** update if the final configuration changes). The submission needs no GPU.

To rebuild the external bench:

```bash
python scripts/prepare_external.py --matches all
git clone https://github.com/metrica-sports/sample-data data/external/metrica/sample-data
python scripts/run_validation.py --config configs/final.yaml
```

`run_validation.py` appends a row to `results/leaderboard.csv`.

## References

1. Bassek, M., Rein, R., Weber, H., & Memmert, D. (2025). An integrated dataset of spatiotemporal and event data in elite soccer. *Scientific Data*, 12, 195. https://doi.org/10.1038/s41597-025-04505-y
2. Bassek, M., Rein, R., Weber, H., & Memmert, D. (2025). An integrated dataset of spatiotemporal and event data in elite soccer [Data set]. figshare. https://doi.org/10.6084/m9.figshare.28196177 (CC BY 4.0). Mirror: https://huggingface.co/datasets/pysport/idsse-data
3. Metrica Sports. Sample data (tracking and event data, 3 anonymised matches). https://github.com/metrica-sports/sample-data
4. Spearman, W. (2018). Beyond expected goals. *MIT Sloan Sports Analytics Conference*. (Pitch-control model used in A2.)
