# Pompeu Data Team: frame-level ball possession

More Than A Hack 2027, Phase 1 · Martí, Sergi, Ander · Code: https://github.com/sergicasesalonso/Pompeu-Data-Team

> DRAFT. Sections marked **TODO** are not final yet. Final scores will come from
> `results/leaderboard.csv`, external bench, `test` split, once the configuration is frozen on 16 November.

## 1. Summary

- **Task:** label each of the 199,319 frames of the tactical video `H`, `A`, `D` or `X`. Submissions are ranked on spellF1.
- **Submission:** `configs/final.yaml`. It currently matches `configs/rules_v0.yaml`, i.e. A1 cleaning, the A3 dead-ball HMM and the A4 possession state machine. **TODO:** final method name and its external `test` spellF1 and macroF1.
- **Approach.** First we clean the automatic tracking. A two-state HMM then segments live and dead ball. Next we detect touches and sustained controls with their team, and turn them into possessions. That step uses either a rule-based state machine that encodes the challenge's control rule, or a hidden semi-Markov model (HSMM) that combines the same cues with ball-free evidence. Every comparison between methods is made on a bench of **other** matches with provider labels (DFL/IDSSE, Metrica); the few values that were set from the public window are listed in §5 (`a3_segment_llr`) and §6 (label and degradation conventions). Before scoring, those matches are degraded until they look like our tracking. Line B adds ball observations taken from the video and the broadcast by code. Measured on the external bench, it only matters if it fills many gaps (§4).

## 2. Approach and why

There are two lines, joined by one data contract. `Tracking` (`pompeu/io/contract.py`) holds positions in metres on a 105.3 × 68.0 pitch, with one row per video frame, a confidence and a source per ball observation, and invariants checked on construction.

- **Line A, labelling from the provided assets:** A1 cleaning → A2 per-frame features (about 200 columns) → A3 dead-ball detector → A4 touches/controls and the `rules_v0` state machine → A5 `hsmm_v1` → **TODO** A7 learned sequence model, A8 calibration, A9 decoding for spellF1, A10 final label policy.
- **Line B, more ball from the video:** B0 video checks (done), B1 ball pseudo-labels (done), B8 broadcast alignment (done), B3 tracker and fusion into an improved tracking ("ATD+", done) → **TODO** B2 ball detector and B7 referee whistle from audio (not yet in the final configuration).

Why this design:

- **spellF1 scores possessions as objects.** Predicted runs are matched to true runs by temporal IoU. Per-frame voting fragments possessions: `baseline_smooth50` produces 594 possessions on the full match, while `rules_v0` and `hsmm_v1` produce 250 and 238 (`full_poss` in `results/leaderboard.csv`). The output therefore has to be a segmentation with explicit durations and transitions, which is why we use an HMM, a state machine and an HSMM rather than a per-frame classifier.
- **The ball is missing on 41% of graded frames** (CHALLENGE.md), and those gaps are where possession changes are lost. Line A uses ball-free evidence (collective shape, Smart Tagging, dead-ball structure). Line B tries to fill the gaps.
- **Every module reads and writes the same `Tracking`.** Moving to the Line B tracking only means changing `tracking:` in the YAML. No Line A code changes.

## 3. Assets used and dropped

| Asset | Used for | Status |
|---|---|---|
| Tactical ATD (players and ball) | Everything in Line A; source of B1 pseudo-labels | Used |
| Tactical homography | Metres to pixels: B0 check, B1 pseudo-labels, B3 | Used. Valid on every frame |
| Smart Tagging | Team evidence (attacking phases mean the team has the ball; defending phases mean it does not; zone codes do not vote), set-piece restarts, the `GOALS` phase | Used |
| Tactical video | B0: decoded frame count equals the ATD's (199,319). B1/B2: ball detector | B0, B1 done; **TODO** B2 |
| Tactical audio (AAC, 48 kHz) | B7: referee whistle | Present; **TODO** |
| Broadcast ATD and homography | B8: extra ball observations after alignment to the tactical clock on player positions (median fit 0.29 m) | Used in the ATD+ experiments; not in the final configuration (§4) |
| Broadcast video | Not used: B8 needs only the broadcast ATD | Dropped |
| External open data (DFL/IDSSE, Metrica) | Validation bench; fitting the few model weights; training the learned components | Used, see §6 |

Tried and dropped, or no effect on its own:

- A1 cleaning alone does not change the baseline. Of 2,043 ball jumps above 40 m/s, 1,911 are removed, but majority voting was already absorbing them. The value of cleaning is in the A2–A4 features and events.
- An HSMM with a single `D` state let team evidence during stoppages (for example, a goalkeeper placing the ball) override dead ball. It was replaced by team-owned dead states (§5).
- Using `rules_v0` as an HSMM expert (`hsmm_v1_rules`) mostly copies `rules_v0`.
- **TODO:** list the remaining dropped variants once the final method is fixed.

## 4. Handling missing and noisy tracking

Cleaning is done entirely by code (`pompeu/clean/`, A1). Nothing is corrected by hand.

- **Ball:** we remove points that imply more than 40 m/s relative to both neighbours (repeated until none remain), drop fragments shorter than 3 frames, and fill gaps shorter than 0.5 s with a cubic spline unless the fill would need more than 40 m/s. Filled frames are marked `interp` with confidence 0.5.
- **False ("phantom") balls:** some ATD balls sit still in open play with the nearest player 7–10 m away. Every event requires a player within 2 m, so these balls never cause a change of possession.
- **Players:**
  - A slot is flagged as frozen (confidence 0.3, not deleted) when it moves less than 0.2 m in 10 s while the other players move at a median speed above 1 m/s. Slots 24 and 46 turned out to be the goalkeepers, frozen by the tracker when off-screen.
  - Duplicates (same team, within 0.5 m for more than 1 s) are removed from the shorter tracklet.
  - Tracklets are stitched with the Hungarian algorithm within each connected component when they are less than 2 s and less than `9 m/s·Δt + 1 m` apart. Gaps of up to 1 s are filled linearly.
- **Velocities** are recomputed from positions, because the ATD velocity column is empty on 89% of frames.
- **No ball:** features that need the ball are left as NaN and never imputed, and the length and age of each gap are exposed as features. Collective features (centroids, lines, compactness, pressing counts), Smart Tagging and the dead-ball structure all work without the ball. The state machine keeps the current possession through a gap. The HSMM still receives ball-free evidence there.
- **Line B, more ball from video:**
  - B0: the ATD ball projects inside the image on 99.91% of the frames where it is observed, so the frames, tracking and homography are aligned.
  - B1: pseudo-labels for a ball detector are generated by code (§6).
  - B8/B3: the broadcast ATD is aligned to the tactical clock from player positions alone (median distance 0.29 m), replays are discarded, and its ball enters the fusion as a candidate. Ball coverage of graded frames goes from 59.2% to 62.9% (ATD+ v1) and to 62.1% with candidate filters (v2). The two trackers disagree by about 0.7 m on the ball, plus a tail of fast or airborne balls and false ones; it is not a clock offset (lag 0 is the best).
  - **Value of coverage** (`scripts/coverage_sweep.py`, external `val` match DFL J03WN1, 5 seeds; spellF1 of `rules_v0` / `hsmm_v1` as the share of in-half frames without ball changes):

    | Ball absent | `rules_v0` | `hsmm_v1` |
    |---|---|---|
    | 50% | 0.387 | 0.377 |
    | 41% (our ATD) | 0.396 | 0.390 |
    | 37% | 0.409 | 0.400 |
    | 30% | 0.423 | 0.410 |
    | 20% | 0.438 | 0.421 |

    The levels are nested: each match is degraded once at 20% and more ball is then erased to reach each higher rate, with a log-normal stand-in for the ATD's gap lengths. The 41% row is therefore not the same run as the 5 s column of the `min_possession_s` table in §6 (0.400 / 0.386), which degrades directly; only differences within one table are comparable. One point of coverage is worth about +0.0025 spellF1. The broadcast adds about 3 points, roughly +0.01, which is within the noise. Line B therefore matters only if the ball detector (B2) fills many gaps. **TODO:** coverage and spellF1 with B2, if it enters.

## 5. Possession rules implemented

**Dead ball (A3, `pompeu/events/deadball.py`).** A two-state live/dead HMM with sticky transitions (mean live period 40 s, mean dead period 20 s). Each cue adds log-odds per frame, scaled by 1/(0.5 s·fps) because consecutive frames are not independent. The cues are:

- mean field speed over 2 s (threshold 1.4 m/s);
- ball more than 0.5 m outside the pitch: +8;
- ball not visible: +0.5;
- ball still for more than 1 s at a restart spot: +2 (+0.5 elsewhere);
- a Smart Tagging set piece within 3 s: +1;
- the `GOALS` phase: +2;
- before each half's kick-off: +8;
- a restart kick (ball goes from slow or invisible to above 4 m/s within 0.5 s, with a player within 2 m): −4 for 1.5 s.

After Viterbi decoding, live stretches shorter than 5 s between two dead periods are closed. Each dead period starts when the ball crosses the line, or otherwise when the field starts to slow down, and ends at the restart kick. All parameters are physical values. None was fitted on the provided match.

**Restarts.** A possession starts with the restarting team:

- ball out over a touchline: a throw-in for the opponent of the last nearby player;
- over the goal line between the posts: a goal, and the conceding team restarts;
- over the goal line outside the posts: a corner if a defender touched the ball last, otherwise a goal kick;
- ball not seen going out: classified from where the ball is placed, and a free kick if nothing fits.

The restarting team comes from Smart Tagging first, then geometry, then the player who kicks.

**Events (A4, `pompeu/events/touches.py`).**

- A *touch* is a change of ball speed above 3 m/s or of direction above 30° with a player within 2 m. Its team probability is 80% the distance difference and 20% pitch control.
- A *control* is the ball within 1.5 m of the same player for at least 0.6 s.

**State machine (`rules_v0`, applied within each live stretch).**

- **Change of possession** needs control by the opponent: a second opponent touch within 3 s, or an opponent control.
- **Deflections:** a single opponent touch stays pending. A later event by the team in possession cancels it.
- **Aerial or fly-by balls:** a touch with the ball above 15 m/s both before and after is ignored.
- **Uncertain team:** events with P(home) between 0.35 and 0.65 are ignored.
- **Loose, contested or invisible ball:** possession stays with the last team in control.
- **Timing:** a confirmed change is dated back to the start of the streak in which the opponent was the team nearest the ball, never before the last event of the team in possession. This follows the challenge reference, which dates a change at the new team's first touch.

**HSMM (`hsmm_v1`, `pompeu/models/hsmm.py`).**

- **States:** four hidden states with explicit durations, `HL`/`AL` (live, team in possession) and `HD`/`AD` (dead, team to restart). Team-owned dead states keep team evidence and live/dead evidence factorised, and they make the restart part of the structure (`HD → HL`).
- **Durations** are log-normal: possessions have a median of 8 s and a minimum of 0.2 s, stoppages a median of 20 s and a minimum of 1 s.
- **Transitions:** a possession ends in a change of possession with probability 0.7 and in a stoppage with probability 0.3. After a stoppage, play restarts with the same team with probability 0.98.
- **Experts:**
  - the A3 cues and A3's final segmentation (`a3_segment_llr`);
  - ball proximity, used only when the ball is within 4 m of a player and not contested (no evidence when both teams are within 2.5 m);
  - A4 events, excluding fly-by balls;
  - Smart Tagging votes, with a 5 s correlation;
  - the A3 restart team.
- **Decoding:** exact semi-Markov Viterbi and forward-backward (numba). Both are tested against exhaustive enumeration on small problems.

**Weights and the public window.** The HSMM weights are being fitted only on the external `val` split (`scripts/fit_hsmm.py`, coordinate descent on spellF1). This includes `a3_segment_llr`, whose current value (3) was originally chosen on the stressed public window. **TODO:** fitted values and their `test` score. The fit's output, `configs/hsmm_v1_fit.yaml`, does not exist yet.

**Final label policy (A10).** No `X` on graded frames. **TODO.**

## 6. Training and validation data

**Rules we follow:**

- No human labels on the provided match are used, in any amount or for any purpose.
- The five public minutes are used only to check the pipeline: frame alignment, file format, and what the labels mean. They are never used for training, tuning or selection. Before the rule clarification of 22–23 October 2026, a stressed copy of the public window was part of our selection rule. It is no longer used, and every comparison in this document is on the external bench.
- External data is about other games only.
- Every tracking improvement on the provided match comes from code.

| Dataset | Source and licence | Labels: who and how | How much | Used for |
|---|---|---|---|---|
| DFL / IDSSE open data [1, 2] | figshare, doi:10.6084/m9.figshare.28196177 (files fetched from the identical Hugging Face mirror `pysport/idsse-data` by `scripts/prepare_external.py`). **CC BY 4.0**; data owner: Deutsche Fußball Liga | Provider labels: per-frame `BallStatus` (alive/dead) and `BallPossession` (home/away) in the official positional data (TRACAB optical tracking, 25 fps). No labels of ours. Mapped to the challenge's control rule by `align_labels` (below) | 7 full matches, 2022/23 season: 2 Bundesliga, 5 2. Bundesliga | `train` / `val` / `test` |
| Metrica Sports sample data [3] | github.com/metrica-sports/sample-data. **No licence file**; the repository asks users to acknowledge the source and "be responsible with the use of this data" | Event data collected manually by the provider. We derive per-frame labels by code: the team of the last control event (pass, carry, recovery, set piece, shot), and `D` from ball out, foul or goal to the next restart | 3 anonymised full matches, 25 fps tracking (games 1–2 CSV, game 3 FIFA EPTS) | `train` / `val` / `test` |
| Ball pseudo-labels (B1) | Our code (`pompeu/vision/pseudolabels.py`) on the provided match | No person involved: cleaned ATD ball positions, observed by the tracker (not interpolated), projected through the frame's homography | 88,937 frames kept: 70,449 `train` / 18,488 `val` | Training the ball detector (B2) |

**B1 filters.** A frame is kept only if:

- the ball lands at least 16 px inside the image;
- the frame's reprojection error is at most 0.357, the 95% quantile over graded frames (computed from the homography, not from any label);
- the ball speed is below 35 m/s.

25,936 kept frames have the ball within 1 m of a player and are flagged for down-weighting. The split is temporal: 1-minute blocks, one block in five for `val`. The table stays in `cache/` and is never published.

**Splits** are by whole match and fixed (`pompeu/external/__init__.py`):

- `test` (DFL J03WMX, J03WPY, Metrica 3) decides and is never trained or tuned on;
- `val` (DFL J03WN1; Metrica 2 is planned but could not be downloaded, so every `val` number in this document is from J03WN1 alone) fits the few weights and is used for early stopping;
- `train` (DFL J03WOH, J03WQQ, J03WOY, J03WR9, Metrica 1) is used by everything that learns.

The test Metrica match is game 3 because games 1 and 2 have the ball glued to the carrier: the median distance to the nearest player is 0.28 m, against 1.4 m on DFL and 1.9 m on our ATD.

**Label conventions.**

- **DFL:** `BallPossession` changes on almost every touch, 8.4 possessions per live minute against 4.2 in the challenge's public reference (17 possessions in 4.0 live minutes). `align_labels` gives a possession shorter than `min_possession_s` = 5 s, sandwiched between two possessions of the other team with no stoppage in between, to the surrounding team. This brings DFL to 4.26 possessions per live minute. The 5 s value was set to match the possession rate of the public reference, so it does come from the public window, although no method was tuned on it. It matters: it changes the ranking between the top methods (see the sensitivity below), so we report the 0–8 s range and not only the 5 s result.
- **Degradation:** the ball-loss weighting towards dead frames (`dead_weight`) was first set from a rate measured on the public window: 34% of dead frames have no ball, against about 6% of live ones. It has been re-measured without any label, using our own A3 dead/live segmentation on the full match (`scripts/ball_missing_by_state.py`): the ball is absent on 74.7% of dead frames and 12.0% of live ones, a ratio of 6.2. **TODO:** state which value the final configuration uses (the current default in `degrade.py` is 5.0).
- **Sensitivity to `min_possession_s`** (`scripts/min_possession_sensitivity.py`, external `val`, 5 seeds). The threshold cannot be fixed from the public window, and it moves the ranking of the three best methods:

  | Method | 0 s | 2 s | 5 s | 8 s |
  |---|---|---|---|---|
  | `rules_v0` | 0.315 | 0.393 | 0.400 | 0.351 |
  | `hsmm_v1` | 0.266 | 0.344 | 0.386 | 0.354 |
  | `hsmm_v1_rules` | 0.298 | 0.380 | 0.401 | 0.352 |

  Aligned possessions at 0, 2, 5 and 8 s: 500, 358, 242 and 194. `rules_v0` is the most robust: first at 0 and 2 s (by 0.017 and 0.013 over `hsmm_v1_rules`, no intervals computed), tied at 5 s, and 0.003 behind at 8 s. Absolute external scores are only comparable at the same `min_possession_s`. **TODO:** final choice and its justification from the challenge text (CHALLENGE.md: a player keeps the ball for a few seconds).

**Degradation** (`pompeu/external/degrade.py`, seeded). Clean external tracking is made to look like our ATD before any method sees it:

- the ball is erased in runs drawn from the ATD's own gap lengths (capped at 105 s) until 41% of in-half frames have no ball;
- about 0.12% of ball frames are displaced by 10–40 m;
- ball jitter of 5 cm;
- identity swaps between crossing teammates;
- far-from-ball players frozen;
- heavy-tailed tracklets with gaps;
- clustered dropouts of far players;
- ghost duplicates;
- slot re-packing like the ATD's 46 slots.

Smart Tagging is not simulated, so tagging cues are silent on external matches. `scripts/degradation_fit.py` compares ten statistics between the degraded matches and our ATD using KS distances. **TODO:** report its table.

**Learned components** train only on degraded external matches (`pompeu/external/examples.py`: degrade, clean with A1, build A2 features, aligned labels; Smart Tagging columns are excluded). The A3 live/dead LightGBM classifier (`scripts/train_deadball_gbm.py`) is implemented but off by default (`gbm_weight: 0`). **TODO:** its `test` result and the A7 sequence model.

**Metrics** come from the organisers' `evaluation/possession_metrics.py`, never reimplemented, with block-bootstrap 5–95% intervals.

## 7. Results

External bench, `test` split (DFL J03WMX, J03WPY, Metrica 3), 2 degradation seeds per match, from `results/leaderboard.csv`. "Possessions" is the predicted count over the test matches, against 692 in the aligned truth. "Full match" is the possession count on the provided match (no ground truth).

| Method (config) | spellF1 [5–95%] | macroF1 | Possessions (truth 692) | Full match |
|---|---|---|---|---|
| `baseline_smooth50_clean` | 0.253 [0.237, 0.270] | 0.626 | 1367 | 594 |
| `deadball_vote125` | 0.412 [0.386, 0.439] | 0.802 | 800.5 | 284 |
| `rules_v0` (current final) | **TODO** | **TODO** | **TODO** | 250 |
| `hsmm_v1` | **TODO** | **TODO** | **TODO** | 238 |
| `hsmm_v1_fit` | **TODO** | **TODO** | **TODO** | **TODO** |
| **final** | **TODO** | **TODO** | **TODO** | **TODO** |

The leaderboard also holds an earlier, DFL-only bench (J03WMX, J03WPY), on which `hsmm_v1` scored 0.440 and `deadball_vote125` 0.390. That bench is superseded by the split above and the scores are not comparable. We accept a change only if it improves on the external `test` bench with non-overlapping intervals. Differences between intervals that overlap are not treated as real. (The organisers' 0.04 noise floor in EVALUATION.md refers to the full provided match, not to this bench.)

## 8. Limitations

- **Changes inside long ball gaps** are still missed. Without the ball there are no touch events, so the state machine under-segments and short possessions inside a gap disappear. On the external `val` match, `rules_v0` goes from 0.438 with 20% of the ball missing to 0.387 with 50% (§4). Adding the broadcast ball recovers only about 3 points of coverage.
- **Touches are sparse** (about 8 per minute), so dating changes relies on nearest-team streaks rather than on the touch itself.
- **The external bench is imperfect:** different stadiums, trackers and label conventions. The degradation reproduces our gap and noise statistics, not our camera. Smart Tagging and team-assignment errors are not simulated.
- **The 5 s label mapping** is a single global threshold matched to a 17-possession reference (about ±25%), and it changes the ranking between the top methods (§6).
- **Metrica labels** are only as precise as the hand-collected events that open each possession (about one per second).
- **TODO:** final failure analysis. With more time we would add ball height (B4), player re-detection (B5) and a larger learned sequence model.

## 9. Reproduction

Python 3.11, `pip install -r requirements.lock` (on Windows, put the virtual environment on a short path so numba's DLL loads; `container/pompeu.def` builds the same environment with Apptainer). Unpack the match bundle into `data/` (see `data/README.md`). The repository contains no match data. Then:

```bash
python scripts/make_submission.py --config configs/final.yaml --out PompeuDataTeam.csv
python evaluation/validate_submission.py PompeuDataTeam.csv
```

`make_submission.py` prints the sha256. Expected: **TODO** (currently `bfd5ff16006b1e5c50371abeb528195cdc7a60397ec8bed90a8fccc468269f41` for `rules_v0`).

To rebuild the external bench:

```bash
python scripts/prepare_external.py --matches all
git clone https://github.com/metrica-sports/sample-data data/external/metrica/sample-data
python scripts/run_validation.py --config configs/final.yaml
```

The DFL files are downloaded by `prepare_external.py`, and the Metrica repository is cloned into `data/external/metrica/sample-data`. `run_validation.py` appends a row to `results/leaderboard.csv`.

**TODO:** GPU steps (detector weights, determinism settings) if Line B enters the final configuration. The Line B steps that do not need a GPU (broadcast alignment and fusion, `scripts/build_atd_plus.py`) are numpy-only, and the ATD+ they write is hashed over its arrays, so two runs can be compared (**to verify:** two runs on the tower give the same hash).

## References

1. Bassek, M., Rein, R., Weber, H., & Memmert, D. (2025). An integrated dataset of spatiotemporal and event data in elite soccer. *Scientific Data*, 12, 195. https://doi.org/10.1038/s41597-025-04505-y
2. Bassek, M., Rein, R., Weber, H., & Memmert, D. (2025). An integrated dataset of spatiotemporal and event data in elite soccer [Data set]. figshare. https://doi.org/10.6084/m9.figshare.28196177 (CC BY 4.0). Mirror: https://huggingface.co/datasets/pysport/idsse-data
3. Metrica Sports. Sample data (tracking and event data, 3 anonymised matches). https://github.com/metrica-sports/sample-data
4. Spearman, W. (2018). Beyond expected goals. *MIT Sloan Sports Analytics Conference*. (Pitch-control model used in A2.)
