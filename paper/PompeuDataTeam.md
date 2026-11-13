# Pompeu Data Team: frame-level ball possession

More Than A Hack 2027, Phase 1 · Martí, Sergi, Ander · Code: https://github.com/sergicasesalonso/Pompeu-Data-Team

Implementation detail for every module (all parameters, file by file) is in `ARQUITECTURA.md` in the repository.

<!-- FINAL: every block marked FINAL depends on the configuration frozen on Monday 16-11 12:00.
Candidates: rules_v0 (current safety submission), hsmm_v1_fit_mbr, a temporal model.
Fill each block from the clean leaderboard row of the chosen config, then delete these comments. -->

## 1. Summary

- **Task:** label each of the 199,319 frames of the tactical video `H`, `A`, `D` or `X`; ranked on spellF1.
- **Approach.** Clean the tracking, segment live and dead ball with an HMM, detect touches and controls with their team, and turn them into possessions, either with a state machine encoding the challenge's control rule (`rules_v0`) or with a hidden semi-Markov model (HSMM) that combines the same cues with ball-free evidence and explicit possession durations. Methods are compared only on **other** matches with provider labels (DFL/IDSSE, Metrica), degraded to look like our tracking, under a selection rule fixed before the test split was read (§6).
- **Submission:** `configs/final.yaml`. <!-- FINAL: name of the chosen config (rules_v0 / hsmm_v1_fit_mbr / temporal model), one sentence on what it is, and its external test spellF1 [5–95%] and macroF1 next to rules_v0's 0.513 [0.485, 0.541] and 0.823. -->
- **Ball from the video (Line B)** was explored and is not submitted: the broadcast adds about 3 points of ball coverage, worth about +0.01 spellF1, within the noise (§4).

## 2. Approach and why

Every module reads and writes one data contract, `Tracking` (`pompeu/io/contract.py`): metres on a 105.3 × 68.0 pitch, one row per video frame, a confidence and source per ball observation. **Line A** labels from the provided assets: A1 cleaning → A2 per-frame features (about 200 columns) → A3 dead-ball detector → A4 touches, controls and the `rules_v0` state machine → A5 HSMM. **Line B** adds ball from the video and broadcast into an augmented tracking, "ATD+"; switching to it only changes `tracking:` in the YAML.

- **spellF1 scores possessions as objects** (runs matched by temporal IoU). Per-frame voting fragments them: on the external `test` bench `baseline_smooth50_clean` predicts 1,367 possessions against 692 true ones (§7). Hence a segmentation with explicit durations and transitions.
- **The ball is missing on 41% of graded frames** (CHALLENGE.md), where possession changes are lost. Line A uses ball-free evidence (collective shape, Smart Tagging, dead-ball structure); Line B tried to fill the gaps.

## 3. Assets used and dropped

| Asset | Used for | Status |
|---|---|---|
| Tactical ATD (players and ball) | Everything in Line A; source of B1 pseudo-labels | Used |
| Tactical homography | Metres to pixels for B0, B1, B3 | Line B only |
| Smart Tagging | Team evidence (attacking phases vote for the team, defending phases against; zone codes do not vote), set-piece restarts, the `GOALS` phase | Used |
| Tactical video | B0: decoded frame count equals the ATD's (199,319); B1 pseudo-labels for a ball detector | Explored, not submitted |
| Tactical audio (AAC, 48 kHz) | Referee whistle | Present; not used |
| Broadcast ATD and homography | Extra ball observations after alignment to the tactical clock | Explored, not submitted (§4) |
| Broadcast video | Not needed: alignment uses only the broadcast ATD | Dropped |
| External open data | Validation bench, model weights, extra training (§6) | Used |

**Tried and dropped** (all on external data; `val` = DFL J03WN1 + Metrica 2, 5 degradation seeds, unless stated):

- **Smoothed per-frame voting** (`baseline_smooth25/50/125`): 746, 594 and 398 possessions on the full match against 250 for `rules_v0`; 0.253 spellF1 on `test`. **Voting with dead-ball segmentation** (`deadball_vote125`): 0.412 on `test`.
- **HSMM with a single `D` state:** team evidence during stoppages (a goalkeeper placing the ball) overrode dead ball; replaced by team-owned dead states (§5).
- **Gap-team expert** (`hsmm_v2`): a ridge logistic regression on ball-free collective features (centroids, lines, compactness, pressing counts, 3 s means), trained on the DFL `train` matches, giving team log-odds on frames without ball. It names the right team on 69.8% of DFL `val` gap frames. **Augmenting** its training set (y-mirror, home/away swap, degradation seeds 0–9, labels at 2/5/8 s; 748,164 rows against 56,587) raises that to 78.0%, and recovers 75.4% of the gaps that contain a change (67.3% before), but inside the HSMM the gain is only +0.001 to +0.003 spellF1: the bottleneck is how the HSMM uses the evidence, not the evidence.
- **HSMM weights fitted on `val` at the 5 s convention** (`hsmm_v2_fit`): `test` spellF1 0.568 [0.536, 0.602], but on `val` it loses to `rules_v0` at the 2 s convention (0.468 against 0.505) because it predicts too few, long possessions; it fails rule B (§6). Refitting on the mean of 2/5/8 s (`hsmm_v2_fitavg`, `fitavg2` with the augmented expert and wider grids) narrows the 2 s loss to −0.010 and −0.026 (mean 0.496 and 0.499 against 0.470); never run on `test`.
- **Self-training** the gap-team expert on algorithmic pseudo-labels of the provided match (frames where `rules_v0` and `hsmm_v1` agree with the ball visible, public window and 5 s margin excluded): the teacher is 84–88% precise on DFL `val`, the pseudo-labels fall on easy frames, and recovered gap changes drop from 0.693 to 0.62–0.69. Negative.
- **Synthetic Smart Tagging** for the external matches, from their events and phases with an error model measured on the provided match against confident `rules_v0`/`hsmm_v1` stretches (no ground truth, public window excluded): HSMM mean spellF1 over 2/5/8 s 0.506 against 0.498 without tags, +0.008 with overlapping intervals; weights ≥ 4 hurt. The idea came from another participant's public description [6].
- **SkillCorner Open Data** [5]: a converter (10 fps broadcast tracking resampled to 25 fps; labels from phases of play) for 20 A-League 2024/25 matches, used only as extra training data in exploratory gap-team variants, never in `val` or `test`.

## 4. Handling missing and noisy tracking

Cleaning is done entirely by code (`pompeu/clean/`, A1); nothing is corrected by hand.

- **Ball:** points implying more than 40 m/s to both neighbours are removed, fragments under 3 frames dropped, and gaps under 0.5 s spline-filled (marked `interp`, confidence 0.5). This removes 1,911 of the 2,043 jumps above 40 m/s (`results/cleaning_report.md`). Phantom balls never change possession, because every event needs a player within 2 m.
- **Players:** frozen slots are flagged (confidence 0.3), not deleted; same-team duplicates are removed; tracklets are stitched with the Hungarian algorithm under a speed bound. Velocities are recomputed from positions (the ATD column is empty on 89% of frames).
- **No ball:** ball features stay NaN, never imputed; gap length and age are features. The state machine keeps the current possession through a gap; the HSMM still receives ball-free evidence there.

**Line B (explored, not submitted).** B0: the decoded video has exactly the ATD's 199,319 frames and the ATD ball projects inside the image on 99.91% of observed frames, so video, tracking and homography are aligned. B1: 88,937 ball pseudo-labels (observed, cleaned ATD ball projected through the homography; jumps, high reprojection error and crowded balls removed; temporal train/val split), kept in `cache/`, for a ball detector. B8: the broadcast ATD is aligned to the tactical clock from player positions alone (median 0.29 m), replays are discarded. B3: candidates enter a physically constrained fusion; coverage of graded frames goes from 59.2% to 62.9% (ATD+ v1) or 62.1% with filters (v2), and ATD+ regenerates with the same hash. The trackers disagree by about 0.7 m on the ball, plus fast, airborne or false balls; lag 0 is best.

**Value of coverage** (`scripts/coverage_sweep.py`, DFL J03WN1, 5 seeds, 5 s convention; nested levels, only differences comparable):

| Ball absent | 50% | 41% (our ATD) | 37% | 30% | 20% |
|---|---|---|---|---|---|
| `rules_v0` | 0.387 | 0.396 | 0.409 | 0.423 | 0.438 |
| `hsmm_v1` | 0.377 | 0.390 | 0.400 | 0.410 | 0.421 |

One point of coverage is worth about +0.0025 spellF1, so the broadcast's 3 points give about +0.01; a video detector would have to fill a large share of the gaps to matter.

## 5. Possession rules implemented

**Dead ball (A3, `pompeu/events/deadball.py`).** A sticky two-state live/dead HMM (mean live period 40 s, dead 20 s). Cues add per-frame log-odds: slow field, ball out or not visible, ball still at a restart spot, Smart Tagging set pieces and `GOALS`, pre-kick-off, and a restart kick as strong live evidence. After Viterbi, live stretches under 5 s between dead periods are closed. A dead period runs from the ball crossing the line (or the field slowing) to the restart kick. <!-- FINAL: if the chosen config uses the LightGBM live/dead expert (deadball.gbm_weight), add one sentence: trained on external train only, early stopping on val, dead F1 on test 0.911 combined (results/deadball_gbm_report.md); artifact and lightgbm version pinned. -->

**Restarts.** The restarting team owns the next possession: throw-in for the opponent of the last nearby player; after a goal, the conceding team; over the goal line, corner or goal kick depending on who touched last; otherwise classified from where the ball is placed (free kick by default). The team comes from Smart Tagging first, then geometry, then the kicker.

**Events (A4, `pompeu/events/touches.py`).** A *touch* is a ball speed change above 3 m/s or direction change above 30° with a player within 2 m; its team probability is 80% distance difference and 20% pitch control [4]. A *control* is the ball within 1.5 m of the same player for at least 0.6 s.

**State machine (`rules_v0`, within each live stretch).**

- **Change of possession** needs control by the opponent: a second opponent touch within 3 s, or an opponent control.
- **Deflections and challenges:** a single opponent touch stays pending and is cancelled by a later event of the team in possession.
- **Aerial or fly-by balls:** a touch with the ball above 15 m/s before and after is ignored.
- **Uncertain team:** events with P(home) between 0.35 and 0.65 are ignored.
- **Loose, contested or invisible ball:** possession stays with the last team in control.
- **Timing:** a confirmed change is dated back to the start of the opponent's nearest-to-ball streak, never before the last event of the team in possession.

**HSMM (`pompeu/models/hsmm.py`).** States `HL`/`AL` (live, team in possession) and `HD`/`AD` (dead, team to restart) with log-normal durations; team-owned dead states make the restart structural (`HD → HL`, same team with probability 0.98). Experts: A3 cues and segmentation, uncontested ball proximity, A4 events, the `rules_v0` output, Smart Tagging and the A3 restart team. Exact semi-Markov Viterbi, tested against exhaustive enumeration. Expert weights and duration parameters are fitted on external `val` only (`scripts/fit_hsmm.py`, coordinate descent).

<!-- FINAL: describe the chosen config here. For rules_v0: delete the HSMM weights sentence above or say the HSMM is not submitted. For hsmm_v1_fit_mbr: fitted values (w_events 0.5, w_dead 2.0, rules_llr 0.5, a3_segment_llr 1.0, w_proximity 0, poss_median_s 12, p_poss_to_dead 0.4, gbm_weight 1.0) and MBR decoding (200 posterior samples, greedy boundary search on expected spellF1). For a temporal model: architecture, inputs, training set and how its output is decoded. -->

**Final label policy.** No `X` on graded frames (`full_x_on_graded` = 0 in `results/leaderboard.csv`); `X` outside the halves. <!-- FINAL: state the policy of the chosen config (e.g. A10: no runs of 1–2 frames) if it differs. -->

## 6. Training and validation data

**Rules we follow.** No human labels on the provided match, for any purpose. External data is about other games only. **The five public minutes were not used to train, tune or select anything**; they only check the pipeline (alignment, format, label meaning). Two values first read off that window were removed or re-justified without it: the A3 segmentation weight is refitted on `val`, and the degradation's `dead_weight` is checked by a label-free measurement (below).

| Dataset | Source and licence | Labels: who and how | How much | Used for |
|---|---|---|---|---|
| DFL / IDSSE open data [1, 2] | figshare, doi:10.6084/m9.figshare.28196177 (mirror `pysport/idsse-data`). **CC BY 4.0**; owner Deutsche Fußball Liga | Provider: per-frame `BallStatus` and `BallPossession` (TRACAB, 25 fps), mapped by `align_labels` | 7 full matches, 2022/23 | `train` / `val` / `test` |
| Metrica Sports sample data [3] | github.com/metrica-sports/sample-data; no licence file, the repository asks users to acknowledge the source | Provider events, collected manually; per-frame labels derived by code (team of the last control event; `D` from stoppage to restart) | 3 anonymised full matches, 25 fps | `train` / `val` / `test` |
| SkillCorner Open Data [5] | github.com/SkillCorner/opendata, **MIT**; credit requested | Provider phases of play (team in possession); `D` = gaps between phases | 20 A-League 2024/25 matches | Extra training only, exploratory (§3) |
| Ball pseudo-labels (B1) | Our code on the provided match | No person: projected ATD ball, filtered | 88,937 frames, never published | Not used by the submitted file |

**Splits** are by whole match and fixed (`pompeu/external/__init__.py`): `train` = DFL J03WOH, J03WQQ, J03WOY, J03WR9 and Metrica 1 (everything that learns); `val` = DFL J03WN1 and Metrica 2 (weights, early stopping, selection); `test` = DFL J03WMX, J03WPY and Metrica 3, never trained or tuned on. Metrica 3 is the test match because games 1 and 2 have the ball glued to the carrier (median 0.28 m to the nearest player, against 1.4 m on DFL and 1.9 m on our ATD).

**Degradation** (`pompeu/external/degrade.py`, seeded). External tracking is made to look like our ATD: the ball is erased in runs drawn from the ATD's gap lengths until **41%** of in-half frames have no ball, runs starting `dead_weight` = 5 times likelier on dead frames; 0.12% of ball frames are displaced 10–40 m; players get identity swaps, freezes, gaps, dropouts, ghosts and slot re-packing. The value 5 was first read off the public window, so we checked it without any label: with our own A3 segmentation of the full match (`scripts/ball_missing_by_state.py`) the ball is absent on 74.7% of dead frames and 12.0% of live ones (ratio 6.2). `dead_weight` is not that ratio (it weights where a gap starts; in the match gaps start only 1.17 times more often on dead frames but last much longer), but the measurement confirms that gaps concentrate in stoppages to the degree assumed, so we keep 5 rather than retune the bench every number depends on. Smart Tagging is not simulated, so tagging cues are silent on external matches. <!-- FINAL: if the final bench uses the later simulator version (ATD-like ball blur), say so and make sure every number in §7 comes from that same version. -->

**Label convention `min_possession_s`.** DFL's `BallPossession` changes on almost every touch (8.4 possessions per live minute). `align_labels` gives a possession shorter than `min_possession_s`, sandwiched between two of the other team with no stoppage, to the surrounding team. The convention of the hidden ground truth is unknown, and the value that matches the public reference's possession rate (5 s, 4.26 per live minute) would be read off the public window, so **no single value is trusted**: every candidate is scored at 2, 5 and 8 s. The convention moves the ranking (`val`, 5 seeds, spellF1):

| Method | 2 s | 5 s | 8 s | Mean | Possessions per run |
|---|---|---|---|---|---|
| `rules_v0` | **0.505** | 0.495 | 0.409 | 0.470 | 420 |
| `hsmm_v2` (gap-team expert, default weights) | 0.474 | 0.502 | 0.418 | 0.465 | – |
| `hsmm_v2_fit` (fitted at 5 s) | 0.468 | 0.538 | 0.483 | 0.496 | 365 |
| `hsmm_v2_fitavg` (fitted on the mean) | 0.495 | 0.533 | 0.459 | 0.496 | 377 |

True possessions per run: 609, 429 and 343 at 2, 5 and 8 s. The fitted HSMMs gain at 5 and 8 s and lose at 2 s because they under-segment.

**Selection rule (rule B), fixed before the final `test` pass.** A candidate replaces `rules_v0` only if (1) its mean spellF1 over 2/5/8 s on `val` is higher, (2) it loses no more than 0.02 to `rules_v0` at any of the three conventions, (3) all its weights come from `train`/`val`, and (4) it then improves on `rules_v0` in a **single** `test` pass (spellF1 with non-overlapping intervals, or a tie with better macroF1), and regenerates the same sha256 from a clean clone. Otherwise `rules_v0` is submitted. Earlier configurations were scored on `test` before this rule was fixed (§7); they are reported, not selected from.

**Metrics** come from the organisers' `evaluation/possession_metrics.py`, never reimplemented, with block-bootstrap 5–95% intervals.

## 7. Results

External bench, `test` split (DFL J03WMX, J03WPY, Metrica 3), 2 degradation seeds per match, 5 s convention, from `results/leaderboard.csv`. "Possessions" is the predicted count over the test matches (692 true); "Full match" is the count on the provided match (no ground truth).

| Method (config) | spellF1 [5–95%] | macroF1 | Possessions | Full match |
|---|---|---|---|---|
| `baseline_smooth50_clean` | 0.253 [0.237, 0.270] | 0.626 | 1367 | 594 |
| `deadball_vote125` | 0.412 [0.386, 0.439] | 0.802 | 800.5 | 284 |
| `rules_v0` (safety submission) | 0.513 [0.485, 0.541] | 0.823 | 633 | 250 |
| `hsmm_v1` | 0.496 [0.467, 0.526] | 0.816 | 563.5 | 238 |
| `hsmm_v2` | 0.533 [0.505, 0.564] | 0.836 | 604 | 248 |
| `hsmm_v2_fit` (fails rule B on `val`) | 0.568 [0.536, 0.602] | 0.834 | 513.5 | 203 |

<!-- FINAL: add the clean row (committed tree, not +dirty) of the chosen config from its single test pass, plus its 2/5/8 s val row in §6. Candidate numbers not yet usable: hsmm_v1_fit_mbr test 0.594 [0.561, 0.624], macroF1 0.848 (row from an uncommitted tree); val 2/5/8 s 0.470 / 0.561 / 0.511 against rules_v0 0.440 / 0.471 / 0.413 on a later simulator version, in-sample. -->

`test` was read once per configuration, but for several configurations, so the table carries some selection; differences with overlapping intervals are not treated as real. All configurations predict fewer possessions than the truth.

## 8. Limitations

- **No ball on 41% of frames.** Changes inside long gaps are missed: no ball, no touch events. On `val`, `rules_v0` goes from 0.438 with 20% of the ball missing to 0.387 with 50% (§4); ball-free evidence helps the expert more than the decoder.
- **Under-segmentation**, worst at the 2 s convention: fitted HSMMs predict 346–377 possessions per `val` run against 609 true at 2 s, and every method is below the true count on `test`.
- **In-sample `val` numbers.** Weights, early stopping and selection all use the same two `val` matches, one of them Metrica with a ball partly rebuilt from events, so `val` gains are optimistic; only the single `test` pass is out of sample.
- **Unknown label convention.** The 2–8 s range is a hedge, not a measurement; the ranking depends on it.
- **The external bench is imperfect:** other stadiums, trackers and label conventions; Smart Tagging is not simulated; Metrica labels are only as precise as the hand-collected events.
- **With more time:** a video ball detector good enough to fill a large share of gaps, ball height, and a sequence model that predicts possession counts closer to the truth. <!-- FINAL: one line of failure analysis for the chosen config. -->

## 9. Reproduction

Python 3.11, `pip install -r requirements.lock` (on Windows, use a short virtual-environment path so numba's DLL loads; `container/pompeu.def` builds the same environment with Apptainer). Unpack the match bundle into `data/` (see `data/README.md`); the repository contains no match data. Then:

```bash
python scripts/make_submission.py --config configs/final.yaml --out PompeuDataTeam.csv
python evaluation/validate_submission.py PompeuDataTeam.csv
```

`make_submission.py` prints the sha256, verified from a fresh clone with an empty cache: `bfd5ff16006b1e5c50371abeb528195cdc7a60397ec8bed90a8fccc468269f41` <!-- FINAL: replace with the chosen config's sha256 (must equal README.md), and list any artifact it needs (e.g. the LightGBM model) and pinned dependency. -->. The submission needs no GPU.

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
5. SkillCorner. Open data (broadcast tracking, phases of play and dynamic events). https://github.com/SkillCorner/opendata (MIT licence).
6. Idea only, no code reused: another participant's public repository (github.com/agamemnon-labs/football-analytics, MIT) reports a dilated temporal CNN over per-team features with Smart Tagging as input and synthetic tags for external matches; it prompted our synthetic-tagging experiment (§3). <!-- FINAL: keep only if the idea is still credited in the final text; cite it in §5 if the submitted temporal model follows it. -->
