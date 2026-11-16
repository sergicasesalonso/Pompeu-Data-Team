# Pompeu Data Team: frame-level ball possession

More Than A Hack 2027, Phase 1 · Martí, Sergi, Ander · Code: https://github.com/sergicasesalonso/Pompeu-Data-Team

Implementation detail for every module (all parameters, file by file) is in `ARQUITECTURA.md` in the repository.

## 1. Summary

- **Task:** label each of the 199,319 frames of the tactical video `H`, `A`, `D` or `X`; ranked on spellF1.
- **Submission: `hsmm_tcn`** (`configs/final.yaml`). A hidden semi-Markov model (HSMM) segments the match into team-owned live and dead periods with explicit durations. A dilated temporal convolutional network (TCN), trained only on public data from other matches, is added to it as one more expert. On our external bench (other matches with provider labels, degraded to look like our tracking) its single pre-registered `test` pass gives **spellF1 0.678 [0.647, 0.709]** and macroF1 0.884. The previous candidates get 0.539 (`hsmm_v1_fit_mbr`) and 0.459 (`rules_v0`, the rule-based safety submission) on the same bench (§7).
- **Ball from the video (Line B)** was explored and is not submitted: the broadcast adds about 3 points of ball coverage, worth about +0.01 spellF1, within the noise (§4).

## 2. Approach and why

Every module reads and writes one data contract, `Tracking` (`pompeu/io/contract.py`): metres on a 105.3 × 68.0 pitch, one row per video frame, and a confidence and source for each ball observation. **Line A** labels from the provided assets: A1 cleaning → A2 per-frame features (about 200 columns) → A3 dead-ball detector → A4 touches, controls and the `rules_v0` state machine → A5 HSMM → TCN expert. **Line B** adds ball from the video and broadcast into an augmented tracking, "ATD+". Switching to it only changes `tracking:` in the YAML.

- **spellF1 scores possessions as objects**: runs matched by temporal IoU. Per-frame voting fragments them. On the external `test` bench, smoothed voting predicted twice as many possessions as there are (§3). Hence a segmentation with explicit durations and transitions.
- **The ball is missing on 41% of graded frames** (CHALLENGE.md), and that is where possession changes are lost. Touch events cannot see them. The TCN reads about ±51 s of collective context, plus the Smart Tagging phases, and gives the HSMM team evidence inside those gaps.

## 3. Assets used and dropped

| Asset | Used for | Status |
|---|---|---|
| Tactical ATD (players and ball) | Everything in Line A; source of B1 pseudo-labels | Used |
| Smart Tagging | TCN inputs only (possession phases and set pieces, §5). In the submitted model the HSMM and A3 run without tags, the condition validated on the external bench | Used |
| Tactical homography | Metres to pixels for B0, B1, B3 | Line B only |
| Tactical video | B0: decoded frame count equals the ATD's (199,319); B1 pseudo-labels for a ball detector | Explored, not submitted |
| Tactical audio (AAC, 48 kHz) | Referee-whistle candidates | Explored, not used |
| Broadcast ATD and homography | Extra ball observations after alignment to the tactical clock | Explored, not submitted (§4) |
| Broadcast video | Not needed: alignment uses only the broadcast ATD | Dropped |
| External open data | Validation bench, model weights, TCN training (§6) | Used |

**Tried and dropped** (all on external data; `val` = DFL J03WN1 + Metrica 2):

- **Smoothed per-frame voting** (`baseline_smooth25/50/125`): 1,367 possessions predicted on `test` against 692 true ones, 0.253 spellF1; with dead-ball segmentation (`deadball_vote125`) 0.412. These numbers are from an earlier simulator version (§6).
- **HSMM with a single `D` state:** team evidence during stoppages (a goalkeeper placing the ball) overrode dead ball, so it was replaced by team-owned dead states (§5).
- **Gap-team expert** (`hsmm_v2`): a ridge logistic regression on ball-free collective features (centroids, lines, compactness, pressing counts), giving team log-odds on frames without ball. It names the right team on 69.8% of DFL `val` gap frames, 78.0% after augmentation, but inside the HSMM the gain was only +0.001 to +0.003 spellF1. The TCN replaces it: it learns the same ball-free evidence jointly with the dead-ball and possession structure.
- **HSMM weights fitted at a single label convention** (`hsmm_v2_fit`): it gains at 5 s and loses to `rules_v0` at 2 s on `val`, because it predicts too few, long possessions. It fails rule B (§6).
- **Self-training** the gap-team expert on algorithmic pseudo-labels of the provided match: the pseudo-labels fall on easy frames, and recovered gap changes drop from 0.693 to 0.62–0.69. Negative.
- **The HSMM reading Smart Tagging directly**, with synthetic tags on the external matches: +0.008 with overlapping intervals; weights ≥ 4 hurt. In the final model the HSMM gets no tagging; tags reach it only through the TCN.

## 4. Handling missing and noisy tracking

Cleaning is done entirely by code (`pompeu/clean/`, A1); nothing is corrected by hand.

- **Ball:** a point implying more than 40 m/s to both neighbours is removed, fragments under 3 frames are dropped, and gaps under 0.5 s are spline-filled (marked `interp`, confidence 0.5). This removes 1,911 of the 2,043 jumps above 40 m/s (`results/cleaning_report.md`). Phantom balls never change possession, because every event needs a player within 2 m.
- **Players:** frozen slots are flagged (confidence 0.3), not deleted; same-team duplicates are removed; tracklets are stitched with the Hungarian algorithm under a speed bound. Velocities are recomputed from positions, since the ATD column is empty on 89% of frames.
- **No ball:** ball features stay NaN, never imputed; gap length and age are features, and every feature that can be NaN gets a missing flag in the TCN.

**Line B (explored, not submitted).**
- **B0:** the decoded video has exactly the ATD's 199,319 frames, and the ATD ball projects inside the image on 99.91% of observed frames.
- **B1:** 88,937 ball pseudo-labels: the observed, cleaned ATD ball projected through the homography and filtered.
- **B2:** a temporal heatmap ball detector was built and piloted, but CPU inference over the full match did not fit the deadline.
- **B8:** the broadcast ATD is aligned to the tactical clock from player positions alone (median 0.29 m), and replays are discarded.
- **B3:** physically constrained fusion. Coverage of graded frames goes from 59.2% to 62.1–62.9%, and ATD+ regenerates with the same hash.

A coverage sweep on DFL J03WN1 (earlier simulator; only differences are comparable) puts one point of coverage at about +0.0025 spellF1 for `rules_v0` and `hsmm_v1`. The broadcast's 3 points are worth about +0.01, so a video detector would have had to fill a large share of the gaps to matter.

## 5. Possession rules implemented

**Dead ball (A3, `pompeu/events/deadball.py`).** A sticky two-state live/dead HMM (mean live period 40 s, dead 20 s). Cues add per-frame log-odds:
- a slow field;
- the ball out or not visible;
- the ball still at a restart spot;
- Smart Tagging set pieces and `GOALS`, when tagging is given (the submitted model gives none to A3);
- pre-kick-off;
- a restart kick, as strong live evidence.

After Viterbi, live stretches under 5 s between dead periods are closed. A dead period runs from the ball crossing the line (or the field slowing) to the restart kick. A LightGBM live/dead expert adds to it (`artifacts/deadball_gbm_v1.txt`, lightgbm 4.6.0 pinned). It was trained on external `train` only (1,425,888 rows, 185 features), with early stopping on `val` at iteration 50. Combined with A3, its dead-ball F1 on `test` is 0.911, against 0.877 with the rules alone (`results/deadball_gbm_report.md`).

**Restarts.** The restarting team owns the next possession:
- throw-in: the opponent of the last nearby player;
- after a goal: the conceding team;
- over the goal line: corner or goal kick, depending on who touched last;
- otherwise, the restart is classified from where the ball is placed (free kick by default).

The team comes from Smart Tagging first when it is given, then geometry, then the kicker.

**Events (A4, `pompeu/events/touches.py`).**
- A *touch* is a ball speed change above 3 m/s or a direction change above 30°, with a player within 2 m. Its team probability is 80% distance difference and 20% pitch control [4].
- A *control* is the ball within 1.5 m of the same player for at least 0.6 s.

**`rules_v0` (the state machine; it is also an HSMM expert).**
- A change of possession needs control by the opponent: a second opponent touch within 3 s, or an opponent control.
- A single opponent touch (a deflection or a challenge) stays pending and is cancelled by a later event of the team in possession.
- Aerial balls (above 15 m/s before and after the touch) are ignored.
- Events with P(home) between 0.35 and 0.65 are ignored.
- A loose or invisible ball stays with the last team in control.

**HSMM (`pompeu/models/hsmm.py`).**
- **States:** `HL`/`AL` (live, team in possession) and `HD`/`AD` (dead, team to restart), with log-normal durations. Team-owned dead states make the restart structural (`HD → HL`, same team with probability 0.98).
- **Experts:** A3 cues and segmentation, A4 events, the `rules_v0` output, the A3 restart team and the TCN.
- **Fitting:** weights and duration parameters are fitted on external `val` only, by coordinate descent: w_events 0.5, w_dead 2.0, rules_llr 0.5, a3_segment_llr 1.0 (from a 0/1/3/6 grid), w_proximity 0, poss_median_s 12, p_poss_to_dead 0.4, gbm_weight 1.0.
- **Decoding:** minimum Bayes risk (MBR) over 200 posterior samples, with a greedy boundary search on expected spellF1.

**TCN expert (`pompeu/experimental/tcn.py`, `hsmm_tcn.py`).**
- **Architecture:** a non-causal dilated TCN at 5 Hz, with 96 channels, kernel 3 and dilations 1–128 (about ±51 s of context), about 0.5 M parameters. It has three output heads, one per label convention (2/5/8 s, §6), averaged at inference.
- **Inputs:** the A2 features, standardised, with missing flags, plus 18 Smart Tagging context columns.
- **Training:** on external `train` (4 DFL matches and Metrica 1, degradation seeds 0–3) and the 20 SkillCorner matches, each with 4 augmentations (team swap and y-mirror, recomputed from the clean match). Settings: AdamW, one-cycle, 2,500 steps, dropout 0.3, input dropout 0.2. The checkpoint is chosen by cross-entropy on `val` items built from degradation seeds 10–11, not the scored ones.
- **Tags in training:** the external matches have no Smart Tagging, so the training inputs use *synthetic* tags. They are generated from the labels with an error model measured on the provided match against confident `rules_v0` stretches, with no ground truth.
- **Tags on our match:** the real tags are restricted to the codes the generator produces: possession phases and four set-piece types. Zone, goal and kick-off codes never appeared in training.
- **How it enters the HSMM:** through a product of experts. Its team log-odds log P(H) − log P(A) (weight 1) and its dead log-odds log P(D) − log(P(H)+P(A)) (weight 2) are added to the HSMM's per-frame evidence. Both weights were picked on `val`.
- **Reproducibility:** training is deterministic on CPU. The committed weights (`artifacts/tcn_v1.pt`) are pinned by hash (`ceecf8915830d7c0`), and the pipeline stops if they differ.

**Final label policy.** There is no `X` on graded frames, and `X` outside the halves. The A10 policy (`pompeu/decode/policy.py`) removes runs of 1–2 frames. For `hsmm_v1_fit_mbr` on `val` it changes no graded label.

## 6. Training and validation data

**Rules we follow.**
- No human labels on the provided match, for any purpose.
- External data is about other games only.
- **The five public minutes were not used to train, tune or select anything.** They only check the pipeline: alignment, format and label meaning. Two values first read off that window were removed or re-justified without it: the A3 segmentation weight is refitted on `val`, and the degradation's `dead_weight` is checked by a label-free measurement (below).

| Dataset | Source and licence | Labels: who and how | How much | Used for |
|---|---|---|---|---|
| DFL / IDSSE open data [1, 2] | figshare, doi:10.6084/m9.figshare.28196177. **CC BY 4.0**; owner Deutsche Fußball Liga | Provider: per-frame `BallStatus` and `BallPossession` (TRACAB, 25 fps), mapped by `align_labels` | 7 full matches, 2022/23 | `train` / `val` / `test` |
| Metrica Sports sample data [3] | github.com/metrica-sports/sample-data; no licence file, the repository asks users to acknowledge the source | Provider events, collected manually; per-frame labels derived by code | 3 anonymised full matches | `train` / `val` / `test` |
| SkillCorner Open Data [5] | github.com/SkillCorner/opendata, **MIT** | Provider phases of play; `D` = gaps between phases | 20 A-League 2024/25 matches | TCN training only |
| Ball pseudo-labels (B1) | Our code on the provided match | No person: projected ATD ball, filtered | 88,937 frames, never published | Not used by the submission |

**Splits.** They are by whole match and fixed (`pompeu/external/__init__.py`):

| Split | Matches | Used for |
|---|---|---|
| `train` | DFL J03WOH, J03WQQ, J03WOY, J03WR9; Metrica 1 | everything that learns |
| `val` | DFL J03WN1; Metrica 2 | weights, early stopping, selection |
| `test` | DFL J03WMX, J03WPY; Metrica 3 | never trained or tuned on |

Metrica 3 is the test match because games 1 and 2 have the ball glued to the carrier.

**Degradation** (`pompeu/external/degrade.py`, seeded). External tracking is made to look like our ATD:
- **Ball erased:** in runs drawn from the ATD's gap lengths, until **41%** of in-half frames have no ball. Runs start `dead_weight` = 5 times likelier on dead frames.
- **Ball blurred:** a Gaussian within each observed run (0.27 s on DFL, 0.40 s on Metrica). It was calibrated so that A4's touch rate on degraded matches approaches the ATD's (about 8 per minute, against 16–22 on raw external tracking).
- **Ball displaced:** 0.12% of ball frames are moved 10–40 m.
- **Players:** identity swaps, freezes, gaps, dropouts and ghosts.

The value 5 was first read off the public window, so we checked it without any label: with our own A3 segmentation of the full match, the ball is absent on 74.7% of dead frames and 12.0% of live ones. That confirms that gaps concentrate in stoppages, so we keep 5 rather than retune the bench every number depends on. Every number in §6–§7 comes from this simulator version, except where marked as earlier.

**Label convention `min_possession_s`.** DFL's `BallPossession` changes on almost every touch. `align_labels` gives a possession shorter than `min_possession_s`, sandwiched between two of the other team, to the surrounding team. The convention of the hidden ground truth is unknown, so **every candidate is scored at 2, 5 and 8 s**. Some candidates swap places in the ranking depending on the convention (`val`, 5 seeds, spellF1):

| Method | 2 s | 5 s | 8 s | Mean | Possessions per run |
|---|---|---|---|---|---|
| `rules_v0` | 0.440 | 0.471 | 0.413 | 0.441 | 395 |
| `hsmm_v2_fit` (fitted at 5 s) | 0.428 | 0.515 | 0.469 | 0.471 | 350 |
| `hsmm_v1_fit_mbr` | 0.470 | 0.561 | 0.511 | 0.514 | 343 |
| **`hsmm_tcn`** (submitted) | **0.545** | **0.658** | **0.602** | **0.602** | 360 |

True possessions per run: 609, 429 and 343 at 2, 5 and 8 s. On `val` the TCN reads synthetic tags, and its weights, smoothing and checkpoint were picked on these same matches, so its `val` row is optimistic.

**Selection rule (rule B).** A candidate replaces `rules_v0` only if all of these hold:
1. its mean spellF1 over 2/5/8 s on `val` is higher;
2. it loses no more than 0.02 at any convention;
3. all its weights come from `train`/`val`;
4. a **single** `test` pass shows spellF1 better with non-overlapping intervals (or a tie with better macroF1), and a full-match sanity check that does not get worse;
5. it regenerates the same sha256 from a clean clone.

**What `test` has seen.** We say it as it happened. Before the rule was fixed, five HSMM configurations were scored on `test` on this simulator (§7, first five rows); `hsmm_v1_fit_mbr` was then chosen on `val`. For `hsmm_tcn`, the candidate set, the decision metric and the criterion were written down before its pass. It was then run once on `test`, together with `rules_v0` and `hsmm_v1_fit_mbr` on identical degraded matches, and nothing was changed afterwards.

**Metrics** come from the organisers' `evaluation/possession_metrics.py`, never reimplemented, with block-bootstrap 5–95% intervals.

## 7. Results

External bench, `test` split (DFL J03WMX, J03WPY, Metrica 3), 2 degradation seeds per match, 5 s convention. "Possessions" is the mean predicted count over the test matches (692 true). "Full match" is the count on the provided match, which has no ground truth.

| Method (config) | spellF1 [5–95%] | macroF1 | Possessions | Full match |
|---|---|---|---|---|
| `rules_v0` (safety submission) | 0.459 [0.426, 0.490] | 0.803 | 598 | 250 |
| `hsmm_v2_fit` | 0.527 [0.489, 0.564] | 0.818 | – | – |
| `hsmm_v1_fit` | 0.536 [0.500, 0.570] | 0.831 | – | – |
| `hsmm_v1_fit_avg` | 0.536 [0.503, 0.568] | 0.831 | – | – |
| `hsmm_v1_fit_mbr` | 0.539 [0.503, 0.574] | 0.832 | 479 | 204 |
| **`hsmm_tcn` (submitted)** | **0.678 [0.647, 0.709]** | **0.884** | 555 | 206 |

At the other conventions, the single pass gives:

| Convention | `rules_v0` | `hsmm_v1_fit_mbr` | `hsmm_tcn` |
|---|---|---|---|
| 2 s | 0.443 | 0.453 | **0.573** |
| 8 s | 0.430 | 0.537 | **0.639** |

The intervals of `hsmm_tcn` do not overlap with the others' at any of the three conventions. Re-scored in the same run, `rules_v0` and `hsmm_v1_fit_mbr` reproduce their earlier rows exactly (0.4588 and 0.5392), so the bench is unchanged (`scripts/experiments/test_pass.py`).

On `test` the external matches have no Smart Tagging, so the TCN runs there with an empty tag table. Its 0.678 is therefore a no-tags figure. On `val`, removing the tag channels costs about 0.05 for the TCN alone and about 0.03 for HSMM+TCN (`results/tcn_report.md`), so tags help, but their transfer to the real match is uncertain (§8).

## 8. Limitations

- **Under-segmentation.** Every method predicts fewer possessions than the truth: 555 against 692 on `test` for the TCN, and 206 on the full match, below our prior expectation of 300–400. The TCN eases this, but it does not remove it. Missing possessions are the main error, and they cost most at the 2 s convention.
- **No ball on 41% of frames.** Changes inside long gaps rely on ball-free evidence alone. On `val`, `rules_v0` goes from 0.438 with 20% of the ball missing to 0.387 with 50%.
- **Smart Tagging transfer.** The TCN learned from synthetic tags. The provided match's real tags agree with the tracking at least as well as the synthetic ones (team agreement 0.85 against 0.68), but they are sparser: they cover 27% of ball-missing frames against 43%. The gain on the real match may be smaller than on `val`, and it is not measured on `test`, which has no tags.
- **In-sample `val` numbers.** Weights, the TCN checkpoint and selection all use the same two `val` matches. One of them is Metrica, with a ball partly rebuilt from events. Only the `test` pass is out of sample, and `test` was read for five earlier configurations (§6).
- **Unknown label convention.** The 2–8 s range is a hedge, not a measurement.
- **An imperfect external bench:** other stadiums, trackers and label conventions, and Metrica labels only as precise as hand-collected events.
- **With more time:**
  - a temporal model that predicts possession counts closer to the truth;
  - richer real tags, or a model trained on the provided match's tag statistics;
  - a video ball detector able to fill a large share of the gaps.

## 9. Reproduction

1. **Environment:** Python 3.11 and `pip install -r requirements-cpu-torch.lock`. It is `requirements.lock` plus torch 2.9.1 for CPU, and it pins numpy 2.0.2 and lightgbm 4.6.0. On Windows, use a short virtual-environment path so numba's DLL loads.
2. **Data:** unpack the match bundle into `data/` (see `data/README.md`). The repository contains no match data.
3. **Run:**

   ```bash
   python scripts/make_submission.py --config configs/final.yaml --out PompeuDataTeam.csv
   python evaluation/validate_submission.py PompeuDataTeam.csv
   ```

Expected sha256: `8959a2ce91303dbb0d6d05c516af80c771266eda9491dd12cb223d0de72a2379`, reproduced with empty caches on CPU. The run needs the committed artifacts `artifacts/tcn_v1.pt` (pinned by hash) and `artifacts/deadball_gbm_v1.txt`, and no GPU. To retrain the TCN deterministically on CPU instead (same weights hash), follow "Reproduce the shipped model" in `results/tcn_report.md`.

To rebuild the external bench and score a config: `python scripts/prepare_external.py --matches all`, clone the Metrica sample data into `data/external/metrica/sample-data`, then `python scripts/run_validation.py --config configs/final.yaml`.

## References

1. Bassek, M., Rein, R., Weber, H., & Memmert, D. (2025). An integrated dataset of spatiotemporal and event data in elite soccer. *Scientific Data*, 12, 195. https://doi.org/10.1038/s41597-025-04505-y
2. Bassek, M., Rein, R., Weber, H., & Memmert, D. (2025). An integrated dataset of spatiotemporal and event data in elite soccer [Data set]. figshare. https://doi.org/10.6084/m9.figshare.28196177 (CC BY 4.0).
3. Metrica Sports. Sample data (tracking and event data, 3 anonymised matches). https://github.com/metrica-sports/sample-data
4. Spearman, W. (2018). Beyond expected goals. *MIT Sloan Sports Analytics Conference*. (Pitch-control model used in A2.)
5. SkillCorner. Open data (broadcast tracking, phases of play and dynamic events). https://github.com/SkillCorner/opendata (MIT licence).
6. Idea only, no code reused: another participant's public repository (github.com/agamemnon-labs/football-analytics, MIT) describes a dilated temporal CNN over per-team features, with Smart Tagging as input and synthetic tags for external matches. It prompted our synthetic-tagging experiment and the TCN expert (§5).
