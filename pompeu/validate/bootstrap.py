"""Confidence intervals for spellF1 and macroF1, built on possession_metrics.

One prediction: exactly the organisers' intervals -- pm.bootstrap_spell_f1 over possessions and
pm.bootstrap_macro_f1 over segments, called as pm.evaluate calls them.

Several predictions of the same window (the stress bench: one per degradation seed): the quantity
of interest is the seed-averaged score, and it has two sources of noise, which ground-truth
possessions happen to be in the window and which gaps the seeds drew. Each draw resamples both --
the seeds with replacement, and the possessions (or segments) with replacement, the same picks for
every seed -- then averages the per-seed scores. The per-draw scores use the same formulas as
pm's own bootstrap; tests/test_bootstrap.py checks that with no resampling they equal pm.evaluate.
"""

import numpy as np

from pompeu import paths  # noqa: F401
import possession_metrics as pm


def _spell_tables(gt_codes, pred_codes, runs):
    """(flags (n_iou, n_gt) bool, n_pred) for one prediction, as pm.bootstrap_spell_f1 has them."""
    filled = pm.fill_abstentions(pred_codes, runs)
    gt_spells = pm.possession_spells(gt_codes, runs)
    pred_spells = pm.possession_spells(filled, runs)
    overlaps = pm.spell_overlaps(gt_spells, pred_spells)
    flags = np.vstack([pm.matched_gt_flags(overlaps, len(gt_spells), threshold)
                       for threshold in pm.SPELL_IOUS])
    return flags, len(pred_spells)


def spell_f1_from_flags(flags, n_pred, picks):
    """Mean spell F1 over IoU thresholds for the possessions in `picks` (pm's bootstrap formula)."""
    n_gt = len(picks)
    f1s = []
    for row in flags:
        true_positive = int(row[picks].sum())
        false_positive = max(0, n_pred - true_positive)
        false_negative = n_gt - true_positive
        f1s.append(2.0 * true_positive / (2 * true_positive + false_positive + false_negative)
                   if true_positive else 0.0)
    return float(np.mean(f1s))


def _segment_tables(gt_codes, pred_codes, runs):
    """(gt code per GT segment, per-segment prediction histogram), as pm.bootstrap_macro_f1."""
    gt_segments = pm.segments(gt_codes, runs)
    gt_of_segment = np.array([code for _, _, code in gt_segments], dtype=np.int64)
    contributions = np.zeros((len(gt_segments), pm.N_LABELS), dtype=np.int64)
    for index, (start, stop, _) in enumerate(gt_segments):
        contributions[index] = np.bincount(pred_codes[start:stop].astype(np.int64),
                                           minlength=pm.N_LABELS)
    return gt_of_segment, contributions


def macro_f1_from_segments(gt_of_segment, contributions, picks):
    cm3 = np.zeros((pm.N_SCORED, pm.N_LABELS), dtype=np.int64)
    np.add.at(cm3, gt_of_segment[picks], contributions[picks])
    return pm.macro_f1(cm3)


def intervals(gt_codes, preds, n_draws, seed=pm.BOOTSTRAP_SEED,
              percentiles=pm.BOOTSTRAP_PERCENTILES):
    """{'spell': (low, high), 'macro': (low, high)} for one prediction array or a list of them."""
    if not n_draws:
        return {"spell": (None, None), "macro": (None, None)}
    runs = pm.valid_runs(gt_codes)

    if not isinstance(preds, (list, tuple)):
        filled = pm.fill_abstentions(preds, runs)
        return {"spell": pm.bootstrap_spell_f1(gt_codes, filled, runs, n_draws, seed, percentiles),
                "macro": pm.bootstrap_macro_f1(gt_codes, preds, runs, n_draws, seed, percentiles)}

    spell_tables = [_spell_tables(gt_codes, pred, runs) for pred in preds]
    segment_tables = [_segment_tables(gt_codes, pred, runs) for pred in preds]
    n_gt = spell_tables[0][0].shape[1]
    n_segments = len(segment_tables[0][0])
    if not n_gt or not n_segments:
        return {"spell": (None, None), "macro": (None, None)}

    rng = np.random.RandomState(seed)
    spell_scores = np.empty(n_draws)
    macro_scores = np.empty(n_draws)
    for draw in range(n_draws):
        seeds = rng.randint(0, len(preds), len(preds))
        spell_picks = rng.randint(0, n_gt, n_gt)
        segment_picks = rng.randint(0, n_segments, n_segments)
        spell_scores[draw] = np.mean([spell_f1_from_flags(*spell_tables[s], spell_picks)
                                      for s in seeds])
        macro_scores[draw] = np.mean([macro_f1_from_segments(*segment_tables[s], segment_picks)
                                      for s in seeds])
    return {"spell": tuple(float(v) for v in np.percentile(spell_scores, percentiles)),
            "macro": tuple(float(v) for v in np.percentile(macro_scores, percentiles))}
