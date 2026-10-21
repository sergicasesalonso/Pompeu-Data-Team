"""Pompeu Data Team -- possession inference for More than a Hack 2027, Phase 1.

Everything we write lives in this package. evaluation/ and starter/ belong to the organisers and
are only imported, never edited.

    pompeu.io         load the provided assets; the Tracking contract between Line A and Line B
    pompeu.validate   the validation harness: public_clean, public_stress, external, sanity
    pompeu.pipeline   config -> per-frame H/A/D labels

Entry points are in scripts/:

    python scripts/run_validation.py --config configs/baseline_smooth50.yaml
    python scripts/make_submission.py --config configs/baseline_smooth50.yaml
"""
