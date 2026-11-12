"""A6: external matches with real labels -- one registry, one split, whatever their source.

    from pompeu import external
    external.resolve("test")             # ['dfl:J03WMX', 'dfl:J03WPY', 'metrica:3'] if present
    tracking, labels = external.load("metrica:3")

Names are "<source>:<match>". Sources:

    dfl      IDSSE / DFL open data (pompeu.external.dfl): 7 Bundesliga matches, per-frame labels
    metrica  Metrica Sports sample data (pompeu.external.metrica): 3 matches, labels from events

THE SPLIT is by whole match, fixed, so no frame of a match is ever both trained on and scored
on (plan, stage A6 step 4):

    test   the `external` bench that decides. Never trained on, never tuned on.
    val    fitting the few weights (HSMM experts, calibration temperature), early stopping.
    train  everything that learns: the live/dead classifier (A3), the sequence model (A7).

Both sources and both German divisions sit in test. A bare DFL id ("J03WPY") is accepted as
"dfl:J03WPY".
"""

from pompeu.external import dfl, metrica

SOURCES = {"dfl": dfl, "metrica": metrica}
SPLITS = {
    # Metrica games 1 and 2 have the ball glued to the carrier (median 0.28 m to the nearest
    # player, against 1.4 m on DFL and 1.9 m on our ATD): their ball is partly rebuilt from the
    # events. A test match like that would flatter every method that trusts ball proximity, so
    # the test Metrica match is game 3 (EPTS, a realistic 1.3 m).
    "test": ["dfl:J03WMX", "dfl:J03WPY", "metrica:3"],
    "val": ["dfl:J03WN1", "metrica:2"],
    "train": ["dfl:J03WOH", "dfl:J03WQQ", "dfl:J03WOY", "dfl:J03WR9", "metrica:1"],
}
ALL = SPLITS["train"] + SPLITS["val"] + SPLITS["test"]


def parse(name):
    """(source module, match key) for a registry name."""
    if ":" not in name:
        name = "dfl:" + name
    source, key = name.split(":", 1)
    if source not in SOURCES:
        raise KeyError("unknown external source {!r}; have {}".format(source, sorted(SOURCES)))
    return SOURCES[source], (int(key) if source == "metrica" else key)


def canonical(name):
    return name if ":" in name else "dfl:" + name


def available(name):
    module, key = parse(name)
    if module is dfl:
        try:
            dfl.files(key)
            return True
        except IOError:
            return False
    return metrica.available(key)


def load(name):
    """(Tracking, labels) with the source's own labels (not yet convention-aligned)."""
    module, key = parse(name)
    return module.load(key)


def resolve(setting):
    """Registry names for a setting: a split name, "all", "auto" (= every match present), or a
    list. Only matches whose files are present are returned."""
    if isinstance(setting, str):
        if setting in SPLITS:
            names = SPLITS[setting]
        elif setting in ("all", "auto"):
            names = ALL
        else:
            names = [setting]
    else:
        names = list(setting or [])
    return [canonical(name) for name in names if available(canonical(name))]


def align_labels(labels, min_possession_s, fps=25):
    """The challenge's control convention on any source's labels (see dfl.align_labels)."""
    return dfl.align_labels(labels, min_possession_s, fps)
