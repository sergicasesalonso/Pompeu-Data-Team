"""The data contract between the two lines: one Tracking class, whatever produced it.

Line A only ever reads a Tracking. Line B only ever writes one. That is what lets the two work in
parallel: when Line B's ATD+ arrives, Line A runs it by changing one word in a config.

    from pompeu.io.contract import load_tracking
    tr = load_tracking("atd")                 # the provided ATD
    tr = load_tracking("cache/atd_plus.npz")  # anything saved with Tracking.save()

| field         | shape     | meaning                                                   |
|---------------|-----------|-----------------------------------------------------------|
| players_xy    | (N, K, 2) | metres, NaN where the slot is not observed                |
| players_team  | (N, K)    | 0 home, 1 away, -1 unknown                                |
| players_conf  | (N, K)    | 0..1; 1 for an ATD observation, 0 where not observed      |
| ball_xy       | (N, 2)    | ball on the ground plane, metres, NaN if absent           |
| ball_z        | (N,)      | height in metres, NaN when unknown (always, for the ATD)  |
| ball_conf     | (N,)      | 0..1; 0 where absent                                      |
| ball_source   | (N,)      | code into BALL_SOURCES: none, atd, video, fused, interp   |

Row i is video frame frame0 + i. frame0 is 0 for a full match and moves only when crop() cuts a
window out, so code that needs the half or the absolute frame stays correct on a crop.

Coordinates are the ATD's: x along the length 0..105.3, y across 0..68.0, not flipped per half.
attack_sign() says which way a team attacks; normalising is A1's job, not the contract's.

The invariants are checked on construction, so a Tracking that exists is a valid one: a Line B
writer that breaks the contract fails where it writes, not three stages later in Line A.
"""

import os
from dataclasses import dataclass, field, replace

import numpy as np

from pompeu import paths  # noqa: F401  (makes games importable)
import games

# Params
TEAM_HOME, TEAM_AWAY, TEAM_UNKNOWN = 0, 1, -1
# Index = code stored in ball_source. "none" is the code wherever there is no ball.
BALL_SOURCES = ("none", "atd", "video", "fused", "interp")
SRC_NONE, SRC_ATD, SRC_VIDEO, SRC_FUSED, SRC_INTERP = range(len(BALL_SOURCES))
ARRAY_FIELDS = ("players_xy", "players_team", "players_conf",
                "ball_xy", "ball_z", "ball_conf", "ball_source")
DTYPES = {"players_xy": np.float64, "players_team": np.int8, "players_conf": np.float32,
          "ball_xy": np.float64, "ball_z": np.float32, "ball_conf": np.float32,
          "ball_source": np.int8}


@dataclass
class Tracking:
    players_xy: np.ndarray
    players_team: np.ndarray
    players_conf: np.ndarray
    ball_xy: np.ndarray
    ball_z: np.ndarray
    ball_conf: np.ndarray
    ball_source: np.ndarray
    frame0: int = 0
    # game, source, fps, pitch, team_names, home_attack_sign. Plain JSON-able values only.
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        for name in ARRAY_FIELDS:
            setattr(self, name, np.asarray(getattr(self, name), dtype=DTYPES[name]))
        self.frame0 = int(self.frame0)
        self.validate()

    # -- shape and invariants ------------------------------------------------------------------

    @property
    def n_frames(self):
        return self.players_xy.shape[0]

    @property
    def n_slots(self):
        return self.players_xy.shape[1]

    @property
    def frames(self):
        """Absolute video frame of each row."""
        return np.arange(self.frame0, self.frame0 + self.n_frames)

    @property
    def ball_present(self):
        return ~np.isnan(self.ball_xy[:, 0])

    @property
    def players_present(self):
        return ~np.isnan(self.players_xy[:, :, 0])

    def validate(self):
        """Raise ValueError naming the first broken invariant."""
        n, k = self.players_xy.shape[:2] if self.players_xy.ndim == 3 else (None, None)
        expected = {"players_xy": (n, k, 2), "players_team": (n, k), "players_conf": (n, k),
                    "ball_xy": (n, 2), "ball_z": (n,), "ball_conf": (n,), "ball_source": (n,)}
        for name, shape in expected.items():
            if getattr(self, name).shape != shape:
                raise ValueError("{} has shape {}, expected {}".format(
                    name, getattr(self, name).shape, shape))

        if not np.isin(self.players_team, (TEAM_HOME, TEAM_AWAY, TEAM_UNKNOWN)).all():
            raise ValueError("players_team must be 0, 1 or -1")
        if not np.isin(self.ball_source, np.arange(len(BALL_SOURCES))).all():
            raise ValueError("ball_source must be a code into BALL_SOURCES")
        for name in ("players_conf", "ball_conf"):
            values = getattr(self, name)
            if np.isnan(values).any() or (values < 0).any() or (values > 1).any():
                raise ValueError("{} must be in [0, 1], without NaN".format(name))

        # x and y are missing together, and "missing" means the same thing in every field.
        player_x_nan = np.isnan(self.players_xy[:, :, 0])
        if (player_x_nan != np.isnan(self.players_xy[:, :, 1])).any():
            raise ValueError("players_xy has x without y or y without x")
        if (self.players_conf[player_x_nan] != 0).any():
            raise ValueError("players_conf must be 0 where players_xy is NaN")
        ball_nan = np.isnan(self.ball_xy[:, 0])
        if (ball_nan != np.isnan(self.ball_xy[:, 1])).any():
            raise ValueError("ball_xy has x without y or y without x")
        if (self.ball_source[ball_nan] != SRC_NONE).any() or \
                (self.ball_source[~ball_nan] == SRC_NONE).any():
            raise ValueError("ball_source must be 'none' exactly where ball_xy is NaN")
        if (self.ball_conf[ball_nan] != 0).any():
            raise ValueError("ball_conf must be 0 where ball_xy is NaN")

    # -- match context -------------------------------------------------------------------------

    def half(self):
        """(N,) int8: 1 or 2 inside a half (graded frames), 0 outside."""
        out = np.zeros(self.n_frames, dtype=np.int8)
        frames = self.frames
        for half, first, last in games.half_video_blocks(self.meta.get("game", games.GAME)):
            out[(frames >= first) & (frames <= last)] = half
        return out

    def attack_sign(self, team, half):
        """+1 if `team` (0 home, 1 away) attacks towards +x in `half`, -1 if towards -x."""
        sign = int(self.meta["home_attack_sign"])
        sign = sign if half == 1 else -sign
        return sign if team == TEAM_HOME else -sign

    # -- derived copies ------------------------------------------------------------------------

    def crop(self, start, stop):
        """Rows for absolute video frames [start, stop), clipped to what this Tracking holds."""
        lo = max(0, start - self.frame0)
        hi = min(self.n_frames, stop - self.frame0)
        return replace(self, frame0=self.frame0 + lo, meta=dict(self.meta),
                       **{name: getattr(self, name)[lo:hi].copy() for name in ARRAY_FIELDS})

    def without_ball(self, mask):
        """Copy with the ball removed on the rows where `mask` is True, invariants kept."""
        mask = np.asarray(mask, dtype=bool)
        ball_xy, ball_z = self.ball_xy.copy(), self.ball_z.copy()
        ball_conf, ball_source = self.ball_conf.copy(), self.ball_source.copy()
        ball_xy[mask] = np.nan
        ball_z[mask] = np.nan
        ball_conf[mask] = 0
        ball_source[mask] = SRC_NONE
        return replace(self, ball_xy=ball_xy, ball_z=ball_z, ball_conf=ball_conf,
                       ball_source=ball_source, meta=dict(self.meta))

    # -- persistence ---------------------------------------------------------------------------

    def save(self, path):
        """One .npz. It is derived from the match data, so it never goes into git."""
        folder = os.path.dirname(path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        meta = np.array(repr(self.meta))
        with open(path + ".part", "wb") as handle:
            np.savez(handle, frame0=self.frame0, meta=meta,
                     **{name: getattr(self, name) for name in ARRAY_FIELDS})
        os.replace(path + ".part", path)
        return path

    @classmethod
    def load(cls, path):
        import ast
        with np.load(path, allow_pickle=False) as stored:
            return cls(frame0=int(stored["frame0"]), meta=ast.literal_eval(str(stored["meta"])),
                       **{name: stored[name] for name in ARRAY_FIELDS})

    @classmethod
    def from_atd(cls, atd):
        """The provided ATD as a Tracking: confidence 1 wherever it observed something."""
        players_xy = atd.P[:, :, :2]
        observed = ~np.isnan(players_xy[:, :, 0])
        ball_xy = atd.B[:, :2]
        ball = ~np.isnan(ball_xy[:, 0])
        return cls(
            players_xy=players_xy,
            players_team=np.broadcast_to(atd.slot_team, observed.shape),
            players_conf=observed.astype(np.float32),
            ball_xy=ball_xy,
            ball_z=np.full(atd.n_frames, np.nan, dtype=np.float32),
            ball_conf=ball.astype(np.float32),
            ball_source=np.where(ball, SRC_ATD, SRC_NONE).astype(np.int8),
            meta={"game": atd.game, "source": "atd:" + atd.source,
                  "fps": games.GAMES[atd.game]["fps"],
                  "pitch": tuple(games.GAMES[atd.game]["pitch"]),
                  "team_names": tuple(atd.team_names),
                  "home_attack_sign": int(atd.home_attack_sign)},
        )


def load_tracking(spec="atd", game=games.GAME, data_dir=None):
    """The single way Line A gets a Tracking.

    spec is "atd" for the provided tactical ATD, or a path to a .npz written by Tracking.save()
    (Line B's ATD+, or anything else that honours the contract).
    """
    if spec == "atd":
        from pompeu.io.atd import load_atd
        return Tracking.from_atd(load_atd(game=game, data_dir=data_dir))
    path = spec if os.path.isabs(spec) else os.path.join(paths.ROOT, spec)
    if not os.path.isfile(path):
        raise IOError("tracking {!r}: not 'atd' and no file at {}".format(spec, path))
    return Tracking.load(path)
