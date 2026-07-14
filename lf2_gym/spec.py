"""Gymnasium space builders for LF2 — pure numpy + gymnasium.

Lets the learner (on Linux) declare ``observation_space`` / ``action_space``
without instantiating a real game controller. The same builders are used by
:class:`~lf2_gym.lf2_envs.base.Lf2EnvBase` and
:class:`~lf2_gym.lf2_envs.base.Lf2ParallelEnvBase` so on-Windows and
off-Windows code agree byte-for-byte on space dtypes / shapes.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from gymnasium import spaces

Mode = Literal["info", "picture", "mix"]


#: Number of scalar fields per player in the info observation: ``[mp, hp,
#: facing, x, y, z]``.
INFO_FIELDS_PER_PLAYER: int = 6

#: Per-field divisor used by :func:`normalize_info` to map raw game-memory
#: ints into a roughly ``[0, 1]``-ish range. Order matches
#: ``[mp, hp, facing, x_pos, y_pos, z_pos]``. Values reflect typical LF2
#: ranges:
#:
#: * ``mp_max`` / ``hp_max`` are 500 across most characters
#: * ``facing`` is already 0/1 (scale of 1.0 is a no-op)
#: * ``x_pos`` reaches ~1500 px on the widest default stages
#: * ``y_pos`` reaches ~600 px (floor-area height)
#: * ``z_pos`` is depth into the screen — non-negative, up to ~1000 px on
#:   typical stages (earlier assumption of ``[-300, 0]`` was wrong)
INFO_SCALES: np.ndarray = np.array(
    [500.0, 500.0, 1.0, 1500.0, 600.0, 1000.0], dtype=np.float32
)

#: Per-field additive bias applied AFTER dividing by :data:`INFO_SCALES`.
#: Currently zero for every column — kept as an explicit vector so future
#: tuning (e.g. re-centering positions) is a one-line change.
INFO_BIAS: np.ndarray = np.array(
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32
)


def normalize_info(raw: np.ndarray) -> np.ndarray:
    """Convert raw ``(num_players, 6) int16`` player state into the
    ``(num_players * 6,) float32`` observation actually fed to the policy.

    Per column the formula is ``raw / INFO_SCALES + INFO_BIAS`` — picked
    so typical LF2 values land in roughly ``[0, 1]``. The cast collapses
    the 2-D layout so RLlib's default ``Catalog`` can auto-build an MLP
    encoder.
    """
    return (raw.astype(np.float32) / INFO_SCALES + INFO_BIAS).reshape(-1)


def build_lf2_info_space(num_players: int, mp_max: int = 500, hp_max: int = 500) -> spaces.Box:
    """Flat ``float32`` info Box matching :func:`normalize_info` output.

    Shape ``(num_players * 6,)``, dtype ``float32``. Bounds are set wide
    (``[-100, 100]``) so ``Preprocessor.check_shape`` never rejects a
    perfectly reasonable normalized value that just happens to sit
    outside a tight declared band — e.g. an unusually wide stage or a
    z_pos beyond our scale estimate. The MLP encoder doesn't use the
    bounds for normalization; the bounds are just metadata for
    ``Box.contains()``.

    ``mp_max`` / ``hp_max`` are kept on the signature for backwards
    compat but do not drive the bounds — the per-field scaling lives
    in :data:`INFO_SCALES`. Change the scale, not the bound, when
    retuning for a different LF2 setup.
    """
    del mp_max, hp_max  # noqa: kept for back-compat, see docstring
    np_type = np.float32
    low = np.full(num_players * INFO_FIELDS_PER_PLAYER, -100.0, dtype=np_type)
    high = np.full(num_players * INFO_FIELDS_PER_PLAYER, 100.0, dtype=np_type)
    return spaces.Box(low=low, high=high, dtype=np_type)


def build_lf2_image_space(channels: int, img_h: int, img_w: int) -> spaces.Box:
    """Channels-first stacked-image observation."""
    return spaces.Box(
        low=0,
        high=255,
        shape=(channels, img_h, img_w),
        dtype=np.uint8,
    )


def build_lf2_obs_space(
    *,
    mode: Mode,
    num_players: int,
    mp_max: int,
    hp_max: int,
    channels: int,
    img_h: int,
    img_w: int,
) -> spaces.Space:
    """Compose the observation space according to ``mode``."""
    info = build_lf2_info_space(num_players, mp_max, hp_max)
    image = build_lf2_image_space(channels, img_h, img_w)

    if mode == "mix":
        return spaces.Dict({"Info": info, "Game_Screen": image})
    if mode == "info":
        return info
    if mode == "picture":
        return image
    raise ValueError(f"Unsupported mode: {mode!r}")


def build_lf2_act_space(num_actions: int) -> spaces.Discrete:
    """Discrete action space sized by the character's move set."""
    return spaces.Discrete(num_actions)
