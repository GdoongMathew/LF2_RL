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


def build_lf2_info_space(num_players: int, mp_max: int, hp_max: int) -> spaces.Box:
    """Per-player ``[mp, hp, facing, x, y, z]`` info Box, one row per player.

    Bounds use finite ``int16`` limits so the resulting Box dtype matches the
    declared dtype — using ``np.inf`` here would silently widen things and
    trip RLlib / SB3 space validators.
    """
    np_type = np.int16
    info_min = np.iinfo(np_type).min
    info_max = np.iinfo(np_type).max
    low = [[0, 0, 0, 0, 0, info_min]] * num_players
    high = [[mp_max, hp_max, 1, info_max, info_max, 0]] * num_players
    return spaces.Box(
        low=np.array(low, dtype=np_type),
        high=np.array(high, dtype=np_type),
        dtype=np_type,
    )


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
