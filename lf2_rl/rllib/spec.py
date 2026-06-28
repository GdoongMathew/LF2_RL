"""Pure RLlib-side helpers — env name, policy mapping, space construction.

The learner (on Linux / DGX) imports this module only; it never touches the
Windows game I/O. Worker registration of the actual env creator lives in
:mod:`lf2_rl.rllib.worker_register`, which is loaded on rollout workers via
the Ray runtime-env ``worker_process_setup_hook``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import spaces
from pettingzoo.utils.wrappers import BaseParallelWrapper

from lf2_gym.characters import CHARACTER_MOVES, Characters
from lf2_gym.spec import (
    Mode,
    build_lf2_act_space,
    build_lf2_obs_space,
)

LF2_PARALLEL_ENV = "lf2_parallel"
SHARED_POLICY_ID = "shared_policy"

# Standard LF2 window size on which the controller crops the gaming area.
# Override via ``env_config["native_window_height" / "native_window_width"]``
# if the deployment uses a non-default LF2 setup.
LF2_NATIVE_WINDOW_HEIGHT = 600
LF2_NATIVE_WINDOW_WIDTH = 800


def policy_mapping_fn(agent_id, *args, **kwargs) -> str:
    """All in-window players share a single policy."""
    return SHARED_POLICY_ID


# ---------------------------------------------------------------- spaces
def estimate_img_hw(
    downscale: int,
    native_h: int = LF2_NATIVE_WINDOW_HEIGHT,
    native_w: int = LF2_NATIVE_WINDOW_WIDTH,
) -> tuple[int, int]:
    """Approximate the ``(H, W)`` the controller's image pipeline will emit.

    Mirrors the crop math in
    :meth:`lf2_gym.windows.controller.Lf2GameController.update_game_img` —
    gaming area = ``(width, height * 2/3 - 2)`` of the LF2 window — then
    divides by ``downscale``. Used only to declare obs shape on the
    learner; the worker still computes the *real* dims from the live
    window and the two should agree to the int floor.
    """
    gaming_h = int(native_h * 2 / 3) - 2
    gaming_w = native_w
    return gaming_h // downscale, gaming_w // downscale


def _resolve_num_actions(env_config: dict[str, Any]) -> int:
    """How many discrete actions the shared policy emits.

    Priority: explicit ``env_config["num_actions"]`` > moves-of(``character``)
    > ``CHARACTER_MOVES[Characters.Template]`` (the 9 basic moves).
    """
    if "num_actions" in env_config:
        return int(env_config["num_actions"])
    char_name = env_config.get("character", Characters.Template)
    char = char_name if isinstance(char_name, Characters) else Characters(char_name)
    return len(CHARACTER_MOVES[char])


def build_spaces_from_config(
    env_config: dict[str, Any],
    *,
    channels_last: bool = True,
) -> tuple[spaces.Space, spaces.Space]:
    """Declare ``(obs_space, act_space)`` for the shared policy purely from config.

    The driver hands these directly to RLlib's ``.environment()`` and
    ``.multi_agent(policies=...)`` so that no env is ever constructed on
    the learner side just to probe spaces. Workers on Windows still build
    real envs through the registered creator.
    """
    mode: Mode = env_config.get("mode", "mix")
    player_ids = env_config.get("player_ids") or (0, 1, 2, 3)
    num_players = len(tuple(player_ids))

    downscale = int(env_config.get("downscale", 2))
    img_h = int(env_config.get("img_h") or 0)
    img_w = int(env_config.get("img_w") or 0)
    if not img_h or not img_w:
        img_h, img_w = estimate_img_hw(
            downscale,
            native_h=int(env_config.get("native_window_height", LF2_NATIVE_WINDOW_HEIGHT)),
            native_w=int(env_config.get("native_window_width", LF2_NATIVE_WINDOW_WIDTH)),
        )

    gray_scale = bool(env_config.get("gray_scale", True))
    channels = 1 if gray_scale else 3
    mp_max = int(env_config.get("mp_max", 500))
    hp_max = int(env_config.get("hp_max", 500))

    obs_space = build_lf2_obs_space(
        mode=mode,
        num_players=num_players,
        mp_max=mp_max,
        hp_max=hp_max,
        channels=channels,
        img_h=img_h,
        img_w=img_w,
    )
    if channels_last:
        obs_space = _to_channels_last_space(obs_space)

    act_space = build_lf2_act_space(_resolve_num_actions(env_config))
    return obs_space, act_space


def _to_channels_last_space(space: spaces.Space) -> spaces.Space:
    """``(C, H, W)`` -> ``(H, W, C)`` for the image leaf of the obs space."""
    if isinstance(space, spaces.Dict) and "Game_Screen" in space.spaces:
        img = space.spaces["Game_Screen"]
        c, h, w = img.shape
        new_img = spaces.Box(low=0, high=255, shape=(h, w, c), dtype=img.dtype)
        return spaces.Dict({**space.spaces, "Game_Screen": new_img})
    if isinstance(space, spaces.Box) and space.dtype == np.uint8 and len(space.shape) == 3:
        c, h, w = space.shape
        return spaces.Box(low=0, high=255, shape=(h, w, c), dtype=space.dtype)
    return space


# --------------------------------------------------------------- wrapper
class ChannelsLastWrapper(BaseParallelWrapper):
    """Transpose ``Game_Screen`` obs from ``(C, H, W)`` to ``(H, W, C)``.

    Pure PettingZoo wrapper — no Windows deps in its load graph, so the
    learner can also import and use it directly if it ever wants to handle
    obs locally.
    """

    def __init__(self, env):
        super().__init__(env)
        self._space_cache: dict[str, Any] = {}

    @staticmethod
    def _to_hwc(obs):
        if isinstance(obs, dict) and "Game_Screen" in obs:
            obs = dict(obs)
            obs["Game_Screen"] = np.transpose(obs["Game_Screen"], (1, 2, 0))
        return obs

    def observation_space(self, agent):
        if agent not in self._space_cache:
            self._space_cache[agent] = _to_channels_last_space(self.env.observation_space(agent))
        return self._space_cache[agent]

    def reset(self, seed=None, options=None):
        obs, infos = self.env.reset(seed=seed, options=options)
        return {a: self._to_hwc(o) for a, o in obs.items()}, infos

    def step(self, actions):
        obs, rew, term, trunc, infos = self.env.step(actions)
        return {a: self._to_hwc(o) for a, o in obs.items()}, rew, term, trunc, infos
