"""Env construction + registration helpers for RLlib.

Wraps the PettingZoo :class:`Lf2ParallelEnv` so RLlib treats the 4 in-window
players as agents that all map to one shared policy. Also adapts the observation
image layout from channels-first ``(C, H, W)`` (what the game controller
produces) to channels-last ``(H, W, C)`` which RLlib's default vision network
expects.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pettingzoo.utils.wrappers import BaseParallelWrapper

from lf2_gym.lf2_envs.parallel_env import Lf2ParallelEnv

LF2_PARALLEL_ENV = "lf2_parallel"
SHARED_POLICY_ID = "shared_policy"


class ChannelsLastWrapper(BaseParallelWrapper):
    """Transpose ``Game_Screen`` obs from (C, H, W) to (H, W, C) for RLlib."""

    def __init__(self, env: Lf2ParallelEnv):
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
            from gymnasium import spaces

            space = self.env.observation_space(agent)
            if isinstance(space, spaces.Dict) and "Game_Screen" in space.spaces:
                img = space.spaces["Game_Screen"]
                c, h, w = img.shape
                new_img = spaces.Box(low=0, high=255, shape=(h, w, c), dtype=img.dtype)
                space = spaces.Dict({**space.spaces, "Game_Screen": new_img})
            self._space_cache[agent] = space
        return self._space_cache[agent]

    def reset(self, seed=None, options=None):
        obs, infos = self.env.reset(seed=seed, options=options)
        return {a: self._to_hwc(o) for a, o in obs.items()}, infos

    def step(self, actions):
        obs, rew, term, trunc, infos = self.env.step(actions)
        return {a: self._to_hwc(o) for a, o in obs.items()}, rew, term, trunc, infos


def make_lf2_parallel_env(config: dict | None = None):
    """RLlib env creator: returns a ``ParallelPettingZooEnv``-wrapped LF2 env."""
    from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv

    config = dict(config or {})
    channels_last = config.pop("channels_last", True)
    env = Lf2ParallelEnv(**config)
    if channels_last:
        env = ChannelsLastWrapper(env)
    return ParallelPettingZooEnv(env)


def policy_mapping_fn(agent_id, *args, **kwargs) -> str:
    """All in-window players share a single policy."""
    return SHARED_POLICY_ID


def register(env_name: str = LF2_PARALLEL_ENV) -> str:
    """Register the LF2 parallel env with Ray Tune's registry."""
    from ray.tune.registry import register_env

    register_env(env_name, make_lf2_parallel_env)
    return env_name
