"""Multi-agent PettingZoo environment for Little Fighter 2.

A single LF2 window hosts up to 4 human-controllable player slots. This
environment exposes each of those slots as an independent agent that *shares a
single, centrally-trained policy*. Every step the controller sends the chosen
keys for every agent to the (one) game window and returns a per-agent
observation / reward, giving ~4x experience throughput per window without any
extra processes (the practical replacement for ``SubprocVecEnv`` here).

The shared screen image (``Game_Screen``) is identical for all agents, while the
``Info`` array and reward are agent-centric (the acting player is listed first).
"""

from __future__ import annotations

import functools
import time
from typing import Any

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from lf2_gym.characters import LogicBtn, Move
from lf2_gym.lf2_envs.controller import Lf2GameController
from lf2_gym.lf2_envs.env import compute_reward
from lf2_gym.lf2_envs.utils import KeyMap, Player, resolve_move_key


def _agent_id(player_id: int) -> str:
    return f"player_{player_id}"


class Lf2ParallelEnv(ParallelEnv):
    """PettingZoo ``ParallelEnv`` wrapping one LF2 window with N agents."""

    metadata = {"render_modes": ["human", "rgb_array"], "name": "littlefighter2_parallel_v0"}

    def __init__(
        self,
        windows_name: str = "Little Fighter 2",
        player_ids: tuple[int, ...] | list[int] | None = None,
        downscale: int = 2,
        frame_stack: int = 4,
        frame_skip: int = 1,
        reset_skip_sec: int = 2,
        gray_scale: bool = True,
        mode: str = "mix",
        controller: Lf2GameController | None = None,
    ):
        """
        :param player_ids: which human player slots (0-3) are agent-controlled.
            Defaults to all four slots.
        :param mode: observation mode ("mix", "picture" or "info").
        :param controller: optional shared game controller (created if omitted).
        """
        super().__init__()
        if mode not in ("mix", "picture", "info"):
            raise ValueError(f"Unsupported mode: {mode}")
        self.mode = mode
        self.reset_skip_sec = reset_skip_sec

        self.controller = controller or Lf2GameController(
            windows_name=windows_name,
            downscale=downscale,
            frame_stack=frame_stack,
            frame_skip=frame_skip,
            gray_scale=gray_scale,
        )

        if player_ids is None:
            player_ids = tuple(range(4))
        self._player_ids: tuple[int, ...] = tuple(player_ids)

        self._players: dict[str, Player] = {}
        for pid in self._player_ids:
            player = self.controller.players[pid]
            if player is None or not player.is_active:
                raise ValueError(f"Player slot {pid} is not an active human player.")
            self._players[_agent_id(pid)] = player

        self.possible_agents: list[str] = [_agent_id(pid) for pid in self._player_ids]
        self.agents: list[str] = list(self.possible_agents)

        # per-agent attack counters for reward shaping
        self._bot_attack: dict[str, int] = {a: 0 for a in self.possible_agents}

        self._obs_spaces = {a: self._build_obs_space(self._players[a]) for a in self.possible_agents}
        self._act_spaces = {a: spaces.Discrete(len(self._players[a].moves)) for a in self.possible_agents}

    # --------------------------------------------------------------- spaces
    def _build_obs_space(self, player: Player) -> spaces.Space:
        num_players = self.controller.num_players
        low = [[0, 0, 0, 0, 0, -np.inf]] * num_players
        high = [[player.mp_max, player.hp_max, 1, np.inf, np.inf, 0]] * num_players
        info = spaces.Box(low=np.array(low), high=np.array(high), dtype=np.int16)
        image = spaces.Box(
            low=0,
            high=255,
            shape=(self.controller.channels, self.controller.img_h, self.controller.img_w),
            dtype=np.uint8,
        )
        if self.mode == "mix":
            return spaces.Dict({"Info": info, "Game_Screen": image})
        if self.mode == "picture":
            return image
        return info

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str) -> spaces.Space:
        return self._obs_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str) -> spaces.Space:
        return self._act_spaces[agent]

    # ----------------------------------------------------------- observations
    def _agent_obs(self, agent: str):
        player = self._players[agent]
        if self.mode == "picture":
            return self.controller.get_image_obs()
        if self.mode == "info":
            return self.controller.get_players_state(player)
        return {
            "Game_Screen": self.controller.get_image_obs(),
            "Info": self.controller.get_players_state(player),
        }

    def _all_obs(self) -> dict[str, Any]:
        return {a: self._agent_obs(a) for a in self.agents}

    # ------------------------------------------------------------------- step
    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, dict]]:
        self.agents = list(self.possible_agents)
        self._bot_attack = {a: 0 for a in self.possible_agents}

        # Reset the round using the first agent's confirm key.
        first_agent = self.possible_agents[0]
        first_player = self._players[first_agent]
        default_ok = None
        if isinstance(options, dict):
            default_ok = options.get("default_ok")
        if default_ok is None:
            default_ok = resolve_move_key(
                Move(name="attack", sequence=(LogicBtn.Attack,)),
                KeyMap.for_player(first_player.index),
                facing=first_player.facing,
            )
        time.sleep(self.reset_skip_sec)
        self.controller.reset_round(default_ok)

        observations = self._all_obs()
        infos = {a: {} for a in self.agents}
        return observations, infos

    def step(
        self, actions: dict[str, int]
    ) -> tuple[dict[str, Any], dict[str, float], dict[str, bool], dict[str, bool], dict[str, dict]]:
        # Send keys for every acting agent to the single game window.
        for agent, action in actions.items():
            self.controller.press_player_action(self._players[agent], action)

        observations = self._all_obs()

        rewards: dict[str, float] = {}
        for agent in self.agents:
            reward, self._bot_attack[agent] = compute_reward(
                self._players[agent],
                self.controller.active_players,
                self._bot_attack[agent],
            )
            rewards[agent] = reward

        game_over = self.controller.game_over
        terminations = {a: game_over for a in self.agents}
        truncations = {a: False for a in self.agents}
        infos = {a: {"GameOver": game_over} for a in self.agents}

        if game_over:
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    # ----------------------------------------------------------------- render
    def render(self):
        return self.controller.gaming_screen

    def close(self):
        self.controller.close()


def parallel_env(**kwargs) -> Lf2ParallelEnv:
    """Factory matching the PettingZoo ``parallel_env`` convention."""
    return Lf2ParallelEnv(**kwargs)
