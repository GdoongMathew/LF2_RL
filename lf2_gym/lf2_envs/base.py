"""Cross-platform LF2 env logic — pure-Python core.

Holds the RL interface (gymnasium / PettingZoo) plus step / reset /
observation / reward flow, all expressed against a *duck-typed controller*
that owns OS-specific game I/O. Concrete subclasses
(:class:`~lf2_gym.windows.env.Lf2Env`,
:class:`~lf2_gym.windows.parallel_env.Lf2ParallelEnv`) build the Windows
controller and inject it here.

Importing this module is safe on Linux: no ``win32`` / ``pymem`` /
``pyautogui`` / ``cv2`` / ``mss`` references anywhere in its load graph.
"""

from __future__ import annotations

import functools
import time
from typing import Any, Literal, Protocol

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from lf2_gym.characters import LogicBtn, Move
from lf2_gym.keymap import KeyMap, PlayerProtocol, resolve_move_key
from lf2_gym.loggers import get_logger
from lf2_gym.rewards import RewardState, compute_reward
from lf2_gym.spec import (
    Mode,
    build_lf2_act_space,
    build_lf2_obs_space,
    normalize_info,
)

logger = get_logger(__name__)


class ControllerProtocol(Protocol):
    """Duck-typed interface required from a LF2 game controller.

    The real implementation lives on the Windows side
    (:class:`lf2_gym.windows.controller.Lf2GameController`); tests provide
    a ``FakeController`` that satisfies the same interface.
    """

    img_h: int
    img_w: int
    gray_scale: bool
    gaming_screen: Any
    game_over: bool
    players: list  # list[PlayerProtocol | None] of length 8

    @property
    def channels(self) -> int: ...

    @property
    def num_players(self) -> int: ...

    @property
    def active_players(self) -> tuple[PlayerProtocol, ...]: ...

    def get_image_obs(self) -> np.ndarray: ...
    def get_players_state(self, my_player: PlayerProtocol) -> np.ndarray: ...
    def press_player_action(self, player: PlayerProtocol, action_id: int) -> None: ...
    def reset_round(self, default_ok: list[str]) -> None: ...
    def close(self) -> None: ...


def _default_ok_for(player: PlayerProtocol) -> list[str]:
    """Default "confirm" key sequence used by ``reset()``."""
    return resolve_move_key(
        Move(name="attack", sequence=(LogicBtn.Attack,)),
        KeyMap.for_player(player.index),
        facing=player.facing,
    )


# ============================================================================
# Single-agent base
# ============================================================================
class Lf2EnvBase(gym.Env):
    """Cross-platform single-agent ``gym.Env`` around an injected controller."""

    metadata = {"render.modes": ["console", "rgb_array"]}

    def __init__(
        self,
        *,
        controller: ControllerProtocol,
        player_id: int = 1,
        reset_skip_sec: int = 2,
        mode: Mode = "mix",
        basic_action: bool = False,
    ):
        super().__init__()

        self.controller = controller
        self.basic_action = basic_action
        self.my_player_id = player_id
        self.my_player: PlayerProtocol = controller.players[player_id]

        self.reset_skip_sec = reset_skip_sec
        self.mode = mode
        self.reward = 0.0
        self.reward_state = RewardState()

        self.action_space = build_lf2_act_space(len(self.my_player.moves))
        self.observation_space = build_lf2_obs_space(
            mode=mode,
            num_players=self.num_players,
            mp_max=self.my_player.mp_max,
            hp_max=self.my_player.hp_max,
            channels=controller.channels,
            img_h=controller.img_h,
            img_w=controller.img_w,
        )
        logger.info("Lf2EnvBase initialized.")

    # ------------------------------------------------------- controller proxies
    @property
    def num_players(self) -> int:
        return self.controller.num_players

    @property
    def active_players(self) -> tuple[PlayerProtocol, ...]:
        return self.controller.active_players

    @property
    def img_h(self) -> int:
        return self.controller.img_h

    @property
    def img_w(self) -> int:
        return self.controller.img_w

    @property
    def gray_scale(self) -> bool:
        return self.controller.gray_scale

    @property
    def gaming_screen(self):
        return self.controller.gaming_screen

    @property
    def game_over(self) -> bool:
        return self.controller.game_over

    @game_over.setter
    def game_over(self, value: bool) -> None:
        self.controller.game_over = value

    # ------------------------------------------------------------ observations
    def get_state(self):
        if self.mode == "picture":
            return self.controller.get_image_obs()
        if self.mode == "mix":
            return {
                "Game_Screen": self.controller.get_image_obs(),
                "Info": self.get_players_state(),
            }
        return self.get_players_state()

    def get_players_state(self) -> np.ndarray:
        # Normalize the controller's (num_players, 6) int16 buffer to a
        # 1-D float32 vector in roughly [0, 1] — matches the space built
        # by ``lf2_gym.spec.build_lf2_info_space`` and gives the MLP
        # encoder a network-friendly signal (raw values span [0, 1500]+
        # across columns, which would otherwise saturate the first layer).
        return normalize_info(self.controller.get_players_state(self.my_player))

    # ------------------------------------------------------------------- API
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        super().reset(seed=seed)
        default_ok = options.get("default_ok", None) if isinstance(options, dict) else None
        if default_ok is None:
            default_ok = _default_ok_for(self.my_player)
        time.sleep(self.reset_skip_sec)
        self.controller.reset_round(default_ok)

        self.reward = 0.0
        self.reward_state = RewardState()
        logger.info("Environment reset.")
        return self.get_state(), self.get_info()

    def step(self, action_id):
        self.controller.press_player_action(self.my_player, action_id)

        ob = self.get_state()
        reward = self.get_reward()
        info = self.get_info()
        if self.game_over:
            logger.info("Game Over!")
        return ob, reward, self.game_over, False, info

    def render(self, mode="human"):
        """Render — ``human`` mode is concrete-subclass-specific (needs cv2)."""
        if mode == "console":
            print(f"My player HP: {self.my_player.hp}, Reward: {self.get_reward()}")
        elif mode == "rgb_array":
            return self.gaming_screen
        else:
            super().render(mode=mode)

    def get_reward(self) -> float:
        self.reward, self.reward_state = compute_reward(
            self.my_player,
            self.active_players,
            self.reward_state,
        )
        return self.reward

    def get_info(self) -> dict[str, Any]:
        return {"GameOver": self.game_over}

    def close(self) -> None:
        self.controller.close()

# ============================================================================
# Multi-agent base
# ============================================================================
def _agent_id(player_id: int) -> str:
    return f"player_{player_id}"


class Lf2ParallelEnvBase(ParallelEnv):
    """Cross-platform PettingZoo ``ParallelEnv`` around an injected controller."""

    metadata = {"render_modes": ["human", "rgb_array"], "name": "littlefighter2_parallel_v0"}

    def __init__(
        self,
        *,
        controller: ControllerProtocol,
        player_ids: tuple[int, ...] | list[int] | None = None,
        reset_skip_sec: int = 2,
        mode: Mode = "mix",
    ):
        super().__init__()
        if mode not in ("mix", "picture", "info"):
            raise ValueError(f"Unsupported mode: {mode}")

        self.mode: Mode = mode
        self.reset_skip_sec = reset_skip_sec
        self.controller = controller

        if player_ids is None:
            player_ids = tuple(range(4))
        self._player_ids: tuple[int, ...] = tuple(player_ids)

        self._players: dict[str, PlayerProtocol] = {}
        for pid in self._player_ids:
            player = controller.players[pid]
            if player is None or not player.is_active:
                raise ValueError(f"Player slot {pid} is not an active human player.")
            self._players[_agent_id(pid)] = player

        self.possible_agents: list[str] = [_agent_id(pid) for pid in self._player_ids]
        self.agents: list[str] = list(self.possible_agents)

        # Per-agent reward-shaping accumulators (HP / attack-counter / alive
        # snapshot from the previous step). Reset to fresh ``RewardState()``
        # in ``reset()``.
        self._reward_state: dict[str, RewardState] = {a: RewardState() for a in self.possible_agents}

        # All in-window agents share a single policy, so RLlib's
        # ``agent_to_module_mapping`` requires identical obs/action spaces
        # across agents AND between the driver-declared space and the
        # actual env's space. Two consequences:
        #
        # 1. Use one shared ``Box`` / ``Discrete`` instance for every
        #    agent (avoid per-agent ``mp_max`` / ``hp_max`` reads from
        #    live game memory, which would vary by character).
        # 2. Use the **same constant** bounds the driver-side
        #    ``lf2_rl.rllib.spec.build_spaces_from_config`` defaults to —
        #    500 / 500. If you train across mixed characters with HP > 500
        #    pass matching ``--hp-max`` / ``--mp-max`` to train.py *and*
        #    plumb them through ``env_config``.
        shared_mp_max = 500
        shared_hp_max = 500
        shared_num_moves = max(len(p.moves) for p in self._players.values())

        shared_obs_space = build_lf2_obs_space(
            mode=mode,
            num_players=controller.num_players,
            mp_max=shared_mp_max,
            hp_max=shared_hp_max,
            channels=controller.channels,
            img_h=controller.img_h,
            img_w=controller.img_w,
        )
        shared_act_space = build_lf2_act_space(shared_num_moves)

        self._obs_spaces = {a: shared_obs_space for a in self.possible_agents}
        self._act_spaces = {a: shared_act_space for a in self.possible_agents}

    # ----------------------------------------------------------------- spaces
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
        info = normalize_info(self.controller.get_players_state(player))
        if self.mode == "info":
            return info
        return {
            "Game_Screen": self.controller.get_image_obs(),
            "Info": info,
        }

    def _all_obs(self) -> dict[str, Any]:
        return {a: self._agent_obs(a) for a in self.agents}

    # ------------------------------------------------------------------- API
    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, dict]]:
        self.agents = list(self.possible_agents)
        self._reward_state = {a: RewardState() for a in self.possible_agents}

        first_agent = self.possible_agents[0]
        first_player = self._players[first_agent]
        default_ok = None
        if isinstance(options, dict):
            default_ok = options.get("default_ok")
        if default_ok is None:
            default_ok = _default_ok_for(first_player)
        time.sleep(self.reset_skip_sec)
        self.controller.reset_round(default_ok)

        observations = self._all_obs()
        infos = {a: {} for a in self.agents}
        return observations, infos

    def step(
        self, actions: dict[str, int]
    ) -> tuple[
        dict[str, Any],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict],
    ]:
        for agent, action in actions.items():
            self.controller.press_player_action(self._players[agent], action)

        observations = self._all_obs()

        rewards: dict[str, float] = {}
        for agent in self.agents:
            reward, self._reward_state[agent] = compute_reward(
                self._players[agent],
                self.controller.active_players,
                self._reward_state[agent],
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
