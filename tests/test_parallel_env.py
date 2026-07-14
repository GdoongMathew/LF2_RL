"""Logic tests for the multi-agent LF2 env using a fake controller.

These run anywhere (no live game needed, no Windows libs required): a stub
controller / players replace all game I/O, so we exercise the PettingZoo
wiring, spaces, reward mapping and episode-termination behaviour
deterministically against the pure :class:`Lf2ParallelEnvBase`.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pettingzoo")

# Import the pure base class (no win32 / pymem / mss / cv2 in its load
# graph) so this test runs on Linux CI too. The concrete Windows wrapper
# ``lf2_gym.windows.parallel_env.Lf2ParallelEnv`` only adds controller
# construction on top.
from lf2_gym.lf2_envs.base import Lf2ParallelEnvBase as Lf2ParallelEnv  # noqa: E402


class FakePlayer:
    def __init__(self, index, team, *, hp=100, mp=100, attacks=0):
        self.index = index
        self.team = team
        self.hp = hp
        self.hp_max = 100
        self.mp = mp
        self.mp_max = 500
        self.attacks = attacks
        self.is_active = True
        self.facing = "right"
        self._facing_bytes = b"\x00"
        # a tiny, fixed move set
        self.moves = [object(), object(), object()]

    @property
    def is_alive(self):
        return self.hp > 0

    def action_keys(self, action_index):
        return ["A"]


class FakeController:
    """Duck-typed stand-in for Lf2GameController (no real game I/O)."""

    def __init__(self, players, img_h=8, img_w=12, gray_scale=True):
        self.players = list(players) + [None] * (8 - len(players))
        self.img_h = img_h
        self.img_w = img_w
        self.gray_scale = gray_scale
        self.game_over = False
        self.gaming_screen = np.zeros((img_h, img_w, 3), np.uint8)
        self.pressed = []
        self.reset_calls = 0

    @property
    def channels(self):
        return 1 if self.gray_scale else 3

    @property
    def active_players(self):
        return tuple(p for p in self.players if p is not None and p.is_active)

    @property
    def num_players(self):
        return len(self.active_players)

    def get_image_obs(self):
        return np.zeros((self.channels, self.img_h, self.img_w), np.uint8)

    @staticmethod
    def player_state(player):
        return [player.mp, player.hp, 0, 0, 0, 0]

    def get_players_state(self, my_player):
        return np.asarray(
            [self.player_state(my_player)]
            + [self.player_state(p) for p in self.active_players if p is not my_player],
            dtype=np.int16,
        )

    def press_player_action(self, player, action_id):
        self.pressed.append((player.index, action_id))

    def reset_round(self, default_ok):
        self.reset_calls += 1
        self.game_over = False


@pytest.fixture
def controller():
    players = [
        FakePlayer(0, team=0),
        FakePlayer(1, team=1),
        FakePlayer(2, team=0),
        FakePlayer(3, team=1),
    ]
    return FakeController(players)


@pytest.fixture
def env(controller):
    return Lf2ParallelEnv(player_ids=(0, 1, 2, 3), mode="mix", reset_skip_sec=0, controller=controller)


def test_agents_and_spaces(env):
    assert env.possible_agents == ["player_0", "player_1", "player_2", "player_3"]
    for agent in env.possible_agents:
        assert env.action_space(agent).n == 3
        space = env.observation_space(agent)
        assert "Game_Screen" in space.spaces and "Info" in space.spaces


def test_reset_returns_per_agent_obs(env, controller):
    obs, infos = env.reset(options={"default_ok": ["A"]})
    assert set(obs) == set(env.possible_agents)
    assert controller.reset_calls == 1
    for agent in env.possible_agents:
        assert obs[agent]["Game_Screen"].shape == (1, 8, 12)
        # info: self first then 3 others, flattened + normalized to
        # (num_players * 6,) float32 — see ``lf2_gym.spec.normalize_info``.
        assert obs[agent]["Info"].shape == (4 * 6,)
        assert obs[agent]["Info"].dtype == np.float32
    assert set(infos) == set(env.possible_agents)


def test_step_presses_keys_for_all_agents(env, controller):
    env.reset(options={"default_ok": ["A"]})
    actions = {a: i % 3 for i, a in enumerate(env.agents)}
    obs, rewards, terms, truncs, infos = env.step(actions)
    pressed_indices = {p[0] for p in controller.pressed}
    assert pressed_indices == {0, 1, 2, 3}
    assert set(rewards) == set(env.possible_agents)
    assert all(t is False for t in truncs.values())


def test_game_over_clears_agents(env, controller):
    env.reset(options={"default_ok": ["A"]})
    controller.game_over = True
    actions = {a: 0 for a in env.agents}
    _, _, terms, _, _ = env.step(actions)
    assert all(terms.values())
    assert env.agents == []


def test_info_mode_obs_is_array(controller):
    env = Lf2ParallelEnv(player_ids=(0, 1), mode="info", reset_skip_sec=0, controller=controller)
    obs, _ = env.reset(options={"default_ok": ["A"]})
    assert isinstance(obs["player_0"], np.ndarray)
    # 4 active players in the fake controller, 6 fields each, flattened
    # and normalized to float32 by ``lf2_gym.spec.normalize_info``.
    assert obs["player_0"].shape == (4 * 6,)
    assert obs["player_0"].dtype == np.float32


def test_inactive_slot_raises(controller):
    controller.players[2].is_active = False
    with pytest.raises(ValueError):
        Lf2ParallelEnv(player_ids=(2,), reset_skip_sec=0, controller=controller)
