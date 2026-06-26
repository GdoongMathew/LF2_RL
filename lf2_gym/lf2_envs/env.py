import time
from typing import Any, Literal, cast

import cv2
import gymnasium as gym
import numpy as np
import pyautogui
from gymnasium import spaces

from lf2_gym.characters import LogicBtn, Move
from lf2_gym.lf2_envs.controller import Lf2GameController, split_one
from lf2_gym.lf2_envs.utils import KeyMap, Player, resolve_move_key
from lf2_gym.lf2_envs.winguiauto import winguiauto as winauto

pyautogui.FAILSAFE = False

__all__ = ["Lf2Env", "compute_reward", "split_one"]


def compute_reward(
    my_player: Player,
    active_players: tuple[Player, ...],
    prev_attacks: int,
) -> tuple[float, int]:
    """Per-player reward shaping shared by single- and multi-agent envs.

    Returns ``(reward, updated_attack_count)``.
    """
    enemy_hp = []
    team_hp = []

    for player in active_players:
        hp_norm = player.hp * 10 / player.hp_max
        if player.team == my_player.team:
            team_hp.append(hp_norm)
        else:
            enemy_hp.append(hp_norm)

    team_avg = sum(team_hp) / len(team_hp) if team_hp else 0.0
    enemy_avg = sum(enemy_hp) / len(enemy_hp) if enemy_hp else 0.0
    reward = team_avg - enemy_avg
    mp_reward = (my_player.mp_max - my_player.mp) / my_player.mp_max
    reward += mp_reward

    # death penalty
    if not my_player.is_alive:
        reward -= 50

    # increase reward when increasing attacks.
    if my_player.attacks != prev_attacks:
        reward += (my_player.attacks - prev_attacks) / 10
        prev_attacks = my_player.attacks

    return reward, prev_attacks


class Lf2Env(gym.Env):
    """
    Crop a image from the gaming window, and return all players info as well as
    the current image shown on the display.

    Game I/O (window capture, memory reads, key sending) is delegated to a
    shared :class:`~lf2_gym.lf2_envs.controller.Lf2GameController` so the same
    machinery can back the multi-agent environment as well.
    """

    metadata = {"render.modes": ["human", "console", "rgb_array"]}

    def __init__(
        self,
        windows_name="Little Fighter 2",
        player_id=1,
        downscale=2,
        frame_stack=4,
        frame_skip=1,
        reset_skip_sec=2,
        gray_scale=True,
        mode: Literal["info", "picture", "mix"] = "mix",
        basic_action=False,
        controller: Lf2GameController | None = None,
    ):
        """
        Initialize the gym environment
        :param windows_name: name of the windows
        :param player_id: training player id
        :param downscale: downscale ratio
        :param frame_stack: number of frames should be stacked
        :param frame_skip: number of frames to skip before stacking.
        :param reset_skip_sec: immortal time (sec)
        :param gray_scale: convert recording image from rgb to gray scale
        :param mode: observation output mode.
        :param basic_action: only output L,R,U,D,A,D,J
        :param controller: optional shared game controller (created if omitted).
        """
        super(Lf2Env, self).__init__()

        self.controller = controller or Lf2GameController(
            windows_name=windows_name,
            downscale=downscale,
            frame_stack=frame_stack,
            frame_skip=frame_skip,
            gray_scale=gray_scale,
        )
        self.basic_action = basic_action

        self.my_player_id = player_id
        self.my_player: Player = cast(Player, self.controller.players[player_id])

        self.reset_skip_sec = reset_skip_sec

        self.action_space = spaces.Discrete(len(self.my_player.moves))
        self.mode = mode
        self.reward = 0
        self.bot_attack = 0

        channels = self.controller.channels
        # my_mp, my_hp, my_facing, my_x, my_y, my_z, [enemy_x, enemy_y, enemy_z]
        low = [[0, 0, 0, 0, 0, -np.inf]] * self.num_players
        high = [[self.my_player.mp_max, self.my_player.hp_max, 1, np.inf, np.inf, 0]] * self.num_players
        info = spaces.Box(low=np.array(low), high=np.array(high), dtype=np.int16)
        image = spaces.Box(
            low=0,
            high=255,
            shape=(channels, self.img_h, self.img_w),
            dtype=np.uint8,
        )

        if self.mode == "mix":
            self.observation_space = spaces.Dict({"Info": info, "Game_Screen": image})
        elif self.mode == "info":
            self.observation_space = info
        elif self.mode == "picture":
            self.observation_space = image
        else:
            raise ValueError("Not Supported mode.... Exiting.")
        print("Lf2 Environment initialized.")

    # ------------------------------------------------------- controller proxies
    @property
    def num_players(self) -> int:
        return self.controller.num_players

    @property
    def active_players(self) -> tuple[Player, ...]:
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

    def get_state(self):
        # return the current state of the game
        if self.mode == "picture":
            ob = self.controller.get_image_obs()
        elif self.mode == "mix":
            ob = dict(Game_Screen=self.controller.get_image_obs(), Info=self.get_players_state())
        else:
            # info mode
            ob = self.get_players_state()

        return ob

    def get_players_state(self) -> np.ndarray:
        return self.controller.get_players_state(self.my_player)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        """
        Restart the game
        Currently still need the game to be on the active window.
        """
        super().reset(seed=seed)
        default_ok = options.get("default_ok", None) if isinstance(options, dict) else None
        if default_ok is None:
            default_ok = resolve_move_key(
                Move(name="attack", sequence=(LogicBtn.Attack,)),
                KeyMap.for_player(self.my_player_id),
                facing=self.my_player.facing,
            )
        time.sleep(self.reset_skip_sec)
        self.controller.reset_round(default_ok)

        self.reward = 0
        self.bot_attack = 0
        print("Env reset.")
        return self.get_state(), self.get_info()

    def step(self, action_id):
        """
        Take an action within the environment
        :param action_id: an action id from the action space
        :return: observation, reward, done, info
        """
        self.controller.press_player_action(self.my_player, action_id)

        ob = self.get_state()
        reward = self.get_reward()

        # currently nothing will return in info
        info = self.get_info()

        return ob, reward, self.game_over, False, info

    def render(self, mode="human"):
        if mode == "console":
            reward = self.get_reward()
            print(f"My player HP: {self.my_player.hp}, Reward: {reward}")
        elif mode == "human":
            if self.gaming_screen is not None:
                cv2.imshow("lf2_render", self.gaming_screen)
                cv2.waitKey(1)
        elif mode == "rgb_array":
            return self.gaming_screen
        else:
            super(Lf2Env, self).render(mode=mode)

    def get_reward(self):
        """
        Calculate the corresponding rewards of the current state.
        :return: reward
        """
        self.reward, self.bot_attack = compute_reward(
            self.my_player,
            self.active_players,
            self.bot_attack,
        )
        return self.reward

    def get_info(self):
        """
        get additional information.
        :return:
        """
        info = dict()
        info["GameOver"] = self.game_over
        # info['episode'] =
        return info

    def close(self):
        self.controller.close()

    def seed(self, seed=None):
        pass


if __name__ == "__main__":

    hwnd = winauto.findTopWindow(wantedText="Little Fighter 2")

    my_player = Player(game_hwnd=hwnd, index=0, is_computer=False)
    my_player_1 = Player(game_hwnd=hwnd, index=1, is_computer=True)

    print(my_player.character)
    print(my_player_1.character)

    now = time.time()
    while 1:

        my_player.update()
        my_player_1.update()

        print(my_player.hp)
        print(my_player_1.hp)
        time.sleep(1)

        if time.time() - now >= 12000:
            break
