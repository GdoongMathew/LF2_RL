from typing import Literal, cast, Any

from lf2_gym.characters import Move, LogicBtn
from lf2_gym.lf2_envs.winguiauto import winguiauto as winauto
from lf2_gym.lf2_envs.utils import Player, press_key, resolve_move_key, KeyMap
from mss import MSS
from win32api import GetSystemMetrics
import numpy as np
from collections import deque
import win32gui
import win32con
import win32ui
import pyautogui
import time
import cv2
import threading

from gymnasium import spaces
import gymnasium as gym

pyautogui.FAILSAFE = False


def split_one(num_interval=1):
    num_list = np.sinh(list(i + 1 for i in range(num_interval)), dtype=float)
    return num_list / np.sum(num_list)


class Lf2Env(gym.Env):
    """
    Crop a image from the gaming window, and return all players info as well as
    the current image shown on the display.
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
        """
        super(Lf2Env, self).__init__()

        self.window_name = windows_name
        self.game_hwnd = winauto.findTopWindow(wantedText=windows_name)
        self.PyCWnd1 = win32ui.FindWindow(None, windows_name)
        self.PyCWnd1.SetForegroundWindow()
        self.PyCWnd1.SetFocus()
        self.kill_thread = False
        self.basic_action = basic_action
        self.sct = MSS()

        self.players: list[None | Player] = [None] * 8
        self.find_players()

        self.my_player_id = player_id
        self.my_player: Player = cast(Player, self.players[player_id])

        self.gaming_screen = None
        self.game_over = False
        self.restart = True

        self.img_h = 0
        self.img_w = 0
        self.frame_skip = frame_skip
        self.downscale = downscale
        self.gray_scale = gray_scale

        self.frame_stack = frame_stack
        self.img_weights = split_one(self.frame_stack)
        self.frames = deque([], maxlen=self.frame_stack)
        # Immortal seconds before every rounds.
        self.reset_skip_sec = reset_skip_sec

        self.recording_thread = threading.Thread(target=self.update_game_img, daemon=True)
        self.recording_thread.start()
        self.player_thread = threading.Thread(target=self.update_players, daemon=True)
        self.player_thread.start()

        self.action_space = spaces.Discrete(len(self.my_player.moves))
        self.mode = mode
        self.reward = 0
        self.bot_attack = 0
        while True:
            channels = 1 if self.gray_scale else 3
            if len(self.frames) != 0:
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
                break
        print("Lf2 Environment initialized.")

    @property
    def num_players(self) -> int:
        return len(self.active_players)

    @property
    def active_players(self) -> tuple[Player, ...]:
        return tuple(p for p in self.players if p is not None and p.is_active)

    def find_players(self):
        for index, player in enumerate(self.players):
            computer, human = Player(
                game_hwnd=self.game_hwnd,
                index=index,
                is_computer=True,
            ), Player(
                game_hwnd=self.game_hwnd,
                index=index,
                is_computer=False,
            )
            if computer.is_active:
                self.players[index] = computer
            elif human.is_active:
                self.players[index] = human
            else:
                self.players[index] = None

    def get_state(self):
        # return the current state of the game
        if self.mode in ["picture", "mix"]:
            while len(self.frames) < self.frame_stack:
                time.sleep(0.001)

        if self.mode == "picture":
            img_stack = np.stack(self.frames, axis=-1)
            img_stack = np.multiply(img_stack, split_one(self.frame_stack))
            ob = np.sum(img_stack, axis=-1)
            if not self.gray_scale:
                ob = np.transpose(ob, (2, 0, 1))
            else:
                ob = ob[None, ...]

        elif self.mode == "mix":
            # my_mp, my_hp, my_facing, my_x, my_y, my_z, [enemy_x, enemy_y, enemy_z]
            # img_stack = np.stack(self.frames, axis=-1)
            _imgs = np.stack(self.frames)
            img_stack = np.tensordot(self.img_weights, _imgs, axes=([0], [0]))
            if not self.gray_scale:
                img_stack = np.transpose(img_stack, (2, 0, 1))
            else:
                img_stack = img_stack[None, ...]
            ob = dict(Game_Screen=img_stack.astype(np.uint8), Info=self.get_players_state())

        else:
            # info mode
            ob = self.get_players_state()

        return ob

    def get_players_state(self) -> np.ndarray:
        def _player_state(player: Player) -> list[int]:
            return [
                player.mp,
                player.hp,
                int(bool.from_bytes(player._facing_bytes)),
                player.x_pos,
                player.y_pos,
                player.z_pos,
            ]

        return np.asarray(
            [
                _player_state(self.my_player),
                *[_player_state(p) for p in self.active_players if p is not self.my_player],
            ],
            dtype=np.int16,
        )

    def update_game_img(self):
        """
        Update the current gaming scene
        """
        skip_i = 0
        last_frame = np.array([0])
        while not self.kill_thread:
            tup = win32gui.GetWindowPlacement(self.game_hwnd)

            # check if the windows is in max size.
            if tup[1] == win32con.SW_SHOWMAXIMIZED:
                w = GetSystemMetrics(0)
                h = GetSystemMetrics(1)
                rect = [0, 0, w, h]
            elif tup[1] == win32con.SW_SHOWNORMAL:
                rect = list(win32gui.GetWindowRect(self.game_hwnd))
            else:
                continue
            h = rect[3] - rect[1]
            pos = {
                "top": int(rect[1] + h * 0.266),
                "left": int(rect[0] + 1),
                "height": int(((rect[3] - rect[1]) * 2 / 3) - 2),
                "width": int(rect[2] - rect[0]),
            }

            # img color in BGR order
            screen_shot = np.array(self.sct.grab(pos), np.uint8)[:, :, :3]

            if np.array_equal(last_frame, screen_shot):
                # refresh until new frame exists.
                continue
            self.gaming_screen = screen_shot

            if not self.img_h:
                shape = np.array(np.shape(self.gaming_screen)[:2]) / self.downscale
                self.img_h = int(shape[0])  # 500
                self.img_w = int(shape[1])  # 996
                print("img dimension: H {} W {}".format(self.img_h, self.img_w))

            frame = cv2.resize(self.gaming_screen, (self.img_w, self.img_h))
            if self.gray_scale:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # skip this frame
            if skip_i >= self.frame_skip:
                self.frames.append(frame)
                skip_i = 0
            else:
                skip_i += 1
            last_frame = screen_shot.copy()
            time.sleep(0.01)

    def update_players(self):
        """
        Return player status
        """
        while not self.kill_thread:
            time.sleep(0.01)
            team = []
            for player in filter(lambda p: p is not None, self.players):
                player.update()
                if player.is_active and player.is_alive:
                    team.append(player.team)
            self.restart = False
            self.game_over = len(team) > 0 and len(set(team)) == 1

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
        press_key(["f4", *default_ok], interval=1.0)
        # Todo figure out how to send keyboard event to a non-active windows.
        # chile_hwnd = win32gui.GetWindow(self.game_hwnd, win32con.GW_CHILD)
        # PostMessage(chile_hwnd, win32con.WM_KEYDOWN, win32con.VK_F4, 0)
        # PostMessage(chile_hwnd, win32con.WM_KEYUP, win32con.VK_F4, 0)
        # PostMessage(chile_hwnd, win32con.WM_CHAR, default_ok, 0)

        self.restart = True
        self.game_over = False
        self.frames.clear()
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
        press_key(self.my_player.action_keys(action_id))

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
        enemy_hp = []
        team_hp = []

        for player in self.active_players:
            hp_norm = player.hp * 10 / player.hp_max
            if player.team == self.my_player.team:
                team_hp.append(hp_norm)
            else:
                enemy_hp.append(hp_norm)

        team_avg = sum(team_hp) / len(team_hp) if team_hp else 0.0
        enemy_avg = sum(enemy_hp) / len(enemy_hp) if enemy_hp else 0.0
        self.reward = team_avg - enemy_avg
        mp_reward = (self.my_player.mp_max - self.my_player.mp) / self.my_player.mp_max
        self.reward += mp_reward

        # death penalty
        if not self.my_player.is_alive:
            self.reward -= 50

        # increase reward when increasing attacks.
        if self.my_player.attacks != self.bot_attack:
            self.reward += (self.my_player.attacks - self.bot_attack) / 10
            self.bot_attack = self.my_player.attacks

        # Most simple reward?
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
        cv2.destroyAllWindows()
        self.kill_thread = True

    def seed(self, seed=None):
        pass


if __name__ == "__main__":

    hwnd = winauto.findTopWindow(wantedText="Little Fighter 2")

    my_player = Player(game_hwnd=hwnd, index=0, is_computer=False)
    my_player_1 = Player(game_hwnd=hwnd, index=1, is_computer=True)

    print(my_player.character)
    print(my_player_1.character)

    now = time.time()
    # att_1 = ply1.sp_attact6()
    while 1:

        my_player.update()
        my_player_1.update()

        print(my_player.hp)
        print(my_player_1.hp)
        # print(com_player.Hp)
        time.sleep(1)

        if time.time() - now >= 12000:
            break
