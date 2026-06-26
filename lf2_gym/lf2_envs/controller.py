"""Game-I/O controller for Little Fighter 2.

`Lf2GameController` owns every interaction with the running game / OS:

* window handle + focus,
* screen capture (``mss``) and the frame-stacking image pipeline,
* per-player memory reads (via :class:`~lf2_gym.lf2_envs.utils.Player`),
* keystroke delivery for *any* of the (up to 4) human player slots.

It is deliberately agnostic of the RL interface so it can be shared by:

* the single-agent :class:`~lf2_gym.lf2_envs.env.Lf2Env`, and
* the multi-agent :class:`~lf2_gym.lf2_envs.parallel_env.Lf2ParallelEnv`,

where the 4 in-window players are each driven by an agent that shares one
centrally-trained policy.
"""

from __future__ import annotations

import threading
import time
from collections import deque

import cv2
import numpy as np
import win32con
import win32gui
import win32ui
from mss import MSS
from win32api import GetSystemMetrics

from lf2_gym.lf2_envs.utils import Player, press_key
from lf2_gym.lf2_envs.winguiauto import winguiauto as winauto


def split_one(num_interval: int = 1) -> np.ndarray:
    """Return normalized ``sinh`` weights used to collapse a frame stack."""
    num_list = np.sinh(list(i + 1 for i in range(num_interval)), dtype=float)
    return num_list / np.sum(num_list)


class Lf2GameController:
    """Owns the LF2 window and all shared game I/O for one game instance."""

    def __init__(
        self,
        windows_name: str = "Little Fighter 2",
        downscale: int = 2,
        frame_stack: int = 4,
        frame_skip: int = 1,
        gray_scale: bool = True,
    ):
        self.window_name = windows_name
        self.game_hwnd = winauto.findTopWindow(wantedText=windows_name)
        self.PyCWnd1 = win32ui.FindWindow(None, windows_name)
        self.PyCWnd1.SetForegroundWindow()
        self.PyCWnd1.SetFocus()

        self.kill_thread = False
        self.sct = MSS()

        self.players: list[None | Player] = [None] * 8
        self.find_players()

        self.gaming_screen = None

        self.img_h = 0
        self.img_w = 0
        self.downscale = downscale
        self.gray_scale = gray_scale
        self.frame_skip = frame_skip
        self.frame_stack = frame_stack
        self.img_weights = split_one(self.frame_stack)
        self.frames = deque([], maxlen=self.frame_stack)

        self.game_over = False
        self.restart = True

        self.recording_thread = threading.Thread(target=self.update_game_img, daemon=True)
        self.recording_thread.start()
        self.player_thread = threading.Thread(target=self.update_players, daemon=True)
        self.player_thread.start()

        # Block until the image pipeline has produced its first frame so that
        # img_h / img_w are known to callers building observation spaces.
        while not self.img_h:
            time.sleep(0.001)

    # ------------------------------------------------------------------ players
    @property
    def channels(self) -> int:
        return 1 if self.gray_scale else 3

    @property
    def num_players(self) -> int:
        return len(self.active_players)

    @property
    def active_players(self) -> tuple[Player, ...]:
        return tuple(p for p in self.players if p is not None and p.is_active)

    def find_players(self) -> None:
        for index in range(len(self.players)):
            computer = Player(game_hwnd=self.game_hwnd, index=index, is_computer=True)
            human = Player(game_hwnd=self.game_hwnd, index=index, is_computer=False)
            if computer.is_active:
                self.players[index] = computer
            elif human.is_active:
                self.players[index] = human
            else:
                self.players[index] = None

    # ------------------------------------------------------------- observations
    def wait_for_frames(self) -> None:
        """Block until the frame stack is full."""
        while len(self.frames) < self.frame_stack:
            time.sleep(0.001)

    def get_image_obs(self) -> np.ndarray:
        """Return the weighted, channels-first ``uint8`` stacked image."""
        self.wait_for_frames()
        _imgs = np.stack(self.frames)
        img_stack = np.tensordot(self.img_weights, _imgs, axes=([0], [0]))
        if not self.gray_scale:
            img_stack = np.transpose(img_stack, (2, 0, 1))
        else:
            img_stack = img_stack[None, ...]
        return img_stack.astype(np.uint8)

    @staticmethod
    def player_state(player: Player) -> list[int]:
        # mp, hp, facing, x, y, z
        return [
            player.mp,
            player.hp,
            int(bool.from_bytes(player._facing_bytes)),
            player.x_pos,
            player.y_pos,
            player.z_pos,
        ]

    def get_players_state(self, my_player: Player) -> np.ndarray:
        """Per-agent info array, with ``my_player`` first then the rest."""
        return np.asarray(
            [
                self.player_state(my_player),
                *[self.player_state(p) for p in self.active_players if p is not my_player],
            ],
            dtype=np.int16,
        )

    # --------------------------------------------------------------- key sending
    def press_player_action(self, player: Player, action_id: int) -> None:
        """Send the keystrokes for ``player``'s chosen action to the window."""
        press_key(player.action_keys(action_id))

    # ------------------------------------------------------------------- threads
    def update_game_img(self) -> None:
        """Continuously capture the gaming scene into the frame stack."""
        skip_i = 0
        last_frame = np.array([0])
        while not self.kill_thread:
            tup = win32gui.GetWindowPlacement(self.game_hwnd)

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
                self.img_h = int(shape[0])
                self.img_w = int(shape[1])
                print("img dimension: H {} W {}".format(self.img_h, self.img_w))

            frame = cv2.resize(self.gaming_screen, (self.img_w, self.img_h))
            if self.gray_scale:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if skip_i >= self.frame_skip:
                self.frames.append(frame)
                skip_i = 0
            else:
                skip_i += 1
            last_frame = screen_shot.copy()
            time.sleep(0.01)

    def update_players(self) -> None:
        """Continuously refresh player memory state and game-over detection."""
        while not self.kill_thread:
            time.sleep(0.01)
            team = []
            for player in filter(lambda p: p is not None, self.players):
                player.update()
                if player.is_active and player.is_alive:
                    team.append(player.team)
            self.restart = False
            self.game_over = len(team) > 0 and len(set(team)) == 1

    # -------------------------------------------------------------------- rounds
    def reset_round(self, default_ok: list[str]) -> None:
        """Press F4 (+ confirmation keys) to restart the round and clear frames."""
        press_key(["f4", *default_ok], interval=1.0)
        self.restart = True
        self.game_over = False
        self.frames.clear()

    def close(self) -> None:
        cv2.destroyAllWindows()
        self.kill_thread = True
