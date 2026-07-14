"""Game-I/O controller for Little Fighter 2 (Windows-only).

:class:`Lf2GameController` owns every interaction with the running game / OS:

* window handle + focus,
* screen capture (``mss``) and the frame-stacking image pipeline,
* per-player memory reads (via :class:`~lf2_gym.windows.player.Player`),
* keystroke delivery for *any* of the (up to 4) human player slots.

It is deliberately agnostic of the RL interface so it can be shared by:

* the single-agent :class:`~lf2_gym.windows.env.Lf2Env`, and
* the multi-agent :class:`~lf2_gym.windows.parallel_env.Lf2ParallelEnv`,

where the 4 in-window players are each driven by an agent that shares one
centrally-trained policy. The RL logic itself lives in the pure
:mod:`lf2_gym.lf2_envs.base` module, which receives this controller via
constructor injection — i.e. nothing in ``lf2_gym/lf2_envs/`` ever imports
this module.
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

from lf2_gym.loggers import get_logger
from lf2_gym.windows.keys import press_key
from lf2_gym.windows.player import Player
from lf2_gym.windows.winguiauto import winguiauto as winauto

logger = get_logger(__name__)


def split_one(num_interval: int = 1) -> np.ndarray:
    """Return normalized ``sinh`` weights used to collapse a frame stack."""
    num_list = np.sinh(list(i + 1 for i in range(num_interval)), dtype=np.float32)
    return num_list / np.sum(num_list)


class Lf2GameController:
    """Owns the LF2 window and all shared game I/O for one game instance."""

    #: Default seconds to wait for the first captured frame after the
    #: recording thread starts. Override via the ``first_frame_timeout``
    #: ctor kwarg; set to ``None`` (or non-positive) to wait forever
    #: (legacy behaviour).
    DEFAULT_FIRST_FRAME_TIMEOUT: float = 30.0

    def __init__(
        self,
        windows_name: str = "Little Fighter 2",
        downscale: int = 2,
        frame_stack: int = 4,
        frame_skip: int = 1,
        gray_scale: bool = True,
        first_frame_timeout: float | None = DEFAULT_FIRST_FRAME_TIMEOUT,
    ):
        self.window_name = windows_name
        self.game_hwnd = winauto.findTopWindow(wantedText=windows_name)
        self.PyCWnd1 = win32ui.FindWindow(None, windows_name)
        self.PyCWnd1.SetForegroundWindow()
        self.PyCWnd1.SetFocus()
        self.first_frame_timeout = first_frame_timeout

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
        self.img_weights: np.ndarray = split_one(self.frame_stack)
        self.frames = deque([], maxlen=self.frame_stack)
        # Pre-composed channels-first uint8 observation maintained by the
        # capture thread so that step() never has to stack/weight frames.
        self._composed_obs: np.ndarray | None = None
        self._active_players: tuple[Player, ...] | None = None

        # Pre-allocated per-player info buffer ([mp, hp, facing, x, y, z] per
        # slot). The player thread writes rows in-place; step() just
        # fancy-indexes the active rows out of it.
        self._state_buf = np.zeros((len(self.players), 6), dtype=np.int16)

        self.game_over = False
        self.restart = True

        self._stop_event = threading.Event()

        self.recording_thread = threading.Thread(target=self.update_game_img, daemon=True)
        self.recording_thread.start()
        self.player_thread = threading.Thread(target=self.update_players, daemon=True)
        self.player_thread.start()

        # Block until the image pipeline has produced its first frame so that
        # img_h / img_w are known to callers building observation spaces.
        #
        # Fail fast if LF2 never produces a frame (e.g. LF2 not running, not
        # focused, or window minimized). Without a deadline a stuck Ray
        # worker would silently never report Ready, hanging the whole
        # training run.
        deadline = (
            time.monotonic() + self.first_frame_timeout
            if self.first_frame_timeout and self.first_frame_timeout > 0
            else None
        )
        while not self.img_h:
            if deadline is not None and time.monotonic() > deadline:
                raise RuntimeError(
                    f"Lf2GameController: no frame captured from window "
                    f"{windows_name!r} within {self.first_frame_timeout}s. "
                    f"Make sure LF2 is running, focused and visible."
                )
            time.sleep(0.01)

    # ------------------------------------------------------------------ players
    @property
    def channels(self) -> int:
        return 1 if self.gray_scale else 3

    @property
    def num_players(self) -> int:
        return len(self.active_players)

    @property
    def active_players(self) -> tuple[Player, ...]:
        if not self._active_players:
            self._active_players = tuple(p for p in self.players if p is not None and p.is_active)
        return self._active_players

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
        """Block until the composed image observation is ready."""
        while self._composed_obs is None:
            time.sleep(0.001)

    def _compose_obs(self) -> None:
        """Compute the weighted channels-first uint8 obs from ``self.frames``.

        Runs in the capture thread so the hot ``get_image_obs`` path doesn't
        pay for ``np.stack`` + a float ``tensordot`` + a uint8 cast.
        """
        _imgs = np.stack(self.frames)
        img_stack = np.tensordot(self.img_weights, _imgs, axes=([0], [0])).astype(np.uint8)
        if self.gray_scale:
            img_stack = img_stack[None, ...]
        else:
            img_stack = np.transpose(img_stack, (2, 0, 1))
        # Publish via reference swap (atomic in CPython). Consumers see a
        # fully-formed array without needing a lock.
        self._composed_obs = img_stack

    def get_image_obs(self) -> np.ndarray:
        """Return the weighted, channels-first ``uint8`` stacked image."""
        self.wait_for_frames()
        # ``.copy()`` keeps the "fresh array per call" contract — consumers
        # (gym wrappers, replay buffers) historically assumed they own the
        # returned buffer.
        return self._composed_obs.copy()

    def get_players_state(self, my_player: Player) -> np.ndarray:
        """Per-agent info array, with ``my_player`` first then the rest."""
        my_idx = my_player.index
        rows = [my_idx]
        rows.extend(i for i in range(len(self.active_players)) if i != my_idx)
        # Fancy index returns a fresh contiguous array — safe to hand back.
        return self._state_buf[rows]

    # --------------------------------------------------------------- key sending
    @staticmethod
    def press_player_action(player: Player, action_id: int) -> None:
        """Send the keystrokes for ``player``'s chosen action to the window."""
        press_key(player.action_keys(action_id))

    # ------------------------------------------------------------------- threads
    def update_game_img(self) -> None:
        """Continuously capture the gaming scene into the frame stack."""
        skip_i = 0
        last_frame = np.array([0])
        while not self._stop_event.is_set():
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
            screen_sample = screen_shot[::5, ::5]

            if np.array_equal(last_frame, screen_sample):
                # refresh until new frame exists.
                continue
            last_frame = screen_sample.copy()
            self.gaming_screen = screen_shot

            if not self.img_h:
                shape = (np.array(np.shape(self.gaming_screen)[:2]) / self.downscale).astype(int)
                self.img_h, self.img_w = shape
                logger.info("Screen capture initialized: H {} W {}".format(self.img_h, self.img_w))

            frame = cv2.resize(self.gaming_screen, (self.img_w, self.img_h))
            if self.gray_scale:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if skip_i >= self.frame_skip:
                self.frames.append(frame)
                skip_i = 0
                if len(self.frames) == self.frame_stack:
                    self._compose_obs()
            else:
                skip_i += 1

            self._stop_event.wait(0.02)

    def update_players(self) -> None:
        """Continuously refresh player memory state and game-over detection."""
        buf = self._state_buf
        while not self._stop_event.is_set():
            self._stop_event.wait(0.02)
            if self.game_over:
                continue
            team = []
            human_alive = False
            for i, player in enumerate(self.active_players):
                player.update()
                # Mirror the refreshed Python attrs into the contiguous int16
                # buffer so callers can fancy-index without rebuilding lists.
                buf[i] = np.asarray(
                    [player.mp, player.hp, player.facing_int, player.x_pos, player.y_pos, player.z_pos],
                    dtype=np.int16,
                )
                if player.is_alive:
                    team.append(player.team)
                human_alive |= player.is_alive and not player.is_computer
            self.restart = False
            self.game_over = (len(team) > 0 and len(set(team)) == 1) or not human_alive
            if self.game_over and not self.restart:
                logger.info("Game over detected: all active players are on the same team.")

    # -------------------------------------------------------------------- rounds
    def reset_round(self, default_ok: list[str]) -> None:
        """Press F4 (+ confirmation keys) to restart the round and clear frames."""
        press_key(["f4", *default_ok], interval=1.0)
        self.restart = True
        self.game_over = False
        self.frames.clear()
        self._active_players = None
        self.find_players()
        # Force the next ``get_image_obs`` to block until a fresh stack has
        # been composed post-restart, matching the original semantics of
        # ``wait_for_frames`` after a deque clear.
        self._composed_obs = None

    def close(self) -> None:
        cv2.destroyAllWindows()
        self._stop_event.set()
