"""Single-agent Windows-side LF2 env (live game I/O).

Thin wrapper around :class:`lf2_gym.lf2_envs.base.Lf2EnvBase`:

* builds the Windows :class:`~lf2_gym.windows.controller.Lf2GameController`
  (``mss`` + ``win32*`` + ``cv2`` + ``pymem`` + ``pyautogui``),
* enables the ``cv2``-backed ``human`` render mode.

Nothing in ``lf2_gym.lf2_envs.*`` imports this module — so the learner side
(Linux DGX) can load the pure base classes without dragging in any of the
Windows libraries above.
"""

from __future__ import annotations

import time
from typing import Literal

import cv2
import pyautogui

from lf2_gym.lf2_envs.base import Lf2EnvBase
from lf2_gym.loggers import get_logger
from lf2_gym.rewards import compute_reward  # noqa: F401 (back-compat re-export)
from lf2_gym.windows.controller import Lf2GameController, split_one  # noqa: F401
from lf2_gym.windows.player import Player
from lf2_gym.windows.winguiauto import winguiauto as winauto

pyautogui.FAILSAFE = False

__all__ = ["Lf2Env", "compute_reward", "split_one"]

logger = get_logger(__name__)


class Lf2Env(Lf2EnvBase):
    """Single-agent LF2 env that owns a Windows :class:`Lf2GameController`.

    Game I/O (window capture, memory reads, key sending) is delegated to a
    :class:`~lf2_gym.windows.controller.Lf2GameController`; the RL logic
    (step / reset / observation / reward) lives in
    :class:`~lf2_gym.lf2_envs.base.Lf2EnvBase` so it can be unit-tested
    cross-platform.
    """

    metadata = {"render.modes": ["human", "console", "rgb_array"]}

    def __init__(
        self,
        windows_name: str = "Little Fighter 2",
        player_id: int = 1,
        downscale: int = 2,
        frame_stack: int = 4,
        frame_skip: int = 1,
        reset_skip_sec: int = 2,
        gray_scale: bool = True,
        mode: Literal["info", "picture", "mix"] = "mix",
        basic_action: bool = False,
        controller: Lf2GameController | None = None,
    ):
        """
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
        controller = controller or Lf2GameController(
            windows_name=windows_name,
            downscale=downscale,
            frame_stack=frame_stack,
            frame_skip=frame_skip,
            gray_scale=gray_scale,
        )
        super().__init__(
            controller=controller,
            player_id=player_id,
            reset_skip_sec=reset_skip_sec,
            mode=mode,
            basic_action=basic_action,
        )

    def render(self, mode: str = "human"):
        if mode == "human":
            if self.gaming_screen is not None:
                cv2.imshow("lf2_render", self.gaming_screen)
                cv2.waitKey(1)
        else:
            super().render(mode=mode)


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
