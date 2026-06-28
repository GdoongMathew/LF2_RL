"""Multi-agent Windows-side PettingZoo env for Little Fighter 2.

Thin wrapper around :class:`lf2_gym.lf2_envs.base.Lf2ParallelEnvBase` that
builds the Windows :class:`~lf2_gym.windows.controller.Lf2GameController`.

A single LF2 window hosts up to 4 human-controllable player slots; the base
class exposes each slot as an independent agent that shares one centrally
trained policy. The shared screen image (``Game_Screen``) is identical for
all agents, while the ``Info`` array and reward are agent-centric (acting
player listed first).
"""

from __future__ import annotations

from typing import Any

from lf2_gym.lf2_envs.base import Lf2ParallelEnvBase
from lf2_gym.windows.controller import Lf2GameController


class Lf2ParallelEnv(Lf2ParallelEnvBase):
    """PettingZoo ``ParallelEnv`` wrapping one LF2 window with N agents."""

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
        controller = controller or Lf2GameController(
            windows_name=windows_name,
            downscale=downscale,
            frame_stack=frame_stack,
            frame_skip=frame_skip,
            gray_scale=gray_scale,
        )
        super().__init__(
            controller=controller,
            player_ids=player_ids,
            reset_skip_sec=reset_skip_sec,
            mode=mode,  # type: ignore[arg-type]
        )


def parallel_env(**kwargs: Any) -> Lf2ParallelEnv:
    """Factory matching the PettingZoo ``parallel_env`` convention."""
    return Lf2ParallelEnv(**kwargs)
