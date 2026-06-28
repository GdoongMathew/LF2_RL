"""Keyboard mapping helpers — pure Python, no Windows deps.

Extracted from ``lf2_gym.lf2_envs.utils`` so cross-platform code (training
learner on Linux, unit tests, etc.) can resolve action -> key strings without
loading any of the OS-specific machinery (``pymem``, ``pyautogui``,
``win32*``) that the live game I/O code requires.

The only thing this module *does not* do is press the keys. That stays on the
Windows side because ``pyautogui.keyDown`` / ``keyUp`` need a real OS keyboard.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

from lf2_gym.characters import LogicBtn, Move, vk
from lf2_gym.config import LF2_GYM_CONFIG


@runtime_checkable
class PlayerProtocol(Protocol):
    """Subset of player attributes that pure code (reward, base env) needs.

    The concrete ``lf2_gym.windows.player.Player`` (memory-reading) and the
    ``FakePlayer`` used by tests both satisfy this protocol.
    """

    index: int
    facing: Literal["left", "right"]

    hp: int
    hp_max: int
    mp: int
    mp_max: int

    attacks: int
    team: int

    is_active: bool
    is_alive: bool
    moves: Sequence[Move]

    def action_keys(self, action_index: int) -> list[str]: ...


class KeyMap:
    """Converting :class:`LogicBtn` to actual physical key button strings."""

    _cache: dict[int, dict[LogicBtn, str]] = {}

    @classmethod
    def for_player(cls, player_id: int) -> dict[LogicBtn, str]:
        if player_id not in cls._cache:
            cls._cache[player_id] = cls._parse_control_text(
                player_id,
                control_txt=LF2_GYM_CONFIG.control_txt,
            )
        return cls._cache[player_id]

    @staticmethod
    def _parse_control_text(player_id: int, control_txt: Path) -> dict[LogicBtn, str]:
        if player_id > 3:
            raise ValueError(f"Player ID must be less than 3, get {player_id}.")

        with open(control_txt.as_posix(), "r") as f:
            for i, line in enumerate(f):
                if i == player_id:
                    codes = [int(x) for x in line.split(" ") if x not in (" ", "\n")]
                    break

        return {
            LogicBtn.Up: vk[codes[1]],
            LogicBtn.Down: vk[codes[2]],
            LogicBtn.Left: vk[codes[3]],
            LogicBtn.Right: vk[codes[4]],
            LogicBtn.Attack: vk[codes[5]],
            LogicBtn.Jump: vk[codes[6]],
            LogicBtn.Defend: vk[codes[7]],
        }


def resolve_move_key(
    move: Move,
    keymap: dict[LogicBtn, str],
    facing: Literal["left", "right"],
) -> list[str]:
    """Translate a logical :class:`Move` sequence into physical key strings."""
    out: list[str] = []
    for btn in move.sequence:
        if btn is LogicBtn.Dir:
            btn = LogicBtn(facing.lower())
        out.append(keymap[btn])
    return out
