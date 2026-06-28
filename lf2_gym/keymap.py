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
from types import MappingProxyType

from lf2_gym.characters import LogicBtn, Move
from lf2_gym.config import LF2_GYM_CONFIG

vk = MappingProxyType({
    0x31: "1",
    0x32: "2",
    0x33: "3",
    0x34: "4",
    0x35: "5",
    0x36: "6",
    0x37: "7",
    0x38: "8",
    0x39: "9",
    0x30: "0",
    0x41: "A",
    0x42: "B",
    0x43: "C",
    0x44: "D",
    0x45: "E",
    0x46: "F",
    0x47: "G",
    0x48: "H",
    0x49: "I",
    0x4A: "J",
    0x4B: "K",
    0x4C: "L",
    0x4D: "M",
    0x4E: "N",
    0x4F: "O",
    0x50: "P",
    0x51: "Q",
    0x52: "R",
    0x53: "S",
    0x54: "T",
    0x55: "U",
    0x56: "V",
    0x57: "W",
    0x58: "X",
    0x59: "Y",
    0x5A: "Z",
    0x70: "F1",
    0x71: "F2",
    0x72: "F3",
    0x73: "F4",
    0x74: "F5",
    0x75: "F6",
    0x76: "F7",
    0x77: "F8",
    0x78: "F9",
    0x79: "F10",
    0x7A: "F11",
    0x7B: "F12",
    0x26: "UP",
    0x25: "LEFT",
    0x27: "RIGHT",
    0x28: "DOWN",
    0x1B: "ESC",
    0x20: "SPACE",
    0x0D: "ENTER",
    0x2D: "INSERT",
    0x2E: "DELETE",
    0x09: "TAB",
    0xA2: "CTRL",
    0xA3: "CONTROL",
    0xA0: "SHIFT",
    0x14: "CAPSLOCK",
    0xBD: "SUBTRACT",
    0xDB: "[",
    0xDD: "]",
    0xBA: ";",
    0xDE: "'",
    0xC0: "`",
    0xDC: "\\",
    0xBC: ",",
    0xBE: ".",
    0xBF: "/",
    0xBB: "=",
})

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
                    codes = [int(x) for x in line.split()]
                    break
        if len(codes) < 8:
            raise ValueError(f"player {player_id} control line has {len(codes)} fields, need ≥ 8")

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
