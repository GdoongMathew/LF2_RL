# All following code is inspired by LF2 Dashboard
# LF2 Dashboard github: https://github.com/xmfcx/LF2-Dashboard/tree/master/LF2Dashboard

from ctypes.wintypes import BOOL
from ctypes.wintypes import DWORD
from ctypes.wintypes import HANDLE
from collections.abc import Sequence
from pathlib import Path
from typing import Literal
from dataclasses import field, dataclass
from sys import byteorder
import struct
from ..config import LF2_GYM_CONFIG


from lf2_gym.characters import Characters, CHARACTER_MOVES, LogicBtn, Move, vk
from lf2_gym.const import (
    LF2AbsAddress,
    LF2PlayerAddressOffset,
    LF2_BLOCK_POS,
    LF2_BLOCK_STAT,
    PLAYER_ADDRESSES,
    COMPUTER_ADDRESSES,
    CPLAYER_IN_GAME,
    PLAYER_IN_GAME,
    DATA_FILE_COUNT,
)
import pyautogui
import ctypes
import pymem
import win32process

PROCESS_VM_OPERATION = 0x0008
PROCESS_VM_READ = 0x0010
PROCESS_VM_WRITE = 0x0020


def press_key(keys: list[str]):
    for key in keys:
        pyautogui.press(
            key,
        )


class ProcessWR:
    # Reading/Writing process memory from certain memory address

    _ctype_open_process = ctypes.windll.kernel32.OpenProcess
    _ctype_open_process.restype = HANDLE
    _ctype_open_process.argtypes = (DWORD, BOOL, DWORD)

    _ctypes_get_last_error = ctypes.windll.kernel32.GetLastError
    _ctypes_get_last_error.restype = DWORD
    _ctypes_get_last_error.argtypes = ()

    _proc_instance: dict[int, "ProcessWR"] = {}

    def __new__(cls, *args, win_handle: int, **kwargs):
        if win_handle not in cls._proc_instance:
            cls._proc_instance[win_handle] = super().__new__(cls)
        return cls._proc_instance[win_handle]

    def __init__(self, *, win_handle: int):
        self.pid = win32process.GetWindowThreadProcessId(win_handle)[1]
        self.proc_handle = self.get_process_handle(
            self.pid,
            PROCESS_VM_OPERATION | PROCESS_VM_READ | PROCESS_VM_WRITE,
        )

    @staticmethod
    def get_process_handle(
        process_id,
        desired_access,
        inherit_handle: bool = False,
    ):
        handle = ProcessWR._ctype_open_process(desired_access, inherit_handle, process_id)
        if handle is None or handle == 0:
            raise RuntimeError(
                f"Failed to open process with ID {process_id}, Desired access: {desired_access}, "
                f"Inherit handle: {inherit_handle} with error {ProcessWR._ctypes_get_last_error()}."
            )
        return handle

    def read_block(self, base_address: int, size: int) -> bytes:
        return pymem.memory.read_bytes(self.proc_handle, base_address, size)

    def read_bytes(self, lpBaseAddress: int, n_size: int) -> bytes:
        return self.read_block(lpBaseAddress, n_size)

    def read_char(self, lpBaseAddress):
        return pymem.memory.read_char(self.proc_handle, lpBaseAddress)

    def read_int(self, lpBaseAddress):
        return pymem.memory.read_int(self.proc_handle, lpBaseAddress)

    def read_uint(self, lpBaseAddress):
        return pymem.memory.read_uint(self.proc_handle, lpBaseAddress)

    def read_long(self, lpBaseAddress):
        return pymem.memory.read_long(self.proc_handle, lpBaseAddress)

    def read_str(self, lpBaseAddress, n_size=4):
        return pymem.memory.read_string(self.proc_handle, lpBaseAddress, n_size)

    def read_float(self, lpBaseAddress):
        return pymem.memory.read_float(self.proc_handle, lpBaseAddress)

    def read_ushort(self, lpBaseAddress):
        return pymem.memory.read_ushort(self.proc_handle, lpBaseAddress)

    def write_int(self, lpBaseAddress, data):
        return pymem.memory.write_int(self.proc_handle, lpBaseAddress, data)


@dataclass(kw_only=True)
class Player:

    game_hwnd: int
    index: int
    is_computer: bool
    character: str = field(init=False)

    address: int = field(init=False)
    data_address: int = field(init=False)

    kills: int = field(init=False)
    attacks: int = field(init=False)

    hp: int = field(init=False)
    hp_dark: int = field(init=False)
    hp_lost: int = field(init=False)

    mp: int = field(init=False)
    mp_max: int = field(default=500)  # not sure about the number
    mp_usage: int = field(init=False)

    picking: int = field(init=False)
    owner: int = field(init=False)
    enemy: int = field(init=False)
    team: int = field(init=False)

    x_pos: int = field(default=None, init=False)
    y_pos: int = field(default=None, init=False)
    z_pos: int = field(default=None, init=False)

    _facing_bytes: bytes = field(init=False, repr=False)

    def __post_init__(self):
        self._game_reading = ProcessWR(win_handle=self.game_hwnd)

        address_table = COMPUTER_ADDRESSES if self.is_computer else PLAYER_ADDRESSES
        self.address = self._game_reading.read_int(address_table[self.index])
        self.data_address = self._game_reading.read_int(
            self.address_shift(LF2PlayerAddressOffset.PDATA_POINTER)
        )
        character = self._get_player_character()
        assert character is not None, "Failed to get character"
        self.character: Characters = character
        self.update()

    def address_shift(self, shift: int):
        return self.address + shift

    @property
    def facing(self) -> Literal["right", "left"]:
        return "left" if bool.from_bytes(self._facing_bytes) else "right"

    @property
    def hp_max(self) -> int:
        return self._game_reading.read_int(
            self.address_shift(LF2PlayerAddressOffset.HP_MAX),
        )

    @property
    def is_active(self) -> bool:
        address_table = CPLAYER_IN_GAME if self.is_computer else PLAYER_IN_GAME
        return bool.from_bytes(
            self._game_reading.read_bytes(address_table[self.index], 1),
            byteorder,
        )

    @property
    def is_alive(self) -> bool:
        return self.hp > 0

    def _parse_stat_buffer(self, stat_buffer: bytes, base: int):
        for attr, offset in (
            ("hp", LF2PlayerAddressOffset.HP),
            ("hp_dark", LF2PlayerAddressOffset.HP_DARK),
            ("hp_lost", LF2PlayerAddressOffset.HP_LOST),
            ("mp", LF2PlayerAddressOffset.MP),
            ("mp_usage", LF2PlayerAddressOffset.MP_USAGE),
            ("kills", LF2PlayerAddressOffset.KILLS),
            ("attacks", LF2PlayerAddressOffset.ATTACK),
            ("picking", LF2PlayerAddressOffset.PICKING),
            ("owner", LF2PlayerAddressOffset.OWNER),
            ("enemy", LF2PlayerAddressOffset.ENEMY),
            ("team", LF2PlayerAddressOffset.TEAM),
        ):
            rel = offset - base
            value = struct.unpack_from("<i", stat_buffer, rel)[0]
            setattr(self, attr, value)

        self.hp = max(0, self.hp)

    def _parse_pos_buffer(self, pos_buffer: bytes, base: int):
        for attr, offset in (
            ("x_pos", LF2PlayerAddressOffset.X_POS),
            ("y_pos", LF2PlayerAddressOffset.Y_POS),
            ("z_pos", LF2PlayerAddressOffset.Z_POS),
        ):
            rel = offset - base
            value = struct.unpack_from("<i", pos_buffer, rel)[0]
            setattr(self, attr, value)

    def update(self):
        """Update the player status from the game memory."""

        base_stat_bytes = self.address_shift(LF2_BLOCK_STAT[0])
        stat_buffer = self._game_reading.read_block(base_stat_bytes, LF2_BLOCK_STAT[1])
        self._parse_stat_buffer(stat_buffer, LF2_BLOCK_STAT[0])

        base_pos_bytes = self.address_shift(LF2_BLOCK_POS[0])
        pos_buffer = self._game_reading.read_block(base_pos_bytes, LF2_BLOCK_POS[1])
        self._parse_pos_buffer(pos_buffer, LF2_BLOCK_POS[0])

        self._facing_bytes = self._game_reading.read_bytes(
            self.address_shift(LF2PlayerAddressOffset.FACING),
            1,
        )

    def _get_player_character(self) -> Characters | None:
        """Get the character name of the player by comparing the data address with the character data addresses."""
        _data_address = self._game_reading.read_int(LF2AbsAddress.DATA_POINTER)

        for i, name in enumerate(Characters):
            address = self._game_reading.read_int(_data_address + i * 4)
            if address == self.data_address:
                return name
        return None

    @property
    def moves(self) -> Sequence[Move]:
        """Return the list of moves for the player's character."""
        return CHARACTER_MOVES[self.character]

    def action_keys(self, action_index: int) -> list[str]:
        """Return the key for the given action."""
        move = self.moves[action_index]
        return resolve_move_key(move, KeyMap.for_player(self.index), self.facing)


class KeyMap:
    """Converting LogicBtn to actual physical key button"""

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
            line = f.read().splitlines()[player_id + 1]

        codes = [int(x) for x in line.split() if x.strip()]
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
    out = []
    for btn in move.sequence:
        if btn is LogicBtn.Dir:
            btn = LogicBtn(facing.capitalize())
        out.append(keymap[btn])
    return out
