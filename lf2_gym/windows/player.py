"""Live LF2 player — owns the memory reads against the running game.

Concrete implementation of :class:`lf2_gym.keymap.PlayerProtocol` backed by
:class:`lf2_gym.windows.process.ProcessWR`. The base envs in
:mod:`lf2_gym.lf2_envs.base` only ever see the protocol; this class is what
the Windows-side controller actually instantiates.
"""

from __future__ import annotations

import struct
from collections.abc import Sequence
from dataclasses import dataclass, field
from sys import byteorder
from typing import Literal

from lf2_gym.characters import CHARACTER_MOVES, Characters, Move
from lf2_gym.const import (
    COMPUTER_ADDRESSES,
    CPLAYER_IN_GAME,
    LF2_BLOCK_POS,
    LF2_BLOCK_STAT,
    LF2AbsAddress,
    LF2PlayerAddressOffset,
    PLAYER_ADDRESSES,
    PLAYER_IN_GAME,
)
from lf2_gym.keymap import KeyMap, resolve_move_key
from lf2_gym.loggers import get_logger
from lf2_gym.windows.process import ProcessWR


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

    _facing_bytes: bytes = field(default=b"\x00", init=False, repr=False)

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

    @property
    def logger(self):
        if not hasattr(self, "_logger"):
            self._logger = get_logger(f"Player {self.character}")
        return self._logger

    def address_shift(self, shift: int):
        return self.address + shift

    @property
    def facing_int(self) -> int:
        return 1 if self._facing_bytes != b"\x00" else 0

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
        self.logger.info(f"Action {action_index}: {move.name} -> {move.sequence}")
        return resolve_move_key(move, KeyMap.for_player(self.index), self.facing)
