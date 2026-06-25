from typing import Final
from enum import IntEnum


BackGroundCode = {
    "HK Coliseum": 0,
    "Lion Forest": 1,
    "Stanley Prison": 2,
    "The Great Wall": 3,
    "Queen's Island": 4,
    "Forbidden Tower": 5,
    "BrokeBack Cliff": 6,
    "CUHK": 7,
    "Uai Hom Village": 8,
    "Template1": 9,
    "Template2": 10,
    "Template3": 11,
    "Random": 100,
}


class Difficulty(IntEnum):
    EASY = 2
    NORMAL = 1
    DIFFICULT = 0
    CRAZY = -1


class Mode(IntEnum):
    VS = 0
    STAGE = 1
    ONE_VS_ONE = 2
    TWO_VS_TWO = 3
    BATTLE = 4
    DEMO = 5
    PLAYBACK = 6
    QUIT = 7

    @property
    def label(self):
        labels = {
            Mode.VS: "VS",
            Mode.STAGE: "Stage",
            Mode.ONE_VS_ONE: "1v1",
            Mode.TWO_VS_TWO: "2v2",
            Mode.BATTLE: "Battle",
            Mode.DEMO: "Demo",
            Mode.PLAYBACK: "PlayBack",
            Mode.QUIT: "Quit",
        }
        return labels[self]


# Address Offset Constant relative to player base address
class LF2PlayerAddressOffset(IntEnum):

    X_POS = 0x10
    Y_POS = 0x14
    Z_POS = 0x18
    X_POS_F = 0x58
    Y_POS_F = 0x60
    Z_POS_F = 0x68

    KILLS = 0x358
    ATTACK = 0x348

    HP = 0x2FC
    HP_DARK = 0x300
    HP_MAX = 0x304
    HP_LOST = 0x32C
    MP = 0x308
    MP_USAGE = 0x350

    PICKING = 0x35C
    OWNER = 0x354
    ENEMY = 0x360
    TEAM = 0x364
    PDATA_POINTER = 0x368

    INVINCIBLE = 0x8
    FACING = 0x80


def _span(start: LF2PlayerAddressOffset, end: LF2PlayerAddressOffset) -> tuple[int, int]:
    """回傳 (起始 offset, 位元組長度)，長度涵蓋 end 這個 int32 本身。"""
    return int(start), int(end) - int(start) + 4


LF2_BLOCK_POS: Final[tuple[int, int]] = _span(
    LF2PlayerAddressOffset.X_POS,
    LF2PlayerAddressOffset.Z_POS,
)

LF2_BLOCK_POS_F: Final[tuple[int, int]] = _span(
    LF2PlayerAddressOffset.X_POS_F,
    LF2PlayerAddressOffset.Z_POS_F,
)

LF2_BLOCK_STAT: Final[tuple[int, int]] = _span(
    LF2PlayerAddressOffset.HP,
    LF2PlayerAddressOffset.PDATA_POINTER,
)


class LF2AbsAddress(IntEnum):
    # global absolute address
    DATA_POINTER = 0x4592D4
    GAME_STATE = 0x44D020
    TIME = 0x450BBC
    TOTAL_TIME = 0x450B8C
    BACKGROUND = 0x44D024
    DIFFICULTY = 0x450C30
    MODE = 0x451160


# player array address
PLAYER_ADDRESSES: Final[tuple[int, ...]] = tuple(0x458C94 + i * 4 for i in range(8))
COMPUTER_ADDRESSES: Final[tuple[int, ...]] = tuple(0x458CBC + i * 4 for i in range(8))
ACTIVE_PLAYERS: Final[tuple[int, ...]] = tuple(0x458B04 + i for i in range(8))
SELECTED_PLAYERS: Final[tuple[int, ...]] = tuple(0x451288 + i * 4 for i in range(8))
PLAYER_IN_GAME: Final[tuple[int, ...]] = tuple(0x458B04 + i for i in range(8))
CPLAYER_IN_GAME: Final[tuple[int, ...]] = tuple(0x458B0E + i for i in range(8))

NAMES: Final[tuple[int, ...]] = tuple(0x44FCC0 + i * 11 for i in range(11))

DATA_FILE_COUNT: Final[int] = 65
