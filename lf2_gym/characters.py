# http://www.cheatbook.de/files/littlef2.htm
from dataclasses import dataclass
from enum import StrEnum
from collections import defaultdict

vk = {
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
    0xBD: "subtract",
    0xDB: "[",
    0xDD: "]",
    0xBA: ";",
    0xDE: "'",
    0xC0: "`",
    0xDC: "\\",
    0xBC: ",",
    0xBE: ".",
    0xBF: "/",
}


class Characters(StrEnum):
    """Characters in Little Fighter 2.

    DO NOT change the ordering of the following members, as it's directly mapped to the
    character ID in LF2. The character ID is used to identify the character in the game memory.
    """
    Template = "Template"
    Julian = "Julian"
    Firzen = "Firzen"
    LouisEX = "LouisEX"
    Bat = "Bat"
    Justin = "Justin"
    Knight = "Knight"
    Jan = "Jan"
    Monk = "Monk"
    Sorcerer = "Sorcerer"
    Jack = "Jack"
    Mark = "Mark"
    Hunter = "Hunter"
    Bandit = "Bandit"
    Deep = "Deep"
    John = "John"
    Henry = "Henry"
    Rudolf = "Rudolf"
    Louis = "Louis"
    Firen = "Firen"
    Freeze = "Freeze"
    Dennis = "Dennis"
    Woody = "Woody"
    Davis = "Davis"


class LogicBtn(StrEnum):
    Up = "up"
    Down = "down"
    Left = "left"
    Right = "right"
    Attack = "attack"
    Jump = "jump"
    Defend = "defend"

    Dir = "dir"  # would be switched with character's facing direction


@dataclass(frozen=True, kw_only=True)
class Move:
    name: str
    sequence: tuple[LogicBtn, ...]

    @property
    def need_direction(self) -> bool:
        return any(move is LogicBtn.Dir for move in self.sequence)


_Run = Move(name="run", sequence=(LogicBtn.Dir, LogicBtn.Dir))


BASIC_MOVES = (
    Move(name="idle", sequence=()),
    Move(name="up", sequence=(LogicBtn.Up,)),
    Move(name="down", sequence=(LogicBtn.Down,)),
    Move(name="right", sequence=(LogicBtn.Right,)),
    Move(name="left", sequence=(LogicBtn.Left,)),
    Move(name="attack", sequence=(LogicBtn.Attack,)),
    Move(name="defend", sequence=(LogicBtn.Defend,)),
    Move(name="jump", sequence=(LogicBtn.Jump,)),
    _Run,
)


CHARACTER_MOVES = defaultdict(lambda: BASIC_MOVES)

CHARACTER_MOVES.update(
    **{
        Characters.Template: BASIC_MOVES,
        Characters.John: (
            *BASIC_MOVES,
            Move(
                name="energy_blast",
                sequence=(
                    LogicBtn.Defend,
                    LogicBtn.Dir,
                    LogicBtn.Attack,
                ),
            ),
            Move(
                name="heal_others",
                sequence=(
                    LogicBtn.Defend,
                    LogicBtn.Up,
                    LogicBtn.Jump,
                ),
            ),
            Move(
                name="energy_shield",
                sequence=(
                    LogicBtn.Defend,
                    LogicBtn.Dir,
                    LogicBtn.Jump,
                ),
            ),
            Move(
                name="energy_disk",
                sequence=(
                    LogicBtn.Defend,
                    LogicBtn.Up,
                    LogicBtn.Attack,
                ),
            ),
            Move(
                name="heal_myself",
                sequence=(
                    LogicBtn.Defend,
                    LogicBtn.Down,
                    LogicBtn.Jump,
                ),
            ),
        ),
        Characters.Deep: (
            *BASIC_MOVES,
            Move(
                name="energy_blast",
                sequence=(
                    LogicBtn.Defend,
                    LogicBtn.Dir,
                    LogicBtn.Attack,
                ),
            ),
            Move(
                name="strike",
                sequence=(
                    LogicBtn.Defend,
                    LogicBtn.Down,
                    LogicBtn.Attack,
                ),
            ),
            Move(
                name="leap_attack",
                sequence=(
                    LogicBtn.Defend,
                    LogicBtn.Up,
                    LogicBtn.Jump,
                    LogicBtn.Attack,
                ),
            ),
            Move(
                name="leap_attack2",
                sequence=(
                    LogicBtn.Defend,
                    LogicBtn.Down,
                    LogicBtn.Attack,
                    LogicBtn.Jump,
                    LogicBtn.Attack,
                ),
            ),
            Move(
                name="dash_strafe",
                sequence=(
                    LogicBtn.Defend,
                    LogicBtn.Dir,
                    LogicBtn.Jump,
                ),
            ),
        ),
        Characters.Henry: (
            *BASIC_MOVES,
            Move(
                name="dragon_palm",
                sequence=(
                    LogicBtn.Defend,
                    LogicBtn.Dir,
                    LogicBtn.Attack,
                ),
            ),
            Move(
                name="multiple_shot",
                sequence=(
                    LogicBtn.Defend,
                    LogicBtn.Jump,
                    LogicBtn.Attack,
                    LogicBtn.Attack,
                    LogicBtn.Attack,
                ),
            ),
            Move(
                name="critical_show",
                sequence=(
                    LogicBtn.Defend,
                    LogicBtn.Dir,
                    LogicBtn.Jump,
                ),
            ),
            Move(
                name="sonata_of_the_death",
                sequence=(
                    LogicBtn.Defend,
                    LogicBtn.Up,
                    LogicBtn.Jump,
                ),
            ),
        ),
        Characters.Rudolf: (
            *BASIC_MOVES,
            Move(
                name="leap_attack",
                sequence=(
                    LogicBtn.Defend,
                    LogicBtn.Dir,
                    LogicBtn.Jump,
                ),
            ),
            Move(
                name="multiple_ninja_star",
                sequence=(
                    LogicBtn.Defend,
                    LogicBtn.Dir,
                    LogicBtn.Attack,
                ),
            ),
            Move(
                name="hide",
                sequence=(
                    LogicBtn.Defend,
                    LogicBtn.Up,
                    LogicBtn.Jump,
                ),
            ),
            Move(
                name="double",
                sequence=(
                    LogicBtn.Defend,
                    LogicBtn.Down,
                    LogicBtn.Jump,
                ),
            ),
        ),
        Characters.Louis: (
            *BASIC_MOVES,
            Move(
                name="thunder_punch",
                sequence=(
                    *_Run.sequence,
                    LogicBtn.Attack,
                ),
            ),
            Move(
                name="thunder_punch2",
                sequence=(
                    *_Run.sequence,
                    LogicBtn.Jump,
                    LogicBtn.Attack,
                ),
            ),
            Move(
                name="thunder_kick",
                sequence=(
                    LogicBtn.Defend,
                    LogicBtn.Dir,
                    LogicBtn.Jump,
                ),
            ),
            Move(
                name="whirlwind_throw",
                sequence=(
                    LogicBtn.Defend,
                    LogicBtn.Up,
                    LogicBtn.Jump,
                ),
            ),
            Move(
                name="phoenix_palm",
                sequence=(
                    LogicBtn.Defend,
                    LogicBtn.Dir,
                    LogicBtn.Attack,
                ),
            ),
        ),
        Characters.Firen: (
            *BASIC_MOVES,
            Move(
                name="fire_ball",
                sequence=(
                    LogicBtn.Defend,
                    LogicBtn.Dir,
                    LogicBtn.Attack,
                    LogicBtn.Attack,
                    LogicBtn.Attack,
                ),
            ),
            Move(
                name="blaze",
                sequence=(
                    LogicBtn.Defend,
                    LogicBtn.Dir,
                    LogicBtn.Jump,
                ),
            ),
            Move(
                name="inferno",
                sequence=(
                    LogicBtn.Defend,
                    LogicBtn.Down,
                    LogicBtn.Jump,
                ),
            ),
            Move(
                name="explosion",
                sequence=(
                    LogicBtn.Defend,
                    LogicBtn.Up,
                    LogicBtn.Jump,
                ),
            ),
        ),
        Characters.Freeze: (
            *BASIC_MOVES,
            Move(
                name="ice_blast",
                sequence=(
                    LogicBtn.Defend,
                    LogicBtn.Dir,
                    LogicBtn.Attack,
                    LogicBtn.Attack,
                    LogicBtn.Attack,
                ),
            ),
            Move(
                name="ice_sword",
                sequence=(
                    LogicBtn.Defend,
                    LogicBtn.Down,
                    LogicBtn.Jump,
                ),
            ),
            Move(
                name="icicle",
                sequence=(
                    LogicBtn.Defend,
                    LogicBtn.Dir,
                    LogicBtn.Jump,
                ),
            ),
            Move(
                name="whirlwind",
                sequence=(
                    LogicBtn.Defend,
                    LogicBtn.Up,
                    LogicBtn.Jump,
                ),
            ),
        ),
        Characters.Dennis: (
            *BASIC_MOVES,
            Move(
                name="energy_blast",
                sequence=(
                    LogicBtn.Defend,
                    LogicBtn.Dir,
                    LogicBtn.Attack,
                ),
            ),
            Move(
                name="strafe",
                sequence=(
                    LogicBtn.Defend,
                    LogicBtn.Down,
                    LogicBtn.Attack,
                ),
            ),
            Move(
                name="whirlwind_kick",
                sequence=(
                    LogicBtn.Defend,
                    LogicBtn.Dir,
                    LogicBtn.Jump,
                ),
            ),
            Move(
                name="chasing_blast",
                sequence=(
                    LogicBtn.Defend,
                    LogicBtn.Up,
                    LogicBtn.Attack,
                ),
            ),
        ),
        Characters.Woody: (
            *BASIC_MOVES,
            Move(
                name="flip_kick",
                sequence=(
                    LogicBtn.Defend,
                    LogicBtn.Up,
                    LogicBtn.Attack,
                ),
            ),
            Move(
                name="turning_kick",
                sequence=(
                    LogicBtn.Defend,
                    LogicBtn.Down,
                    LogicBtn.Attack,
                ),
            ),
            Move(
                name="teleport_to_enemy",
                sequence=(
                    LogicBtn.Defend,
                    LogicBtn.Up,
                    LogicBtn.Jump,
                ),
            ),
            Move(
                name="teleport_to_friend",
                sequence=(
                    LogicBtn.Defend,
                    LogicBtn.Down,
                    LogicBtn.Jump,
                ),
            ),
            Move(
                name="energy_blast",
                sequence=(
                    LogicBtn.Defend,
                    LogicBtn.Dir,
                    LogicBtn.Attack,
                ),
            ),
            Move(
                name="tiger_dash",
                sequence=(
                    LogicBtn.Defend,
                    LogicBtn.Dir,
                    LogicBtn.Jump,
                ),
            ),
        ),
        Characters.Davis: (
            *BASIC_MOVES,
            Move(
                name="leap_attack",
                sequence=(
                    LogicBtn.Defend,
                    LogicBtn.Up,
                    LogicBtn.Jump,
                    LogicBtn.Attack,
                ),
            ),
            Move(
                name="energy_blast",
                sequence=(
                    LogicBtn.Defend,
                    LogicBtn.Dir,
                    LogicBtn.Attack,
                ),
            ),
            Move(
                name="strafe",
                sequence=(
                    LogicBtn.Defend,
                    LogicBtn.Down,
                    LogicBtn.Attack,
                ),
            ),
            Move(
                name="dragon_punch",
                sequence=(
                    LogicBtn.Defend,
                    LogicBtn.Up,
                    LogicBtn.Attack,
                ),
            ),
        ),
        Characters.Jan: (
            *BASIC_MOVES,
            Move(
                name="devils_judgement",
                sequence=(
                    LogicBtn.Defend,
                    LogicBtn.Up,
                    LogicBtn.Attack,
                ),
            ),
            Move(
                name="angels_blessing",
                sequence=(
                    LogicBtn.Defend,
                    LogicBtn.Up,
                    LogicBtn.Jump,
                ),
            ),
        ),
        Characters.Bat: (
            *BASIC_MOVES,
            Move(
                name="speed_punch",
                sequence=(
                    LogicBtn.Defend,
                    LogicBtn.Dir,
                    LogicBtn.Jump,
                ),
            ),
            Move(
                name="eye_laser",
                sequence=(
                    LogicBtn.Defend,
                    LogicBtn.Dir,
                    LogicBtn.Attack,
                ),
            ),
            Move(
                name="sommon_bats",
                sequence=(
                    LogicBtn.Defend,
                    LogicBtn.Up,
                    LogicBtn.Jump,
                ),
            ),
        ),
        Characters.Julian: (
            *BASIC_MOVES,
            Move(
                name="soul_punch",
                sequence=(
                    *_Run.sequence,
                    LogicBtn.Attack,
                ),
            ),
            Move(
                name="uppercut",
                sequence=(
                    LogicBtn.Defend,
                    LogicBtn.Up,
                    LogicBtn.Attack,
                ),
            ),
            Move(
                name="skull_blast",
                sequence=(
                    LogicBtn.Defend,
                    LogicBtn.Dir,
                    LogicBtn.Attack,
                ),
            ),
            Move(
                name="mirror_image",
                sequence=(
                    LogicBtn.Defend,
                    LogicBtn.Jump,
                    LogicBtn.Attack,
                    LogicBtn.Jump,
                ),
            ),
            Move(
                name="big_bang",
                sequence=(
                    LogicBtn.Defend,
                    LogicBtn.Up,
                    LogicBtn.Jump,
                ),
            ),
            Move(
                name="soul_bomb",
                sequence=(
                    LogicBtn.Defend,
                    LogicBtn.Dir,
                    LogicBtn.Jump,
                ),
            ),
        ),
        Characters.Firzen: (
            *BASIC_MOVES,
            Move(
                name="firzen_cannon",
                sequence=(
                    LogicBtn.Defend,
                    LogicBtn.Dir,
                    LogicBtn.Jump,
                ),
            ),
            Move(
                name="overwhelming_disaster",
                sequence=(
                    LogicBtn.Defend,
                    LogicBtn.Up,
                    LogicBtn.Attack,
                ),
            ),
            Move(
                name="arctic_volcano",
                sequence=(
                    LogicBtn.Defend,
                    LogicBtn.Up,
                    LogicBtn.Jump,
                ),
            ),
        ),
        Characters.Jack: (
            *BASIC_MOVES,
            Move(
                name="energy_blast",
                sequence=(
                    LogicBtn.Defend,
                    LogicBtn.Dir,
                    LogicBtn.Attack,
                ),
            ),
            Move(
                name="flash_kick",
                sequence=(
                    LogicBtn.Defend,
                    LogicBtn.Up,
                    LogicBtn.Attack,
                ),
            ),
        ),
        Characters.Justin: (
            *BASIC_MOVES,
            Move(
                name="wolf_punch",
                sequence=(
                    LogicBtn.Defend,
                    LogicBtn.Down,
                    LogicBtn.Attack,
                ),
            ),
            Move(
                name="energy_blast",
                sequence=(
                    LogicBtn.Defend,
                    LogicBtn.Dir,
                    LogicBtn.Attack,
                ),
            ),
        ),
        Characters.LouisEX: (
            *BASIC_MOVES,
            Move(
                name="phoenix_dance",
                sequence=(
                    LogicBtn.Defend,
                    LogicBtn.Down,
                    LogicBtn.Attack,
                ),
            ),
            Move(
                name="thunder_punch",
                sequence=(
                    *_Run.sequence,
                    LogicBtn.Attack,
                ),
            ),
            Move(
                name="thunder_punch2",
                sequence=(
                    *_Run.sequence,
                    LogicBtn.Jump,
                    LogicBtn.Attack,
                ),
            ),
            Move(
                name="phoenix_palm",
                sequence=(
                    LogicBtn.Defend,
                    LogicBtn.Dir,
                    LogicBtn.Attack,
                    LogicBtn.Attack,
                    LogicBtn.Attack,
                ),
            ),
        ),
        Characters.Mark: (
            *BASIC_MOVES,
            Move(
                name="crash_punch",
                sequence=(
                    LogicBtn.Defend,
                    LogicBtn.Dir,
                    LogicBtn.Attack,
                ),
            ),
            Move(
                name="body_attack",
                sequence=(
                    LogicBtn.Defend,
                    LogicBtn.Dir,
                    LogicBtn.Jump,
                ),
            ),
        ),
        Characters.Monk: (
            *BASIC_MOVES,
            Move(
                name="shaolin_palm",
                sequence=(
                    LogicBtn.Defend,
                    LogicBtn.Dir,
                    LogicBtn.Attack,
                ),
            ),
        ),
        Characters.Sorcerer: (
            *BASIC_MOVES,
            Move(
                name="ice_blast",
                sequence=(
                    LogicBtn.Defend,
                    LogicBtn.Dir,
                    LogicBtn.Jump,
                ),
            ),
            Move(
                name="fire_ball",
                sequence=(
                    LogicBtn.Defend,
                    LogicBtn.Dir,
                    LogicBtn.Attack,
                    LogicBtn.Attack,
                    LogicBtn.Attack,
                ),
            ),
            Move(
                name="heal_others",
                sequence=(
                    LogicBtn.Defend,
                    LogicBtn.Up,
                    LogicBtn.Jump,
                ),
            ),
            Move(
                name="heal_myself",
                sequence=(
                    LogicBtn.Defend,
                    LogicBtn.Down,
                    LogicBtn.Jump,
                ),
            ),
        ),
    }
)
