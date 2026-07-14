# http://www.cheatbook.de/files/littlef2.htm
from dataclasses import dataclass
from enum import StrEnum
from collections import defaultdict



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
