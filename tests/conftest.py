"""Shared fixtures for LF2 memory-read tests.

All tests that touch the live game are marked `requires_lf2` and are
auto-skipped when LF2 (or its Windows-only deps) is unavailable, so the
suite still runs in CI with `-m "not requires_lf2"`.
"""
import struct

import pytest

# Windows-only deps: skip the whole module cleanly if missing (e.g. Linux CI).
pytest.importorskip("pymem")
pytest.importorskip("win32process")

from lf2_gym.const import (  # noqa: E402  (after importorskip on purpose)
    LF2PlayerAddressOffset as OFF,
    LF2_BLOCK_POS,
    LF2_BLOCK_STAT,
)

LF2_WINDOW_NAME = "Little Fighter 2"


# ---- field tables shared by verify + benchmark ----
STAT_FIELDS: list[tuple[str, int]] = [
    ("Hp", OFF.HP),
    ("Hp_Dark", OFF.HP_DARK),
    ("Hp_Lost", OFF.HP_LOST),
    ("Mp", OFF.MP),
    ("Mp_Usage", OFF.MP_USAGE),
    ("Kills", OFF.KILLS),
    ("Attack", OFF.ATTACK),
    ("Picking", OFF.PICKING),
    ("Owner", OFF.OWNER),
    ("Enemy", OFF.ENEMY),
    ("Team", OFF.TEAM),
]
POS_FIELDS: list[tuple[str, int]] = [
    ("x_pos", OFF.X_POS),
    ("y_pos", OFF.Y_POS),
    ("z_pos", OFF.Z_POS),
]


def block_len(block: tuple[int, int]) -> int:
    """Return a SAFE byte length for a (start, X) block tuple.

    Handles both conventions:
      * (start, length)  -> use length directly
      * (start, end_off) -> derive length = end - start + 4 (incl. the int32)
    If the 2nd element is >= start it's treated as an *end offset*.
    """
    start, second = int(block[0]), int(block[1])
    if second >= start:  # looks like an end-offset (current const.py case)
        return second - start + 4
    return second  # already a length


def parse_block(buf: bytes, base: int, fields: list[tuple[str, int]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for name, off in fields:
        rel = int(off) - int(base)
        out[name] = struct.unpack_from("<i", buf, rel)[0]
    return out


@pytest.fixture(scope="session")
def lf2_hwnd():
    """Window handle of a running LF2, or skip the test if not found."""
    from lf2_gym.lf2_envs.winguiauto import winguiauto as winauto

    try:
        return winauto.findTopWindow(wantedText=LF2_WINDOW_NAME)
    except Exception as exc:  # WinGuiAutoError or anything else => not available
        pytest.skip(f"Little Fighter 2 not running ({type(exc).__name__}: {exc})")


@pytest.fixture(scope="session")
def proc_wr(lf2_hwnd):
    """A ProcessWR bound to the live game (keyword-only win_handle)."""
    from lf2_gym.lf2_envs.utils import ProcessWR

    return ProcessWR(win_handle=lf2_hwnd)


@pytest.fixture(scope="session")
def player(lf2_hwnd):
    """A com Player(idx=0). Skips if it cannot be constructed/read."""
    from lf2_gym.lf2_envs.utils import Player

    try:
        return Player(game_hwnd=lf2_hwnd, index=0, is_computer=True)
    except Exception as exc:
        pytest.skip(f"Cannot construct Player(0): {type(exc).__name__}: {exc}")