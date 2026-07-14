"""Verify batched read_block + struct parsing equals per-field read_int."""
import pytest

from tests.conftest import (
    POS_FIELDS,
    STAT_FIELDS,
    block_len,
    parse_block,
)
from lf2_gym.const import LF2_BLOCK_POS, LF2_BLOCK_STAT

pytestmark = pytest.mark.requires_lf2


def _read_legacy(proc_wr, base: int) -> dict[str, int]:
    return {name: proc_wr.read_int(base + int(off)) for name, off in STAT_FIELDS + POS_FIELDS}


def _read_batched(proc_wr, base: int) -> dict[str, int]:
    stat_start = int(LF2_BLOCK_STAT[0])
    pos_start = int(LF2_BLOCK_POS[0])
    stat_buf = proc_wr.read_block(base + stat_start, block_len(LF2_BLOCK_STAT))
    pos_buf = proc_wr.read_block(base + pos_start, block_len(LF2_BLOCK_POS))
    return {
        **parse_block(stat_buf, stat_start, STAT_FIELDS),
        **parse_block(pos_buf, pos_start, POS_FIELDS),
    }


def test_block_lengths_are_sane():
    """Guard against the (start, end) vs (start, len) mix-up."""
    assert block_len(LF2_BLOCK_STAT) == 0x368 - 0x2FC + 4 == 0x70
    assert block_len(LF2_BLOCK_POS) == 0x18 - 0x10 + 4 == 0x0C


def test_read_block_returns_expected_size(proc_wr, player):
    base = player.address
    stat_buf = proc_wr.read_block(base + int(LF2_BLOCK_STAT[0]), block_len(LF2_BLOCK_STAT))
    pos_buf = proc_wr.read_block(base + int(LF2_BLOCK_POS[0]), block_len(LF2_BLOCK_POS))
    assert len(stat_buf) == block_len(LF2_BLOCK_STAT)
    assert len(pos_buf) == block_len(LF2_BLOCK_POS)


def test_batched_matches_legacy(proc_wr, player):
    """Same player, two read strategies -> identical values per field.

    Retries a few times to tolerate the rare race when the game state
    changes between the two reads. Run on a menu/paused screen for best
    stability.
    """
    base = player.address

    last_mismatch: dict[str, tuple[int, int]] = {}
    for _ in range(5):
        legacy = _read_legacy(proc_wr, base)
        batched = _read_batched(proc_wr, base)
        last_mismatch = {k: (legacy[k], batched[k]) for k in legacy if legacy[k] != batched[k]}
        if not last_mismatch:
            break

    assert not last_mismatch, (
        "Field mismatch between legacy and batched reads "
        f"(offset math wrong if persistent): {last_mismatch}"
    )


@pytest.mark.parametrize("field_name,offset", STAT_FIELDS)
def test_each_stat_field_individually(proc_wr, player, field_name, offset):
    """Per-field parametrized check -> pinpoints exactly which offset is off."""
    base = player.address
    stat_start = int(LF2_BLOCK_STAT[0])
    stat_buf = proc_wr.read_block(base + stat_start, block_len(LF2_BLOCK_STAT))

    import struct

    batched_val = struct.unpack_from("<i", stat_buf, int(offset) - stat_start)[0]
    legacy_val = proc_wr.read_int(base + int(offset))
    assert batched_val == legacy_val, f"{field_name}: batched={batched_val} legacy={legacy_val}"