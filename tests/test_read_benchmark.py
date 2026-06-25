"""Benchmark legacy per-field reads vs batched block reads.

Marked `benchmark`; by default it only *reports* and asserts a very loose
sanity bound. Set LF2_STRICT_BENCH=1 to enforce a real speedup threshold.
"""

import os
import statistics
import struct
import time

import pytest

from tests.conftest import block_len
from lf2_gym.const import (
    LF2PlayerAddressOffset as OFF,
    LF2_BLOCK_POS,
    LF2_BLOCK_STAT,
)

pytestmark = [pytest.mark.requires_lf2, pytest.mark.benchmark]

N = int(os.environ.get("LF2_BENCH_N", "1000"))
STAT_OFFSETS = [
    OFF.HP,
    OFF.HP_DARK,
    OFF.HP_LOST,
    OFF.MP,
    OFF.MP_USAGE,
    OFF.KILLS,
    OFF.ATTACK,
    OFF.PICKING,
    OFF.OWNER,
    OFF.ENEMY,
    OFF.TEAM,
]
POS_OFFSETS = [OFF.X_POS, OFF.Y_POS, OFF.Z_POS]


def _legacy(pwr, base):
    for off in STAT_OFFSETS:
        pwr.read_int(base + int(off))
    for off in POS_OFFSETS:
        pwr.read_int(base + int(off))
    pwr.read_bytes(base + int(OFF.FACING), 1)


def _batched(pwr, base):
    stat_start = int(LF2_BLOCK_STAT[0])
    pos_start = int(LF2_BLOCK_POS[0])
    stat_buf = pwr.read_block(base + stat_start, block_len(LF2_BLOCK_STAT))
    pos_buf = pwr.read_block(base + pos_start, block_len(LF2_BLOCK_POS))
    for off in STAT_OFFSETS:
        struct.unpack_from("<i", stat_buf, int(off) - stat_start)
    for off in POS_OFFSETS:
        struct.unpack_from("<i", pos_buf, int(off) - pos_start)
    pwr.read_bytes(base + int(OFF.FACING), 1)


def _time(fn, pwr, base, n=N) -> list[float]:
    for _ in range(50):  # warmup
        fn(pwr, base)
    out = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn(pwr, base)
        out.append((time.perf_counter() - t0) * 1e6)  # microseconds
    return out


def test_batched_is_faster(proc_wr, player, capsys):
    base = player.address

    legacy = _time(_legacy, proc_wr, base)
    batched = _time(_batched, proc_wr, base)

    m_legacy = statistics.mean(legacy)
    m_batched = statistics.mean(batched)
    speedup = m_legacy / m_batched if m_batched else float("inf")

    with capsys.disabled():
        print(f"\n[bench N={N}] legacy  mean={m_legacy:8.2f}us " f"median={statistics.median(legacy):8.2f}us")
        print(f"[bench N={N}] batched mean={m_batched:8.2f}us " f"median={statistics.median(batched):8.2f}us")
        print(f"[bench N={N}] speedup={speedup:.2f}x " f"(save {m_legacy - m_batched:.2f}us/call)\n")

    # Loose default sanity: batched must not be dramatically slower.
    assert m_batched <= m_legacy * 1.5

    if os.environ.get("LF2_STRICT_BENCH") == "1":
        assert speedup >= 2.0, f"expected >=2x speedup, got {speedup:.2f}x"
