"""Pure reward shaping for LF2 — no Windows / no live-game I/O dependencies.

Used by both the single-agent and multi-agent envs, and importable on the
Linux learner side without dragging ``win32*`` / ``pymem`` / ``pyautogui``.

Reward design (delta-based, scale-balanced):

* **HP delta** — damage dealt to the enemy team minus damage taken by my
  team, each ``max(0, …)``-clamped so round resets / counter rollbacks
  don't produce phantom signals, then normalized by ``my_player.hp_max``
  so a full-character-HP swing is roughly ``±1``.
* **Hit bonus** — ``+HIT_BONUS`` per increment in ``my_player.attacks``.
  This is the densest *causal* signal (the action just landed a hit) and
  is scaled to the same magnitude as HP-delta so the network actually
  notices it.
* **Death penalty** — *one-shot* ``-DEATH_PENALTY`` only on the step
  ``alive → dead``, not every step the player is dead.
* **Time penalty** — optional small per-step decay; off by default.

Replaces the previous level-based ``team_avg - enemy_avg`` (which paid
out the same big number every step regardless of action), the inverted
``mp_low → +reward`` term, and the repeated ``-50`` death penalty.
"""

from __future__ import annotations

from dataclasses import dataclass

from lf2_gym.keymap import PlayerProtocol

# Reward weights — module-level so callers can monkey-patch / tune them
# without modifying this file or threading them through every step.
HP_DELTA_WEIGHT: float = 1.0  # (enemy_loss − team_loss) / hp_max
HIT_BONUS: float = 0.5  # per increment in my_player.attacks
DEATH_PENALTY: float = 10.0  # one-shot on alive → dead transition
TIME_PENALTY: float = 0.0  # subtracted every step; 0 = disabled


@dataclass(frozen=True)
class RewardState:
    """Cross-step accumulator needed to turn raw player state into deltas.

    The env keeps one of these per agent between ``step()`` calls and a
    fresh ``RewardState()`` after every ``reset()``. :func:`compute_reward`
    is purely functional in ``prev`` and returns the updated state — the
    caller stores it back verbatim.
    """

    enemy_hp: int = 0
    team_hp: int = 0
    attacks: int = 0
    is_alive: bool = True


def compute_reward(
    my_player: PlayerProtocol,
    active_players: tuple[PlayerProtocol, ...],
    prev: RewardState,
) -> tuple[float, RewardState]:
    """Per-player delta-based reward shaping.

    Returns ``(reward, updated_state)``. Pass ``updated_state`` back in
    as ``prev`` on the next step.
    """
    enemy_hp = 0
    team_hp = 0
    for p in active_players:
        if p.team == my_player.team:
            team_hp += p.hp
        else:
            enemy_hp += p.hp

    # Δ damage — clamp to ≥ 0 so round resets / counter rollbacks don't
    # generate phantom penalties / bonuses.
    enemy_loss = max(0, prev.enemy_hp - enemy_hp)
    team_loss = max(0, prev.team_hp - team_hp)
    hit_delta = max(0, my_player.attacks - prev.attacks)

    # Death is a single event: the step that crossed alive → dead.
    just_died = prev.is_alive and not my_player.is_alive

    hp_max = my_player.hp_max or 1  # defensive: avoid div-by-zero pre-init
    reward = (
        HP_DELTA_WEIGHT * (enemy_loss - team_loss) / hp_max
        + HIT_BONUS * hit_delta
        - DEATH_PENALTY * float(just_died)
        - TIME_PENALTY
    )

    return reward, RewardState(
        enemy_hp=enemy_hp,
        team_hp=team_hp,
        attacks=my_player.attacks,
        is_alive=my_player.is_alive,
    )
