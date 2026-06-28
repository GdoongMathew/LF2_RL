"""Physical keystroke delivery via ``pyautogui``.

The pure ``lf2_gym.keymap`` module resolves a logical :class:`Move` into a
list of key strings; this module is what actually pushes those keys to the
OS. Lives on the Windows side because ``pyautogui.keyDown`` / ``keyUp``
require a real OS keyboard subsystem.
"""

from __future__ import annotations

import time

import pyautogui


def press_key(keys: list[str], interval: float = 0.1) -> None:
    """Hold ``keys`` down for ``interval`` seconds, then release all of them.

    Consecutive identical keys get an extra ``keyUp`` first so the second
    ``keyDown`` is registered as a fresh key event by the game.
    """
    last_key = ""
    if keys is None:
        return
    for key in keys:
        if key == last_key:
            # to prevent not sending key event if two consecutive identical keys.
            pyautogui.keyUp(key)
        pyautogui.keyDown(key)
        last_key = key
    time.sleep(interval)
    for key in keys:
        pyautogui.keyUp(key)
