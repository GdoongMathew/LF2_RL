"""Windows-only LF2 game I/O.

This sub-package holds every module that reaches into OS-specific
machinery — ``win32*``, ``pymem``, ``pyautogui``, ``mss``, ``cv2`` — and is
intentionally segregated from the pure ``lf2_gym`` core so the Linux learner
(in async DGX training) never has to load any of it.

Importing ``lf2_gym.windows`` on a non-Windows platform raises immediately
so the mistake surfaces early; cross-platform code that doesn't need live
game I/O should depend on :mod:`lf2_gym.lf2_envs.base` instead and inject
its own controller.
"""

from __future__ import annotations

import sys

if sys.platform != "win32":  # pragma: no cover - guard
    raise ImportError(
        "lf2_gym.windows is a Windows-only sub-package; on this platform "
        "import lf2_gym.lf2_envs.base directly and supply your own "
        "controller implementation."
    )
