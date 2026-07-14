from gymnasium.envs.registration import register

# ``entry_point`` is a string that gym only resolves the first time
# ``gym.make("LittleFighter2-v0")`` is called — so the learner-side
# ``import lf2_gym`` on Linux does *not* trigger any Windows-only imports.
register(
    id="LittleFighter2-v0",
    entry_point="lf2_gym.windows.env:Lf2Env",
)


def make_parallel_env(**kwargs):
    """Construct the multi-agent PettingZoo ``Lf2ParallelEnv``.

    Only usable on Windows: pulls in the live game controller via
    :mod:`lf2_gym.windows.parallel_env`.
    """
    from lf2_gym.windows.parallel_env import Lf2ParallelEnv

    return Lf2ParallelEnv(**kwargs)
