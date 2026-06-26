from gymnasium.envs.registration import register

register(
    id="LittleFighter2-v0",
    entry_point="lf2_gym.lf2_envs.env:Lf2Env",
)


def make_parallel_env(**kwargs):
    """Construct the multi-agent PettingZoo ``Lf2ParallelEnv``.

    Imported lazily so the (optional) ``pettingzoo`` dependency is only required
    when the multi-agent environment is actually used.
    """
    from lf2_gym.lf2_envs.parallel_env import Lf2ParallelEnv

    return Lf2ParallelEnv(**kwargs)
