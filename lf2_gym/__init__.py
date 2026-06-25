from gymnasium.envs.registration import register

register(
    id="LittleFighter2-v0",
    entry_point="lf2_gym.lf2_envs.env:Lf2Env",
)
