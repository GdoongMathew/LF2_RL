import logging
import os
import gymnasium as gym
import lf2_gym

# from lf2_gym.lf2_envs.LF2_Env import Lf2Env


from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3 import A2C, DQN, PPO
from lf2_gym.loggers import get_logger

logger = get_logger()
logger.setLevel(logging.INFO)

# import numpy as np


def make_env(env_id, **kwargs):
    """
    Utility function for multiprocessed env.
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param rank: (int) index of the subprocess
    """

    def _init():
        env = gym.make(env_id, **kwargs)
        return env

    return _init


def main():

    env_id = "LittleFighter2-v0"
    kwargs = dict(
        frame_stack=3,
        frame_skip=1,
        reset_skip_sec=2,
        mode="mix",
        gray_scale=False,
        player_id=0,
    )
    num_cpu = 1

    lf2_env = gym.make(env_id, **kwargs)

    # discount factor
    gamma = 0.95
    # #
    # lf2_env = SubprocVecEnv([make_env(env_id, **karg) for i in range(num_cpu)])
    save_root = r"D:\log\lf2"
    model = PPO(
        "MultiInputPolicy",
        lf2_env,
        verbose=1,
        batch_size=32,
        # prioritized_replay=True,
        gamma=gamma,
        tensorboard_log=os.path.join(save_root, "tensorboard")
    )
    #
    print("Start learning")
    model.learn(total_timesteps=6000000)
    model.save(save_root)
    #

    # model = PPO2.load(save_root)
    obs = lf2_env.reset()

    done = False
    while not done:

        # model prediction
        actions, _states = model.predict(obs)
        obs, reward, done, info = lf2_env.step(actions)
        # print(info)
        lf2_env.render("console")
        if done:
            _ = lf2_env.reset()


if __name__ == "__main__":
    main()
