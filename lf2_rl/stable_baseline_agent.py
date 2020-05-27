import gym
import os
import lf2_gym


# from lf2_gym.lf2_envs.LF2_Env import Lf2Env

from stable_baselines.deepq import LnMlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import A2C, PPO2, DQN
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

    env_id = 'LittleFighter2-v0'
    karg = dict(frame_stack=3, frame_skip=1, reset_skip_sec=2, mode='picture', gray_scale=True)
    num_cpu = 1

    # lf2_env = Lf2Env(**karg)

    # lf2_env = gym.make(env_id, **karg)

    # discount factor
    gamma = 0.95
    # #
    lf2_env = SubprocVecEnv([make_env(env_id, **karg) for i in range(num_cpu)])
    save_root = r'LF2_RL_Model'
    model = DQN(LnMlpPolicy,
                lf2_env,
                verbose=1,
                batch_size=60,
                prioritized_replay=True,
                gamma=gamma,
                full_tensorboard_log=True,
                tensorboard_log=os.path.join(save_root, 'tensorboard')
                )
    #
    print('Start learning')
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
        lf2_env.render('console')
        if done:
            _ = lf2_env.reset()


if __name__ == '__main__':
    main()

