import gym
import lf2_gym
import time
import numpy as np


Total_Episode = 1000


def main():

    lf2_env = gym.make('LittleFighter2-v0', windows_scale=1.25)
    # lf2_env = Lf2Env(windows_name='Little Fighter 2', windows_scale=1.25)
    lf2_env.reset()

    done = False
    while not done:

        # obs, reward, done, info = lf2_env.step(lf2_env.action_space.sample())
        if lf2_env.game_over:
            lf2_env.reset()


if __name__ == '__main__':
    main()

