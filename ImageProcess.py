import gym
import lf2_gym
import time
import numpy as np


def main():

    lf2_env = gym.make('LittleFighter2-v0', windows_scale=1.25)

    # lf2_gym = Lf2Env(windows_name='Little Fighter 2', windows_scale=1.25)
    lf2_env.reset()
    time.sleep(0.5)

    done = False
    now = time.time()

    while not done:
        obs, reward, done, info = lf2_env.step(lf2_env.action_space.sample())

        time.sleep(1)

        if lf2_env.game_over:
            lf2_env.reset()
            print('reset')

        if time.time() - now > 120:
            lf2_env.kill_thread = True
            break


if __name__ == '__main__':
    main()

