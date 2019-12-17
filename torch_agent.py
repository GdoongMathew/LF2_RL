import torch
import gym
import lf2_gym
import time

env_id = 'LittleFighter2-v0'
karg = dict(frame_stack=1, frame_skip=0, reset_skip_sec=2, mode='mixY')


def main():
    lf2_env = gym.make(env_id, **karg)
    now = time.time()
    lf2_env.reset()

    done = False
    while not done:
        obs, reward, done, info = lf2_env.step(0)

        print(obs['Info'])

    lf2_env.close()


if __name__ == '__main__':
    main()