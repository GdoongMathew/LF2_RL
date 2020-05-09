import gym
import lf2_gym

mode = 'mix'
karg = dict(frame_stack=3, frame_skip=1, reset_skip_sec=2, mode=mode)
lf2_env = gym.make('LittleFighter2-v0', **karg)

