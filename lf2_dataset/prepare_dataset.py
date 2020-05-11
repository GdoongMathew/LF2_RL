import gym
import lf2_gym
import cv2

class DataCapture:
    def __init__(self, path):
        self.path = path

    def action_output(self, basic_action=False):
        if basic_action:

mode = 'mix'
karg = dict(frame_stack=3, frame_skip=1, reset_skip_sec=2, mode=mode)
lf2_env = gym.make('LittleFighter2-v0', **karg)
