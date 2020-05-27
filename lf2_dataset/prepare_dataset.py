from getkeys import key_check
import os
import gym
import lf2_gym
import cv2
import json
from datetime import datetime
import keyboard


class DataConverter:
    def __init__(self, env, path, filename, video_type='mp4'):
        self.env = env
        self.path = path
        self.stop = False
        self.filename = filename
        self.video_type = video_type
        self.file_num = 0

        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def start_recording(self):
        img_list = []
        state_list = []
        reward_list = []
        while not self.stop:
            img_list.append(lf2_env.render(mode='rgb_array'))
            state_list.append(lf2_env.get_state().tolist())
            reward_list.append(lf2_env.get_reward())
            if self.env.game_over:
                self.write_file(img_list, state_list, reward_list, None)
                img_list.clear()
                state_list.clear()
                reward_list.clear()
                self.env.reset()

    def action_output(self):
        pass

    def write_file(self, imgs, states, reward, key_press):
        json_item = json.dumps(states)
        now = datetime.now()
        file_name = f"{self.filename}_{self.file_num}_{now.strftime('%Y_%m_%d_%H_%M')}"
        size = imgs[0].shape[:2]
        print(size)
        out = cv2.VideoWriter(os.path.join(self.path, f'{file_name}.{self.video_type}'),
                              cv2.VideoWriter_fourcc(*'MP4V'), 30, (size[0], size[1]))
        for im in imgs:
            out.write(im)

        print('Video wrote.')
        out.release()

        with open(os.path.join(self.path, f'{file_name}.json'), 'w') as file:
            json.dump(json_item, file)
        print('State json wrote.')

mode = 'info'
karg = dict(frame_stack=1, frame_skip=0, reset_skip_sec=2, mode=mode)
lf2_env = gym.make('LittleFighter2-v0', **karg)

obs = lf2_env.reset()

now = datetime.now()
d_converter = DataConverter(lf2_env, f"E:\LF2_Dataset\{now.strftime('%Y_%m_%d')}", lf2_env.my_player.name)
d_converter.start_recording()