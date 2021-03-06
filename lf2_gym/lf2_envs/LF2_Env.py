from lf2_gym.lf2_envs.LF2_Util import *
from mss import mss
from win32api import GetSystemMetrics
import numpy as np
from collections import deque
import win32gui
import win32con
import win32ui
import pyautogui
import time
import cv2
import threading

from gym import spaces
import gym


pyautogui.FAILSAFE = False


def split_one(num_interval=1):
    num_list = np.sinh(list(i + 1 for i in range(num_interval)), dtype=np.float)
    return num_list / np.sum(num_list)


class Lf2Env(gym.Env):
    """
    Crop a image from the gaming window, and return all players info as well as
    the current image shown on the display.
    """
    metadata = {'render.modes': ['human', 'console', 'rgb_array']}

    def __init__(self,
                 windows_name='Little Fighter 2',
                 player_id=1,
                 downscale=2,
                 frame_stack=4,
                 frame_skip=1,
                 reset_skip_sec=2,
                 gray_scale=True,
                 mode='mix',
                 basic_action=False):
        """
        Initialize the gym environment
        :param windows_name: name of the windows
        :param player_id: training player id
        :param downscale: downscale ratio
        :param frame_stack: number of frames should be stacked
        :param frame_skip: number of frames to skip before stacking.
        :param reset_skip_sec: immortal time (sec)
        :param gray_scale: convert recording image from rgb to gray scale
        :param mode: observation output mode.
        :param basic_action: only output L,R,U,D,A,D,J
        """
        super(Lf2Env, self).__init__()

        self.window_name = windows_name
        self.game_hwnd = winauto.findTopWindow(wantedText=windows_name)
        self.PyCWnd1 = win32ui.FindWindow(None, windows_name)
        self.PyCWnd1.SetForegroundWindow()
        self.PyCWnd1.SetFocus()
        self.kill_thread = False
        self.basic_action = basic_action
        self.sct = mss()

        self.players = {0: None, 1: None, 2: None, 3: None,
                        4: None, 5: None, 6: None, 7: None}

        num_players = self.find_players()
        self.my_player_id = player_id
        self.my_player = self.players[player_id]

        self.gaming_screen = None
        self.game_over = False
        self.restart = True

        self.img_h = 0
        self.img_w = 0
        self.frame_skip = frame_skip
        self.downscale = downscale
        self.gray_scale = gray_scale

        self.frame_stack = frame_stack
        self.img_weights = split_one(self.frame_stack)
        self.frames = deque([], maxlen=self.frame_stack)
        # Immortal seconds before every rounds.
        self.reset_skip_sec = reset_skip_sec

        self.recording_thread = threading.Thread(target=self.update_game_img, daemon=True).start()
        self.player_thread = threading.Thread(target=self.update_players, daemon=True).start()

        self.action_space = spaces.Discrete(len(self.get_action_space()))
        self.mode = mode
        self.reward = 0
        self.bot_attack = 0
        while True:
            if self.gray_scale:
                img_last_dim = 1
            else:
                img_last_dim = 3
            if len(self.frames) != 0:
                # my_mp, my_hp, my_facing, my_x, my_y, my_z, [enemy_x, enemy_y, enemy_z]
                low = [0, 0, 0, 0, 0, -np.inf] * num_players
                high = [self.my_player.Mp_Max, self.my_player.Hp_Max, 1, np.inf, np.inf, 0] * num_players
                info = spaces.Box(low=np.array(low), high=np.array(high), dtype=np.int32)

                if self.mode == 'mix':
                    self.observation_space = spaces.Dict({
                        'Info': info,
                        'Game_Screen': spaces.Box(low=0, high=255,
                                                  shape=(self.img_h, self.img_w, img_last_dim))
                    })
                elif self.mode == 'info':
                    self.observation_space = info
                elif self.mode == 'picture':
                    self.observation_space = spaces.Box(low=0, high=255,
                                                        shape=(self.img_h, self.img_w, img_last_dim))
                else:
                    raise ValueError('Not Supported mode.... Exiting.')
                break
        print('Lf2 Environment initialized.')

    def find_players(self):
        num_player = 0
        for i, item in enumerate(self.players.items()):
            p_c = Player(self.game_hwnd, i, com=True)
            p_h = Player(self.game_hwnd, i)
            if p_c.is_active:
                # computer player
                self.players[i] = p_c
                num_player += 1
            elif p_h.is_active:
                # human player
                self.players[i] = p_h
                num_player += 1
            else:
                self.players[i] = None
        del p_c, p_h
        return num_player

    def get_state(self):
        # return the current state of the game
        if self.mode in ['picture', 'mix']:
            while len(self.frames) < self.frame_stack:
                time.sleep(0.001)

        if self.mode == 'picture':
            img_stack = np.stack(self.frames, axis=-1)
            img_stack = np.multiply(img_stack, split_one(self.frame_stack))
            ob = np.sum(img_stack, axis=-1)

        elif self.mode == 'mix':
            # my_mp, my_hp, my_facing, my_x, my_y, my_z, [enemy_x, enemy_y, enemy_z]
            info = [self.my_player.Mp, self.my_player.Hp, int(self.my_player.facing_bool),
                    self.my_player.x_pos, self.my_player.y_pos, self.my_player.z_pos]
            for i in self.players:
                if self.players[i] is None or i == self.my_player_id:
                    continue

                info += [self.players[i].Mp, self.players[i].Hp, int(self.players[i].facing_bool),
                         self.players[i].x_pos, self.players[i].y_pos, self.players[i].z_pos]

            # img_stack = np.stack(self.frames, axis=-1)
            _imgs = np.array(self.frames)
            img_stack = self.img_weights[0] * _imgs[0] + self.img_weights[1] * _imgs[1] + self.img_weights[2] * _imgs[2]
            # cv2.imshKey(1)
            ob = dict(Game_Screen=img_stack.astype(np.int8),
                      Info=np.array(info))

        else:
            # info mode
            info = [self.my_player.Mp, self.my_player.Hp, int(self.my_player.facing_bool),
                    self.my_player.x_pos, self.my_player.y_pos, self.my_player.z_pos]
            for i in self.players:
                if self.players[i] is None or i == self.my_player_id:
                    continue
                info += [self.players[i].Mp, self.players[i].Hp, int(self.players[i].facing_bool),
                         self.players[i].x_pos, self.players[i].y_pos, self.players[i].z_pos]

            ob = np.array(info)

        return ob

    def update_game_img(self):
        """
        Update the current gaming scene
        """
        skip_i = 0
        last_frame = np.array([0])
        while not self.kill_thread:
            tup = win32gui.GetWindowPlacement(self.game_hwnd)

            # check if the windows is in max size.
            if tup[1] == win32con.SW_SHOWMAXIMIZED:
                w = GetSystemMetrics(0)
                h = GetSystemMetrics(1)
                rect = [0, 0, w, h]
            elif tup[1] == win32con.SW_SHOWNORMAL:
                rect = list(win32gui.GetWindowRect(self.game_hwnd))
            else:
                continue
            h = rect[3] - rect[1]
            pos = {'top': int(rect[1] + h * 0.266),
                   'left': int(rect[0] + 1),
                   'height': int(((rect[3] - rect[1]) * 2 / 3) - 2),
                   'width': int(rect[2] - rect[0])}

            # img color in BGR order
            screen_shot = np.array(self.sct.grab(pos), np.uint8)[:, :, :3]

            if np.array_equal(last_frame, screen_shot):
                # refresh until new frame exists.
                continue
            self.gaming_screen = screen_shot

            if not self.img_h:
                shape = np.array(np.shape(self.gaming_screen)[:2]) / self.downscale
                self.img_h = int(shape[0])  # 500
                self.img_w = int(shape[1])  # 996
                print('img dimension: H {} W {}'.format(self.img_h, self.img_w))

            frame = cv2.resize(self.gaming_screen, (self.img_w, self.img_h))
            if self.gray_scale:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # skip this frame
            if skip_i >= self.frame_skip:
                self.frames.append(frame)
                skip_i = 0
            else:
                skip_i += 1
            last_frame = screen_shot.copy()
            time.sleep(0.01)

    def update_players(self):
        """
        Return player status
        """
        while not self.kill_thread:
            time.sleep(0.01)
            team = []
            for i in self.players:
                player_i = self.players[i]
                if player_i is None:
                    continue
                player_i.update_status(reset=self.restart)
                if player_i.is_active and player_i.is_alive:
                    team.append(player_i.Team)
            self.restart = False
            self.game_over = True if len(set(team)) == 1 else False

    def reset(self, default_ok=None):
        """
        Restart the game
        Currently still need the game to be on the active window.
        :param default_ok: default ok key
        """

        if default_ok is None:
            default_ok = self.my_player.perform_action('attack')
        press_key(['f4', default_ok])
        # Todo figure out how to send keyboard event to a non-active windows.
        # chile_hwnd = win32gui.GetWindow(self.game_hwnd, win32con.GW_CHILD)
        # PostMessage(chile_hwnd, win32con.WM_KEYDOWN, win32con.VK_F4, 0)
        # PostMessage(chile_hwnd, win32con.WM_KEYUP, win32con.VK_F4, 0)
        # PostMessage(chile_hwnd, win32con.WM_CHAR, default_ok, 0)

        self.restart = True
        self.game_over = False
        self.frames.clear()
        time.sleep(self.reset_skip_sec)
        self.reward = 0
        self.bot_attack = 0
        print('Env reset.')
        return self.get_state()

    def step(self, action_id):
        """
        Take an action within the environment
        :param action_id: an action id from the action space
        :return: observation, reward, done, info
        """
        act_name = self.get_action_space()[action_id]
        press_key(self.my_player.perform_action(act_name))
        ob = self.get_state()
        reward = self.get_reward()

        # currently nothing will return in info
        info = self.get_info()

        return ob, reward, self.game_over, info

    def render(self, mode='human'):
        if mode == 'console':
            reward = self.get_reward()
            print('My player Hp: {}. Reward: {}'.format(self.my_player.Hp, reward))
        elif mode == 'human':
            if self.gaming_screen is not None:
                cv2.imshow('lf2_render', self.gaming_screen)
                cv2.waitKey(1)
        elif mode == 'rgb_array':
            return self.gaming_screen
        else:
            super(Lf2Env, self).render(mode=mode)

    def get_reward(self):
        """
        Calculate the corresponding rewards of the current state.
        :return: reward
        """
        enemy_hp = []
        team_hp = []

        for i in self.players:
            if self.players[i] is None:
                continue
            hp_norm = self.players[i].Hp * 10 / self.players[i].Hp_Max
            if self.players[i].Team == self.my_player.Team:
                team_hp.append(hp_norm)
            else:
                enemy_hp.append(hp_norm)

        self.reward = (sum(team_hp) / len(team_hp)) - (sum(enemy_hp) / len(enemy_hp))
        mp_reward = (self.my_player.Mp_Max - self.my_player.Mp) / self.my_player.Mp_Max
        self.reward += mp_reward

        # death penalty
        if not self.my_player.is_alive:
            self.reward -= 50

        # increase reward when increasing attacks.
        if self.my_player.Attack != self.bot_attack:
            self.reward += (self.my_player.Attack - self.bot_attack) / 10
            self.bot_attack = self.my_player.Attack

        # Most simple reward?
        return self.reward

    def get_action_space(self):
        """
        get a list of the current player actions
        :return: a list of actions
        """
        return self.my_player.get_action_list(basic_action=self.basic_action)

    def get_info(self):
        """
        get additional information.
        :return:
        """
        info = dict()
        info['GameOver'] = self.game_over
        # info['episode'] =
        return info

    def close(self):
        cv2.destroyAllWindows()
        self.kill_thread = True

    def seed(self, seed=None):
        pass


if __name__ == '__main__':

    hwnd = winauto.findTopWindow(wantedText='Little Fighter 2')

    my_player = Player(hwnd, 0)
    my_player_1 = Player(hwnd, 1)
    com_player = Player(hwnd, 2, com=True)

    print(my_player.name)
    print(my_player_1.name)

    ply1 = globals()['Julian']()

    now = time.time()
    # att_1 = ply1.sp_attact6()
    while 1:

        my_player.update_status()
        my_player_1.update_status()

        print(my_player.Hp)
        print(my_player_1.Hp)
        # print(com_player.Hp)
        time.sleep(1)

        if time.time() - now >= 12000:
            break





