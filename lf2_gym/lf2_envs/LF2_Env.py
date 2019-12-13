from lf2_gym.lf2_envs.LF2_Util import *
from mss import mss
from win32api import GetSystemMetrics
import numpy as np
from lf2_gym.lf2_envs.winguiauto import winguiauto as winauto
from collections import deque
import win32gui
import win32con
import pyautogui
import time
import cv2
import threading

import gym
from gym import spaces

pyautogui.FAILSAFE = False


class Lf2Env(gym.Env):
    """
    Crop a image from the gaming window, and return all players info as well as
    the current image shown on the display.
    """
    metadata = {'render.modes': ['human', 'console', 'rgb_array']}

    def __init__(self,
                 windows_name='Little Fighter 2',
                 windows_scale=1.25,
                 player_id=1,
                 show=True,
                 downscale=2,
                 frame_stack=4,
                 frame_skip=1,
                 reset_skip_sec=3):
        """
        Initialize the gym environment
        :param windows_name: name of the windows
        :param windows_scale: windows scaling ratio, can be found in the settings.
        :param player_id: training player id
        :param show: Showing cropping image
        :param downscale: downscale ratio
        :param frame_stack: number of frames should be stacked
        :param frame_skip: number of frames to skip before stacking.
        """
        super(Lf2Env, self).__init__()

        self.window_name = windows_name
        self.window_scale = windows_scale
        self.show = show
        self.game_hwnd = winauto.findTopWindow(wantedText=windows_name)
        self.kill_thread = False
        self.sct = mss()

        self.players = {0: None, 1: None, 2: None, 3: None,
                        4: None, 5: None, 6: None, 7: None}

        for i, item in enumerate(self.players.items()):
            self.players[i] = Player(self.game_hwnd, i)

        self.my_player_id = player_id
        self.my_player = self.players[player_id]

        self.gaming_screen = None
        self.game_over = False
        self.restart = True

        self.img_h = 0
        self.img_w = 0
        self.frame_stack = frame_stack
        self.frame_skip = frame_skip
        self.downscale = downscale

        self.frames = deque([], maxlen=self.frame_stack)
        self.reset_skip_sec = reset_skip_sec

        self.recording_thread = threading.Thread(target=self.update_game_img, daemon=True)
        self.player_thread = threading.Thread(target=self.update_players, daemon=True)

        self.recording_thread.start()
        self.player_thread.start()

        self.action_space = spaces.Discrete(len(self.get_action_space()))
        while True:
            if self.img_h != 0:

                inf = np.array([np.inf] * 3)

                self.observation_space = spaces.Dict({
                    'Positions': spaces.Box(low=-inf, high=inf, dtype=np.int32),
                    'Hp': spaces.Discrete(self.my_player.Hp_Max),
                    'Mp': spaces.Discrete(self.my_player.Mp_Max),
                    'Game_Screen': spaces.Box(low=0, high=255,
                                              shape=(self.img_h, self.img_w, self.frame_stack))
                })
                break

        print('Lf2 Environment initialized.')

    def get_state(self, player_id=None):
        # return the current state of the game

        if player_id:
            if not isinstance(player_id, int):
                raise TypeError('id must be integer.')
        else:
            player_id = self.my_player_id

        ob = dict(Game_Screen=self.frames,
                  Positions=np.array([self.players[player_id].x_pos,
                                      self.players[player_id].y_pos,
                                      self.players[player_id].z_pos]),
                  Hp=self.players[player_id].Hp,
                  Mp=self.players[player_id].Mp)
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
            # cut off windows title  from the image
            rect[1] = rect[1] + 44

            pos = {'top': rect[1],
                   'left': rect[0] + 3,
                   'height': rect[3] - rect[1] - 50,
                   'width': rect[2] - rect[0] - 6}

            screen_shot = self.sct.grab(pos)
            screen_shot = np.array(screen_shot)

            if np.array_equal(last_frame, screen_shot):
                # refresh until new frame exists.
                continue

            last_frame = screen_shot.copy()
            img_h, img_w = np.shape(screen_shot)[:2]
            info_scale = int(img_h * 0.23175)
            # gaming_info_img = screen_shot[:info_scale]
            self.gaming_screen = screen_shot[info_scale:]

            if not self.img_h:
                shape = np.array(np.shape(self.gaming_screen)[:2]) / self.downscale
                self.img_h = int(shape[0])
                self.img_w = int(shape[1])
                print('img dimension: H {} W {}'.format(self.img_h, self.img_w))

            frame = cv2.resize(self.gaming_screen, (self.img_w, self.img_h))
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # skip this frame
            if skip_i >= self.frame_skip:
                self.frames.append(frame)
                skip_i = 0
            else:
                skip_i += 1
            if self.show:
                cv2.imshow('resize', frame)
                cv2.imshow('lf2_gym', self.gaming_screen)
                cv2.waitKey(1)

    def update_players(self):
        """
        Return player status
        """
        while not self.kill_thread:
            time.sleep(0.01)
            team = []
            for i in self.players:
                player_i = self.players[i]
                player_i.update_status(reset=self.restart)
                if player_i.is_active and player_i.is_alive:
                    team.append(player_i.Team)
            self.restart = False
            if len(set(team)) == 1:
                self.game_over = True

    def reset(self, default_ok='a'):
        """
        Restart the game
        Currently still need the game to be on the active window.
        :param default_ok: default ok key
        """
        self.press_key(['f4', default_ok])
        # Todo figure out how to send keyboard event to a non-active windows.
        # chile_hwnd = win32gui.GetWindow(self.game_hwnd, win32con.GW_CHILD)
        # PostMessage(chile_hwnd, win32con.WM_KEYDOWN, win32con.VK_F4, 0)
        # PostMessage(chile_hwnd, win32con.WM_KEYUP, win32con.VK_F4, 0)
        # PostMessage(chile_hwnd, win32con.WM_CHAR, default_ok, 0)

        self.game_over = False
        self.restart = True
        time.sleep(self.reset_skip_sec)
        print('Env reset.')
        return self.get_state()

    @staticmethod
    def press_key(keys):
        last_key = ''
        if keys is not None:
            for key in keys:
                if key == last_key:
                    # to prevent not sending key event if two consecutive identical keys.
                    pyautogui.keyUp(key)
                pyautogui.keyDown(key)
                last_key = key
            time.sleep(0.1)
            for key in keys:
                pyautogui.keyUp(key)

    def step(self, action_id):
        """
        Take an action within the environment
        :param action_id: an action id from the action space
        :return: observation, reward, done, info
        """
        act_name = self.get_action_space()[action_id]
        print(act_name)
        self.press_key(self.my_player.perform_action(act_name))

        ob = self.get_state()
        reward = self.get_reward()

        # currently nothing will return in info
        info = ()

        return ob, reward, self.game_over, info

    def render(self, mode='human'):
        # todo need to be modified...
        if mode == 'console':
            reward = self.get_reward()
            print('My player Hp: {}. Reward: {}'.format(my_player.Hp, reward))
        elif self.gaming_screen is not None:
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

            hp_norm = self.players[i].Hp / self.players[i].Hp_Max
            if self.players[i].Team == self.my_player.Team:
                team_hp.append(hp_norm)
            else:
                enemy_hp.append(hp_norm)

        # Most simple reward?
        return (sum(team_hp) / len(team_hp)) - (sum(enemy_hp) / len(enemy_hp))

    def get_action_space(self):
        """
        get a list of the current player actions
        :return: a list of actions
        """
        return self.my_player.get_action_list()


if __name__ == '__main__':

    hwnd = winauto.findTopWindow(wantedText='Little Fighter 2')

    my_player = Player(hwnd, 0)
    my_player_1 = Player(hwnd, 1)
    com_player = Player(hwnd, 2, com=True)

    # print(my_player.name)
    # print(my_player_1.name)

    ply1 = globals()['Julian']()

    now = time.time()
    # att_1 = ply1.sp_attact6()
    while 1:

        my_player.update_status()
        my_player_1.update_status()
        com_player.update_status()


        # print(my_player.Hp)
        # print(my_player_1.Hp)
        # print(com_player.Hp)
        time.sleep(1)

        if time.time() - now >= 12000:
            break





