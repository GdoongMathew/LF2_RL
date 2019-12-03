from LF2_Util import *
from LF2_char import *
from mss import mss
from win32api import GetSystemMetrics
import numpy as np
import winguiauto.winguiauto as winauto
import win32gui
import win32con
import pyautogui
import time
import threading


def press_key(keys):
    last_key = ''
    for key in keys:
        if key == last_key:
            # to prevent not sending key event if two consecutive identical keys.
            # time.sleep(0.1)
            pyautogui.keyUp(key)
        pyautogui.keyDown(key)
        last_key = key
    time.sleep(0.1)
    for key in keys:
        pyautogui.keyUp(key)


class LF2Env:
    """
    Crop a image from the gaming window, and return all players info as well as
    the current image shown on the display.
    """

    def __init__(self, windows_name, windows_scale=1.0, player_id=2):
        """
        Initialize some parameter.
        :param windows_name: windows name that we want to crop image from
        :param windows_scale: Windows scaling param.( Can be seen in the setting Display area.
        """
        self.window_name = windows_name
        self.window_scale = windows_scale
        self.game_hwnd = winauto.findTopWindow(wantedText=windows_name)
        self.kill_thread = False
        self.sct = mss()

        self.players = {0: None, 1: None, 2: None, 3: None,
                        4: None, 5: None, 6: None, 7: None}

        for i, item in enumerate(self.players.items()):
            self.players[i] = Player(self.game_hwnd, i)

        self.my_player_id = player_id
        self.my_player = self.players[player_id - 1]

        self.gaming_screen = None

        self.recording_thread = threading.Thread(target=self.screen_recording, daemon=True)
        self.player_thread = threading.Thread(target=self.player_state, daemon=True)

        self.recording_thread.start()
        self.player_thread.start()

    def get_state(self, id=None):
        # return the current state of the game
        if id:
            if not isinstance(id, int):
                raise TypeError('id must be integer.')
            return self.gaming_screen, self.players[id]
        else:
            return self.gaming_screen, self.players

    def screen_recording(self):
        """
        Update the current gaming scene
        """
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
            rect[1] = rect[1] + 28

            pos = {'top': rect[1],
                   'left': rect[0] + 3,
                   'height': rect[3] - rect[1] - 3,
                   'width': rect[2] - rect[0] - 6}

            screen_shot = self.sct.grab(pos)
            screen_shot = np.array(screen_shot)

            h, w = np.shape(screen_shot)[:2]
            info_scale = int(h * 0.23175)
            gaming_info_img = screen_shot[:info_scale]
            self.gaming_screen = screen_shot[info_scale:]
            time.sleep(0.01)

    def player_state(self):
        """
        Return player status
        """
        while not self.kill_thread:
            for i in self.players:
                self.players[i].update_status()
                time.sleep(0.01)

    @staticmethod
    def reset(default_ok='a'):
        """
        Restart the game
        :param default_ok: default ok key
        :return:
        """
        press_key(['f4', default_ok])

    def reward(self):
        """
        Calculate the corresponding rewards of the current state.
        :return: reward
        """
        enemy_hp = 0
        team_hp = 0

        for i in self.players:
            if self.players[i].Team == self.my_player.Team:
                team_hp += self.players[i].Hp
            else:
                enemy_hp += self.players[i].Hp

        # Most simple reward?
        return team_hp - enemy_hp


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
        com_player.update_status()

        print(my_player.Hp)
        print(my_player_1.Hp)
        print(com_player.Hp)
        time.sleep(5)

        if time.time() - now >= 12000:
            break





