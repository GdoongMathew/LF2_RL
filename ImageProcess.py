import winguiauto.winguiauto as winauto
from win32api import GetSystemMetrics
import tensorflow as tf
from mss import mss
from lf2_state import Player
import logging
import win32gui
import win32con
import numpy as np
import cv2
import time


def screen_record(windows_name, set_window_resize=1.0):
    hwnd = winauto.findTopWindow(wantedText=windows_name)
    screen_num = 0
    sct = mss()
    now_time = time.time()

    player_state = Player(hwnd, 1)

    while(1):

        tup = win32gui.GetWindowPlacement(hwnd)
        player_state.update_status()

        if tup[1] == win32con.SW_SHOWMAXIMIZED:
            w = GetSystemMetrics(0)
            h = GetSystemMetrics(1)
            rect = [0, 0, w, h]
            print(rect)

        elif tup[1] == win32con.SW_SHOWNORMAL:
            rect = list(win32gui.GetWindowRect(hwnd))

        else:
            continue

        # cut off windows title  from the image
        rect[1] = rect[1] + 28

        pos = {'top': rect[1], 'left': rect[0] + 3,
               'height': rect[3] - rect[1] - 3, 'width':rect[2] - rect[0] - 6}

        screen_shot = sct.grab(pos)
        screen_shot = np.array(screen_shot)

        h, w = np.shape(screen_shot)[:2]
        info_scale = int(h * 0.23175)
        gaming_info_img = screen_shot[:info_scale]
        gaming_screen = screen_shot[info_scale:]

        cv2.imshow('screen', gaming_screen)
        cv2.imshow('info', gaming_info_img)

        c = cv2.waitKey(1)
        logging.info('Loop took {} s.'.format(time.time() - now_time))
        now_time = time.time()
        screen_num += 1
        if (c == 27 & 0xEFFFFF):
            break

def main():
    screen_record('Little Fighter 2', set_window_resize=1.25)

if __name__ == '__main__':
    main()

