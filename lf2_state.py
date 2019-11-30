from lf2_util import *
from LF2_char import *
import pyautogui
import time

def press_key(keys):
    last_key = ''
    for key in keys:
        if key == last_key:
            # time.sleep(0.1)
            pyautogui.keyUp(key)
        pyautogui.keyDown(key)
        last_key = key
    time.sleep(0.1)
    for key in keys:
        pyautogui.keyUp(key)


if __name__ == '__main__':

    import winguiauto.winguiauto as winauto
    hwnd = winauto.findTopWindow(wantedText='Little Fighter 2')

    my_player = Player(hwnd, 0)
    com_player = Player(hwnd, 1, com=False)

    print(my_player.name)
    print(com_player.name)

    ply1 = globals()['Julian']()

    now = time.time()
    # att_1 = ply1.sp_attact6()
    while 1:

        my_player.update_status()
        com_player.update_status()

        print(com_player.Hp)
        time.sleep(5)

        if time.time() - now >= 12000:
            break





