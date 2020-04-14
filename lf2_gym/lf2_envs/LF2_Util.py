# All following code is inspired by LF2 Dashboard
# LF2 Dashboard github: https://github.com/xmfcx/LF2-Dashboard/tree/master/LF2Dashboard

from ctypes.wintypes import BOOL
from ctypes.wintypes import DWORD
from ctypes.wintypes import HANDLE
from configobj import ConfigObj
from sys import byteorder
from inspect import signature

from lf2_envs.LF2_char import *
from lf2_envs.winguiauto import winguiauto as winauto
from collections import OrderedDict
import time
import pyautogui
import functools
import ctypes
import pymem
import win32process
import win32ui


PROCESS_VM_OPERATION = 0x0008
PROCESS_VM_READ = 0x0010
PROCESS_VM_WRITE = 0x0020

Char_Name = OrderedDict([
    ("Template", 0), ("Julian",    0), ("Firzen",    0), ("LouisEX",   0),
    ("Bat",      0), ("Justin",    0), ("Knight",    0), ("Jan",       0),
    ("Monk",     0), ("Sorcerer",  0), ("Jack",      0), ("Mark",      0),
    ("Hunter",   0), ("Bandit",    0), ("Deep",      0), ("John",      0),
    ("Henry",    0), ("Rudolf",    0), ("Louis",     0), ("Firen",     0),
    ("Freeze",   0), ("Dennis",    0), ("Woody",     0), ("Davis",     0)]
)

BackGround_code = {
    'HK Coliseum': 0,       'Lion Forest': 1,       'Stanley Prison': 2,    'The Great Wall': 3,
    "Queen's Island": 4,    'Forbidden Tower': 5,   'BrokeBack Cliff': 6,   'CUHK': 7,
    'Uai Hom Village': 8,   'Template1': 9,         'Template2': 10,        'Template3': 11,
    'Random': 100
}

Difficulty = {2: 'Easy', 1: 'Normal', 0: 'Difficult', -1: 'Crazy'}
Mode = {0: 'VS', 1: 'Stage', 2: '1v1', 3: '2v2', 4: 'Battle', 5: 'Demo', 6: 'PlayBack', 7: 'Quit'}


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


class Lf2AddressTable:
    # All shifting address value
    Kills = 0x358
    Attack = 0x348
    Hp = 0x2FC
    Hp_Dark = 0x300
    Hp_Max = 0x304
    Hp_Lost = 0x32C
    Mp = 0x308
    Mp_Usage = 0x350

    x_pos = 0x10
    y_pos = 0x14
    z_pos = 0x18

    x_pos_f = 0x58
    y_pos_f = 0x60
    z_pos_f = 0x68

    Picking = 0x35C
    Owner = 0x354
    Enemy = 0x360
    Team = 0x364
    Invincible = 0x8

    Facing = 0x80

    PDataPointer = 0x368
    DataPointer = 0x4592D4
    GameState = 0x44D020
    Time = 0x450BBC
    TotalTime = 0x450B8C

    Player = [0x458C94 + i * 4 for i in range(8)]
    Computer = [0x458CBC + i * 4 for i in range(8)]
    ActivePlayers = [0x458B04 + i for i in range(8)]
    SelectedPlayers = [0x451288 + i * 4 for i in range(8)]
    PlayerInGame = [0x458B04 + i for i in range(8)]
    CPlayerInGame = [0x458B0E + i for i in range(8)]

    Names = [0x44FCC0 + i * 11 for i in range(11)]
    DataFile = [None] * 65

    BackGround = 0x44D024
    Difficulty = 0x450C30
    Mode = 0x451160


class ProcessWR:
    # Reading/Writing process memory from certain memory address
    def __init__(self, win_handle):
        # Windows
        self.win_handle = win_handle
        self.pid = win32process.GetWindowThreadProcessId(self.win_handle)[1]
        self.OpenProcess = ctypes.windll.kernel32.OpenProcess
        self.OpenProcess.restype = HANDLE
        self.OpenProcess.argtypes = (DWORD, BOOL, DWORD)

        self.GetLastError = ctypes.windll.kernel32.GetLastError
        self.GetLastError.restype = DWORD
        self.GetLastError.argtypes = ()

        self.proc_handle = self.get_process_handle(self.pid, PROCESS_VM_OPERATION | PROCESS_VM_READ | PROCESS_VM_WRITE)

    def get_process_handle(self, dwProcessId, dwDesiredAccess, bInheritHandle=False):
        handle = self.OpenProcess(dwDesiredAccess, bInheritHandle, dwProcessId)
        if handle is None or handle == 0:
            raise Exception('Error: %s' % self.GetLastError())
        return handle

    def read_bytes(self, lpBaseAddress, n_size):
        return pymem.memory.read_bytes(self.proc_handle, lpBaseAddress, n_size)

    def read_char(self, lpBaseAddress):
        return pymem.memory.read_char(self.proc_handle, lpBaseAddress)

    def read_int(self, lpBaseAddress):
        return pymem.memory.read_int(self.proc_handle, lpBaseAddress)

    def read_uint(self, lpBaseAddress):
        return pymem.memory.read_uint(self.proc_handle, lpBaseAddress)

    def read_long(self, lpBaseAddress):
        return pymem.memory.read_long(self.proc_handle, lpBaseAddress)

    def read_str(self, lpBaseAddress, n_size=4):
        return pymem.memory.read_string(self.proc_handle, lpBaseAddress, n_size)

    def read_float(self, lpBaseAddress):
        return pymem.memory.read_float(self.proc_handle, lpBaseAddress)

    def read_ushort(self, lpBaseAddress):
        return pymem.memory.read_ushort(self.proc_handle, lpBaseAddress)

    def write_int(self, lpBaseAddress, data):
        return pymem.memory.write_int(self.proc_handle, lpBaseAddress, data)


class Player:
    def __init__(self, game_proc_handle, idx, com=False):
        self.game_reading = ProcessWR(game_proc_handle)
        self.idx = idx  # 0 ~ 7
        self.is_computer = com
        self.player_address = self.game_reading.read_int(Lf2AddressTable.Player[self.idx]
                                                         if not self.is_computer
                                                         else Lf2AddressTable.Computer[self.idx])

        self.DataFiles = []
        self.DataAddress = self.game_reading.read_int(self.address_shift(Lf2AddressTable.PDataPointer))
        self.name = self.get_player_char()
        self.lf2_char = globals()[self.name]()

        self.Kills = 0
        self.Attack = 0
        self.Hp = 0
        self.Hp_Dark = 0
        self.Hp_Max = self.game_reading.read_int(self.address_shift(Lf2AddressTable.Hp_Max))
        self.Hp_Lost = 0
        self.Mp = 0
        self.Mp_Max = 500  # not so sure about this number.
        self.Mp_Usage = 0

        self.Picking = 0
        self.Owner = 0
        self.Enemy = 0
        self.Team = 0

        self.act_add = Lf2AddressTable.CPlayerInGame[self.idx] if self.is_computer \
            else Lf2AddressTable.PlayerInGame[self.idx]

        self.is_active = bool.from_bytes(self.game_reading.read_bytes(self.act_add, 1), byteorder)
        self.is_alive = False

        self.x_pos = None
        self.y_pos = None
        self.z_pos = None

        self.game_state = None
        self.time = 0
        self.total_time = 0

        self.facing_bool = False
        self.Facing = 'right'

    def address_shift(self, shift):
        """
        Shifting the address by adding the player address.
        :param shift: desired shifting address value
        :return: shifted address
        """
        return self.player_address + shift

    def update_status(self, reset=False):

        if reset:
            self.name = self.get_player_char()
            self.lf2_char = globals()[self.name]()
            self.DataAddress = self.game_reading.read_int(self.address_shift(Lf2AddressTable.PDataPointer))
            self.Team = self.game_reading.read_int(self.address_shift(Lf2AddressTable.Team))

        self.game_state = self.game_reading.read_ushort(Lf2AddressTable.GameState)
        self.Attack = self.game_reading.read_int(self.address_shift(Lf2AddressTable.Attack))
        self.Kills = self.game_reading.read_int(self.address_shift(Lf2AddressTable.Kills))

        hp = self.game_reading.read_int(self.address_shift(Lf2AddressTable.Hp))
        self.Hp = hp if hp >= 0 else 0  # Just for the sake of not letting health point below 0
        self.Hp_Dark = self.game_reading.read_int(self.address_shift(Lf2AddressTable.Hp_Dark))
        self.Hp_Lost = self.game_reading.read_int(self.address_shift(Lf2AddressTable.Hp_Lost))

        self.Mp = self.game_reading.read_int(self.address_shift(Lf2AddressTable.Mp))
        self.Mp_Usage = self.game_reading.read_int(self.address_shift(Lf2AddressTable.Mp_Usage))

        self.x_pos = self.game_reading.read_int(self.address_shift(Lf2AddressTable.x_pos))
        self.y_pos = self.game_reading.read_int(self.address_shift(Lf2AddressTable.y_pos))
        self.z_pos = self.game_reading.read_int(self.address_shift(Lf2AddressTable.z_pos))

        facing_byte = self.game_reading.read_bytes(self.address_shift(Lf2AddressTable.Facing), 1)
        self.facing_bool = bool.from_bytes(facing_byte, byteorder)
        self.Facing = 'left' if self.facing_bool else 'right'

        self.Enemy = self.game_reading.read_int(self.address_shift(Lf2AddressTable.Enemy))
        self.Picking = self.game_reading.read_int(self.address_shift(Lf2AddressTable.Picking))

        self.is_active = bool.from_bytes(self.game_reading.read_bytes(self.act_add, 1), byteorder)
        self.is_alive = self.Hp > 0 if self.is_active else False

    def get_player_char(self):
        """
        Get the character of the player.
        :return: the name of the character
        """
        _data_address = self.game_reading.read_int(Lf2AddressTable.DataPointer)
        for i in range(len(Lf2AddressTable.DataFile)):
            Lf2AddressTable.DataFile[i] = self.game_reading.read_int(_data_address + i * 4)

        for i, item in enumerate(Char_Name):
            Char_Name[item] = Lf2AddressTable.DataFile[i]

        for name, i in Char_Name.items():
            if i == self.DataAddress:
                return name
        return ''

    def get_action_list(self, basic_action=False):
        """
        return list of action space if exists, else return None
        :return: list of available actions.
        """
        assert len(self.name), 'No character name {} found.'.format(self.name)
        lf2_char = Template(player_id=self.idx) if basic_action \
            else globals()[self.name](player_id=self.idx)
        action_s = lf2_char.action_space()
        if basic_action:
            action_s.remove('run')
        return action_s

    def perform_action(self, action_str):
        """
        :param action_str:
        :return:
        """
        act_func = getattr(self.lf2_char, action_str)
        sig = str(signature(act_func))
        if 'direction' in sig:
            return functools.partial(act_func, self.Facing)()
        return getattr(self.lf2_char, action_str)()

    def change_char(self, char_name):
        assert char_name in Char_Name.keys()


class PlayGround:
    """
    All Lf2 Environment Properties.
    """
    def __init__(self, config_path):
        self.gym_config = ConfigObj(config_path)['Gym_Parameter']
        self.lf2_path = ConfigObj(config_path)['LF2']['ShortCut']
        ctrl = ConfigObj(config_path)['LF2']['CtrlFile']
        update_ctrl(ctrl)
        self.mode = self.gym_config['Mode']
        self.com_num = self.gym_config['Com_Player_Num']
        self.fighter = self.gym_config['Fighter']
        self.bg = self.gym_config['Background']
        assert self.bg in list(BackGround_code.keys())

        self.lf2_hwnd, self.lf2_win_hwnd = self.run_app()
        self.process_wr = ProcessWR(self.lf2_hwnd)

    def run_app(self):
        """
        run Little Fighter if it isn't running and automate all setup.
        :return: windows hwnd
        """
        windows_name = 'Little Fighter 2'
        try:
            hwnd = winauto.findTopWindow(wantedText=windows_name)
        except winauto.WinGuiAutoError:
            import os
            os.startfile(self.lf2_path)
            hwnd = []
            while not hwnd:
                hwnd = winauto.findTopWindows(wantedText=windows_name)
            hwnd = hwnd[0]

        win_hwnd = win32ui.FindWindow(None, windows_name)
        win_hwnd.SetForegroundWindow()
        win_hwnd.SetFocus()
        return hwnd, win_hwnd

    def set_property(self):
        if Mode[self.process_wr.read_int(0x451160)] != self.mode:
            self.process_wr.write_int(0x451160, 0)
        press_key(Template().attack())



    def find_property(self):
        pass

    def find_player(self):
        pass


if __name__ == '__main__':
    lf2_pgd = PlayGround(r'D:\Python Project\LF2_RL\lf2_gym\lf2_envs\config\config_1.ini')
    time.sleep(5)
    lf2_pgd.set_property()

