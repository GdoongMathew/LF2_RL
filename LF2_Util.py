from ctypes.wintypes import BOOL
from ctypes.wintypes import DWORD
from ctypes.wintypes import HANDLE
import ctypes
import pymem
import win32process


Char_Name = {
    "Template": 0, "Julian":    0, "Firzen":    0, "LouisEX":   0,
    "Bat":      0, "Justin":    0, "Knight":    0, "Jan":       0,
    "Monk":     0, "Sorcerer":  0, "Jack":      0, "Mark":      0,
    "Hunter":   0, "Bandit":    0, "Deep":      0, "John":      0,
    "Henry":    0, "Rudolf":    0, "Louis":     0, "Firen":     0,
    "Freeze":   0, "Dennis":    0, "Woody":     0, "Davis":     0
}

class Lf2AddressTable:
    kills = 0x358
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


class ProcessReading:
    # Reading process memory from certain memory address
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

        PROCESS_VM_OPERATION = 0x0008
        PROCESS_VM_READ = 0x0010

        self.proc_handle = self.get_process_handle(self.pid, PROCESS_VM_OPERATION | PROCESS_VM_READ)

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


class Player:
    def __init__(self, game_proc_handle, idx, com=False):
        self.game_reading = ProcessReading(game_proc_handle)
        self.player_address = self.game_reading.read_int(Lf2AddressTable.Player[idx] if not com
                                                         else Lf2AddressTable.Computer[idx])

        self.DataFiles = []
        self.data_address = self.address_shift(Lf2AddressTable.PDataPointer)
        self.name = self.get_player_char()

        self.kills = 0
        self.Attack = 0
        self.Hp = 0
        self.Hp_Dark = 0
        self.Hp_Max = self.game_reading.read_int(self.address_shift(Lf2AddressTable.Hp_Max))
        self.Hp_Lost = 0
        self.Mp = 0
        self.Mp_Usage = 0

        self.Picking = 0
        self.Owner = 0
        self.Enemy = 0
        self.Team = 0
        self.is_active = False
        self.is_alive = False

        self.x_pos = None
        self.y_pos = None
        self.z_pos = None

        self.game_state = None
        self.time = 0
        self.total_time = 0

    def address_shift(self, shift):
        # return a shifted address by adding a player address
        return self.player_address + shift

    def update_status(self):

        self.name = self.get_player_char()

        self.game_state = self.game_reading.read_ushort(Lf2AddressTable.GameState)

        self.Attack = self.game_reading.read_int(self.address_shift(Lf2AddressTable.Attack))
        self.kills = self.game_reading.read_int(self.address_shift(Lf2AddressTable.kills))
        self.Hp = self.game_reading.read_int(self.address_shift(Lf2AddressTable.Hp))
        self.Hp_Dark = self.game_reading.read_int(self.address_shift(Lf2AddressTable.Hp_Dark))
        self.Hp_Lost = self.game_reading.read_int(self.address_shift(Lf2AddressTable.Hp_Lost))

        self.Mp = self.game_reading.read_int(self.address_shift(Lf2AddressTable.Mp))
        self.Mp_Usage = self.game_reading.read_int(self.address_shift(Lf2AddressTable.Mp_Usage))

        self.x_pos = self.game_reading.read_int(self.address_shift(Lf2AddressTable.x_pos))
        self.y_pos = self.game_reading.read_int(self.address_shift(Lf2AddressTable.y_pos))
        self.z_pos = self.game_reading.read_int(self.address_shift(Lf2AddressTable.z_pos))

        self.Enemy = self.game_reading.read_int(self.address_shift(Lf2AddressTable.Enemy))
        self.Picking = self.game_reading.read_int(self.address_shift(Lf2AddressTable.Picking))
        self.Team = self.game_reading.read_int(self.address_shift(Lf2AddressTable.Team))

    def get_player_char(self):
        # get current player character
        _data_address = self.game_reading.read_int(Lf2AddressTable.DataPointer)
        for i in range(len(Lf2AddressTable.DataFile)):
            self.DataFiles.append(self.game_reading.read_int(_data_address + i * 4))

        for i, item in enumerate(Char_Name):
            Char_Name[item] = self.DataFiles[i]

        for name, i in Char_Name.items():
            if i == self.data_address:
                return name
        return ''



