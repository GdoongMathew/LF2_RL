"""Win32 process memory reader.

Wraps ``pymem`` + ``ctypes.windll.kernel32.OpenProcess`` so the rest of the
Windows-side code can do batched ``read_block`` / typed reads against the
LF2 process. Extracted from the legacy ``lf2_gym.lf2_envs.utils`` module so
the dependency on ``pymem`` / ``win32process`` is colocated with everything
else that's OS-specific.
"""

from __future__ import annotations

import ctypes
from ctypes.wintypes import BOOL, DWORD, HANDLE

import pymem
import win32process

PROCESS_VM_OPERATION = 0x0008
PROCESS_VM_READ = 0x0010
PROCESS_VM_WRITE = 0x0020


class ProcessWR:
    """Read/write a target process's virtual memory by ``win_handle``.

    Instances are cached per ``win_handle`` (a hwnd) so multiple ``Player``
    objects sharing a window collapse to one OS handle.
    """

    _ctype_open_process = ctypes.windll.kernel32.OpenProcess
    _ctype_open_process.restype = HANDLE
    _ctype_open_process.argtypes = (DWORD, BOOL, DWORD)

    _ctypes_get_last_error = ctypes.windll.kernel32.GetLastError
    _ctypes_get_last_error.restype = DWORD
    _ctypes_get_last_error.argtypes = ()

    _proc_instance: dict[int, "ProcessWR"] = {}

    def __new__(cls, *args, win_handle: int, **kwargs):
        if win_handle not in cls._proc_instance:
            cls._proc_instance[win_handle] = super().__new__(cls)
        return cls._proc_instance[win_handle]

    def __init__(self, *, win_handle: int):
        self.pid = win32process.GetWindowThreadProcessId(win_handle)[1]
        self.proc_handle = self.get_process_handle(
            self.pid,
            PROCESS_VM_OPERATION | PROCESS_VM_READ | PROCESS_VM_WRITE,
        )

    @staticmethod
    def get_process_handle(
        process_id,
        desired_access,
        inherit_handle: bool = False,
    ):
        handle = ProcessWR._ctype_open_process(desired_access, inherit_handle, process_id)
        if handle is None or handle == 0:
            raise RuntimeError(
                f"Failed to open process with ID {process_id}, Desired access: {desired_access}, "
                f"Inherit handle: {inherit_handle} with error {ProcessWR._ctypes_get_last_error()}."
            )
        return handle

    def read_block(self, base_address: int, size: int) -> bytes:
        return pymem.memory.read_bytes(self.proc_handle, base_address, size)

    def read_bytes(self, lpBaseAddress: int, n_size: int) -> bytes:
        return self.read_block(lpBaseAddress, n_size)

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
