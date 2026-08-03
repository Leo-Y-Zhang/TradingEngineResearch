"""An in-process memory guard for long unattended runs.

Adapted from an earlier internal utility, written after an external PID-based
watchdog twice failed to arm and let a run reach 4.4 GB unprotected. The research
scripts here have the same exposure: they load large panels unattended.

Why this exists. An external watchdog has to be handed a PID, and capturing that
PID races the process it is meant to protect: twice in a single unattended run
the guard silently failed to arm and a job climbed to 4.4 GB unprotected while
free RAM fell to 1.5 GB. A guard that is only *usually* armed is not a guard.

This one cannot fail to arm, because it lives inside the process it protects. It
polls free physical memory from a daemon thread and, on breach, flushes stdout
and terminates the process itself. Losing the run is the intended outcome: the
alternative is paging the whole machine, which in practice means a freeze.

`os._exit` is used deliberately. A normal exception could be swallowed by the
computation's own error handling, and by the time memory is this tight there is
no safe work left to do -- the terms already printed are the ones that survive.
"""
from __future__ import annotations

import ctypes
import os
import sys
import threading
import time


class _MemoryStatusEx(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def free_gb():
    """Free physical memory in GB, or None where it cannot be read."""
    try:
        status = _MemoryStatusEx()
        status.dwLength = ctypes.sizeof(_MemoryStatusEx)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return None
        return status.ullAvailPhys / (1024 ** 3)
    except Exception:
        return None


def start(floor_gb=1.0, interval=5.0, label="run"):
    """Arm the guard. Returns the thread, which is a daemon and needs no cleanup.

    A poll that cannot read memory is treated as "keep going", not as a breach:
    the failure mode of killing a healthy job on a transient read error is worse
    than a slightly late kill.
    """
    def _watch():
        while True:
            time.sleep(interval)
            free = free_gb()
            if free is None or free >= floor_gb:
                continue
            print(
                f"\nMEMGUARD: free RAM {free:.2f} GB fell below the {floor_gb:.2f} GB "
                f"floor during {label}. Terminating this process so the machine "
                f"stays usable. Everything printed above this line is valid and "
                f"was computed before the breach.",
                flush=True,
            )
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(3)

    thread = threading.Thread(target=_watch, name="memguard", daemon=True)
    thread.start()
    return thread
