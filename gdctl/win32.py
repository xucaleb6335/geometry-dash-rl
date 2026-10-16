"""Win32 helpers via ctypes: processes, listening ports, windows, capture,
focus and synthetic input. Works whether or not the control mod answers.
"""

from __future__ import annotations

import ctypes
import subprocess
import sys
import time
from ctypes import wintypes

if sys.platform != "win32":  # pragma: no cover - the CLI is Windows-only
    raise ImportError("gdctl.win32 needs Windows")

user32 = ctypes.WinDLL("user32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)
iphlpapi = ctypes.WinDLL("iphlpapi", use_last_error=True)

# Physical pixels everywhere (window rects, capture, SendInput).
try:
    user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))  # PER_MONITOR_AWARE_V2
except (AttributeError, OSError):
    pass

# --------------------------------------------------------------------------
# Processes

TH32CS_SNAPPROCESS = 0x2


class PROCESSENTRY32W(ctypes.Structure):
    _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD), ("th32ProcessID", wintypes.DWORD),
                ("th32DefaultHeapID", ctypes.c_size_t), ("th32ModuleID", wintypes.DWORD),
                ("cntThreads", wintypes.DWORD), ("th32ParentProcessID", wintypes.DWORD),
                ("pcPriClassBase", ctypes.c_long), ("dwFlags", wintypes.DWORD),
                ("szExeFile", ctypes.c_wchar * 260)]


kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
kernel32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
kernel32.Process32FirstW.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESSENTRY32W)]
kernel32.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESSENTRY32W)]
kernel32.CloseHandle.argtypes = [wintypes.HANDLE]


def processes(name: str | None = None) -> list[dict]:
    """[{pid, name, ppid}] for every process (or those whose exe name matches)."""
    snap = kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
    if not snap or snap == wintypes.HANDLE(-1).value:
        return []
    out = []
    try:
        entry = PROCESSENTRY32W()
        entry.dwSize = ctypes.sizeof(entry)
        ok = kernel32.Process32FirstW(snap, ctypes.byref(entry))
        while ok:
            exe = entry.szExeFile
            if name is None or exe.lower() == name.lower():
                out.append({"pid": entry.th32ProcessID, "name": exe, "ppid": entry.th32ParentProcessID})
            ok = kernel32.Process32NextW(snap, ctypes.byref(entry))
    finally:
        kernel32.CloseHandle(snap)
    return out


def pid_alive(pid: int) -> bool:
    return any(p["pid"] == pid for p in processes())


def taskkill(pid: int, force: bool = False) -> tuple[int, str]:
    args = ["taskkill", "/PID", str(pid)] + (["/F"] if force else [])
    res = subprocess.run(args, capture_output=True, text=True, timeout=20)
    return res.returncode, (res.stdout + res.stderr).strip()


# --------------------------------------------------------------------------
# Listening TCP ports (no connection made, so the bridge log stays quiet)

TCP_TABLE_OWNER_PID_LISTENER = 3
AF_INET = 2


class MIB_TCPROW_OWNER_PID(ctypes.Structure):
    _fields_ = [("dwState", wintypes.DWORD), ("dwLocalAddr", wintypes.DWORD), ("dwLocalPort", wintypes.DWORD),
                ("dwRemoteAddr", wintypes.DWORD), ("dwRemotePort", wintypes.DWORD), ("dwOwningPid", wintypes.DWORD)]


iphlpapi.GetExtendedTcpTable.argtypes = [ctypes.c_void_p, ctypes.POINTER(wintypes.DWORD), wintypes.BOOL,
                                         wintypes.ULONG, ctypes.c_int, wintypes.ULONG]
iphlpapi.GetExtendedTcpTable.restype = wintypes.DWORD


def listening_ports() -> dict[int, int]:
    """{port: owning pid} for IPv4 TCP listeners."""
    size = wintypes.DWORD(0)
    iphlpapi.GetExtendedTcpTable(None, ctypes.byref(size), False, AF_INET, TCP_TABLE_OWNER_PID_LISTENER, 0)
    for _ in range(3):
        buf = ctypes.create_string_buffer(size.value + 4096)
        size = wintypes.DWORD(ctypes.sizeof(buf))
        res = iphlpapi.GetExtendedTcpTable(buf, ctypes.byref(size), False, AF_INET,
                                           TCP_TABLE_OWNER_PID_LISTENER, 0)
        if res == 0:
            break
    else:
        return {}
    count = ctypes.cast(buf, ctypes.POINTER(wintypes.DWORD))[0]
    rows = ctypes.cast(ctypes.addressof(buf) + 4, ctypes.POINTER(MIB_TCPROW_OWNER_PID))
    out = {}
    for i in range(count):
        row = rows[i]
        port = ((row.dwLocalPort & 0xFF) << 8) | ((row.dwLocalPort >> 8) & 0xFF)
        out[port] = row.dwOwningPid
    return out


# --------------------------------------------------------------------------
# Windows

WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
user32.EnumWindows.argtypes = [WNDENUMPROC, wintypes.LPARAM]
user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
user32.GetWindowThreadProcessId.restype = wintypes.DWORD
user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
user32.IsWindowVisible.argtypes = [wintypes.HWND]
user32.IsIconic.argtypes = [wintypes.HWND]
user32.IsHungAppWindow.argtypes = [wintypes.HWND]
user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
user32.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
user32.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.POINT)]
user32.GetForegroundWindow.restype = wintypes.HWND
user32.SetForegroundWindow.argtypes = [wintypes.HWND]
user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
user32.BringWindowToTop.argtypes = [wintypes.HWND]
user32.PostMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
user32.SendMessageTimeoutW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM,
                                       wintypes.UINT, wintypes.UINT, ctypes.POINTER(ctypes.c_size_t)]
user32.SendMessageTimeoutW.restype = ctypes.c_size_t
user32.AttachThreadInput.argtypes = [wintypes.DWORD, wintypes.DWORD, wintypes.BOOL]
kernel32.GetCurrentThreadId.restype = wintypes.DWORD


def _title(hwnd) -> str:
    n = user32.GetWindowTextLengthW(hwnd)
    buf = ctypes.create_unicode_buffer(n + 1)
    user32.GetWindowTextW(hwnd, buf, n + 1)
    return buf.value


def windows_of(pid: int, visible_only: bool = True) -> list[dict]:
    found = []

    def cb(hwnd, _lparam):
        owner = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
        if owner.value == pid and (not visible_only or user32.IsWindowVisible(hwnd)):
            found.append(hwnd)
        return True

    user32.EnumWindows(WNDENUMPROC(cb), 0)
    return [window_info(h) for h in found]


def window_info(hwnd) -> dict:
    rect = wintypes.RECT()
    user32.GetWindowRect(hwnd, ctypes.byref(rect))
    client = wintypes.RECT()
    user32.GetClientRect(hwnd, ctypes.byref(client))
    return {
        "hwnd": int(hwnd),
        "title": _title(hwnd),
        "visible": bool(user32.IsWindowVisible(hwnd)),
        "minimized": bool(user32.IsIconic(hwnd)),
        "hung": bool(user32.IsHungAppWindow(hwnd)),
        "foreground": int(user32.GetForegroundWindow() or 0) == int(hwnd),
        "rect": [rect.left, rect.top, rect.right, rect.bottom],
        "client_w": client.right - client.left,
        "client_h": client.bottom - client.top,
    }


def main_window(pid: int, title: str | None = None) -> dict | None:
    """The game's top-level window: prefer the exact title, else the largest."""
    wins = windows_of(pid)
    if not wins:
        return None
    if title:
        exact = [w for w in wins if w["title"] == title]
        if exact:
            return exact[0]
    return max(wins, key=lambda w: w["client_w"] * w["client_h"])


def responsive(hwnd: int, timeout_ms: int = 2000) -> bool:
    """WM_NULL round trip through the window's message queue."""
    result = ctypes.c_size_t()
    SMTO_ABORTIFHUNG = 0x2
    ok = user32.SendMessageTimeoutW(hwnd, 0, 0, 0, SMTO_ABORTIFHUNG, timeout_ms, ctypes.byref(result))
    return bool(ok)


def focus(hwnd: int) -> bool:
    """Bring the window to the foreground (restoring it if minimized)."""
    SW_RESTORE = 9
    if user32.IsIconic(hwnd):
        user32.ShowWindow(hwnd, SW_RESTORE)
    # Windows only lets the foreground process change the foreground window.
    # Pressing and releasing Alt, or attaching to the foreground thread's
    # input, lifts that restriction.
    fg = user32.GetForegroundWindow()
    fg_thread = user32.GetWindowThreadProcessId(fg, None) if fg else 0
    me = kernel32.GetCurrentThreadId()
    attached = False
    if fg_thread and fg_thread != me:
        attached = bool(user32.AttachThreadInput(me, fg_thread, True))
    try:
        _key_event(0x12, False)
        _key_event(0x12, True)
        user32.BringWindowToTop(hwnd)
        user32.SetForegroundWindow(hwnd)
    finally:
        if attached:
            user32.AttachThreadInput(me, fg_thread, False)
    time.sleep(0.15)
    return int(user32.GetForegroundWindow() or 0) == int(hwnd)


def close_window(hwnd: int) -> None:
    WM_CLOSE = 0x10
    user32.PostMessageW(hwnd, WM_CLOSE, 0, 0)


# --------------------------------------------------------------------------
# Capture


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [("biSize", wintypes.DWORD), ("biWidth", ctypes.c_long), ("biHeight", ctypes.c_long),
                ("biPlanes", wintypes.WORD), ("biBitCount", wintypes.WORD), ("biCompression", wintypes.DWORD),
                ("biSizeImage", wintypes.DWORD), ("biXPelsPerMeter", ctypes.c_long),
                ("biYPelsPerMeter", ctypes.c_long), ("biClrUsed", wintypes.DWORD),
                ("biClrImportant", wintypes.DWORD)]


user32.GetDC.argtypes = [wintypes.HWND]
user32.GetDC.restype = wintypes.HDC
user32.ReleaseDC.argtypes = [wintypes.HWND, wintypes.HDC]
user32.PrintWindow.argtypes = [wintypes.HWND, wintypes.HDC, wintypes.UINT]
gdi32.CreateCompatibleDC.argtypes = [wintypes.HDC]
gdi32.CreateCompatibleDC.restype = wintypes.HDC
gdi32.CreateCompatibleBitmap.argtypes = [wintypes.HDC, ctypes.c_int, ctypes.c_int]
gdi32.CreateCompatibleBitmap.restype = wintypes.HBITMAP
gdi32.SelectObject.argtypes = [wintypes.HDC, wintypes.HGDIOBJ]
gdi32.SelectObject.restype = wintypes.HGDIOBJ
gdi32.BitBlt.argtypes = [wintypes.HDC, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, wintypes.HDC,
                         ctypes.c_int, ctypes.c_int, wintypes.DWORD]
gdi32.GetDIBits.argtypes = [wintypes.HDC, wintypes.HBITMAP, wintypes.UINT, wintypes.UINT, ctypes.c_void_p,
                            ctypes.c_void_p, wintypes.UINT]
gdi32.DeleteObject.argtypes = [wintypes.HGDIOBJ]
gdi32.DeleteDC.argtypes = [wintypes.HDC]


def capture_client(hwnd: int, method: str = "print") -> tuple[int, int, bytes]:
    """Client area pixels as (width, height, top-down BGRA).

    method "print" uses PrintWindow(PW_CLIENTONLY | PW_RENDERFULLCONTENT),
    which works for a covered window; "screen" copies from the screen
    (the window must be visible and on top).
    """
    client = wintypes.RECT()
    user32.GetClientRect(hwnd, ctypes.byref(client))
    w, h = client.right, client.bottom
    if w <= 0 or h <= 0:
        raise RuntimeError("window has an empty client area (minimized?)")
    if method == "screen":
        src_dc = user32.GetDC(None)
        origin = wintypes.POINT(0, 0)
        user32.ClientToScreen(hwnd, ctypes.byref(origin))
    else:
        src_dc = user32.GetDC(hwnd)
        origin = None
    mem_dc = gdi32.CreateCompatibleDC(src_dc)
    bmp = gdi32.CreateCompatibleBitmap(src_dc, w, h)
    old = gdi32.SelectObject(mem_dc, bmp)
    try:
        if method == "screen":
            SRCCOPY = 0x00CC0020
            if not gdi32.BitBlt(mem_dc, 0, 0, w, h, src_dc, origin.x, origin.y, SRCCOPY):
                raise RuntimeError("BitBlt failed")
        else:
            if not user32.PrintWindow(hwnd, mem_dc, 0x1 | 0x2):
                raise RuntimeError("PrintWindow failed")
        info = BITMAPINFOHEADER()
        info.biSize = ctypes.sizeof(info)
        info.biWidth = w
        info.biHeight = -h  # top-down
        info.biPlanes = 1
        info.biBitCount = 32
        buf = ctypes.create_string_buffer(w * h * 4)
        if not gdi32.GetDIBits(mem_dc, bmp, 0, h, buf, ctypes.byref(info), 0):
            raise RuntimeError("GetDIBits failed")
        return w, h, buf.raw
    finally:
        gdi32.SelectObject(mem_dc, old)
        gdi32.DeleteObject(bmp)
        gdi32.DeleteDC(mem_dc)
        user32.ReleaseDC(hwnd if method != "screen" else None, src_dc)


# --------------------------------------------------------------------------
# Synthetic input (needs the window in the foreground)

INPUT_MOUSE, INPUT_KEYBOARD = 0, 1
KEYEVENTF_KEYUP = 0x2
MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP = 0x2, 0x4


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long), ("dy", ctypes.c_long), ("mouseData", wintypes.DWORD),
                ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.c_size_t)]


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [("wVk", wintypes.WORD), ("wScan", wintypes.WORD), ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.c_size_t)]


class _INPUTUNION(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT), ("ki", KEYBDINPUT), ("pad", ctypes.c_byte * 32)]


class INPUT(ctypes.Structure):
    _fields_ = [("type", wintypes.DWORD), ("u", _INPUTUNION)]


user32.SendInput.argtypes = [wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int]
user32.SetCursorPos.argtypes = [ctypes.c_int, ctypes.c_int]

VK = {"space": 0x20, "enter": 0x0D, "return": 0x0D, "escape": 0x1B, "esc": 0x1B, "back": 0x1B, "tab": 0x09,
      "backspace": 0x08, "shift": 0x10, "ctrl": 0x11, "alt": 0x12, "left": 0x25, "up": 0x26, "right": 0x27,
      "down": 0x28, "delete": 0x2E, "home": 0x24, "end": 0x23, "pageup": 0x21, "pagedown": 0x22}


def vk_code(name: str) -> int:
    n = name.lower()
    if n in VK:
        return VK[n]
    if len(n) == 1 and n.isalnum():
        return ord(n.upper())
    if n.startswith("f") and n[1:].isdigit() and 1 <= int(n[1:]) <= 12:
        return 0x70 + int(n[1:]) - 1
    raise ValueError(f"unknown key: {name}")


def _key_event(vk: int, up: bool) -> None:
    inp = INPUT(type=INPUT_KEYBOARD)
    inp.u.ki = KEYBDINPUT(wVk=vk, wScan=0, dwFlags=KEYEVENTF_KEYUP if up else 0, time=0, dwExtraInfo=0)
    user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))


def send_key(vk: int, hold: float = 0.05) -> None:
    _key_event(vk, False)
    time.sleep(hold)
    _key_event(vk, True)


def send_click(hwnd: int, cx: int, cy: int, hold: float = 0.05) -> tuple[int, int]:
    """Left click at client coordinates (pixels). Returns the screen point."""
    pt = wintypes.POINT(cx, cy)
    user32.ClientToScreen(hwnd, ctypes.byref(pt))
    user32.SetCursorPos(pt.x, pt.y)
    time.sleep(0.03)
    for flag in (MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP):
        inp = INPUT(type=INPUT_MOUSE)
        inp.u.mi = MOUSEINPUT(dx=0, dy=0, mouseData=0, dwFlags=flag, time=0, dwExtraInfo=0)
        user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))
        time.sleep(hold)
    return pt.x, pt.y


# --------------------------------------------------------------------------
# User presence


class LASTINPUTINFO(ctypes.Structure):
    _fields_ = [("cbSize", wintypes.UINT), ("dwTime", wintypes.DWORD)]


kernel32.GetTickCount.restype = wintypes.DWORD


def user_idle_seconds() -> float:
    """Seconds since the last keyboard/mouse input on this machine (any app).

    A small value means a person may be using the computer (or the game).
    """
    info = LASTINPUTINFO(cbSize=ctypes.sizeof(LASTINPUTINFO))
    if not user32.GetLastInputInfo(ctypes.byref(info)):
        return -1.0
    return ((kernel32.GetTickCount() - info.dwTime) & 0xFFFFFFFF) / 1000.0
