"""Desktop launcher: starts the Time Cube server (if needed) and opens the app window."""
import ctypes
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PORT = int(os.environ.get("TIME_CUBE_PORT") or 47963)
URL = f"http://127.0.0.1:{PORT}/"
LOCAL = os.environ.get("LOCALAPPDATA", APP_DIR)


def probe():
    """'ours' if Time Cube answers on the port, 'other' if something else does, None if free."""
    try:
        with urllib.request.urlopen(URL + "api/ping", timeout=0.8) as r:
            return "ours" if b"time-cube" in r.read() else "other"
    except urllib.error.HTTPError:
        return "other"
    except Exception:
        return None


def find_browser():
    pf, pf86 = os.environ.get("ProgramFiles", r"C:\Program Files"), os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    for p in (os.path.join(pf, r"Google\Chrome\Application\chrome.exe"),
              os.path.join(pf86, r"Google\Chrome\Application\chrome.exe"),
              os.path.join(LOCAL, r"Google\Chrome\Application\chrome.exe"),
              os.path.join(pf86, r"Microsoft\Edge\Application\msedge.exe"),
              os.path.join(pf, r"Microsoft\Edge\Application\msedge.exe")):
        if os.path.exists(p):
            return p
    return None


def open_window():
    exe = find_browser()
    if not exe:
        webbrowser.open(URL)
        return
    profile = os.path.join(LOCAL, "TimeCube", "window")
    # Ask for a normal, visible window: without this Chrome can inherit a minimised show state from its parent.
    si = None
    if hasattr(subprocess, "STARTUPINFO"):
        si = subprocess.STARTUPINFO()
        si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        si.wShowWindow = 1  # SW_SHOWNORMAL
    subprocess.Popen([exe, f"--app={URL}", f"--user-data-dir={profile}", "--no-first-run",
                      "--no-default-browser-check", "--disable-features=Translate", "--window-size=1500,940",
                      "--autoplay-policy=no-user-gesture-required"],
                     startupinfo=si)


def main():
    who = probe()
    if who == "other":
        ctypes.windll.user32.MessageBoxW(None, f"Port {PORT} is used by another program, so Time Cube cannot "
                                         "start. Close that program and try again.", "Time Cube", 0x10)
        return
    if who == "ours":
        open_window()
        return

    sys.path.insert(0, APP_DIR)
    import server

    data = server.DATA_DIR
    os.makedirs(data, exist_ok=True)
    if sys.stdout is None or sys.stderr is None:  # pythonw has no console
        sys.stdout = sys.stderr = open(os.path.join(data, "app.log"), "a", buffering=1, encoding="utf-8")

    def opener():
        for _ in range(300):
            if probe() == "ours":
                break
            time.sleep(0.1)
        open_window()

    threading.Thread(target=opener, daemon=True).start()
    server.serve(auto_exit=True)


if __name__ == "__main__":
    main()
