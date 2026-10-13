"""CanoPie launcher: `python main.py`, `python -m canopie` or the `canopie`
command after `pip install .`."""
import logging
import os
import sys


def _windows_setup():
    """Windows only: a console without QuickEdit (a click in it would pause the
    app) and an AppUserModelID (the taskbar groups CanoPie on its own, not with
    python.exe)."""
    import ctypes
    from ctypes import wintypes
    try:
        kernel32 = ctypes.windll.kernel32
        h = kernel32.GetStdHandle(-10)                         # STD_INPUT_HANDLE
        mode = wintypes.DWORD()
        if kernel32.GetConsoleMode(h, ctypes.byref(mode)):
            kernel32.SetConsoleMode(h, mode.value & ~0x0040 & ~0x0020)   # QUICK_EDIT, INSERT
    except Exception:
        pass
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("canopie.app.main.1.0")
    except Exception:
        pass


def _logo_path():
    """The logo next to the package (a source checkout or a Nuitka onefile),
    whatever the case of its extension."""
    here = os.path.dirname(os.path.abspath(__file__))
    for d in (os.path.dirname(here), here, os.path.dirname(os.path.abspath(sys.argv[0] or "."))):
        for name in ("logo.png", "logo.PNG"):
            p = os.path.join(d, name)
            if os.path.exists(p):
                return p
    return None


def main():
    if os.name == "nt":
        _windows_setup()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    from PyQt5 import QtGui
    from PyQt5.QtWidgets import QApplication
    from canopie.main_window import MainWindow
    app = QApplication(sys.argv)
    logo = _logo_path()
    icon = QtGui.QIcon(logo) if logo else None
    if icon is not None:
        app.setWindowIcon(icon)
    window = MainWindow()
    if icon is not None:
        window.setWindowIcon(icon)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
