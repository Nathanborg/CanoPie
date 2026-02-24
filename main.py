import logging
from PyQt5.QtWidgets import QApplication
import sys
import os
import ctypes
from ctypes import wintypes
from PyQt5 import QtGui

# Import your main window
from canopie.main_window import MainWindow 

def disable_quick_edit():
    kernel32 = ctypes.windll.kernel32
    STD_INPUT_HANDLE = -10
    ENABLE_QUICK_EDIT_MODE = 0x0040
    ENABLE_INSERT_MODE = 0x0020
    hStdin = kernel32.GetStdHandle(STD_INPUT_HANDLE)
    mode = wintypes.DWORD()
    if kernel32.GetConsoleMode(hStdin, ctypes.byref(mode)):
        new_mode = mode.value & ~ENABLE_QUICK_EDIT_MODE & ~ENABLE_INSERT_MODE
        kernel32.SetConsoleMode(hStdin, new_mode)

try:
    disable_quick_edit()
except Exception:
    pass

def main():
    # --- CRITICAL FIX 1: The AppID ---
    # Without this, Windows groups your app with "python" or generic processes
    myappid = 'canopie.app.main.1.0' 
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
    # ---------------------------------

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # --- CRITICAL FIX 2: Correct Path for Nuitka Onefile ---
    # In Nuitka Onefile, __file__ points to the temp directory where assets are unpacked.
    # We use that to find the bundled logo.png.
    base_dir = os.path.dirname(os.path.abspath(__file__))
    logo_path = os.path.join(base_dir, "logo.png")

    app = QApplication(sys.argv)
    
    # Load icon
    if os.path.exists(logo_path):
        app_icon = QtGui.QIcon(logo_path)
        app.setWindowIcon(app_icon)
    else:
        # Debug print if it fails (will show if you run in console mode)
        print(f"WARNING: Could not find icon at {logo_path}")

    window = MainWindow()
    if os.path.exists(logo_path):
        window.setWindowIcon(app_icon)

    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()