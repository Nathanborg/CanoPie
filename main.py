import logging
from PyQt5.QtWidgets import QApplication
import sys

from canopie.main_window import MainWindow
from PyQt5 import QtWidgets, QtGui
import os

def main():
    # Configure global logging to stderr only
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Path to logo relative to this file
    base_dir = os.path.dirname(os.path.abspath(__file__))
    logo_path = os.path.join(base_dir, "logo.png")

    app = QApplication(sys.argv)
    app.setWindowIcon(QtGui.QIcon(logo_path))  # app icon

    from canopie.main_window import MainWindow  
    window = MainWindow()
    window.setWindowIcon(QtGui.QIcon(logo_path))  # taskbar/alt-tab icon

    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()

