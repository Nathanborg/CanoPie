import logging
from PyQt5.QtWidgets import QApplication
import sys

from canopie.main_window import MainWindow


def main():
    # Configure global logging to stderr only
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
