# log_window.py
import logging, os, sys, tempfile
from PyQt5 import QtCore, QtWidgets, QtGui

class QtLogHandler(logging.Handler, QtCore.QObject):
    sig = QtCore.pyqtSignal(str, int)  # (formatted message, level)
    def __init__(self, parent=None):
        QtCore.QObject.__init__(self, parent)
        logging.Handler.__init__(self)
    def emit(self, record):
        try:
            msg = self.format(record)
        except Exception:
            msg = record.getMessage()
        self.sig.emit(msg, record.levelno)

class LogWindow(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("CanoPie — Logs")
        self.resize(900, 500)
        self.view = QtWidgets.QPlainTextEdit(readOnly=True)
        self.view.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)

        self.pause = QtWidgets.QCheckBox("Pause autoscroll")
        self.clear_btn = QtWidgets.QPushButton("Clear")
        self.save_btn  = QtWidgets.QPushButton("Save…")

        hl = QtWidgets.QHBoxLayout()
        hl.addWidget(self.pause); hl.addStretch(1); hl.addWidget(self.clear_btn); hl.addWidget(self.save_btn)

        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(self.view); lay.addLayout(hl)

        self.clear_btn.clicked.connect(self.view.clear)
        self.save_btn.clicked.connect(self._save)

    @QtCore.pyqtSlot(str, int)
    def append(self, text, level):
        # color by level
        color = {logging.DEBUG:"#7f8c8d", logging.INFO:"#dcdcdc",
                 logging.WARNING:"#f1c40f", logging.ERROR:"#e74c3c",
                 logging.CRITICAL:"#c0392b"}.get(level, "#dcdcdc")
        self.view.appendHtml(f'<pre style="margin:0;color:{color}">{QtGui.QGuiApplication.translate("", text)}</pre>')
        if not self.pause.isChecked():
            self.view.moveCursor(QtGui.QTextCursor.End)

    def _save(self):
        fp, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save log", "canopie.log", "Log (*.log);;Text (*.txt)")
        if fp:
            with open(fp, "w", encoding="utf-8") as f:
                f.write(self.view.toPlainText())

def setup_logging_to_gui(app_name="CanoPie", level=logging.INFO, parent=None):
    """Returns (logger, gui_handler, log_window). Call once in MainWindow/init."""
    logger = logging.getLogger()
    logger.setLevel(level)

    fmt = logging.Formatter("%(asctime)s  %(levelname)-8s %(name)s: %(message)s", "%H:%M:%S")

    # GUI handler
    gui_handler = QtLogHandler(parent)
    gui_handler.setFormatter(fmt)
    logger.addHandler(gui_handler)

    # File handler (rotating)
    from logging.handlers import RotatingFileHandler
    logdir = os.path.join(tempfile.gettempdir(), app_name)
    os.makedirs(logdir, exist_ok=True)
    logfile = os.path.join(logdir, "app.log")
    fh = RotatingFileHandler(logfile, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(level)
    logger.addHandler(fh)

    # Redirect print()/stderr to logging
    class _StreamToLogger:
        def __init__(self, lvl): self.lvl = lvl
        def write(self, buf):
            for line in buf.splitlines():
                if line.strip(): logger.log(self.lvl, line)
        def flush(self): pass
    sys.stdout = _StreamToLogger(logging.INFO)
    sys.stderr = _StreamToLogger(logging.ERROR)

    # Uncaught exceptions → logging
    def _excepthook(etype, value, tb):
        logging.getLogger("uncaught").exception("Uncaught exception", exc_info=(etype, value, tb))
        # Show the window on crash, if available
        try: logwin.show(); logwin.raise_()
        except Exception: pass
    sys.excepthook = _excepthook

    # Create window and wire it
    logwin = LogWindow(parent=parent)
    gui_handler.sig.connect(logwin.append)

    return logger, gui_handler, logwin, logfile






def setup_logging_light(app_name="CanoPie", level=logging.INFO):
    logger = logging.getLogger()
    logger.setLevel(level)

    # rotating file handler in temp dir
    from logging.handlers import RotatingFileHandler
    logdir = os.path.join(tempfile.gettempdir(), app_name)
    os.makedirs(logdir, exist_ok=True)
    logfile = os.path.join(logdir, "app.log")

    fmt = logging.Formatter("%(asctime)s  %(levelname)-8s %(name)s: %(message)s", "%H:%M:%S")
    fh = RotatingFileHandler(logfile, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(level)
    logger.addHandler(fh)

    # also print to console in dev runs (PyInstaller --windowed ignores console)
    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setFormatter(fmt)
    sh.setLevel(level)
    logger.addHandler(sh)

    # redirect print()/stderr into logging so you don’t miss anything
    class _StreamToLogger:
        def __init__(self, lvl): self.lvl = lvl
        def write(self, buf):
            for line in buf.splitlines():
                if line.strip():
                    logger.log(self.lvl, line)
        def flush(self): pass
    sys.stdout = _StreamToLogger(logging.INFO)
    sys.stderr = _StreamToLogger(logging.ERROR)

    # uncaught exceptions → log
    def _excepthook(etype, value, tb):
        logging.getLogger("uncaught").exception("Uncaught exception", exc_info=(etype, value, tb))
    sys.excepthook = _excepthook

    return logger, logfile
