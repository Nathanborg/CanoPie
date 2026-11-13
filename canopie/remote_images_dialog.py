"""
File > Add Remote Images (URLs)...: open Cloud-Optimized GeoTIFFs that live on
a web server or Google Drive, without downloading them.

For each link the dialog checks the file (name, size, header -- a few hundred
KB at most), writes a small ``<name>.cogref`` pointer into a folder of the
user's choice, and hands the pointers to ProjectTab.open_image_paths, which
groups them into roots and viewers exactly like File > Open Image files.

The Google Drive API key is kept in this computer's CanoPie settings (when
"Remember" is ticked) and handed to remote_raster for the session. It is never
written into a pointer, so pointers can travel with a project folder.
"""

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor

from PyQt5 import QtCore, QtWidgets

from . import remote_raster as rm

_SETTINGS = ("CanoPie", "CanoPie")
KEY_SETTING = "gdrive_api_key"
DIR_SETTING = "remote_pointer_dir"
BATCH_SETTING = "last_batch_size"          # shared with Open Image files
KEY_HELP_URL = "https://developers.google.com/workspace/guides/create-credentials#api-key"


def apply_saved_api_key():
    """Start-up: give remote_raster the Drive key remembered in the settings,
    so projects with Drive pointers open without visiting the dialog."""
    try:
        key = QtCore.QSettings(*_SETTINGS).value(KEY_SETTING, "", type=str)
    except Exception:
        key = ""
    if key:
        rm.set_default_api_key(key)
    return bool(key)


def default_pointer_dir(project_tab=None):
    try:
        d = QtCore.QSettings(*_SETTINGS).value(DIR_SETTING, "", type=str)
    except Exception:
        d = ""
    if d:
        return d
    pf = getattr(project_tab, "project_folder", None)
    if pf:
        return os.path.join(os.fspath(pf), "remote_images")
    return os.path.join(os.path.expanduser("~"), "CanoPie remote images")


def _mb(n):
    return "{:,.1f} MB".format(n / 1e6) if n < 1e9 else "{:,.2f} GB".format(n / 1e9)


def describe(info):
    """One line for the results list."""
    text = "{name}: {w} x {h} px, {b} band(s), {lv} level(s), {size}".format(
        name=info["name"], w=info["width"], h=info["height"], b=info["bands"],
        lv=info["levels"], size=_mb(info["size"]))
    if info.get("warning"):
        text += "  [slow: {}]".format(info["warning"])
    return text


def parse_urls(text):
    """The links in ``text``: one per line, blank lines and ``#`` comments
    skipped, duplicates dropped, order kept."""
    out = []
    for line in str(text or "").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and line not in out:
            out.append(line)
    return out


class RemoteImagesDialog(QtWidgets.QDialog):
    """Collects links, checks them, writes pointers. After ``exec_()`` returns
    Accepted, ``result_paths`` holds the pointers to open (in link order) and
    ``batch_size`` the images per root."""

    def __init__(self, parent=None, project_tab=None, batch_size=None):
        super().__init__(parent)
        self.setWindowTitle("Add Remote Images (URLs)")
        self.setWindowFlags(self.windowFlags() & ~QtCore.Qt.WindowContextHelpButtonHint)
        self.resize(760, 560)
        self.result_paths = []
        self.batch_size = None
        self.results = []          # (url, info or None, error or None)
        settings = QtCore.QSettings(*_SETTINGS)

        lay = QtWidgets.QVBoxLayout(self)
        intro = QtWidgets.QLabel(
            "Paste links to Cloud-Optimized GeoTIFFs, one per line: an https:// link "
            "(a web server or a public bucket) or a Google Drive share link. CanoPie "
            "reads only the tiles it shows; nothing is downloaded.")
        intro.setWordWrap(True)
        lay.addWidget(intro)

        self.urls_edit = QtWidgets.QPlainTextEdit()
        self.urls_edit.setPlaceholderText(
            "https://example.org/orthomosaics/plot_A.tif\n"
            "https://drive.google.com/file/d/FILE_ID/view?usp=sharing")
        lay.addWidget(self.urls_edit, 2)

        form = QtWidgets.QFormLayout()
        row = QtWidgets.QHBoxLayout()
        self.folder_edit = QtWidgets.QLineEdit(default_pointer_dir(project_tab))
        browse = QtWidgets.QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        row.addWidget(self.folder_edit, 1)
        row.addWidget(browse)
        form.addRow("Save pointers in", row)

        self.batch_spin = QtWidgets.QSpinBox()
        self.batch_spin.setRange(1, 1000)
        if batch_size is None:
            batch_size = settings.value(BATCH_SETTING, 1, type=int)
        self.batch_spin.setValue(max(1, int(batch_size or 1)))
        self.batch_spin.setToolTip("How many images each root shows side by side "
                                   "(as in File > Open Image files)")
        form.addRow("Images per root", self.batch_spin)
        lay.addLayout(form)

        drive = QtWidgets.QGroupBox("Google Drive (only for Drive links)")
        dl = QtWidgets.QFormLayout(drive)
        krow = QtWidgets.QHBoxLayout()
        self.key_edit = QtWidgets.QLineEdit(settings.value(KEY_SETTING, "", type=str)
                                            or rm.default_api_key() or "")
        self.key_edit.setEchoMode(QtWidgets.QLineEdit.Password)
        self.key_edit.setPlaceholderText("API key (AIza…)")
        show = QtWidgets.QCheckBox("Show")
        show.toggled.connect(lambda on: self.key_edit.setEchoMode(
            QtWidgets.QLineEdit.Normal if on else QtWidgets.QLineEdit.Password))
        krow.addWidget(self.key_edit, 1)
        krow.addWidget(show)
        dl.addRow("API key", krow)
        self.remember = QtWidgets.QCheckBox(
            "Remember on this computer (kept in CanoPie's settings, never in pointer files)")
        self.remember.setChecked(bool(settings.value(KEY_SETTING, "", type=str)))
        dl.addRow("", self.remember)
        help_lbl = QtWidgets.QLabel(
            'Share each file as <i>Anyone with the link</i>. '
            '<a href="{}">How to get an API key</a>: a Google Cloud project with the '
            'Google Drive API enabled, then Credentials &gt; Create credentials &gt; API key.'
            .format(KEY_HELP_URL))
        help_lbl.setWordWrap(True)
        help_lbl.setOpenExternalLinks(True)
        dl.addRow("", help_lbl)
        lay.addWidget(drive)

        self.results_list = QtWidgets.QListWidget()
        self.results_list.setVisible(False)
        lay.addWidget(self.results_list, 1)

        buttons = QtWidgets.QDialogButtonBox()
        self.open_btn = buttons.addButton("Check and Open", QtWidgets.QDialogButtonBox.AcceptRole)
        buttons.addButton(QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._check_and_open)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)

    # -- helpers ----------------------------------------------------------

    def _browse(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Folder for the pointer files", self.folder_edit.text() or os.path.expanduser("~"))
        if d:
            self.folder_edit.setText(d)

    def _save_key(self, key):
        settings = QtCore.QSettings(*_SETTINGS)
        if key and self.remember.isChecked():
            settings.setValue(KEY_SETTING, key)
        else:
            settings.remove(KEY_SETTING)
        rm.set_default_api_key(key or None)

    def check_links(self, urls, folder, api_key, workers=4):
        """make_pointer for every link, in parallel, behind a progress dialog.
        Fills ``self.results`` in link order; returns False if cancelled."""
        progress = QtWidgets.QProgressDialog("Checking links…", "Cancel", 0, len(urls), self)
        progress.setWindowModality(QtCore.Qt.WindowModal)
        progress.setMinimumDuration(300)
        pool = ThreadPoolExecutor(max_workers=max(1, min(workers, len(urls))),
                                  thread_name_prefix="canopie-remote-check")
        futures = [pool.submit(rm.make_pointer, u, folder, api_key) for u in urls]
        cancelled = False
        try:
            while True:
                done = sum(f.done() for f in futures)
                progress.setValue(done)
                if done == len(futures):
                    break
                if progress.wasCanceled():
                    cancelled = True
                    break
                QtWidgets.QApplication.processEvents()
                time.sleep(0.03)
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
            progress.close()
        if cancelled:
            return False
        self.results = []
        for u, f in zip(urls, futures):
            try:
                self.results.append((u, f.result(), None))
            except Exception as exc:
                logging.warning("[remote] %s: %s", rm.redact(u), exc)
                self.results.append((u, None, rm.redact(str(exc))))
        return True

    def _show_results(self):
        self.results_list.clear()
        for url, info, err in self.results:
            if info is not None:
                item = QtWidgets.QListWidgetItem("OK   " + describe(info))
                item.setToolTip(info["path"])
            else:
                item = QtWidgets.QListWidgetItem("FAILED   {}  --  {}".format(rm.redact(url), err))
                item.setForeground(QtCore.Qt.red)
                item.setToolTip(err)
            self.results_list.addItem(item)
        self.results_list.setVisible(True)

    # -- the button -------------------------------------------------------

    def _check_and_open(self):
        urls = parse_urls(self.urls_edit.toPlainText())
        if not urls:
            QtWidgets.QMessageBox.information(self, "No links", "Paste at least one link.")
            return
        folder = self.folder_edit.text().strip()
        if not folder:
            QtWidgets.QMessageBox.information(self, "No folder", "Choose where to save the pointers.")
            return
        key = self.key_edit.text().strip() or None
        self._save_key(key)
        if not self.check_links(urls, folder, key):
            return
        self._show_results()
        ok = [info["path"] for _u, info, _e in self.results if info is not None]
        failed = len(self.results) - len(ok)
        if not ok:
            QtWidgets.QMessageBox.warning(
                self, "Nothing to open",
                "None of the links could be opened. The reasons are listed in the dialog.")
            return
        if failed:
            ans = QtWidgets.QMessageBox.question(
                self, "Some links failed",
                "{} of {} link(s) could not be opened (reasons in the list).\n\n"
                "Open the {} that worked?".format(failed, len(self.results), len(ok)))
            if ans != QtWidgets.QMessageBox.Yes:
                return
        settings = QtCore.QSettings(*_SETTINGS)
        settings.setValue(DIR_SETTING, folder)
        settings.setValue(BATCH_SETTING, self.batch_spin.value())
        self.result_paths = ok
        self.batch_size = self.batch_spin.value()
        self.accept()
