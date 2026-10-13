"""Without exiftool, CanoPie asks for it only from an explicit EXIF action, once
per session, on the GUI thread -- never while saving polygons or from a
background export (the "Locate exiftool.exe" dialog blocked a polygon save, and
the CI suite, on a machine without exiftool)."""
import threading
import types

import pytest

from .. import project_tab as PT

pytestmark = [pytest.mark.io]


@pytest.fixture
def no_exiftool(monkeypatch, qapp):
    from PyQt5 import QtWidgets
    asked = []
    monkeypatch.setattr(PT, "get_exif_data_exiftool_multiple", lambda *a, **k: {}, raising=False)
    monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: (asked.append(a[1] if len(a) > 1 else "") or ("", ""))))
    monkeypatch.setattr(PT.ProjectTab, "_exiftool_prompt_declined", False)

    def missing(*a, **k):
        raise FileNotFoundError("exiftool")
    fake = types.SimpleNamespace(exiftool_path=None, _extract_exif_via_subprocess=missing,
                                 set_exiftool_path=lambda p: None)
    fake._exiftool_may_ask = lambda ask: PT.ProjectTab._exiftool_may_ask(fake, ask)

    def lookup(files, ask=False):
        return PT.ProjectTab._get_exif_data_with_optional_path(fake, files, ask=ask)
    return lookup, asked


def test_an_incidental_lookup_never_asks(no_exiftool, tmp_path):
    lookup, asked = no_exiftool
    f = str(tmp_path / "a.jpg")
    out = lookup([f])                                   # e.g. the GPS of a root while saving polygons
    assert asked == [] and list(out.values()) == [{}]


def test_an_explicit_export_asks_once_per_session(no_exiftool, tmp_path):
    lookup, asked = no_exiftool
    f = str(tmp_path / "a.jpg")
    assert list(lookup([f], ask=True).values()) == [{}]     # cancelled: goes on without EXIF
    assert len(asked) == 1 and "exiftool" in asked[0].lower()
    lookup([f], ask=True)
    assert len(asked) == 1, "asked again after the user cancelled"


def test_a_background_thread_never_asks(no_exiftool, tmp_path):
    lookup, asked = no_exiftool
    f = str(tmp_path / "a.jpg")
    out = []
    t = threading.Thread(target=lambda: out.append(lookup([f], ask=True)), daemon=True)
    t.start()
    t.join(10)
    assert out and asked == []
