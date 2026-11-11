"""`CanoPie.exe --selftest` must cover every module of the package and pass from source."""
import subprocess

from .. import selftest


def test_module_list_matches_the_tracked_package():
    out = subprocess.run(["git", "ls-files", "canopie/*.py"], capture_output=True, text=True,
                         cwd=__file__.rsplit("canopie", 1)[0])
    if out.returncode != 0 or not out.stdout.strip():
        import pytest
        pytest.skip("not a git checkout")
    tracked = {p.split("/")[-1][:-3] for p in out.stdout.split()
               if p.count("/") == 1 and not p.split("/")[-1].startswith("_")}
    tracked.discard("selftest")
    assert tracked == set(selftest.CANOPIE_MODULES), (
        "update selftest.CANOPIE_MODULES: missing %s, stale %s" % (
            sorted(tracked - set(selftest.CANOPIE_MODULES)),
            sorted(set(selftest.CANOPIE_MODULES) - tracked)))


def test_selftest_passes_from_source(tmp_path):
    report = tmp_path / "selftest.txt"
    code = selftest.run(str(report))
    text = report.read_text(encoding="utf-8")
    assert code == 0, text
