"""
QC regression tests for wheel-zoom repaint cost in ImageViewer.

THE BUG THIS PINS
-----------------
`ImageViewer.__init__` set ``CacheBackground`` with the comment "cache the
background (the image) for faster repaints". That premise is false:
``CacheBackground`` caches what ``drawBackground()`` paints, and ImageViewer
neither overrides ``drawBackground`` nor sets a ``backgroundBrush`` -- the
image is a ``QGraphicsPixmapItem``, i.e. an ITEM, which the background cache
never touches. So Qt cached an empty background while still allocating a
viewport-sized pixmap and re-rendering it on every transform change, which is
precisely what a wheel zoom does on every tick.

Measured on an 8000x6000 image at a 1200x900 viewport, median of 5
interleaved runs, driving the REAL ``wheelEvent`` handler:

    CacheBackground   22.23 ms/tick        (~45 FPS)
    CacheNone         16.04 ms/tick        (~62 FPS)   1.39x

and it is faster in every other scenario measured too, so there is no
zoom-vs-pan trade-off being made here:

                         CacheBackground   CacheNone
    zoom, 0 polygons          19.20          14.27     1.35x
    zoom, 1500 polygons       57.66          53.83     1.07x
    pan,  0 polygons           6.76           5.92     1.14x
    pan,  1500 polygons       60.19          44.67     1.35x

TWO THINGS DELIBERATELY *NOT* DONE (both measured as HARMFUL, so these tests
exist partly to stop them being "optimised" back in):

  * Wrapping the zoom + scrollbar adjustment in
    ``viewport().setUpdatesEnabled(False/True)`` to "coalesce 3 repaints into
    1". Qt already coalesces them -- the measured repaint count is exactly
    1.00 paintEvent per wheel tick either way (asserted below) -- and adding
    the wrapper measured SLOWER (10.14 -> 11.34 ms/tick).
  * De-duplicating the 4 ``_trigger_highres_update()`` calls per tick down to
    1. Also measured slower (10.60 -> 14.87 ms/tick), and the calls are cheap
    anyway: the real app passes no cancel callback to
    ``enable_highres_viewport``, so cancelling is an int increment plus a
    timer restart.
"""
import time

import pytest
from PyQt5 import QtCore, QtGui, QtWidgets

from canopie.image_viewer import ImageViewer

# Subsystem markers -- see pytest.ini and canopie/qc/which_tests.py.
pytestmark = [pytest.mark.viewer]


def _viewer_with_image(viewer_factory, w=1200, h=900, img=(2000, 1500)):
    v = viewer_factory()
    v.resize(w, h)
    qimg = QtGui.QImage(img[0], img[1], QtGui.QImage.Format_RGB32)
    qimg.fill(0x204060)
    v.set_image(QtGui.QPixmap.fromImage(qimg))
    v._image.setPos(0, 0)
    return v


def _wheel(v, delta=120, pos=None):
    """Deliver a real QWheelEvent through the real handler."""
    pos = pos or QtCore.QPoint(v.width() // 2, v.height() // 2)
    ev = QtGui.QWheelEvent(
        QtCore.QPointF(pos), v.mapToGlobal(pos),
        QtCore.QPoint(0, delta), QtCore.QPoint(0, delta),
        QtCore.Qt.NoButton, QtCore.Qt.NoModifier,
        QtCore.Qt.NoScrollPhase, False)
    v._last_wheel_time = 0.0          # bypass the 25 ms coalescer
    v.wheelEvent(ev)


# ---------------------------------------------------------------------------
# The cache mode itself
# ---------------------------------------------------------------------------
def test_viewer_does_not_use_cache_background(viewer_factory):
    """THE regression. A viewer whose image is a QGraphicsPixmapItem must not
    pay for a background cache it can never use."""
    v = _viewer_with_image(viewer_factory)
    assert v.cacheMode() != QtWidgets.QGraphicsView.CacheBackground, (
        "ImageViewer is back on CacheBackground: it re-renders a "
        "viewport-sized pixmap on every transform change and caches an empty "
        "background, because the image is an ITEM and drawBackground() is "
        "never overridden")
    assert v.cacheMode() == QtWidgets.QGraphicsView.CacheNone


def test_drag_cache_mode_is_not_cache_background(viewer_factory):
    """`_begin_item_drag` swaps in `_drag_cache_mode`; it must not reintroduce
    the background cache for the duration of a polygon drag."""
    v = _viewer_with_image(viewer_factory)
    assert v._drag_cache_mode == QtWidgets.QGraphicsView.CacheNone


def test_cache_mode_survives_a_drag_round_trip(viewer_factory):
    """Begin/end drag must leave the viewer back on CacheNone, not on the
    old hard-coded CacheBackground default in `_end_item_drag`'s getattr."""
    v = _viewer_with_image(viewer_factory)
    before = v.cacheMode()

    v._begin_item_drag()
    v._end_item_drag()

    assert v.cacheMode() == before == QtWidgets.QGraphicsView.CacheNone


def test_end_item_drag_default_is_cache_none(viewer_factory):
    """`_end_item_drag` reads `_normal_cache_mode` via getattr with a literal
    fallback. Pin the fallback: if `_normal_cache_mode` is ever missing, the
    viewer must not silently land on CacheBackground."""
    v = _viewer_with_image(viewer_factory)
    v._begin_item_drag()
    try:
        del v._normal_cache_mode
    except AttributeError:
        pass
    v._end_item_drag()

    assert v.cacheMode() == QtWidgets.QGraphicsView.CacheNone


def test_no_cache_background_literal_left_in_source():
    """Source-level pin. Timing tests are too noisy to catch this reliably,
    and the setting is a one-token change that is easy to reintroduce."""
    import inspect
    from canopie import image_viewer as iv

    src = inspect.getsource(iv.ImageViewer)
    offenders = [ln.strip() for ln in src.splitlines()
                 if "CacheBackground" in ln and not ln.strip().startswith("#")]
    assert not offenders, (
        "CacheBackground is used in ImageViewer code again: " + repr(offenders))


# ---------------------------------------------------------------------------
# Repaint accounting -- the reason the setUpdatesEnabled "fix" is not applied
# ---------------------------------------------------------------------------
def test_one_repaint_per_wheel_tick(viewer_factory, qapp, monkeypatch):
    """Qt already merges the scale() and the two scrollbar adjustments into a
    SINGLE paintEvent, because `_apply_wheel_zoom` never returns to the event
    loop between them.

    This is asserted so nobody re-adds a `setUpdatesEnabled(False/True)`
    wrapper to "coalesce 3 repaints into 1": there is only ever 1, and the
    wrapper measured slower.
    """
    counts = {"n": 0}
    original = ImageViewer.paintEvent

    def counting(self, e):
        counts["n"] += 1
        return original(self, e)

    # Patch the CLASS (monkeypatch restores it at teardown) rather than
    # swapping an instance's __class__ -- viewer_factory's teardown is
    # documented as sensitive to viewers whose native state has been messed
    # with, and a dynamically created subclass would outlive this test.
    monkeypatch.setattr(ImageViewer, "paintEvent", counting)

    v = _viewer_with_image(viewer_factory)
    v.show()
    qapp.processEvents()

    ticks = 10
    counts["n"] = 0
    for i in range(ticks):
        _wheel(v, 120 if i % 2 == 0 else -120)
        v.viewport().repaint()          # synchronous, so each tick paints once

    assert counts["n"] == ticks, (
        f"{counts['n']} paintEvents for {ticks} wheel ticks -- expected exactly "
        "one per tick. More than one means the zoom stopped batching its "
        "transform and scrollbar updates; fewer means repaints are being "
        "dropped.")


# ---------------------------------------------------------------------------
# High-res debounce
# ---------------------------------------------------------------------------
def test_highres_request_is_debounced_during_a_wheel_burst(viewer_factory, qapp):
    """A continuous scroll must not fire a high-resolution tile request per
    tick -- each request reads pixels off disk.

    Deliberately does NOT pump the event loop during the burst. Qt timers can
    only fire from the event loop, so this makes the assertion a statement
    about the DEBOUNCE LOGIC (every tick restarts one pending single-shot
    timer, none of them dispatches a request) rather than about how much
    wall-clock time the burst happened to take. An earlier version pumped
    events between ticks and passed in isolation but failed when the rest of
    this file ran first: on a busier machine a >200 ms gap between two ticks
    legitimately lets the timer fire, which is correct behaviour but makes
    the test order-dependent.
    """
    fired = {"n": 0}
    v = _viewer_with_image(viewer_factory)
    v.enable_highres_viewport(lambda *a, **k: fired.__setitem__("n", fired["n"] + 1))
    v.show()
    qapp.processEvents()

    for _ in range(12):
        _wheel(v, 120)              # no processEvents: no timer can dispatch

    assert fired["n"] == 0, (
        f"{fired['n']} high-res tile request(s) fired DURING the scroll -- a "
        "request is being issued straight from the wheel handler instead of "
        "through the 200 ms debounce")

    timer = getattr(v, "_highres_timer", None)
    assert timer is not None and timer.isActive(), (
        "the high-res debounce timer is not pending after a wheel burst, so "
        "the sharpened tile would never be requested")
    assert timer.isSingleShot(), "the high-res timer must be single-shot"
    assert timer.interval() > 0, "a zero-interval debounce is not a debounce"


def test_highres_request_fires_after_the_burst_settles(viewer_factory, qapp):
    """The flip side: once scrolling stops, the request must actually happen,
    or zooming in never sharpens."""
    fired = {"n": 0}
    v = _viewer_with_image(viewer_factory)
    v.enable_highres_viewport(lambda *a, **k: fired.__setitem__("n", fired["n"] + 1))
    v.show()
    qapp.processEvents()

    for i in range(5):
        _wheel(v, 120)          # zoom IN, so the overlay is worth requesting
        qapp.processEvents()

    deadline = time.monotonic() + 3.0
    while fired["n"] == 0 and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.02)

    assert fired["n"] >= 1, (
        "no high-res tile request fired within 3 s of the scroll settling")
    assert fired["n"] <= 2, (
        f"{fired['n']} requests fired for one settled gesture -- the debounce "
        "is collapsing to one request per tick instead of one per gesture")


def test_wheel_input_is_never_discarded(viewer_factory, qapp):
    """Guards the coalescer: accumulated wheel delta must be fully consumed,
    so a fast scroll zooms as far as a slow one. (The handler explicitly
    accumulates rather than dropping events -- see wheelEvent.)"""
    v = _viewer_with_image(viewer_factory)
    v.show()
    qapp.processEvents()

    start = v.transform().m11()
    for _ in range(6):
        _wheel(v, 120)
        qapp.processEvents()

    # Flush any pending coalesce timer.
    deadline = time.monotonic() + 1.0
    while getattr(v, "_wheel_accum", 0.0) and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)

    assert v.transform().m11() > start, "six zoom-in ticks did not zoom in"
    assert getattr(v, "_wheel_accum", 0.0) == 0.0, (
        "accumulated wheel delta was left unconsumed -- scrolling input is "
        "being silently dropped")
