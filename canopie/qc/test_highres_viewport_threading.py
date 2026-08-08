"""
A high-resolution viewport tile must actually REACH the scene.

THE BUG THIS PINS (reported as "COG zooming is not happening at all -- only the
low resolution image is loaded, zooming keeps it at low resolution, but stats
do retrieve original pixels"):

`ProjectTab._request_highres_viewport_region` renders the zoomed region on a
`ThreadPoolExecutor`, then handed the finished QPixmap to the GUI with

    QTimer.singleShot(0, lambda: viewer.update_highres_overlay(...))

called from the executor's DONE-CALLBACK -- i.e. from a worker thread. A QTimer
takes the affinity of the thread that CREATES it, and that worker has no Qt
event loop, so the timer never fired.

The failure was completely silent, and misleading in a specific way: every
upstream step SUCCEEDED. Logging on the real 9.54 GB / 5-level COG showed

    [highres_viewport] arr shape: (1054, 1362, 3)   region read at a finer level
    [highres_viewport] uint8_arr: True              stretch applied
    [highres_viewport] pm: <QPixmap>                pixmap built

and then nothing. So CanoPie was doing the full amount of work on every zoom
and throwing the result away -- no error, no warning, just a permanently blurry
image. And because the export path reads level 0 independently of the viewer,
"the stats are right but the picture never sharpens" was exactly the expected
symptom, which makes it very hard to attribute.

Fixed with a Qt signal (`ImageViewer.highres_tile_ready`): emitting across
threads queues the call onto the receiver's thread, with no assumption about
the sender having an event loop. Note the 3-argument
`QTimer.singleShot(msec, context, slot)` overload -- the other obvious fix --
is NOT available in this PyQt5 build; it raises "arguments did not match any
overloaded call".
"""
import ast
import inspect
import textwrap
import threading

import numpy as np
import pytest
from PyQt5 import QtCore, QtGui

from ..image_viewer import ImageViewer
from ..project_tab import ProjectTab

pytestmark = [pytest.mark.viewer, pytest.mark.contract]


def test_viewer_exposes_a_thread_safe_tile_signal():
    """The hand-off must be a signal, which is thread-affinity-agnostic."""
    assert hasattr(ImageViewer, "highres_tile_ready"), (
        "ImageViewer needs a signal to receive tiles from worker threads")


def test_worker_does_not_hand_off_with_a_bare_qtimer():
    """THE regression, at the mechanism level.

    A bare two-argument `QTimer.singleShot` inside the worker callback is
    precisely the construct that silently dropped every tile. Assert it is not
    how the tile gets to the GUI thread.
    """
    src = textwrap.dedent(
        inspect.getsource(ProjectTab._request_highres_viewport_region))
    tree = ast.parse(src)

    # Find the function that handles the future's result.
    on_done = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_on_done":
            on_done = node
    assert on_done is not None, "_on_done callback not found"

    body = ast.dump(on_done)
    assert "singleShot" not in body, (
        "the finished tile is still handed to the GUI with QTimer.singleShot "
        "from a worker thread -- that timer has no event loop and never fires")
    assert "highres_tile_ready" in body, (
        "_on_done must emit highres_tile_ready to cross into the GUI thread")


def test_tile_emitted_from_a_worker_thread_is_applied_on_the_gui_thread(qapp):
    """BEHAVIOURAL proof, and the part a source check alone cannot give.

    Emits the signal from a real non-GUI thread (exactly what the executor
    does) and asserts the slot runs -- and runs on the GUI thread, since it
    touches the QGraphicsScene.
    """
    v = ImageViewer()
    try:
        qimg = QtGui.QImage(64, 64, QtGui.QImage.Format_RGB32)
        qimg.fill(0)
        v.set_image(QtGui.QPixmap.fromImage(qimg))
        v.enable_highres_viewport(lambda *_a, **_k: None)

        gui_thread = QtCore.QThread.currentThread()
        seen = {}

        real = v.update_highres_overlay

        def _spy(pm, sx, sy, scx, scy, request_id=None):
            seen["thread"] = QtCore.QThread.currentThread()
            seen["size"] = (pm.width(), pm.height())
            return real(pm, sx, sy, scx, scy, request_id=request_id)

        v.update_highres_overlay = _spy

        tile = QtGui.QPixmap(32, 24)
        tile.fill(QtGui.QColor("red"))

        def _worker():
            v.highres_tile_ready.emit(tile, 0.0, 0.0, 1.0, 1.0, 1)

        t = threading.Thread(target=_worker)
        t.start()
        t.join()

        # Pump until the queued slot runs.
        for _ in range(200):
            qapp.processEvents()
            if seen:
                break

        assert seen, (
            "a tile emitted from a worker thread never reached the GUI -- this "
            "is the silent drop that kept the viewer permanently blurry")
        assert seen["size"] == (32, 24)
        assert seen["thread"] is gui_thread, (
            "the overlay was applied off the GUI thread; QGraphicsScene "
            "mutation there is undefined behaviour")
    finally:
        v.close()
        v.setParent(None)
        v.deleteLater()
        qapp.processEvents()


def test_highres_tile_sits_below_annotations(viewer_factory):
    """THE stacking regression, reported as "polygons vanish when I zoom in and
    come back when I zoom out".

    Every scene item used to be at the default Z of 0 and was ordered purely by
    INSERTION ORDER. That held only while nothing was inserted between the base
    image and the annotations -- and the high-resolution zoom tile is exactly
    such an item: added later, opaque, and at Z 0.5. So the moment a zoom
    refined the view, the tile painted over every polygon and point. Zooming
    back out clears the tile, which is why they reappeared.

    This became visible only once the tile actually reached the scene; before
    that fix the overlay was silently dropped, so the bad Z was latent.
    """
    from ..image_viewer import (EditablePolygonItem, EditablePointItem,
                                HIGHRES_TILE_Z, POLYGON_Z, TEMP_DRAWING_Z)

    # The annotation items are constructed but deliberately NOT added to the
    # viewer's scene: leaving foreign items in it crashed the interpreter at
    # viewer_factory teardown (rc 127, mid-suite). Their Z is set in
    # __init__, so constructing them is enough to verify it.
    poly = EditablePolygonItem(QtGui.QPolygonF([
        QtCore.QPointF(10, 10), QtCore.QPointF(50, 10), QtCore.QPointF(50, 50)]))
    point = EditablePointItem([(20, 20)])
    try:
        assert poly.zValue() == POLYGON_Z, (
            f"polygons are at z={poly.zValue()}; they must be at {POLYGON_Z}, "
            f"above the high-res tile at {HIGHRES_TILE_Z}, or they vanish on zoom")
        assert point.zValue() == POLYGON_Z, (
            f"point annotations are at z={point.zValue()}")
    finally:
        poly.setParent(None)
        point.setParent(None)

    # The overlay's own Z, via the real code path, inside the viewer's scene.
    v = viewer_factory()
    qimg = QtGui.QImage(200, 200, QtGui.QImage.Format_RGB32)
    qimg.fill(0)
    v.set_image(QtGui.QPixmap.fromImage(qimg))
    v.enable_highres_viewport(lambda *_a, **_k: None)

    tile = QtGui.QPixmap(64, 64)
    tile.fill(QtGui.QColor("red"))
    v.update_highres_overlay(tile, 0, 0, 1.0, 1.0,
                             request_id=v._highres_request_id)

    overlay = v._highres_front_item
    assert overlay is not None, "overlay was not created"
    assert v._image.zValue() < overlay.zValue(), (
        "the high-res tile must be ABOVE the base preview image")
    assert overlay.zValue() < POLYGON_Z, (
        f"the high-res tile (z={overlay.zValue()}) is at or above the polygon "
        f"layer (z={POLYGON_Z}) -- annotations will vanish on zoom")
    assert HIGHRES_TILE_Z < POLYGON_Z < TEMP_DRAWING_Z, (
        "the documented stacking order is inconsistent")


def test_rect_zoom_band_is_given_a_z_above_the_highres_tile():
    """The crop / rectangle-zoom rubber band had no Z either, so a tile
    arriving mid-drag painted over the selection the user was drawing.

    Checked by parsing `mousePressEvent` rather than by synthesising a
    QMouseEvent and calling the handler: feeding a hand-built event into Qt's
    machinery hard-crashed the interpreter here (rc 127, mid-suite), which is
    a much worse outcome than a slightly indirect assertion. The companion
    `test_highres_tile_sits_below_annotations` covers the real stacking
    behaviour with actual scene items.
    """
    from ..image_viewer import HIGHRES_TILE_Z, TEMP_DRAWING_Z, ImageViewer

    assert TEMP_DRAWING_Z > HIGHRES_TILE_Z, (
        "the rubber-band layer must sit above the high-res tile layer")

    src = textwrap.dedent(inspect.getsource(ImageViewer.mousePressEvent))
    tree = ast.parse(src)

    # Find the statement creating _rect_zoom_item, then require a setZValue
    # call on that same attribute somewhere in the handler.
    creates = any(
        isinstance(n, ast.Attribute) and n.attr == "_rect_zoom_item"
        for n in ast.walk(tree))
    assert creates, "mousePressEvent no longer manages _rect_zoom_item"

    sets_z = any(
        isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "setZValue"
        and isinstance(n.func.value, ast.Attribute)
        and n.func.value.attr == "_rect_zoom_item"
        for n in ast.walk(tree))
    assert sets_z, (
        "the rectangle-zoom band is added to the scene without an explicit "
        "setZValue, so it defaults to 0 and the high-res tile (z=0.5) paints "
        "over the selection the user is dragging")


def test_zoom_triggers_a_refinement_request(qapp):
    """The trigger side: zooming past the fit scale must ask for a tile.
    Guards the other half of the chain -- a working hand-off is useless if
    nothing ever requests a tile."""
    v = ImageViewer()
    try:
        qimg = QtGui.QImage(256, 256, QtGui.QImage.Format_RGB32)
        qimg.fill(0)
        v.set_image(QtGui.QPixmap.fromImage(qimg))
        v.resize(400, 300)
        v._cached_pixmap_size = (256, 256)

        asked = []
        v.enable_highres_viewport(lambda vv, rect, req=None: asked.append(rect))

        v.scale(8.0, 8.0)
        v._trigger_highres_update()
        # _trigger_highres_update debounces by 200 ms, so the pump has to
        # actually let wall-clock time pass -- a tight processEvents() loop
        # finishes well inside the debounce and sees nothing.
        import time
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not asked:
            qapp.processEvents()
            time.sleep(0.02)

        assert asked, "zooming in never requested a high-resolution region"
        r = asked[-1]
        assert r.width() > 0 and r.height() > 0
    finally:
        v.close()
        v.setParent(None)
        v.deleteLater()
        qapp.processEvents()
