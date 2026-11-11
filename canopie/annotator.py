"""
Annotator: custom per-polygon properties on top of the Polygon Manager.

The user defines properties ("flower cover", "phenophase", ...) and fills them
in polygon by polygon.

* A property is NUMERIC (a slider with an N/A box) or CATEGORICAL (a dropdown,
  or a set of tags when several may apply at once).
* Its SCOPE is the whole project, one root, or one image. It can also be
  restricted to polygons with one label (the record's ``name``) or apply to
  every polygon.
* Whether a property applies to a polygon is worked out when asked, from the
  polygon's group, image, root and label. So a new property covers every
  existing polygon in its scope, and every polygon drawn later, without
  writing anything to them.

STORAGE
-------
The VALUES live in each polygon's own ``properties`` dict, under the
property's name, in plain types:

* a number (numeric), a choice string (dropdown), or tags joined with ';'
  ("bee;bird");
* ``None`` (JSON null) for N/A, an explicit answer that counts as filled;
* no key at all while pending.

So they are the same thing the viewer's "Edit properties" dialog shows and
edits, and they go out with the polygon wherever its properties go: the
polygon sidecar JSON, the shapefile DBF, and the ``prop_<name>`` CSV columns.
An Annotator property named like an existing property (a shapefile field)
reads and writes that same field.

``project.json["annotator"]`` holds the DEFINITIONS and an index of the
values::

    {"version": 2,
     "properties": [{"id", "name", "kind", "scope", "scope_value", "label",
                     "minimum", "maximum", "step", "choices", "multi",
                     "export"}, ...],
     "values": {group: {normalised filepath: {property id: value}}}}

The index is a cache, never the source of truth. Most records of a large
project are LazyPolygonRecords whose sidecar is read on first access (the BCI
project: 55 s to open one dialog that read them all). Completion stats read a
record's properties when it is already in memory and the index otherwise.
Every read of a record refreshes its index entry. Version 1 projects kept the
values ONLY in the index; they are written into the polygons the first time
the Annotator opens.
"""
import logging
import os
import uuid

from PyQt5 import QtCore, QtGui, QtWidgets

from .polygon_manager import PolygonManager

SCHEMA_VERSION = 2

KIND_NUMERIC = "numeric"
KIND_CATEGORICAL = "categorical"

SCOPE_PROJECT = "project"
SCOPE_ROOT = "root"
SCOPE_IMAGE = "image"

TAG_SEP = ";"

#: get_value() result for a property nobody has filled in yet.
MISSING = object()

CHECK = "✓"   # the OK mark shown next to a complete polygon


def norm_fp(fp):
    """One spelling per file: shapefile imports store os.path.normpath
    (backslashes) while project.json and the viewer use forward slashes."""
    return os.path.normcase(os.path.normpath(str(fp))) if fp else ""


def record_label(group, record):
    """A polygon's label is its record ``name``, the group key otherwise.
    dict.get, so a lazy record is never paged in for it."""
    if isinstance(record, dict):
        name = dict.get(record, "name", None)
        if name not in (None, ""):
            return str(name)
    return str(group)


def _is_annotatable(record):
    if not isinstance(record, dict):
        return False
    # Image-editor mask polygons cut pixels out; they are not objects.
    return not dict.get(record, "is_mask_polygon", False)


def _is_unread_lazy(record):
    """A LazyPolygonRecord whose sidecar has not been read yet."""
    return getattr(record, "is_materialised", True) is False


# ---------------------------------------------------------------------------
# Data model (no Qt, unit-tested in canopie/qc/test_annotator.py)
# ---------------------------------------------------------------------------

class AnnotationStore:
    """Property definitions plus the value index, independent of any widget."""

    def __init__(self, data=None):
        self.properties = []
        self.values = {}
        #: values came from a version-1 project (index only, not yet written
        #: into the polygons' properties)
        self.legacy = False
        if data:
            self.load_dict(data)

    # ---------------------------------------------------------------- (de)serialise
    def to_dict(self):
        return {"version": SCHEMA_VERSION,
                "properties": [dict(p) for p in self.properties],
                "values": {g: {f: dict(v) for f, v in fm.items() if v}
                           for g, fm in self.values.items() if fm}}

    def load_dict(self, data):
        self.properties = []
        self.values = {}
        if not isinstance(data, dict):
            return
        for p in data.get("properties") or []:
            if isinstance(p, dict) and p.get("id") and p.get("name"):
                self.properties.append(self._normalise_def(dict(p)))
        for g, fm in (data.get("values") or {}).items():
            if not isinstance(fm, dict):
                continue
            for f, vals in fm.items():
                if isinstance(vals, dict) and vals:
                    self.values.setdefault(str(g), {})[norm_fp(f)] = dict(vals)
        self.legacy = int(data.get("version") or 1) < 2 and bool(self.values)

    def is_empty(self):
        return not self.properties and not any(self.values.values())

    # ----------------------------------------------------------------- definitions
    @staticmethod
    def _normalise_def(p):
        p.setdefault("kind", KIND_NUMERIC)
        p.setdefault("scope", SCOPE_PROJECT)
        p.setdefault("scope_value", None)
        p.setdefault("label", None)
        p["export"] = bool(p.get("export", True))
        # drawn patches set a numeric value to their % of the crown:
        # None = ask each time, True = always, False = never
        v = p.get("patch_sets_value")
        p["patch_sets_value"] = None if v is None else bool(v)
        if p["kind"] == KIND_NUMERIC:
            p["minimum"] = float(p.get("minimum", 0.0))
            p["maximum"] = float(p.get("maximum", 100.0))
            if p["maximum"] <= p["minimum"]:
                p["maximum"] = p["minimum"] + 1.0
            step = float(p.get("step", 1.0) or 1.0)
            p["step"] = step if step > 0 else 1.0
        else:
            seen, choices = set(), []
            for c in p.get("choices") or []:
                c = str(c).replace(TAG_SEP, " ").strip()
                if c and c not in seen:
                    seen.add(c)
                    choices.append(c)
            p["choices"] = choices
            p["multi"] = bool(p.get("multi", False))
        if p["scope"] == SCOPE_PROJECT:
            p["scope_value"] = None
        elif p["scope"] == SCOPE_IMAGE and p["scope_value"]:
            p["scope_value"] = str(p["scope_value"])
        if p.get("label") in ("",):
            p["label"] = None
        return p

    def add_property(self, name, kind=KIND_NUMERIC, scope=SCOPE_PROJECT,
                     scope_value=None, label=None, minimum=0.0, maximum=100.0,
                     step=1.0, choices=None, multi=False, export=True, patch_sets_value=None):
        name = (name or "").strip()
        if not name:
            raise ValueError("a property needs a name")
        if any(p["name"].lower() == name.lower() for p in self.properties):
            raise ValueError(f"a property named '{name}' already exists")
        if kind not in (KIND_NUMERIC, KIND_CATEGORICAL):
            raise ValueError(f"unknown property kind {kind!r}")
        if scope not in (SCOPE_PROJECT, SCOPE_ROOT, SCOPE_IMAGE):
            raise ValueError(f"unknown scope {scope!r}")
        if scope != SCOPE_PROJECT and not scope_value:
            raise ValueError(f"a {scope} scope needs a {scope} to apply to")
        if kind == KIND_CATEGORICAL and not [c for c in (choices or []) if str(c).strip()]:
            raise ValueError("a categorical property needs at least one choice")
        p = self._normalise_def({
            "id": uuid.uuid4().hex[:12], "name": name, "kind": kind,
            "scope": scope, "scope_value": scope_value, "label": label,
            "minimum": minimum, "maximum": maximum, "step": step,
            "choices": list(choices or []), "multi": multi, "export": export,
            "patch_sets_value": patch_sets_value})
        self.properties.append(p)
        return p["id"]

    def update_property(self, pid, **changes):
        """Change a definition. Returns the old name (the caller renames the
        key in the polygons' properties when it changed)."""
        p = self.property(pid)
        if p is None:
            raise KeyError(pid)
        old_name = p["name"]
        new_name = changes.get("name")
        if new_name is not None:
            new_name = new_name.strip()
            if not new_name:
                raise ValueError("a property needs a name")
            if any(q is not p and q["name"].lower() == new_name.lower() for q in self.properties):
                raise ValueError(f"a property named '{new_name}' already exists")
            changes["name"] = new_name
        changes.pop("id", None)
        p.update(changes)
        self._normalise_def(p)
        return old_name

    def remove_property(self, pid):
        self.properties = [p for p in self.properties if p["id"] != pid]
        for fm in self.values.values():
            for vals in fm.values():
                vals.pop(pid, None)

    def property(self, pid):
        for p in self.properties:
            if p["id"] == pid:
                return p
        return None

    # ------------------------------------------------------------------ scoping
    @staticmethod
    def applies(prop, filepath, label, root):
        scope = prop.get("scope", SCOPE_PROJECT)
        if scope == SCOPE_ROOT:
            if str(root or "") != str(prop.get("scope_value") or ""):
                return False
        elif scope == SCOPE_IMAGE:
            if norm_fp(filepath) != norm_fp(prop.get("scope_value")):
                return False
        want = prop.get("label")
        return not want or str(want) == str(label)

    def properties_for(self, filepath, label, root):
        return [p for p in self.properties if self.applies(p, filepath, label, root)]

    # ------------------------------------------------------------- value index
    def get_value(self, group, filepath, pid):
        vals = self.values.get(str(group), {}).get(norm_fp(filepath))
        if not vals or pid not in vals:
            return MISSING
        return vals[pid]

    def set_value(self, group, filepath, pid, value):
        """Index a value; ``None`` means N/A. Raises ValueError on a value
        the property cannot hold. write_record() puts it on the polygon."""
        p = self.property(pid)
        if p is None:
            raise KeyError(pid)
        value = self.coerce(p, value)
        if value is MISSING:
            self.clear_value(group, filepath, pid)
            return
        self.values.setdefault(str(group), {}).setdefault(norm_fp(filepath), {})[pid] = value

    def set_na(self, group, filepath, pid):
        self.set_value(group, filepath, pid, None)

    def clear_value(self, group, filepath, pid):
        fm = self.values.get(str(group), {})
        key = norm_fp(filepath)
        vals = fm.get(key)
        if vals:
            vals.pop(pid, None)
            if not vals:
                fm.pop(key, None)

    @staticmethod
    def coerce(prop, value):
        if value is None:
            return None                         # N/A
        if prop["kind"] == KIND_NUMERIC:
            v = float(value)
            if v != v:                          # NaN
                raise ValueError("NaN is not a value; use N/A")
            return min(max(v, prop["minimum"]), prop["maximum"])
        choices = prop.get("choices") or []
        if prop.get("multi"):
            items = [value] if isinstance(value, str) else list(value)
            bad = [c for c in items if c not in choices]
            if bad:
                raise ValueError(f"not a choice of '{prop['name']}': {bad}")
            ordered = [c for c in choices if c in items]
            return ordered if ordered else MISSING
        if value not in choices:
            raise ValueError(f"'{value}' is not a choice of '{prop['name']}'")
        return value

    # ------------------------------------------------ polygon properties <-> value
    @staticmethod
    def encode(prop, value):
        """The plain value stored in the polygon's properties."""
        if value is None:
            return None
        if isinstance(value, list):
            return TAG_SEP.join(value)
        return value

    @classmethod
    def decode(cls, prop, raw):
        """A stored property value as an Annotator value; MISSING when it is
        blank or not something this property can hold (e.g. text in a
        numeric field). Such a value is left untouched on the polygon."""
        if raw is None:
            return None
        if isinstance(raw, str):
            raw = raw.strip()
            if raw == "":
                return MISSING
        try:
            if prop["kind"] == KIND_NUMERIC:
                return cls.coerce(prop, float(raw))
            if prop.get("multi"):
                items = raw if isinstance(raw, list) else [t.strip() for t in str(raw).split(TAG_SEP)]
                return cls.coerce(prop, [t for t in items if t])
            return cls.coerce(prop, str(raw))
        except (TypeError, ValueError):
            return MISSING

    def sync_record(self, group, filepath, record, force=False):
        """Refresh the index for one polygon from its properties.

        A lazy record not read yet is skipped unless ``force`` (that is what
        the index is for). Returns ``"skipped"``, ``"synced"`` or, for a
        version-1 project whose value existed only in the index,
        ``"migrated"``: the value was written into the properties, and the
        caller must mark the polygon dirty so it is saved.
        """
        if not isinstance(record, dict) or (_is_unread_lazy(record) and not force):
            return "skipped"
        props = record.get("properties") or {}
        if not isinstance(props, dict):
            props = {}
        migrated = False
        new_props = None
        for p in self.properties:
            name = p["name"]
            if name in props:
                v = self.decode(p, props[name])
                if v is MISSING:
                    self.clear_value(group, filepath, p["id"])
                else:
                    self.values.setdefault(str(group), {}).setdefault(norm_fp(filepath), {})[p["id"]] = v
            else:
                old = self.get_value(group, filepath, p["id"])
                if self.legacy and old is not MISSING:
                    if new_props is None:
                        new_props = dict(props)
                    new_props[name] = self.encode(p, old)
                    migrated = True
                else:
                    self.clear_value(group, filepath, p["id"])
        if migrated:
            record["properties"] = new_props
        return "migrated" if migrated else "synced"

    def write_record(self, group, filepath, record, pid):
        """Put one indexed value on the polygon's properties (pending removes
        the key). The properties dict is REPLACED, never mutated: a viewer
        item may hold the old one."""
        p = self.property(pid)
        if p is None or not isinstance(record, dict):
            return
        props = dict(record.get("properties") or {})    # reads a lazy record first
        v = self.get_value(group, filepath, pid)
        if v is MISSING:
            props.pop(p["name"], None)
        else:
            props[p["name"]] = self.encode(p, v)
        record["properties"] = props

    # ------------------------------------------------------------------- status
    def status(self, group, filepath, label, root):
        """(filled, total) over the properties that apply to one polygon."""
        props = self.properties_for(filepath, label, root)
        vals = self.values.get(str(group), {}).get(norm_fp(filepath)) or {}
        filled = sum(1 for p in props if p["id"] in vals)
        return filled, len(props)

    def is_complete(self, group, filepath, label, root):
        filled, total = self.status(group, filepath, label, root)
        return total > 0 and filled == total

    def missing_properties(self, group, filepath, label, root):
        vals = self.values.get(str(group), {}).get(norm_fp(filepath)) or {}
        return [p for p in self.properties_for(filepath, label, root) if p["id"] not in vals]

    @staticmethod
    def iter_polygons(all_polygons, root_of):
        """(group, filepath, label, root) for every annotatable polygon."""
        for group, file_map in (all_polygons or {}).items():
            if not isinstance(file_map, dict):
                continue
            for fp, rec in file_map.items():
                if not _is_annotatable(rec):
                    continue
                try:
                    root = root_of(fp) if root_of else None
                except Exception:
                    root = None
                yield group, fp, record_label(group, rec), root

    def completion_by_label(self, all_polygons, root_of):
        """{label: {filled, total, polygons, complete, pct}} across images.
        ``pct`` is filled values over values asked for; None when no property
        applies to that label at all."""
        out = {}
        for group, fp, label, root in self.iter_polygons(all_polygons, root_of):
            filled, total = self.status(group, fp, label, root)
            s = out.setdefault(label, {"filled": 0, "total": 0, "polygons": 0, "complete": 0})
            s["filled"] += filled
            s["total"] += total
            s["polygons"] += 1
            if total and filled == total:
                s["complete"] += 1
        for s in out.values():
            s["pct"] = (100.0 * s["filled"] / s["total"]) if s["total"] else None
        return out

    def pending(self, all_polygons, root_of):
        """The to-do list: one entry per polygon with something still missing."""
        todo = []
        for group, fp, label, root in self.iter_polygons(all_polygons, root_of):
            missing = self.missing_properties(group, fp, label, root)
            if missing:
                todo.append({"group": group, "filepath": fp, "label": label, "root": root,
                             "missing": [p["name"] for p in missing],
                             "missing_ids": [p["id"] for p in missing]})
        return todo

    def property_stats(self, all_polygons, root_of):
        """{property id: {"applies", "filled", "na", "missing"}}."""
        stats = {p["id"]: {"applies": 0, "filled": 0, "na": 0, "missing": 0} for p in self.properties}
        for group, fp, label, root in self.iter_polygons(all_polygons, root_of):
            vals = self.values.get(str(group), {}).get(norm_fp(fp)) or {}
            for p in self.properties_for(fp, label, root):
                s = stats[p["id"]]
                s["applies"] += 1
                if p["id"] in vals:
                    s["filled"] += 1
                    if vals[p["id"]] is None:
                        s["na"] += 1
                else:
                    s["missing"] += 1
        return stats

    def prune(self, all_polygons):
        """Drop index entries of polygons that no longer exist."""
        live = {}
        for group, file_map in (all_polygons or {}).items():
            if isinstance(file_map, dict):
                live[str(group)] = {norm_fp(fp) for fp in file_map}
        removed = 0
        for group in list(self.values):
            keep = live.get(group, set())
            fm = self.values[group]
            for key in list(fm):
                if key not in keep:
                    del fm[key]
                    removed += 1
            if not fm:
                del self.values[group]
        return removed


def format_value(prop, value):
    if value is MISSING:
        return "—"
    if value is None:
        return "N/A"
    if prop["kind"] == KIND_NUMERIC:
        step = prop.get("step", 1.0)
        return f"{value:.0f}" if float(step).is_integer() else f"{value:g}"
    if isinstance(value, list):
        return ", ".join(value)
    return str(value)


def describe_scope(prop):
    scope = prop.get("scope", SCOPE_PROJECT)
    if scope == SCOPE_PROJECT:
        where = "Project"
    elif scope == SCOPE_ROOT:
        where = f"Root {prop.get('scope_value')}"
    else:
        where = f"Image {os.path.basename(str(prop.get('scope_value') or ''))}"
    return where + (f" · label '{prop['label']}'" if prop.get("label") else " · all labels")


# ---------------------------------------------------------------------------
# ProjectTab glue
# ---------------------------------------------------------------------------

def store_for(project_tab):
    """The tab's AnnotationStore, created empty on first use."""
    store = getattr(project_tab, "annotator_store", None)
    if store is None:
        store = AnnotationStore()
        project_tab.annotator_store = store
    return store


def find_record(all_polygons, group, filepath):
    """(stored key, record) for a polygon, whichever spelling of the path
    all_polygons holds it under; (filepath, None) when there is none."""
    fm = (all_polygons or {}).get(group) or {}
    if filepath in fm:
        return filepath, fm[filepath]
    want = norm_fp(filepath)
    for k, rec in fm.items():
        if norm_fp(k) == want:
            return k, rec
    return filepath, None


def attach_export_values(project_tab, group, filepath, record):
    """Prepare an EXPORT COPY of a polygon: take out the Annotator properties
    whose "Include in exports" box is off. Everything else is already in the
    polygon's properties and goes out as it is (prop_<name> CSV columns,
    shapefile attributes). The properties dict is replaced, so the live
    record is never touched even through a shallow copy. Returns ``record``."""
    store = getattr(project_tab, "annotator_store", None)
    if store is None or not isinstance(record, dict):
        return record
    hidden = [p["name"] for p in store.properties if not p.get("export", True)]
    if not hidden:
        return record
    props = record.get("properties")
    if isinstance(props, dict) and any(n in props for n in hidden):
        props = {k: v for k, v in props.items() if k not in hidden}
        dict.__setitem__(record, "properties", props)
    return record


# ---------------------------------------------------------------------------
# Brush patches: areas painted inside the crown for one property
# ---------------------------------------------------------------------------
#: record["annotator_patches"][property name] = {
#:     "polygons": [[outer ring, hole ring, ...], ...]   # GeoJSON-style rings
#:     "fraction": painted area / crown area (0..1),
#:     "area_px": painted area, "crown_area_px": crown area   (in the basis below)
#:     "basis": {"w", "h"}   # the crown's own coordinate basis (image_ref_size)
#: }
#: Saved in the polygon's sidecar JSON next to 'points' (see
#: ProjectTab._sidecar_payload), in the same coordinates as 'points'.
PATCHES_KEY = "annotator_patches"


class PatchCanvas:
    """A painted mask over the crown's bounding box, clipped to the crown.

    Pure numpy / OpenCV; coordinates are in the crown's own basis (the same
    numbers as its 'points'). The mask is at most ``max_side`` pixels on its
    long side, so a crown on a full-resolution COG costs the same as a small one.
    """

    def __init__(self, crown_pts, polygons=None, max_side=2048):
        import numpy as np
        import cv2
        pts = np.asarray(crown_pts, dtype=np.float64).reshape(-1, 2)
        if len(pts) < 3:
            raise ValueError("a crown needs at least 3 vertices")
        self.x0, self.y0 = pts.min(axis=0)
        x1, y1 = pts.max(axis=0)
        span = max(x1 - self.x0, y1 - self.y0, 1e-9)
        self.s = float(max_side) / span                 # mask px per basis unit
        self.w = int(np.ceil((x1 - self.x0) * self.s)) + 2
        self.h = int(np.ceil((y1 - self.y0) * self.s)) + 2
        self.crown = np.zeros((self.h, self.w), np.uint8)
        cv2.fillPoly(self.crown, [self._to_mask(pts)], 1)
        self.mask = np.zeros_like(self.crown)
        for poly in polygons or []:
            self._fill_rings(poly, 1)
        self.mask &= self.crown

    def _to_mask(self, pts):
        import numpy as np
        a = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        return np.round((a - (self.x0, self.y0)) * self.s).astype(np.int32)

    def _fill_rings(self, rings, value):
        import cv2
        if not rings:
            return
        cv2.fillPoly(self.mask, [self._to_mask(rings[0])], value)
        for hole in rings[1:]:
            cv2.fillPoly(self.mask, [self._to_mask(hole)], 1 - value)

    def add_stroke(self, points, radius, erase=False):
        """Paint (or erase) a brush stroke: ``points`` in the basis, ``radius``
        in basis units. The result is always kept inside the crown."""
        import cv2
        if not points:
            return
        r = max(1, int(round(radius * self.s)))
        val = 0 if erase else 1
        p = self._to_mask(points)
        for i, (x, y) in enumerate(p):
            cv2.circle(self.mask, (int(x), int(y)), r, val, -1)
            if i:
                cv2.line(self.mask, tuple(map(int, p[i - 1])), (int(x), int(y)), val, 2 * r)
        self.mask &= self.crown

    def clear(self):
        self.mask[:] = 0

    def add_polygons(self, polygons):
        """Add saved patches (rings in the basis), clipped to the crown."""
        for rings in polygons or []:
            self._fill_rings(rings, 1)
        self.mask &= self.crown

    @property
    def fraction(self):
        c = int(self.crown.sum())
        return (float(self.mask.sum()) / c) if c else 0.0

    def area(self):
        return float(self.mask.sum()) / (self.s ** 2)

    def crown_area(self):
        return float(self.crown.sum()) / (self.s ** 2)

    def polygons(self):
        """The painted area as separate polygons (rings in the basis), holes kept."""
        import cv2
        contours, hier = cv2.findContours(self.mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hier is None:
            return []
        hier = hier[0]
        to_basis = lambda c: [[round(float(x) / self.s + self.x0, 3), round(float(y) / self.s + self.y0, 3)]
                              for x, y in c.reshape(-1, 2)]
        out = []
        for i, c in enumerate(contours):
            if hier[i][3] != -1 or len(c) < 3:     # holes are collected under their parent
                continue
            rings = [to_basis(c)]
            j = hier[i][2]
            while j != -1:
                if len(contours[j]) >= 3:
                    rings.append(to_basis(contours[j]))
                j = hier[j][0]
            out.append(rings)
        return out

    def to_record(self, basis_wh):
        return {"polygons": self.polygons(), "fraction": round(self.fraction, 6),
                "area_px": round(self.area(), 3), "crown_area_px": round(self.crown_area(), 3),
                "basis": {"w": basis_wh[0], "h": basis_wh[1]}}


def transfer_patches(polygons, src_crown, dst_crown):
    """Carry patches drawn on one image's crown onto the same canopy's crown
    in another image: the source crown's bounding box is mapped onto the
    target's (scale + shift per axis). That follows a crown that moved between
    flights or sits in a different image size, and is exact for a fixed
    phenocam where the crown did not move. Clipping to the target crown is
    left to PatchCanvas."""
    import numpy as np
    s = np.asarray(src_crown, dtype=np.float64).reshape(-1, 2)
    d = np.asarray(dst_crown, dtype=np.float64).reshape(-1, 2)
    s0, s1 = s.min(axis=0), s.max(axis=0)
    d0, d1 = d.min(axis=0), d.max(axis=0)
    scale = (d1 - d0) / np.maximum(s1 - s0, 1e-12)
    out = []
    for rings in polygons or []:
        out.append([((np.asarray(r, dtype=np.float64).reshape(-1, 2) - s0) * scale + d0).round(3).tolist()
                    for r in rings])
    return out


def crown_basis(record, fallback_wh=None):
    """(w, h) of the coordinate basis a record's points are in."""
    if dict.get(record, "coord_space", "image") == "norm_pixmap":
        return (1.0, 1.0)
    ref = dict.get(record, "image_ref_size") or {}
    w, h = ref.get("w") or 0, ref.get("h") or 0
    if w and h:
        return (float(w), float(h))
    return fallback_wh


def _pixmap_wh(viewer):
    cached = getattr(viewer, "_cached_pixmap_size", None)
    if cached and cached[0] and cached[1]:
        return float(cached[0]), float(cached[1])
    pm = viewer._image.pixmap()
    return float(max(1, pm.width())), float(max(1, pm.height()))


def basis_to_scene(viewer, basis_wh, pts):
    """Crown-basis points -> a scene QPolygonF (the mapping load_polygons and
    PolygonManager._get_polygon_scene_rect draw crowns with)."""
    pw, ph = _pixmap_wh(viewer)
    bw, bh = basis_wh
    local = QtGui.QPolygonF([QtCore.QPointF(float(x) / bw * pw, float(y) / bh * ph) for x, y in pts])
    return viewer._image.mapToScene(local)


def scene_to_basis(viewer, basis_wh, scene_pt):
    pw, ph = _pixmap_wh(viewer)
    bw, bh = basis_wh
    p = viewer._image.mapFromScene(scene_pt)
    return (p.x() / pw * bw, p.y() / ph * bh)


def patches_path(viewer, basis_wh, polygons):
    path = QtGui.QPainterPath()
    path.setFillRule(QtCore.Qt.OddEvenFill)
    for rings in polygons or []:
        for ring in rings:
            path.addPolygon(basis_to_scene(viewer, basis_wh, ring))
            path.closeSubpath()
    return path


_PATCH_COLORS = ["#ff9800", "#e91e63", "#03a9f4", "#ffeb3b", "#9c27b0", "#00e676"]
PATCH_Z = 1e6       # above polygons and the high-res tile


class PatchBrush(QtCore.QObject):
    """Plain brush on one viewer for one polygon + property: drag paints,
    Ctrl+drag erases, the wheel still zooms, the middle button still pans.
    Every finished stroke is written to the polygon at once."""

    changed = QtCore.pyqtSignal(float)      # painted fraction of the crown

    def __init__(self, viewer, record, prop, color, radius_screen=20.0, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        self.record = record
        self.prop = prop
        self.radius_screen = float(radius_screen)
        self.touched = False
        self.basis = crown_basis(record, None)
        if self.basis is None:
            pt = parent._project() if hasattr(parent, "_project") else None
            hw = pt.polygon_basis_hw(viewer) if pt is not None and hasattr(pt, "polygon_basis_hw") else None
            self.basis = (float(hw[1]), float(hw[0])) if hw and hw[0] else _pixmap_wh(viewer)
        existing = ((record.get(PATCHES_KEY) or {}).get(prop["name"]) or {}).get("polygons")
        self.canvas = PatchCanvas(record.get("points") or [], existing)
        self._stroke = None
        self._erase = False
        col = QtGui.QColor(color)
        fill = QtGui.QColor(col)
        fill.setAlpha(110)
        pen = QtGui.QPen(col, 1.5)
        pen.setCosmetic(True)
        scene = viewer.scene()
        self.overlay = QtWidgets.QGraphicsPathItem()
        self.overlay.setPen(pen)
        self.overlay.setBrush(QtGui.QBrush(fill))
        self.overlay.setZValue(PATCH_Z)
        self.preview = QtWidgets.QGraphicsPathItem()
        self.preview.setPen(QtGui.QPen(QtCore.Qt.NoPen))
        self.preview.setBrush(QtGui.QBrush(fill))
        self.preview.setZValue(PATCH_Z + 1)
        cpen = QtGui.QPen(QtGui.QColor("white"), 1.5)
        cpen.setCosmetic(True)
        self.cursor = QtWidgets.QGraphicsEllipseItem()
        self.cursor.setPen(cpen)
        self.cursor.setZValue(PATCH_Z + 2)
        for it in (self.overlay, self.preview, self.cursor):
            scene.addItem(it)
        self._redraw()
        viewer.viewport().installEventFilter(self)
        viewer.viewport().setMouseTracking(True)
        viewer.viewport().setCursor(QtCore.Qt.CrossCursor)

    # -------------------------------------------------------------- geometry
    def _radius_scene(self):
        return self.radius_screen / max(self.viewer.get_current_scale_factor(), 1e-9)

    def _radius_basis(self):
        pw, _ph = _pixmap_wh(self.viewer)
        return self._radius_scene() * self.basis[0] / pw

    def _redraw(self):
        self.overlay.setPath(patches_path(self.viewer, self.basis, self.canvas.polygons()))

    def _move_cursor(self, scene_pt):
        r = self._radius_scene()
        self.cursor.setRect(scene_pt.x() - r, scene_pt.y() - r, 2 * r, 2 * r)
        pen = QtGui.QPen(QtGui.QColor("#ff5252" if self._erase else "white"), 1.5)
        pen.setCosmetic(True)
        self.cursor.setPen(pen)

    def set_radius(self, px):
        self.radius_screen = float(px)
        self._move_cursor(self.viewer.mapToScene(self.viewer.viewport().mapFromGlobal(QtGui.QCursor.pos())))

    # ------------------------------------------------------------------ input
    def eventFilter(self, obj, ev):
        t = ev.type()
        if t == QtCore.QEvent.MouseButtonPress and ev.button() == QtCore.Qt.LeftButton:
            self._erase = bool(ev.modifiers() & QtCore.Qt.ControlModifier)
            self._stroke = [self.viewer.mapToScene(ev.pos())]
            self._update_preview()
            return True
        if t == QtCore.QEvent.MouseMove:
            sp = self.viewer.mapToScene(ev.pos())
            self._erase = bool(ev.modifiers() & QtCore.Qt.ControlModifier)
            self._move_cursor(sp)
            if self._stroke is not None:
                self._stroke.append(sp)
                self._update_preview()
                return True
            return False
        if t == QtCore.QEvent.MouseButtonRelease and ev.button() == QtCore.Qt.LeftButton:
            if self._stroke is not None:
                self.apply_stroke([scene_to_basis(self.viewer, self.basis, p) for p in self._stroke],
                                  self._radius_basis(), self._erase)
                self._stroke = None
                self.preview.setPath(QtGui.QPainterPath())
            return True
        if t == QtCore.QEvent.MouseButtonDblClick and ev.button() == QtCore.Qt.LeftButton:
            return True
        return False

    def _update_preview(self):
        r = self._radius_scene()
        path = QtGui.QPainterPath()
        pts = self._stroke or []
        if len(pts) >= 2:
            line = QtGui.QPainterPath(pts[0])
            for p in pts[1:]:
                line.lineTo(p)
            st = QtGui.QPainterPathStroker()
            st.setWidth(2.0 * r)
            st.setCapStyle(QtCore.Qt.RoundCap)
            st.setJoinStyle(QtCore.Qt.RoundJoin)
            path.addPath(st.createStroke(line))
        elif pts:
            path.addEllipse(pts[0], r, r)
        self.preview.setPath(path)
        self.preview.setBrush(QtGui.QBrush(QtGui.QColor(255, 82, 82, 90) if self._erase
                                           else self.overlay.brush()))

    def apply_stroke(self, basis_points, radius_basis, erase=False):
        self.canvas.add_stroke(basis_points, radius_basis, erase)
        self._store()

    def clear(self):
        self.canvas.clear()
        self._store()

    def load_polygons(self, polygons, replace=True):
        """Bring in patches from elsewhere (another image of this canopy)."""
        if replace:
            self.canvas.clear()
        self.canvas.add_polygons(polygons)
        self._store()

    def _store(self):
        self.touched = True
        patches = dict(self.record.get(PATCHES_KEY) or {})
        polys = self.canvas.polygons()
        if polys:
            patches[self.prop["name"]] = self.canvas.to_record(self.basis)
        else:
            patches.pop(self.prop["name"], None)
        self.record[PATCHES_KEY] = patches
        self._redraw()
        self.changed.emit(self.canvas.fraction)

    def stop(self):
        try:
            self.viewer.viewport().removeEventFilter(self)
            self.viewer.viewport().unsetCursor()
        except RuntimeError:
            pass
        for it in (self.overlay, self.preview, self.cursor):
            try:
                if it.scene() is not None:
                    it.scene().removeItem(it)
            except RuntimeError:
                pass


def persist_to_project_json(project_tab):
    """Write the store into project.json's "annotator" key, leaving every
    other key as it is. A no-op until the project has been saved once."""
    import json
    store = getattr(project_tab, "annotator_store", None)
    pf = getattr(project_tab, "project_folder", None)
    if store is None or not pf:
        return False
    path = os.path.join(os.fspath(pf), "project.json")
    if not os.path.exists(path):
        return False
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f) or {}
        data["annotator"] = store.to_dict()
        tmp = path + ".annotator.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        os.replace(tmp, path)
        return True
    except Exception as e:
        logging.warning("[annotator] could not update project.json: %s", e)
        return False


# ---------------------------------------------------------------------------
# Widgets
# ---------------------------------------------------------------------------

_MUTED = "color:#777;"
_ACCENT = "#2e7d32"
_PENDING = "#c62828"     # the marked viewer while something is still to fill
_CHIP_CSS = """
QToolButton { border:1px solid #bbb; border-radius:9px; padding:2px 8px; background:transparent; }
QToolButton:checked { background:#2e7d32; color:white; border-color:#2e7d32; }
QToolButton:disabled { color:#aaa; border-color:#ddd; }
"""


class _PropertyBlock(QtWidgets.QWidget):
    """One property in the panel: its name with the N/A box on the right,
    the editor below at full width (a dock is narrow)."""

    changed = QtCore.pyqtSignal(object)    # value, None (N/A) or MISSING

    def __init__(self, prop, parent=None):
        super().__init__(parent)
        self.prop = prop
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 2, 0, 6)
        outer.setSpacing(2)
        head = QtWidgets.QHBoxLayout()
        self.name_label = QtWidgets.QLabel(prop["name"])
        self.name_label.setToolTip(describe_scope(prop))
        self.na = QtWidgets.QCheckBox("N/A")
        self.na.setToolTip("Not applicable: saves an empty (null) value")
        self.draw = QtWidgets.QCheckBox("Draw")
        self.draw.setToolTip("Paint patches of this property inside the crown "
                             "(drag paints, Ctrl+drag erases)")
        self.patch_info = QtWidgets.QLabel("")
        self.patch_info.setStyleSheet(_MUTED + "font-size:11px;")
        head.addWidget(self.name_label, 1)
        head.addWidget(self.patch_info)
        head.addWidget(self.draw)
        head.addWidget(self.na)
        outer.addLayout(head)
        self.body = QtWidgets.QHBoxLayout()
        outer.addLayout(self.body)


class _NumericRow(_PropertyBlock):
    def __init__(self, prop, parent=None):
        super().__init__(prop, parent)
        lo, hi, step = prop["minimum"], prop["maximum"], prop["step"]
        self._steps = max(1, int(round((hi - lo) / step)))
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(0, self._steps)
        self.value_label = QtWidgets.QLabel("—")
        self.value_label.setMinimumWidth(48)
        self.value_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.body.addWidget(self.slider, 1)
        self.body.addWidget(self.value_label)
        self.slider.valueChanged.connect(self._on_slider)
        self.na.toggled.connect(self._on_na)

    def _slider_to_value(self, pos):
        return self.prop["minimum"] + pos * self.prop["step"] if pos < self._steps else self.prop["maximum"]

    def set_value(self, value):
        blockers = [QtCore.QSignalBlocker(self.slider), QtCore.QSignalBlocker(self.na)]
        if value is None:
            self.na.setChecked(True)
            self.slider.setEnabled(False)
        else:
            self.na.setChecked(False)
            self.slider.setEnabled(True)
            if value is not MISSING:
                pos = int(round((value - self.prop["minimum"]) / self.prop["step"]))
                self.slider.setValue(max(0, min(self._steps, pos)))
            else:
                self.slider.setValue(0)
        self.value_label.setText(format_value(self.prop, value))
        del blockers

    def _on_slider(self, pos):
        v = self._slider_to_value(pos)
        self.value_label.setText(format_value(self.prop, v))
        self.changed.emit(v)

    def _on_na(self, checked):
        self.slider.setEnabled(not checked)
        if checked:
            self.value_label.setText("N/A")
            self.changed.emit(None)
        else:
            v = self._slider_to_value(self.slider.value())
            self.value_label.setText(format_value(self.prop, v))
            self.changed.emit(v)


class _ChoiceRow(_PropertyBlock):
    CHIPS_PER_LINE = 3

    def __init__(self, prop, parent=None):
        super().__init__(prop, parent)
        self.combo = None
        self.chips = []
        if prop.get("multi"):
            box = QtWidgets.QWidget()
            grid = QtWidgets.QGridLayout(box)
            grid.setContentsMargins(0, 0, 0, 0)
            grid.setSpacing(4)
            for i, c in enumerate(prop["choices"]):
                b = QtWidgets.QToolButton()
                b.setText(c)
                b.setCheckable(True)
                b.setStyleSheet(_CHIP_CSS)
                b.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
                b.toggled.connect(self._on_chips)
                grid.addWidget(b, i // self.CHIPS_PER_LINE, i % self.CHIPS_PER_LINE)
                self.chips.append(b)
            self.body.addWidget(box, 1)
        else:
            self.combo = QtWidgets.QComboBox()
            self.combo.addItem("—", MISSING)
            for c in prop["choices"]:
                self.combo.addItem(c, c)
            self.combo.currentIndexChanged.connect(self._on_combo)
            self.body.addWidget(self.combo, 1)
        self.na.toggled.connect(self._on_na)

    def _editors(self):
        return [self.combo] if self.combo is not None else list(self.chips)

    def set_value(self, value):
        blockers = [QtCore.QSignalBlocker(w) for w in self._editors() + [self.na]]
        self.na.setChecked(value is None)
        for w in self._editors():
            w.setEnabled(value is not None)
        if self.combo is not None:
            idx = self.combo.findData(value) if value not in (None, MISSING) else 0
            self.combo.setCurrentIndex(max(0, idx))
        else:
            chosen = set(value) if isinstance(value, list) else set()
            for b in self.chips:
                b.setChecked(b.text() in chosen)
        del blockers

    def _current(self):
        if self.combo is not None:
            return self.combo.currentData()
        sel = [b.text() for b in self.chips if b.isChecked()]
        return sel if sel else MISSING

    def _on_combo(self, _i):
        self.changed.emit(self._current())

    def _on_chips(self, _checked):
        self.changed.emit(self._current())

    def _on_na(self, checked):
        for w in self._editors():
            w.setEnabled(not checked)
        self.changed.emit(None if checked else self._current())


class PolygonPropertiesPanel(QtWidgets.QWidget):
    """The left-hand "Polygon Properties" panel for one polygon."""

    values_changed = QtCore.pyqtSignal(str, str, str)   # group, filepath, property id
    step_requested = QtCore.pyqtSignal(int)             # -1 / +1: same canopy, other image
    next_pending_requested = QtCore.pyqtSignal()
    draw_toggled = QtCore.pyqtSignal(str, bool)         # property id, on/off
    brush_size_changed = QtCore.pyqtSignal(int)         # screen px
    clear_patches_requested = QtCore.pyqtSignal()
    copy_previous_requested = QtCore.pyqtSignal()

    def __init__(self, store, parent=None):
        super().__init__(parent)
        self.store = store
        self.group = None
        self.filepath = None
        self.label = None
        self.root = None
        self.rows = {}
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)
        self.title = QtWidgets.QLabel("No polygon selected")
        f = self.title.font()
        f.setBold(True)
        f.setPointSize(f.pointSize() + 2)
        self.title.setFont(f)
        self.title.setWordWrap(True)
        self.subtitle = QtWidgets.QLabel("")
        self.subtitle.setStyleSheet(_MUTED)
        self.subtitle.setWordWrap(True)
        self.subtitle.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.status = QtWidgets.QLabel("")
        outer.addWidget(self.title)
        outer.addWidget(self.subtitle)
        outer.addWidget(self.status)

        # --- navigation: the same canopy in the other images, next to fill
        nav = QtWidgets.QHBoxLayout()
        nav.setSpacing(4)
        self.prev_btn = QtWidgets.QToolButton()
        self.prev_btn.setArrowType(QtCore.Qt.LeftArrow)
        self.prev_btn.setToolTip("This canopy in the previous image")
        self.next_btn = QtWidgets.QToolButton()
        self.next_btn.setArrowType(QtCore.Qt.RightArrow)
        self.next_btn.setToolTip("This canopy in the next image")
        self.where = QtWidgets.QLabel("")
        self.where.setAlignment(QtCore.Qt.AlignCenter)
        self.pending_btn = QtWidgets.QPushButton("Next to fill")
        self.pending_btn.setToolTip("Go to the next polygon with a property still to fill in")
        nav.addWidget(self.prev_btn)
        nav.addWidget(self.where, 1)
        nav.addWidget(self.next_btn)
        nav.addWidget(self.pending_btn)
        outer.addLayout(nav)
        self.prev_btn.clicked.connect(lambda: self.step_requested.emit(-1))
        self.next_btn.clicked.connect(lambda: self.step_requested.emit(1))
        self.pending_btn.clicked.connect(self.next_pending_requested.emit)
        self.set_navigation(0, 0)

        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.HLine)
        line.setStyleSheet("color:#ddd;")
        outer.addWidget(line)
        self.form_host = QtWidgets.QWidget()
        self.form = QtWidgets.QVBoxLayout(self.form_host)
        self.form.setContentsMargins(0, 4, 0, 0)
        self.form.addStretch(1)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setWidget(self.form_host)
        outer.addWidget(scroll, 1)
        self.empty_hint = QtWidgets.QLabel("")
        self.empty_hint.setStyleSheet(_MUTED)
        self.empty_hint.setWordWrap(True)
        outer.addWidget(self.empty_hint)

        # --- brush bar, shown while a property's "Draw" is ticked
        self.brush_bar = QtWidgets.QFrame()
        self.brush_bar.setStyleSheet("QFrame { background: rgba(46,125,50,30); border-radius:4px; }")
        bb = QtWidgets.QVBoxLayout(self.brush_bar)
        bb.setContentsMargins(8, 6, 8, 6)
        self.brush_title = QtWidgets.QLabel("")
        self.brush_title.setWordWrap(True)
        bb.addWidget(self.brush_title)
        self.brush_bar.setToolTip("Drag on the image to paint inside the crown; Ctrl+drag erases. "
                                  "Patches need not touch. The wheel still zooms.")
        size_row = QtWidgets.QHBoxLayout()
        size_row.addWidget(QtWidgets.QLabel("Brush"))
        self.brush_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.brush_slider.setRange(3, 200)
        self.brush_slider.setValue(20)
        self.brush_size_label = QtWidgets.QLabel("20 px")
        self.brush_size_label.setMinimumWidth(44)
        size_row.addWidget(self.brush_slider, 1)
        size_row.addWidget(self.brush_size_label)
        bb.addLayout(size_row)
        btn_row = QtWidgets.QHBoxLayout()
        self.clear_btn = QtWidgets.QPushButton("Clear")
        self.clear_btn.setToolTip("Remove every patch of this property on this polygon")
        self.done_btn = QtWidgets.QPushButton("Done")
        erase_hint = QtWidgets.QLabel("Ctrl erases")
        erase_hint.setStyleSheet(_MUTED)
        btn_row.addWidget(erase_hint, 1)
        btn_row.addWidget(self.clear_btn)
        btn_row.addWidget(self.done_btn)
        bb.addLayout(btn_row)
        self.copy_prev_btn = QtWidgets.QPushButton("From previous image")
        self.copy_prev_btn.setToolTip("Copy this property's patches from the same canopy (same polygon "
                                      "name) in the latest earlier image that has them, fitted to this "
                                      "crown -- then edit them with the brush")
        bb.addWidget(self.copy_prev_btn)
        self.copy_prev_btn.clicked.connect(self.copy_previous_requested.emit)
        outer.addWidget(self.brush_bar)
        self.brush_bar.hide()
        self.brush_slider.valueChanged.connect(self._on_brush_size)
        self.clear_btn.clicked.connect(self.clear_patches_requested.emit)
        self.done_btn.clicked.connect(lambda: self.draw_toggled.emit(self.drawing_pid or "", False))
        self.drawing_pid = None

        foot = QtWidgets.QLabel("Saved in the polygon's properties (Edit properties, shapefile, CSV).")
        foot.setStyleSheet(_MUTED + "font-size:11px;")
        foot.setWordWrap(True)
        outer.addWidget(foot)
        self.setMinimumWidth(280)

    def _on_brush_size(self, v):
        self.brush_size_label.setText(f"{v} px")
        self.brush_size_changed.emit(int(v))

    def set_drawing(self, pid, fraction=None):
        """Show which property is being drawn (None: not drawing)."""
        self.drawing_pid = pid
        for p, row in self.rows.items():
            blocker = QtCore.QSignalBlocker(row.draw)
            row.draw.setChecked(p == pid)
            del blocker
        if pid is None:
            self.brush_bar.hide()
            return
        prop = self.store.property(pid)
        pct = "" if fraction is None else f"<br>{100.0 * fraction:.1f}% of the crown"
        self.brush_title.setText(f"<b>Drawing {prop['name'] if prop else ''}</b>{pct}")
        self.brush_bar.show()

    def set_patch_info(self, pid, fraction):
        row = self.rows.get(pid)
        if row is not None:
            row.patch_info.setText("" if fraction is None else f"{100.0 * fraction:.0f}% drawn")

    def set_navigation(self, index, count):
        """``index`` is 1-based among the images holding this canopy."""
        self.where.setText(f"{index} / {count}" if count else "")
        self.prev_btn.setEnabled(count > 1)
        self.next_btn.setEnabled(count > 1)
        self.where.setToolTip("The same canopy (polygon group) in each image that has it")

    def show_polygon(self, group, filepath, label, root):
        self.group, self.filepath, self.label, self.root = group, filepath, label, root
        self.rebuild()

    def rebuild(self):
        for row in self.rows.values():
            # hide + detach now: deleteLater alone left the old polygon's
            # rows painted under the new ones until the next event loop turn
            self.form.removeWidget(row)
            row.hide()
            row.setParent(None)
            row.deleteLater()
        self.rows = {}
        if self.group is None:
            self.title.setText("No polygon selected")
            self.subtitle.setText("")
            self.status.setText("")
            self.set_navigation(0, 0)
            self.empty_hint.setText("Click a polygon in the Annotator or in a viewer.")
            return
        self.title.setText(self.label or self.group)
        img = os.path.basename(str(self.filepath))
        lines = [img + (f"  ·  root {self.root}" if self.root else "")]
        if self.group != self.label:
            lines.append(f"polygon {self.group}")
        self.subtitle.setText("\n".join(lines))
        self.subtitle.setToolTip(str(self.filepath))
        props = self.store.properties_for(self.filepath, self.label, self.root)
        self.empty_hint.setText("" if props else
                                "No property applies to this polygon. Add one in the Annotator.")
        for p in props:
            row = _NumericRow(p) if p["kind"] == KIND_NUMERIC else _ChoiceRow(p)
            row.set_value(self.store.get_value(self.group, self.filepath, p["id"]))
            row.changed.connect(lambda v, pid=p["id"]: self._on_row_changed(pid, v))
            row.draw.toggled.connect(lambda on, pid=p["id"]: self.draw_toggled.emit(pid, on))
            self.form.insertWidget(self.form.count() - 1, row)   # above the stretch
            self.rows[p["id"]] = row
        self._refresh_status()

    def _on_row_changed(self, pid, value):
        try:
            if value is MISSING:
                self.store.clear_value(self.group, self.filepath, pid)
            else:
                self.store.set_value(self.group, self.filepath, pid, value)
        except (ValueError, KeyError) as e:
            logging.warning("[annotator] rejected value for %s: %s", pid, e)
            return
        self._refresh_status()
        self.values_changed.emit(str(self.group), str(self.filepath), str(pid))

    def _refresh_status(self):
        filled, total = self.store.status(self.group, self.filepath, self.label, self.root)
        if not total:
            self.status.setText("")
        elif filled == total:
            self.status.setText(f"<span style='color:{_ACCENT}'>{CHECK} OK, all {total} filled</span>")
        else:
            self.status.setText(f"<span style='color:#b26a00'>{filled}/{total} filled</span>")


class PropertyDefinitionDialog(QtWidgets.QDialog):
    """Create or edit one property definition."""

    def __init__(self, roots, images, labels, prop=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit Property" if prop else "New Property")
        self.setWindowFlags(self.windowFlags() & ~QtCore.Qt.WindowContextHelpButtonHint)
        self._images = list(images)
        form = QtWidgets.QFormLayout(self)

        self.name = QtWidgets.QLineEdit()
        self.name.setPlaceholderText("e.g. flower cover")
        form.addRow("Name", self.name)

        self.kind = QtWidgets.QComboBox()
        self.kind.addItem("Numeric (slider)", KIND_NUMERIC)
        self.kind.addItem("Categorical", KIND_CATEGORICAL)
        form.addRow("Type", self.kind)

        self.num_box = QtWidgets.QWidget()
        nl = QtWidgets.QHBoxLayout(self.num_box)
        nl.setContentsMargins(0, 0, 0, 0)
        self.minimum = QtWidgets.QDoubleSpinBox()
        self.maximum = QtWidgets.QDoubleSpinBox()
        self.step = QtWidgets.QDoubleSpinBox()
        for sb in (self.minimum, self.maximum, self.step):
            sb.setRange(-1e9, 1e9)
            sb.setDecimals(3)
        self.minimum.setValue(0)
        self.maximum.setValue(100)
        self.step.setRange(0.001, 1e9)
        self.step.setValue(1)
        for lbl, sb in (("min", self.minimum), ("max", self.maximum), ("step", self.step)):
            nl.addWidget(QtWidgets.QLabel(lbl))
            nl.addWidget(sb)
        form.addRow("Range", self.num_box)
        self.from_patches = QtWidgets.QComboBox()
        self.from_patches.addItem("Ask", None)
        self.from_patches.addItem("Always", True)
        self.from_patches.addItem("Never", False)
        self.from_patches.setToolTip("When you finish drawing patches for this property, set its value "
                                     "to the patches' percentage of the crown")
        self.from_patches_label = QtWidgets.QLabel("Drawn % sets value")
        form.addRow(self.from_patches_label, self.from_patches)
        if prop is not None:
            self.from_patches.setCurrentIndex(max(0, self.from_patches.findData(prop.get("patch_sets_value"))))

        self.cat_box = QtWidgets.QWidget()
        cl = QtWidgets.QVBoxLayout(self.cat_box)
        cl.setContentsMargins(0, 0, 0, 0)
        self.choices = QtWidgets.QLineEdit()
        self.choices.setPlaceholderText("comma-separated, e.g. bud, flower, fruit")
        self.multi = QtWidgets.QCheckBox("Several may apply (tags)")
        cl.addWidget(self.choices)
        cl.addWidget(self.multi)
        form.addRow("Choices", self.cat_box)

        self.scope = QtWidgets.QComboBox()
        self.scope.addItem("Whole project", SCOPE_PROJECT)
        self.scope.addItem("One root", SCOPE_ROOT)
        self.scope.addItem("One image", SCOPE_IMAGE)
        form.addRow("Scope", self.scope)
        self.scope_value = QtWidgets.QComboBox()
        self.scope_value.setMinimumContentsLength(24)
        form.addRow("", self.scope_value)
        self._roots = [str(r) for r in roots]

        self.label = QtWidgets.QComboBox()
        self.label.addItem("All labels", None)
        for lb in labels:
            self.label.addItem(str(lb), str(lb))
        form.addRow("Applies to", self.label)

        self.export = QtWidgets.QCheckBox("Include in exports (CSV, shapefile)")
        self.export.setToolTip("Off: the value stays on the polygon (and in Edit properties) "
                               "but is left out of CSV and shapefile exports")
        self.export.setChecked(True if prop is None else bool(prop.get("export", True)))
        form.addRow("", self.export)

        note = QtWidgets.QLabel("Applies to every existing polygon in scope and to every one drawn "
                                "later. Values are saved in each polygon's properties under this name.")
        note.setStyleSheet(_MUTED)
        note.setWordWrap(True)
        form.addRow(note)

        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(self._accept)
        bb.rejected.connect(self.reject)
        form.addRow(bb)

        self.kind.currentIndexChanged.connect(self._sync_kind)
        self.scope.currentIndexChanged.connect(self._sync_scope)

        if prop:
            self.name.setText(prop["name"])
            self.kind.setCurrentIndex(self.kind.findData(prop["kind"]))
            if prop["kind"] == KIND_NUMERIC:
                self.minimum.setValue(prop["minimum"])
                self.maximum.setValue(prop["maximum"])
                self.step.setValue(prop["step"])
            else:
                self.choices.setText(", ".join(prop.get("choices") or []))
                self.multi.setChecked(bool(prop.get("multi")))
            self.scope.setCurrentIndex(self.scope.findData(prop["scope"]))
            if prop.get("label") and self.label.findData(prop["label"]) < 0:
                self.label.addItem(prop["label"], prop["label"])
            self.label.setCurrentIndex(max(0, self.label.findData(prop.get("label"))))
            # Changing the type would orphan every stored value.
            self.kind.setEnabled(False)
        self._sync_kind()
        self._sync_scope()
        if prop and prop.get("scope_value"):
            i = self.scope_value.findData(prop["scope_value"])
            if i < 0:
                self.scope_value.addItem(str(prop["scope_value"]), prop["scope_value"])
                i = self.scope_value.count() - 1
            self.scope_value.setCurrentIndex(i)

    def _sync_kind(self):
        numeric = self.kind.currentData() == KIND_NUMERIC
        self.num_box.setVisible(numeric)
        self.from_patches.setVisible(numeric)
        self.from_patches_label.setVisible(numeric)
        self.cat_box.setVisible(not numeric)

    def _sync_scope(self):
        self.scope_value.clear()
        s = self.scope.currentData()
        self.scope_value.setVisible(s != SCOPE_PROJECT)
        if s == SCOPE_ROOT:
            for r in self._roots:
                self.scope_value.addItem(r, r)
        elif s == SCOPE_IMAGE:
            for fp in self._images:
                self.scope_value.addItem(os.path.basename(fp), fp)

    def result_kwargs(self):
        kind = self.kind.currentData()
        out = {"name": self.name.text().strip(), "kind": kind,
               "scope": self.scope.currentData(),
               "scope_value": self.scope_value.currentData() if self.scope.currentData() != SCOPE_PROJECT else None,
               "label": self.label.currentData(),
               "export": self.export.isChecked()}
        if kind == KIND_NUMERIC:
            out.update(minimum=self.minimum.value(), maximum=self.maximum.value(), step=self.step.value(),
                       patch_sets_value=self.from_patches.currentData())
        else:
            out.update(choices=[c.strip() for c in self.choices.text().split(",") if c.strip()],
                       multi=self.multi.isChecked())
        return out

    def _accept(self):
        k = self.result_kwargs()
        problem = None
        if not k["name"]:
            problem = "Give the property a name."
        elif k["kind"] == KIND_NUMERIC and k["maximum"] <= k["minimum"]:
            problem = "The maximum must be larger than the minimum."
        elif k["kind"] == KIND_CATEGORICAL and not k["choices"]:
            problem = "List at least one choice."
        elif k["scope"] != SCOPE_PROJECT and not k["scope_value"]:
            problem = "Pick the root or image this property applies to."
        if problem:
            QtWidgets.QMessageBox.warning(self, "Property", problem)
            return
        self.accept()


class Annotator(PolygonManager):
    """Advanced polygon property editor.

    Inherits PolygonManager for its viewer mechanics only: _iter_viewers,
    _get_polygon_scene_rect, _select_group_in_viewers and zoom_to_groups all
    work off ``self.parent()``, the ProjectTab. PolygonManager.__init__ (its
    import/correction/filter UI) is deliberately not run.
    """

    ROLE_KIND = QtCore.Qt.UserRole + 1
    ROLE_KEY = QtCore.Qt.UserRole + 2

    def __init__(self, parent=None):
        QtWidgets.QDialog.__init__(self, parent)
        self.setWindowTitle("Annotator")
        self.setWindowFlags(self.windowFlags() & ~QtCore.Qt.WindowContextHelpButtonHint)
        self.resize(820, 640)
        self.store = store_for(parent) if parent is not None else AnnotationStore()
        self._persist_timer = QtCore.QTimer(self)
        self._persist_timer.setSingleShot(True)
        self._persist_timer.setInterval(600)
        self._persist_timer.timeout.connect(self.persist)
        self._refresh_timer = QtCore.QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(250)
        self._refresh_timer.timeout.connect(self.refresh)
        # Deferred work lives on timers OWNED by this dialog: a bare
        # QTimer.singleShot(lambda) outlives it and fired into deleted
        # viewers (a native crash) when the Annotator closed meanwhile.
        self._zoom_timer = QtCore.QTimer(self)
        self._zoom_timer.setSingleShot(True)
        self._zoom_timer.setInterval(120)
        self._zoom_timer.timeout.connect(self._run_pending_zoom)
        self._pending_zoom = None
        self._click_timer = QtCore.QTimer(self)
        self._click_timer.setSingleShot(True)
        self._click_timer.setInterval(0)
        self._click_timer.timeout.connect(self._run_pending_click)
        self._pending_click = None
        self._clear_timer = QtCore.QTimer(self)
        self._clear_timer.setSingleShot(True)
        self._clear_timer.setInterval(400)
        self._clear_timer.timeout.connect(self._run_pending_clear)
        self._pending_clear = None

        # One window: Polygon Properties on the left, the list on the right.
        # They move together, and the window moves off the viewer it would
        # cover (_keep_clear_of).
        right = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(right)
        lay.setContentsMargins(4, 0, 0, 0)

        # --- property definitions
        head = QtWidgets.QHBoxLayout()
        t = QtWidgets.QLabel("Properties")
        f = t.font()
        f.setBold(True)
        t.setFont(f)
        head.addWidget(t)
        head.addStretch(1)
        self.add_btn = QtWidgets.QPushButton("Add")
        self.edit_btn = QtWidgets.QPushButton("Edit")
        self.remove_btn = QtWidgets.QPushButton("Remove")
        for b in (self.add_btn, self.edit_btn, self.remove_btn):
            b.setFlat(True)
            head.addWidget(b)
        lay.addLayout(head)
        self.prop_list = QtWidgets.QListWidget()
        self.prop_list.setMaximumHeight(120)
        lay.addWidget(self.prop_list)
        self.add_btn.clicked.connect(self.on_add_property)
        self.edit_btn.clicked.connect(self.on_edit_property)
        self.remove_btn.clicked.connect(self.on_remove_property)
        self.prop_list.itemDoubleClicked.connect(lambda _i: self.on_edit_property())

        # --- polygons / stats
        self.tabs = QtWidgets.QTabWidget()
        lay.addWidget(self.tabs, 1)

        poly_page = QtWidgets.QWidget()
        pl = QtWidgets.QVBoxLayout(poly_page)
        pl.setContentsMargins(0, 6, 0, 0)
        self.search = QtWidgets.QLineEdit()
        self.search.setPlaceholderText("Filter labels or polygons")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._apply_search)
        pl.addWidget(self.search)
        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderLabels(["Polygon", "Filled"])
        self.tree.setRootIsDecorated(True)
        self.tree.setUniformRowHeights(True)
        self.tree.header().setStretchLastSection(False)
        self.tree.header().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        self.tree.header().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        self.tree.itemClicked.connect(self._on_tree_clicked)
        pl.addWidget(self.tree, 1)
        self.tabs.addTab(poly_page, "Polygons")

        stats_page = QtWidgets.QWidget()
        sl = QtWidgets.QVBoxLayout(stats_page)
        sl.setContentsMargins(0, 6, 0, 0)
        self.stats_summary = QtWidgets.QLabel("")
        self.stats_summary.setWordWrap(True)
        sl.addWidget(self.stats_summary)
        self.stats_tree = QtWidgets.QTreeWidget()
        self.stats_tree.setHeaderLabels(["Still to do", "Missing"])
        self.stats_tree.header().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        self.stats_tree.header().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        self.stats_tree.itemClicked.connect(self._on_tree_clicked)
        sl.addWidget(self.stats_tree, 1)
        self.tabs.addTab(stats_page, "Properties Stats")
        self.tabs.currentChanged.connect(lambda _i: self.refresh())

        foot = QtWidgets.QHBoxLayout()
        self.hint = QtWidgets.QLabel("")
        self.hint.setStyleSheet(_MUTED)
        foot.addWidget(self.hint, 1)
        close = QtWidgets.QPushButton("Close")
        close.clicked.connect(self.close)
        foot.addWidget(close)
        lay.addLayout(foot)

        # --- the left-hand panel
        self.panel = PolygonPropertiesPanel(self.store)
        self.panel.values_changed.connect(self._on_values_changed)
        self.panel.step_requested.connect(self.step_same_canopy)
        self.panel.next_pending_requested.connect(self.go_next_pending)
        self.panel.draw_toggled.connect(self._on_draw_toggled)
        self.panel.brush_size_changed.connect(self._on_brush_size)
        self.panel.clear_patches_requested.connect(self._on_clear_patches)
        self.panel.copy_previous_requested.connect(self.copy_patches_from_previous)
        panel_frame = QtWidgets.QFrame()
        panel_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        pf = QtWidgets.QVBoxLayout(panel_frame)
        pf.setContentsMargins(0, 0, 0, 0)
        head_lbl = QtWidgets.QLabel("Polygon Properties")
        head_lbl.setStyleSheet("font-weight:600; padding:6px 10px 0 10px;")
        pf.addWidget(head_lbl)
        pf.addWidget(self.panel, 1)
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.splitter.addWidget(panel_frame)
        self.splitter.addWidget(right)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([330, 490])
        self.splitter.setChildrenCollapsible(False)
        outer = QtWidgets.QHBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.addWidget(self.splitter)
        # → / D next to fill, ← / A previous to fill -- in this window AND in
        # the project's main window (you draw in a viewer, then want the next
        # one). Those keys step roots there normally; while the Annotator is
        # open that is borrowed and given back on close (_borrow_nav_keys).
        # Text boxes keep their own typing and cursor keys.
        self._nav_shortcuts = []
        self._borrowed = []           # [(QAction, its shortcut)]
        for keys, step in (((QtCore.Qt.Key_Right, QtCore.Qt.Key_D), 1),
                           ((QtCore.Qt.Key_Left, QtCore.Qt.Key_A), -1)):
            for k in keys:
                sc = QtWidgets.QShortcut(QtGui.QKeySequence(k), self)
                sc.setContext(QtCore.Qt.ApplicationShortcut)
                sc.setEnabled(False)
                sc.activated.connect(lambda s=step: self._nav_key(s))
                self._nav_shortcuts.append(sc)
        self.panel.pending_btn.setToolTip("Go to the next polygon with a property still to fill in "
                                          "(→ or D; ← or A goes back)")
        self.dock = None              # kept for callers; the panel is no longer docked
        self._brush = None
        self._brush_size = 20
        self._patch_items = []        # display of a polygon's saved patches
        self._tree_rows = {}          # (group, norm_fp) -> polygon row
        self._label_members = {}      # label -> [(group, filepath, root)]
        self._marked_viewers = []     # [(widget, previous stylesheet)]

    # ------------------------------------------------------------ bookkeeping
    def _project(self):
        return self.parent()

    def _all_polygons(self):
        return getattr(self._project(), "all_polygons", {}) or {}

    def _root_of(self, fp):
        pt = self._project()
        fn = getattr(pt, "get_root_by_filepath", None)
        return fn(fp) if fn else None

    def _root_order(self):
        pt = self._project()
        names = list(getattr(pt, "root_names", None) or [])
        return {str(n): i for i, n in enumerate(names)}

    def persist(self):
        pt = self._project()
        if pt is not None:
            persist_to_project_json(pt)

    def schedule_persist(self):
        self._persist_timer.start()

    def schedule_refresh(self):
        """Debounced full refresh: polygons were added, deleted or renamed."""
        self._refresh_timer.start()

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() == QtCore.QEvent.ActivationChange and self.isActiveWindow():
            self.schedule_refresh()

    def _mark_dirty(self, group, key):
        pt = self._project()
        fn = getattr(pt, "_mark_polygon_dirty", None)
        if fn is not None:
            try:
                fn(group, key)
            except Exception:
                logging.debug("[annotator] mark dirty failed", exc_info=True)

    def sync_from_polygons(self):
        """Refresh the value index from every polygon already in memory.
        A version-1 project's values are written into the polygons here,
        once (only the polygons that have values are read for it)."""
        store = self.store
        migrate = store.legacy
        for group, fm in self._all_polygons().items():
            if not isinstance(fm, dict):
                continue
            for key, rec in fm.items():
                force = migrate and bool(store.values.get(str(group), {}).get(norm_fp(key)))
                if store.sync_record(group, key, rec, force=force) == "migrated":
                    self._mark_dirty(group, key)
        if migrate:
            store.legacy = False
            self.schedule_persist()

    def record_changed(self, group, filepath):
        """The polygon's properties changed outside the Annotator (the
        viewer's Edit properties dialog)."""
        key, rec = find_record(self._all_polygons(), group, filepath)
        if rec is None:
            return
        self.store.sync_record(group, key, rec, force=True)
        self.schedule_persist()
        if self.panel.group == group and norm_fp(self.panel.filepath) == norm_fp(key):
            self.panel.rebuild()
            self._apply_mark_colors()
        if not self._update_polygon_row(group, key):
            self.schedule_refresh()

    def _on_values_changed(self, group, filepath, pid):
        key, rec = find_record(self._all_polygons(), group, filepath)
        if rec is not None:
            self.store.write_record(group, key, rec, pid)
            self._mark_dirty(group, key)
            self._push_properties_to_items(group, key, rec)
        self.schedule_persist()
        self._apply_mark_colors()
        # Only this polygon's row and its label's % change. Rebuilding the
        # whole tree per slider tick would stall a project of 10k+ crowns.
        if self.tabs.currentIndex() == 0 and self._update_polygon_row(group, filepath):
            return
        self.schedule_refresh()

    def _push_properties_to_items(self, group, filepath, rec):
        """Viewer items keep their own reference to the properties dict."""
        props = rec.get("properties")
        for w, viewer in self._iter_viewers():
            idata = w.get("image_data") if isinstance(w, dict) else None
            if norm_fp(getattr(idata, "filepath", None)) != norm_fp(filepath):
                continue
            for it in viewer.get_all_polygons() if hasattr(viewer, "get_all_polygons") else []:
                if getattr(it, "name", None) == group and hasattr(it, "properties"):
                    it.properties = props

    def _update_polygon_row(self, group, filepath):
        child = self._tree_rows.get((group, norm_fp(filepath)))
        if child is None:
            return False
        try:
            top = child.parent()
            label = top.data(0, self.ROLE_KEY)
            green = QtGui.QBrush(QtGui.QColor(_ACCENT))
            plain = self.tree.palette().brush(QtGui.QPalette.Text)
            filled = total = 0
            for g, fp, root in self._label_members.get(label, []):
                f_, t_ = self.store.status(g, fp, label, root)
                filled += f_
                total += t_
                if g == group and norm_fp(fp) == norm_fp(filepath):
                    done = bool(t_) and f_ == t_
                    child.setText(1, f"{CHECK} OK" if done else (f"{f_}/{t_}" if t_ else ""))
                    child.setForeground(1, green if done else plain)
            pct = (100.0 * filled / total) if total else None
            top.setText(1, "—" if pct is None else f"{pct:.0f}% filled")
            top.setForeground(1, green if pct is not None and pct >= 100 else plain)
            return True
        except RuntimeError:          # item deleted under us
            return False

    # ------------------------------------------------------------ show / panel
    def showEvent(self, event):
        super().showEvent(event)
        self.store = store_for(self._project()) if self._project() is not None else self.store
        self.panel.store = self.store
        self.ensure_panel()
        self._borrow_nav_keys(True)
        self.refresh()

    def closeEvent(self, event):
        self.finish_drawing(ask=False)
        self._zoom_timer.stop()
        self._click_timer.stop()
        self._clear_timer.stop()
        self._pending_zoom = self._pending_click = self._pending_clear = None
        if self._persist_timer.isActive():
            self._persist_timer.stop()
            self.persist()
        if self.dock is not None:
            self.dock.hide()
        self._unmark_viewers()
        super().closeEvent(event)

    def ensure_panel(self):
        """The panel is part of this window now (left side); nothing to dock."""
        return

    # -------------------------------------------- keep the window off the canopy
    def _keep_clear_of(self, viewer, poly_scene_rect=None):
        """Move this window so it does not cover the viewer showing the polygon,
        or at least the polygon itself. Called after a list click zoomed there;
        does nothing when the window is already clear."""
        if not self.isVisible() or self.isMaximized() or viewer is None:
            return
        try:
            vp = viewer.viewport()
            vg = QtCore.QRect(vp.mapToGlobal(QtCore.QPoint(0, 0)), vp.size())
        except RuntimeError:
            return
        pr = None
        if poly_scene_rect is not None:
            r = viewer.mapFromScene(poly_scene_rect).boundingRect()
            pr = QtCore.QRect(vp.mapToGlobal(r.topLeft()), r.size()).intersected(vg)
        me = self.frameGeometry()
        if not me.intersects(vg):
            return
        screen = QtWidgets.QApplication.screenAt(vg.center()) or QtWidgets.QApplication.primaryScreen()
        avail = screen.availableGeometry()
        w, h = me.width(), me.height()

        def clamp(x, y):
            return (min(max(x, avail.left()), avail.right() - w + 1),
                    min(max(y, avail.top()), avail.bottom() - h + 1))

        y_now = me.top()
        cands = [(vg.left() - w - 8, y_now), (vg.right() + 9, y_now),
                 (avail.left(), y_now), (avail.right() - w + 1, y_now),
                 (me.left(), vg.top() - h - 8), (me.left(), vg.bottom() + 9)]

        def overlap(a, b):
            if b is None:
                return 0
            i = a.intersected(b)
            return max(0, i.width()) * max(0, i.height())

        best, best_score = None, None
        for x, y in cands:
            x, y = clamp(x, y)
            g = QtCore.QRect(x, y, w, h)
            score = overlap(g, pr) * 1000 + overlap(g, vg) + abs(x - me.left()) * 0.01
            if best_score is None or score < best_score:
                best, best_score = (x, y), score
        if best is not None and best != (me.left(), me.top()):
            self.move(best[0], best[1])

    def open_polygon(self, group, filepath, navigate=True):
        """Show one polygon in the panel, highlight it in the list and mark
        the viewer showing its image. With ``navigate``, bring that image on
        screen, then zoom to and centre the polygon the way the Polygon
        Manager does."""
        if self._brush is not None:
            self.finish_drawing()
        key, rec = find_record(self._all_polygons(), group, filepath)
        if rec is not None:
            # the panel always shows what is on the polygon right now
            self.store.sync_record(group, key, rec, force=True)
        label = record_label(group, rec)
        root = self._root_of(key)
        self.ensure_panel()
        self.panel.show_polygon(group, key, label, root)
        siblings = self._canopy_images(group)
        idx = next((i for i, fp in enumerate(siblings) if norm_fp(fp) == norm_fp(key)), -1)
        self.panel.set_navigation(idx + 1, len(siblings))
        self._highlight_current()
        if navigate:
            self._go_to_root(root)
            self._pending_zoom = (group, key)
            self._zoom_timer.start()
        else:
            self._mark_viewers(key)

    def _run_pending_zoom(self):
        job, self._pending_zoom = self._pending_zoom, None
        if job is not None and self.isVisible():
            self._zoom_and_select(*job)

    def _run_pending_click(self):
        job, self._pending_click = self._pending_click, None
        if job is not None and self.isVisible():
            self.open_polygon(*job, navigate=False)

    def _canopy_images(self, group):
        """The images holding this polygon group, in root order then name."""
        order = self._root_order()
        fps = [fp for fp, rec in (self._all_polygons().get(group) or {}).items() if _is_annotatable(rec)]
        return sorted(fps, key=lambda fp: (order.get(str(self._root_of(fp)), 10 ** 9),
                                           os.path.basename(str(fp)).lower()))

    def step_same_canopy(self, delta):
        """The panel's arrows: this canopy in the previous / next image."""
        g = self.panel.group
        if g is None:
            return
        fps = self._canopy_images(g)
        if len(fps) < 2:
            return
        i = next((i for i, fp in enumerate(fps) if norm_fp(fp) == norm_fp(self.panel.filepath)), 0)
        self.open_polygon(g, fps[(i + delta) % len(fps)], navigate=True)

    def _tree_order(self, label, group, fp, root):
        return (PolygonManager.natural_sort_key(str(label)),
                PolygonManager.natural_sort_key(str(group)),
                self._root_order().get(str(root), 10 ** 9),
                os.path.basename(str(fp)).lower())

    _NAV_KEYS = ("Right", "Left", "A", "D")

    def _nav_key(self, step):
        pt = self._project()
        active = QtWidgets.QApplication.activeWindow()
        if active is not self and (pt is None or active is not pt.window()):
            return                    # another window (another tab's dialog...)
        self.go_next_pending(backwards=step < 0)

    def _borrow_nav_keys(self, on):
        """While open, ←/→/A/D mean previous/next to fill; the project
        window's own actions on those keys (root stepping) get them back on
        close."""
        if on and not self._borrowed:
            pt = self._project()
            win = pt.window() if pt is not None else None
            if win is not None:
                for act in win.findChildren(QtWidgets.QAction):
                    seq = act.shortcut().toString()
                    if seq in self._NAV_KEYS:
                        self._borrowed.append((act, act.shortcut()))
                        act.setShortcut(QtGui.QKeySequence())
        elif not on:
            for act, seq in self._borrowed:
                try:
                    act.setShortcut(seq)
                except RuntimeError:
                    pass
            self._borrowed = []
        for sc in self._nav_shortcuts:
            sc.setEnabled(on)

    def hideEvent(self, event):
        self._borrow_nav_keys(False)
        super().hideEvent(event)

    def go_prev_pending(self):
        """← / A: the previous polygon, in list order, still to fill."""
        self.go_next_pending(backwards=True)

    def go_next_pending(self, backwards=False):
        """The panel's "Next to fill" (→ / D): the next polygon, in list
        order, with a property still pending (wrapping round)."""
        todo = self.store.pending(self._all_polygons(), self._root_of)
        if not todo:
            self.hint.setText("Everything is filled in.")
            return
        todo.sort(key=lambda t: self._tree_order(t["label"], t["group"], t["filepath"], t["root"]),
                  reverse=backwards)
        cur = None
        if self.panel.group is not None:
            cur = self._tree_order(self.panel.label, self.panel.group, self.panel.filepath, self.panel.root)

        def beyond(t):
            o = self._tree_order(t["label"], t["group"], t["filepath"], t["root"])
            return o < cur if backwards else o > cur
        nxt = next((t for t in todo if cur is None or beyond(t)), todo[0])
        self.open_polygon(nxt["group"], nxt["filepath"], navigate=True)

    def _go_to_root(self, root):
        pt = self._project()
        if pt is None or root is None:
            return
        names = list(getattr(pt, "multispectral_root_names", None) or getattr(pt, "root_names", None) or [])
        if root not in names:
            return
        idx = names.index(root)
        if idx == getattr(pt, "current_root_index", None):
            return
        jump = getattr(pt, "_jump_to_index", None)
        try:
            if jump is not None:
                jump(idx)
        except Exception:
            logging.debug("[annotator] root jump failed", exc_info=True)

    def _zoom_and_select(self, group, filepath=None):
        try:
            self.zoom_to_groups([group])
            self._select_group_in_viewers(group, additive=False, center=False)
        except Exception:
            logging.debug("[annotator] zoom to %s failed", group, exc_info=True)
        if filepath is not None:
            self._mark_viewers(filepath)
            w, viewer = self._viewer_for(filepath)
            if viewer is not None:
                rect = None
                try:
                    rect = self._get_polygon_scene_rect(w, viewer, group)
                except Exception:
                    logging.debug("[annotator] polygon rect failed", exc_info=True)
                self._keep_clear_of(viewer, rect)
                # the viewer grid may still be laying out after a root jump:
                # look again once it has settled
                self._pending_clear = (group, filepath)
                self._clear_timer.start()

    def _run_pending_clear(self):
        job, self._pending_clear = self._pending_clear, None
        if job is None or not self.isVisible():
            return
        group, filepath = job
        w, viewer = self._viewer_for(filepath)
        if viewer is not None:
            try:
                rect = self._get_polygon_scene_rect(w, viewer, group)
            except Exception:
                rect = None
            self._keep_clear_of(viewer, rect)

    def _viewer_for(self, filepath):
        for w, viewer in self._iter_viewers():
            idata = w.get("image_data") if isinstance(w, dict) else None
            if norm_fp(getattr(idata, "filepath", None)) == norm_fp(filepath):
                return w, viewer
        return None, None

    # ------------------------------------------------------------- patches
    def _patch_color(self, pid):
        ids = [p["id"] for p in self.store.properties]
        return _PATCH_COLORS[ids.index(pid) % len(_PATCH_COLORS)] if pid in ids else _PATCH_COLORS[0]

    def _current_record(self):
        if self.panel.group is None:
            return None, None
        return find_record(self._all_polygons(), self.panel.group, self.panel.filepath)

    def _clear_patch_display(self):
        for it in self._patch_items:
            try:
                if it.scene() is not None:
                    it.scene().removeItem(it)
            except RuntimeError:
                pass
        self._patch_items = []

    def _show_saved_patches(self, viewer):
        """Draw the open polygon's saved patches (every property, its colour)
        on the viewer showing it; the property being drawn is the brush's."""
        self._clear_patch_display()
        key, rec = self._current_record()
        if rec is None or viewer is None:
            return
        patches = rec.get(PATCHES_KEY) or {}
        for p in self.store.properties:
            entry = patches.get(p["name"])
            self.panel.set_patch_info(p["id"], entry.get("fraction") if entry else None)
            if not entry or (self._brush is not None and self._brush.prop["id"] == p["id"]):
                continue
            basis = crown_basis(rec, None) or tuple(entry.get("basis", {}).get(k, 1.0) for k in ("w", "h"))
            col = QtGui.QColor(self._patch_color(p["id"]))
            fill = QtGui.QColor(col)
            fill.setAlpha(70)
            pen = QtGui.QPen(col, 1.0)
            pen.setCosmetic(True)
            item = QtWidgets.QGraphicsPathItem(patches_path(viewer, basis, entry.get("polygons")))
            item.setPen(pen)
            item.setBrush(QtGui.QBrush(fill))
            item.setZValue(PATCH_Z - 1)
            item.setToolTip(f"{p['name']}: {100.0 * entry.get('fraction', 0):.1f}% of the crown")
            viewer.scene().addItem(item)
            self._patch_items.append(item)

    def _on_brush_size(self, px):
        self._brush_size = int(px)
        if self._brush is not None:
            self._brush.set_radius(px)

    def _on_clear_patches(self):
        if self._brush is not None:
            self._brush.clear()

    def previous_patches(self, group, filepath, prop_name):
        """(source filepath, patches mapped onto this crown) from the latest
        EARLIER image of the same canopy (polygon group) holding patches for
        ``prop_name``; (None, None) when there is none."""
        fps = self._canopy_images(group)
        i = next((i for i, fp in enumerate(fps) if norm_fp(fp) == norm_fp(filepath)), None)
        if not i:
            return None, None
        allp = self._all_polygons()
        _k, dst = find_record(allp, group, filepath)
        if dst is None:
            return None, None
        for fp in reversed(fps[:i]):
            _k2, src = find_record(allp, group, fp)
            entry = ((src or {}).get(PATCHES_KEY) or {}).get(prop_name) if src is not None else None
            if entry and entry.get("polygons") and len(src.get("points") or []) >= 3:
                return fp, transfer_patches(entry["polygons"], src.get("points"), dst.get("points") or [])
        return None, None

    def copy_patches_from_previous(self):
        """The brush bar's "From previous image": start from the canopy's last
        drawing, then edit it with the brush."""
        brush = self._brush
        if brush is None:
            return
        g, key = self._brush_key
        name = brush.prop["name"]
        src_fp, polys = self.previous_patches(g, key, name)
        if not polys:
            self.hint.setText(f"No earlier image of {g} has '{name}' patches.")
            return
        replace = True
        if brush.canvas.mask.any():
            box = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Question, "From previous image",
                                        f"This polygon already has '{name}' patches.", parent=self)
            b_rep = box.addButton("Replace", QtWidgets.QMessageBox.AcceptRole)
            b_add = box.addButton("Add to them", QtWidgets.QMessageBox.AcceptRole)
            box.addButton(QtWidgets.QMessageBox.Cancel)
            box.exec_()
            if box.clickedButton() not in (b_rep, b_add):
                return
            replace = box.clickedButton() is b_rep
        brush.load_polygons(polys, replace=replace)
        self.hint.setText(f"'{name}' patches copied from {os.path.basename(str(src_fp))}.")

    def _on_draw_toggled(self, pid, on):
        if not on:
            if self._brush is not None and (not pid or self._brush.prop["id"] == pid):
                self.finish_drawing()
            return
        self.start_drawing(pid)

    def start_drawing(self, pid):
        """Brush on for one property of the open polygon, on its viewer."""
        if self._brush is not None:
            self.finish_drawing()
        prop = self.store.property(pid)
        key, rec = self._current_record()
        w, viewer = self._viewer_for(key) if rec is not None else (None, None)
        if prop is None or rec is None or viewer is None:
            self.panel.set_drawing(None)
            self.hint.setText("Open the polygon's image in a viewer to draw on it.")
            return False
        try:
            self._brush = PatchBrush(viewer, rec, prop, self._patch_color(pid),
                                     radius_screen=self._brush_size, parent=self)
        except ValueError as e:
            self.panel.set_drawing(None)
            self.hint.setText(f"Cannot draw on this polygon: {e}")
            return False
        self._brush_key = (self.panel.group, key)
        self._brush.changed.connect(self._on_patches_changed)
        self._show_saved_patches(viewer)
        self.panel.set_drawing(pid, self._brush.canvas.fraction)
        src_fp, _polys = self.previous_patches(self.panel.group, key, prop["name"])
        self.panel.copy_prev_btn.setEnabled(src_fp is not None)
        self.panel.copy_prev_btn.setToolTip(
            (f"Copy the '{prop['name']}' patches of {os.path.basename(str(src_fp))} "
             "(this canopy's latest earlier drawing), fitted to this crown, then edit them")
            if src_fp else "No earlier image of this canopy has patches for this property")
        return True

    def _on_patches_changed(self, fraction):
        g, key = self._brush_key
        self._mark_dirty(g, key)
        self.panel.set_drawing(self._brush.prop["id"], fraction)
        self.panel.set_patch_info(self._brush.prop["id"], fraction if fraction > 0 else None)

    def finish_drawing(self, ask=True):
        """Brush off. For a numeric property whose patches changed, offer to
        set the value to their percentage of the crown (or do what the
        property's "Drawn % sets value" says)."""
        brush, self._brush = self._brush, None
        if brush is None:
            return
        brush.stop()
        self.panel.set_drawing(None)
        prop = self.store.property(brush.prop["id"])
        g, key = self._brush_key
        if ask and prop is not None and brush.touched and prop["kind"] == KIND_NUMERIC:
            pct = 100.0 * brush.canvas.fraction
            use = prop.get("patch_sets_value")
            if use is None:
                use, remember = self._ask_use_percentage(prop, pct)
                if remember:
                    prop["patch_sets_value"] = use
                    self.schedule_persist()
            if use:
                self.set_value_from_patches(g, key, prop, pct)
        _w, viewer = self._viewer_for(key)
        if viewer is not None and self.panel.group == g:
            self._show_saved_patches(viewer)

    def _ask_use_percentage(self, prop, pct):
        box = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Question, "Drawn patches",
                                    f"The patches cover {pct:.1f}% of the crown.", parent=self)
        box.setInformativeText(f"Use it as the value of '{prop['name']}'?")
        box.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        box.setDefaultButton(QtWidgets.QMessageBox.Yes)
        remember = QtWidgets.QCheckBox("Do the same for this property from now on")
        box.setCheckBox(remember)
        ans = box.exec_()
        return ans == QtWidgets.QMessageBox.Yes, remember.isChecked()

    def set_value_from_patches(self, group, key, prop, pct):
        step = prop.get("step", 1.0) or 1.0
        value = prop["minimum"] + round((pct - prop["minimum"]) / step) * step
        self.store.set_value(group, key, prop["id"], value)
        _k, rec = find_record(self._all_polygons(), group, key)
        if rec is not None:
            self.store.write_record(group, key, rec, prop["id"])
            self._mark_dirty(group, key)
        if self.panel.group == group and norm_fp(self.panel.filepath) == norm_fp(key):
            row = self.panel.rows.get(prop["id"])
            if row is not None:
                row.set_value(self.store.get_value(group, key, prop["id"]))
            self.panel._refresh_status()
            self._apply_mark_colors()
        self.schedule_persist()
        self._update_polygon_row(group, key)

    # ---------------------------------------------------- "which one is it"
    def _unmark_viewers(self):
        self._clear_patch_display()
        for w, css in self._marked_viewers:
            try:
                w.setStyleSheet(css)
            except RuntimeError:       # the viewer was rebuilt meanwhile
                pass
        self._marked_viewers = []

    def _mark_viewers(self, filepath):
        """Frame the viewer showing the polygon's image, and colour its file
        name: with several viewers showing the same canopy (a phenocam root)
        this is the one being annotated."""
        self._unmark_viewers()
        for w, viewer in self._iter_viewers():
            idata = w.get("image_data") if isinstance(w, dict) else None
            if norm_fp(getattr(idata, "filepath", None)) != norm_fp(filepath):
                continue
            self._marked_viewers.append((viewer, viewer.styleSheet()))
            container = w.get("container") if isinstance(w, dict) else None
            lay = container.layout() if container is not None else None
            head = lay.itemAt(0).widget() if lay is not None and lay.count() else None
            if isinstance(head, QtWidgets.QLabel):
                self._marked_viewers.append((head, head.styleSheet()))
            if norm_fp(filepath) == norm_fp(self.panel.filepath):
                self._show_saved_patches(viewer)
        self._apply_mark_colors()

    def _mark_color(self):
        """Green once every property of the open polygon is filled, red while
        anything is pending."""
        p = self.panel
        if p.group is None:
            return _ACCENT
        filled, total = self.store.status(p.group, p.filepath, p.label, p.root)
        return _ACCENT if filled == total else _PENDING

    def _apply_mark_colors(self):
        """(Re)colour the marked viewer's frame and file-name strip; cheap
        enough to run on every value change."""
        col = self._mark_color()
        for w, _css in self._marked_viewers:
            try:
                if isinstance(w, QtWidgets.QGraphicsView):
                    w.setStyleSheet(f"QGraphicsView {{ border: 3px solid {col}; }}")
                else:
                    w.setStyleSheet(f"background:{col}; color:white; font-weight:600; "
                                    "border-radius:3px; padding:1px 6px;")
            except RuntimeError:
                pass

    def _highlight_current(self):
        """Bold + tinted row for the polygon in the panel, its label opened
        and scrolled to; the same canopy's other images get a light tint."""
        cur_g, cur_f = self.panel.group, norm_fp(self.panel.filepath)
        strong = QtGui.QBrush(QtGui.QColor(46, 125, 50, 70))
        light = QtGui.QBrush(QtGui.QColor(46, 125, 50, 22))
        none = QtGui.QBrush()
        current = None
        for (g, f), row in self._tree_rows.items():
            try:
                is_cur = g == cur_g and f == cur_f
                same = g == cur_g and not is_cur
                font = row.font(0)
                font.setBold(is_cur)
                for col in (0, 1):
                    row.setFont(col, font)
                    row.setBackground(col, strong if is_cur else (light if same else none))
                if is_cur:
                    current = row
            except RuntimeError:
                return
        if current is not None:
            current.parent().setExpanded(True)
            self.tree.setCurrentItem(current)
            self.tree.scrollToItem(current, QtWidgets.QAbstractItemView.PositionAtCenter)

    def on_viewer_polygon_clicked(self, item, viewer):
        """A polygon clicked in a viewer while the Annotator is open."""
        if not self.isVisible():
            return False
        idata = getattr(viewer, "image_data", None)
        fp = getattr(idata, "filepath", None)
        group = getattr(item, "name", None)
        if not fp or not group:
            return False
        key, _rec = find_record(self._all_polygons(), group, fp)
        self._pending_click = (group, key)
        self._click_timer.start()
        return True

    # ------------------------------------------------------------- definitions
    def _labels(self):
        return sorted({lb for _g, _f, lb, _r in AnnotationStore.iter_polygons(self._all_polygons(), None)},
                      key=PolygonManager.natural_sort_key)

    def _definition_dialog(self, prop=None):
        pt = self._project()
        roots = list(getattr(pt, "root_names", None) or [])
        images = sorted({fp for fm in self._all_polygons().values() if isinstance(fm, dict) for fp in fm})
        cur = []
        for w, _v in self._iter_viewers():
            fp = getattr(w.get("image_data"), "filepath", None) if isinstance(w, dict) else None
            if fp:
                cur.append(fp)
        images = cur + [fp for fp in images if fp not in cur]
        return PropertyDefinitionDialog(roots, images, self._labels(), prop=prop, parent=self)

    def on_add_property(self):
        dlg = self._definition_dialog()
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return
        try:
            self.store.add_property(**dlg.result_kwargs())
        except ValueError as e:
            QtWidgets.QMessageBox.warning(self, "Property", str(e))
            return
        # A property named like an existing field picks its values up.
        self.sync_from_polygons()
        self._after_definition_change()

    def _selected_pid(self):
        it = self.prop_list.currentItem()
        return it.data(QtCore.Qt.UserRole) if it else None

    def _polygons_with_value(self, pid):
        out = []
        for g, fm in self.store.values.items():
            for nf, vals in fm.items():
                if pid in vals:
                    out.append((g, nf))
        return out

    def _records_for(self, pairs):
        """(group, stored key, record) for (group, normalised path) pairs."""
        allp = self._all_polygons()
        for g, nf in pairs:
            key, rec = find_record(allp, g, nf)
            if rec is not None:
                yield g, key, rec

    def on_edit_property(self):
        pid = self._selected_pid()
        prop = self.store.property(pid) if pid else None
        if prop is None:
            return
        dlg = self._definition_dialog(prop)
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return
        kw = dlg.result_kwargs()
        kw.pop("kind", None)
        holders = self._polygons_with_value(pid)
        try:
            old_name = self.store.update_property(pid, **kw)
        except ValueError as e:
            QtWidgets.QMessageBox.warning(self, "Property", str(e))
            return
        new_name = prop["name"]
        self._drop_invalid_values(pid)
        # Rename the key / rewrite narrowed values on the polygons themselves.
        for g, key, rec in self._records_for(holders):
            props = dict(rec.get("properties") or {})
            if old_name != new_name:
                props.pop(old_name, None)
            v = self.store.get_value(g, key, pid)
            if v is MISSING:
                props.pop(new_name, None)
            else:
                props[new_name] = self.store.encode(prop, v)
            rec["properties"] = props
            self._mark_dirty(g, key)
        if old_name != new_name:
            self._rename_patches(old_name, new_name)
        self._after_definition_change()

    def _records_with_patches(self, name):
        """Polygons in memory holding brush patches for ``name`` (a lazy
        record not read yet is skipped: its patches are matched by name the
        next time it is read)."""
        for g, fm in self._all_polygons().items():
            if not isinstance(fm, dict):
                continue
            for key, rec in fm.items():
                if isinstance(rec, dict) and not _is_unread_lazy(rec):
                    patches = rec.get(PATCHES_KEY)
                    if patches and name in patches:
                        yield g, key, rec

    def _rename_patches(self, old_name, new_name):
        for g, key, rec in list(self._records_with_patches(old_name)):
            patches = dict(rec.get(PATCHES_KEY) or {})
            patches[new_name] = patches.pop(old_name)
            rec[PATCHES_KEY] = patches
            self._mark_dirty(g, key)

    def _drop_invalid_values(self, pid):
        """After an edit narrows a property (fewer choices, smaller range),
        values it can no longer hold go back to pending; numbers are clamped."""
        prop = self.store.property(pid)
        for g, fm in self.store.values.items():
            for key, vals in fm.items():
                if pid in vals and vals[pid] is not None:
                    try:
                        v = self.store.coerce(prop, vals[pid])
                    except (ValueError, TypeError):
                        v = MISSING
                    if v is MISSING:
                        vals.pop(pid, None)
                    else:
                        vals[pid] = v

    def on_remove_property(self):
        pid = self._selected_pid()
        prop = self.store.property(pid) if pid else None
        if prop is None:
            return
        n = len(self._polygons_with_value(pid))
        box = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Question, "Remove property",
                                    f"Remove '{prop['name']}' from the Annotator?", parent=self)
        box.setInformativeText(f"{n} polygon(s) have a value for it in their properties.")
        b_keep = box.addButton("Remove, keep values", QtWidgets.QMessageBox.AcceptRole)
        b_wipe = box.addButton("Remove with values", QtWidgets.QMessageBox.DestructiveRole)
        box.addButton(QtWidgets.QMessageBox.Cancel)
        box.exec_()
        clicked = box.clickedButton()
        if clicked not in (b_keep, b_wipe):
            return
        if clicked is b_wipe:
            for g, key, rec in self._records_for(self._polygons_with_value(pid)):
                props = dict(rec.get("properties") or {})
                if props.pop(prop["name"], MISSING) is not MISSING:
                    rec["properties"] = props
                    self._mark_dirty(g, key)
            for g, key, rec in list(self._records_with_patches(prop["name"])):
                patches = dict(rec.get(PATCHES_KEY) or {})
                patches.pop(prop["name"], None)
                rec[PATCHES_KEY] = patches
                self._mark_dirty(g, key)
        self.store.remove_property(pid)
        self._after_definition_change()

    def _after_definition_change(self):
        self.schedule_persist()
        self.panel.rebuild()
        self.refresh()

    # ---------------------------------------------------------------- refresh
    def refresh(self):
        self.sync_from_polygons()
        self._fill_prop_list()
        if self.tabs.currentIndex() == 0:
            self._fill_polygon_tree()
        else:
            self._fill_stats()
        if self.panel.group is not None:
            self.panel._refresh_status()
        self._apply_mark_colors()

    def _fill_prop_list(self):
        cur = self._selected_pid()
        self.prop_list.clear()
        for p in self.store.properties:
            kind = "numeric" if p["kind"] == KIND_NUMERIC else ("tags" if p.get("multi") else "categorical")
            not_out = "" if p.get("export", True) else " · not exported"
            it = QtWidgets.QListWidgetItem(f"{p['name']}   · {kind} · {describe_scope(p)}{not_out}")
            it.setData(QtCore.Qt.UserRole, p["id"])
            self.prop_list.addItem(it)
            if p["id"] == cur:
                self.prop_list.setCurrentItem(it)
        has = bool(self.store.properties)
        self.edit_btn.setEnabled(has)
        self.remove_btn.setEnabled(has)
        self.hint.setText("" if has else "Add a property to start annotating.")

    def _expanded_labels(self):
        out = set()
        for i in range(self.tree.topLevelItemCount()):
            it = self.tree.topLevelItem(i)
            if it.isExpanded():
                out.add(it.data(0, self.ROLE_KEY))
        return out

    def _fill_polygon_tree(self):
        expanded = self._expanded_labels()
        all_p = self._all_polygons()
        by_label = {}
        for group, fp, label, root in self.store.iter_polygons(all_p, self._root_of):
            by_label.setdefault(label, []).append((group, fp, root))
        comp = self.store.completion_by_label(all_p, self._root_of)
        order = self._root_order()
        self.tree.setUpdatesEnabled(False)
        self.tree.clear()
        self._tree_rows = {}
        self._label_members = by_label
        green = QtGui.QBrush(QtGui.QColor(_ACCENT))
        for label in sorted(by_label, key=PolygonManager.natural_sort_key):
            s = comp.get(label) or {}
            pct = s.get("pct")
            top = QtWidgets.QTreeWidgetItem([f"{label}  ({s.get('polygons', 0)})",
                                             "—" if pct is None else f"{pct:.0f}% filled"])
            top.setData(0, self.ROLE_KIND, "label")
            top.setData(0, self.ROLE_KEY, label)
            if pct is not None and pct >= 100:
                top.setForeground(1, green)
            self.tree.addTopLevelItem(top)
            # Same canopy together, then its images in root order.
            for group, fp, root in sorted(
                    by_label[label],
                    key=lambda t: (PolygonManager.natural_sort_key(str(t[0])),
                                   order.get(str(t[2]), 10 ** 9), os.path.basename(str(t[1])).lower())):
                filled, total = self.store.status(group, fp, label, root)
                name = os.path.basename(str(fp))
                text = (f"{group}  ·  " if group != label else "") + name
                mark = (f"{CHECK} OK" if total and filled == total else
                        (f"{filled}/{total}" if total else ""))
                child = QtWidgets.QTreeWidgetItem([text, mark])
                child.setToolTip(0, f"{group}\n{fp}\nroot {root}")
                child.setData(0, self.ROLE_KIND, "polygon")
                child.setData(0, self.ROLE_KEY, (group, fp))
                if total and filled == total:
                    child.setForeground(1, green)
                top.addChild(child)
                self._tree_rows[(group, norm_fp(fp))] = child
            top.setExpanded(label in expanded)
        self.tree.setUpdatesEnabled(True)
        self._apply_search(self.search.text())
        self._highlight_current()

    def _apply_search(self, text):
        text = (text or "").strip().lower()
        for i in range(self.tree.topLevelItemCount()):
            top = self.tree.topLevelItem(i)
            label_hit = not text or text in top.text(0).lower()
            any_child = False
            for j in range(top.childCount()):
                ch = top.child(j)
                hit = label_hit or text in ch.text(0).lower()
                ch.setHidden(not hit)
                any_child = any_child or hit
            top.setHidden(not (label_hit or any_child))
            if text and any_child and not label_hit:
                top.setExpanded(True)

    def _fill_stats(self):
        all_p = self._all_polygons()
        stats = self.store.property_stats(all_p, self._root_of)
        todo = self.store.pending(all_p, self._root_of)
        n_poly = sum(1 for _ in self.store.iter_polygons(all_p, self._root_of))
        lines = []
        for p in self.store.properties:
            s = stats[p["id"]]
            na = f", {s['na']} N/A" if s["na"] else ""
            lines.append(f"<b>{p['name']}</b>: {s['filled']}/{s['applies']} filled{na}"
                         + (f", <span style='color:#b26a00'>{s['missing']} pending</span>" if s["missing"] else ""))
        head = (f"{len(todo)} of {n_poly} polygon(s) still have something to fill in."
                if self.store.properties else "No properties defined yet.")
        self.stats_summary.setText(head + ("<br>" + "<br>".join(lines) if lines else ""))

        self.stats_tree.setUpdatesEnabled(False)
        self.stats_tree.clear()
        by_root = {}
        for t in todo:
            by_root.setdefault(str(t["root"]) if t["root"] is not None else "(no root)", []).append(t)
        for root in sorted(by_root, key=PolygonManager.natural_sort_key):
            items = by_root[root]
            top = QtWidgets.QTreeWidgetItem([f"Root {root}", str(sum(len(t['missing']) for t in items))])
            top.setData(0, self.ROLE_KIND, "root")
            self.stats_tree.addTopLevelItem(top)
            for t in sorted(items, key=lambda t: (os.path.basename(str(t["filepath"])), t["label"])):
                ch = QtWidgets.QTreeWidgetItem(
                    [f"{t['label']}  ·  {os.path.basename(str(t['filepath']))}",
                     ", ".join(t["missing"])])
                ch.setToolTip(0, f"{t['group']}\n{t['filepath']}")
                ch.setData(0, self.ROLE_KIND, "polygon")
                ch.setData(0, self.ROLE_KEY, (t["group"], t["filepath"]))
                top.addChild(ch)
            top.setExpanded(len(by_root) == 1)
        self.stats_tree.setUpdatesEnabled(True)

    def _on_tree_clicked(self, item, _col):
        if item.data(0, self.ROLE_KIND) != "polygon":
            return
        group, fp = item.data(0, self.ROLE_KEY)
        self.open_polygon(group, fp, navigate=True)

    # PolygonManager hooks ProjectTab may call on any polygon manager.
    def set_polygons(self, polygon_groups):
        self.schedule_refresh()

    def update_selection_based_on_root(self):
        pass
