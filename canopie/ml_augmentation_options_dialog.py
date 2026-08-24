"""Training augmentation + feature-normalization options for
MachineLearningManager.train_models, with a live preview of the configured
augmentation applied to one representative training image.

Shown as one more step in train_models' existing wizard-of-dialogs sequence
(after the hyperparameter-optimization choice, before feature names are
built) -- see train_models' call site for exactly where. Defaults to
everything OFF ("none"/"none"), so accepting the dialog without touching
anything reproduces training exactly as it worked before this feature
existed -- same convention ThumbnailOptionsDialog/ExportImagesOptionsDialog
already establish for this app's newer options-dialog idiom.

The preview MUST call `ml_augmentation.augment_image_for_training` -- the
exact function train_models itself calls -- and nothing else; see
qc/test_ml_preview_matches_training.py's AST guard, which fails the build if
this file grows its own independent brightness/shadow implementation instead
of delegating.
"""
import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QCheckBox, QRadioButton, QButtonGroup, QSpinBox, QDoubleSpinBox,
    QDialogButtonBox
)

from .image_editor_dialog import CollapsibleBox
from .utils import _normalize_for_display
from . import ml_augmentation

#: The values that reproduce "no augmentation, no normalization" -- accepting
#: the dialog untouched must be a behavioral no-op. qc/test_ml_augmentation_
#: options_dialog.py asserts these stay in step with get_options()'s defaults.
DEFAULT_BRIGHTNESS_MODE = "none"
DEFAULT_TILE_SIZE = 64
DEFAULT_MU_JITTER = 0.15
DEFAULT_SD_JITTER = 0.25
DEFAULT_SHADOW_SMOOTHNESS = 75
DEFAULT_ROW_POLICY = "add"
DEFAULT_N_VARIANTS = 1
DEFAULT_NORMALIZATION = "none"

#: Debounce delay for the live preview -- mirrors ThumbnailOptionsDialog's own
#: debounced-estimate pattern (machine_learning_manager.py's CollapsibleBox-
#: adjacent AnalysisOptionsDialog uses the same QTimer idiom at ~line 5500).
_PREVIEW_DEBOUNCE_MS = 250


class MLAugmentationOptionsDialog(QDialog):
    """Brightness/shadow augmentation and L2/z-score normalization options,
    plus a live single-image preview of the augmentation."""

    def __init__(self, entries, get_export_image_fn, channels_in_export_order_fn, parent=None):
        """
        entries: [(group_name, filepath), ...] -- the same list train_models
            already builds, passed in so the preview can offer a
            representative image without re-deriving it.
        get_export_image_fn: bound MachineLearningManager._get_export_image
            (filepath -> (img, C)) -- passed explicitly, not the whole
            manager object, to keep this dialog's dependency surface small.
        channels_in_export_order_fn: bound
            MachineLearningManager._channels_in_export_order -- unused by the
            preview itself (which augments the raw `img`, not per-band
            channels) but accepted for signature symmetry / potential future
            per-band preview modes.
        """
        super().__init__(parent)
        self._entries = list(entries or [])
        self._get_export_image = get_export_image_fn
        self._channels_in_export_order = channels_in_export_order_fn
        self._preview_img_cache = {}  # filepath -> raw img array, loaded once per dialog session

        self.setWindowTitle("Training Augmentation && Normalization")
        self.setMinimumWidth(520)
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(_PREVIEW_DEBOUNCE_MS)
        self._debounce.timeout.connect(self._refresh_preview)

        self.setup_ui()
        self.apply_style()
        self._schedule_preview()

    # ------------------------------------------------------------------
    def setup_ui(self):
        layout = QVBoxLayout(self)

        # -- Brightness Augmentation -------------------------------------
        bright_box = CollapsibleBox("Brightness Augmentation")
        self.brightness_enabled_cb = QCheckBox("Enable brightness augmentation")
        self.brightness_enabled_cb.setChecked(False)
        self.brightness_enabled_cb.setToolTip(
            "Shifts a training image's mean/std toward a randomly perturbed\n"
            "target before pixels are sampled from it -- the model learns\n"
            "patterns that don't depend on absolute brightness.")
        bright_box.content_layout.addWidget(self.brightness_enabled_cb)

        row_mode = QHBoxLayout()
        self.brightness_mode_group = QButtonGroup(self)
        self.image_level_rb = QRadioButton("Image-level")
        self.image_level_rb.setToolTip("One randomly-perturbed brightness shift for the WHOLE image.")
        self.patch_level_rb = QRadioButton("Patch-level")
        self.patch_level_rb.setToolTip(
            "The image is split into a fixed-size grid; each tile gets its\n"
            "OWN independent random brightness shift -- simulates uneven\n"
            "lighting/shadow gradients across a scene.")
        self.image_level_rb.setChecked(True)
        self.brightness_mode_group.addButton(self.image_level_rb)
        self.brightness_mode_group.addButton(self.patch_level_rb)
        row_mode.addWidget(self.image_level_rb)
        row_mode.addWidget(self.patch_level_rb)
        row_mode.addStretch()
        bright_box.content_layout.addLayout(row_mode)

        row_tile = QHBoxLayout()
        row_tile.addWidget(QLabel("Tile size (px):"))
        self.tile_size_spin = QSpinBox()
        self.tile_size_spin.setRange(8, 2048)
        self.tile_size_spin.setValue(DEFAULT_TILE_SIZE)
        self.tile_size_spin.setEnabled(False)
        row_tile.addWidget(self.tile_size_spin)
        row_tile.addStretch()
        bright_box.content_layout.addLayout(row_tile)

        row_jitter = QHBoxLayout()
        row_jitter.addWidget(QLabel("Mean jitter:"))
        self.mu_jitter_spin = QDoubleSpinBox()
        self.mu_jitter_spin.setRange(0.0, 1.0)
        self.mu_jitter_spin.setSingleStep(0.05)
        self.mu_jitter_spin.setDecimals(2)
        self.mu_jitter_spin.setValue(DEFAULT_MU_JITTER)
        self.mu_jitter_spin.setToolTip(
            "Target mean is drawn from source_mean * U(1-jitter, 1+jitter).")
        row_jitter.addWidget(self.mu_jitter_spin)
        row_jitter.addSpacing(12)
        row_jitter.addWidget(QLabel("Std jitter:"))
        self.sd_jitter_spin = QDoubleSpinBox()
        self.sd_jitter_spin.setRange(0.0, 1.0)
        self.sd_jitter_spin.setSingleStep(0.05)
        self.sd_jitter_spin.setDecimals(2)
        self.sd_jitter_spin.setValue(DEFAULT_SD_JITTER)
        row_jitter.addWidget(self.sd_jitter_spin)
        row_jitter.addStretch()
        bright_box.content_layout.addLayout(row_jitter)
        layout.addWidget(bright_box)

        self.brightness_enabled_cb.toggled.connect(self._on_brightness_toggled)
        self.patch_level_rb.toggled.connect(lambda on: self.tile_size_spin.setEnabled(on and self.brightness_enabled_cb.isChecked()))

        # -- Shadow / Illumination Augmentation --------------------------
        shadow_box = CollapsibleBox("Shadow / Illumination Augmentation")
        self.shadow_enabled_cb = QCheckBox("Enable shadow/illumination augmentation")
        self.shadow_enabled_cb.setChecked(False)
        self.shadow_enabled_cb.setToolTip(
            "Overlays a random linear/circular/diagonal illumination gradient\n"
            "(feathered, applied multiplicatively) -- simulates cast shadows\n"
            "and uneven lighting. See "
            "https://github.com/Nathanborg/Cloud_shadow_correction")
        shadow_box.content_layout.addWidget(self.shadow_enabled_cb)

        row_smooth = QHBoxLayout()
        row_smooth.addWidget(QLabel("Feather smoothness:"))
        self.smoothness_spin = QSpinBox()
        self.smoothness_spin.setRange(1, 301)
        self.smoothness_spin.setSingleStep(2)
        self.smoothness_spin.setValue(DEFAULT_SHADOW_SMOOTHNESS)
        self.smoothness_spin.setToolTip(
            "Gaussian-blur kernel size for the shadow mask's edges. Must be\n"
            "odd -- an even value is rounded up automatically.")
        row_smooth.addWidget(self.smoothness_spin)
        row_smooth.addStretch()
        shadow_box.content_layout.addLayout(row_smooth)
        layout.addWidget(shadow_box)

        # -- Row Policy ---------------------------------------------------
        policy_box = CollapsibleBox("Row Policy")
        row_policy = QHBoxLayout()
        self.row_policy_group = QButtonGroup(self)
        self.add_rb = QRadioButton("Add (grow the training set)")
        self.add_rb.setToolTip(
            "Augmented pixels are ADDED alongside the originals -- the\n"
            "sample count increases. Tested/held-out pixels are never\n"
            "augmented either way.")
        self.replace_rb = QRadioButton("Replace (same size)")
        self.replace_rb.setToolTip(
            "Augmented values REPLACE the originals in the training set --\n"
            "no size change.")
        self.add_rb.setChecked(DEFAULT_ROW_POLICY == "add")
        self.replace_rb.setChecked(DEFAULT_ROW_POLICY == "replace")
        self.row_policy_group.addButton(self.add_rb)
        self.row_policy_group.addButton(self.replace_rb)
        row_policy.addWidget(self.add_rb)
        row_policy.addWidget(self.replace_rb)
        row_policy.addStretch()
        policy_box.content_layout.addLayout(row_policy)

        row_n = QHBoxLayout()
        row_n.addWidget(QLabel("Copies per pixel (Add mode):"))
        self.n_variants_spin = QSpinBox()
        self.n_variants_spin.setRange(1, 10)
        self.n_variants_spin.setValue(DEFAULT_N_VARIANTS)
        self.n_variants_spin.setToolTip(
            "How many independently-augmented siblings each sampled pixel\n"
            "gets in Add mode. Larger values grow BOTH the augmentation cost\n"
            "and the final model's training time roughly linearly.")
        row_n.addWidget(self.n_variants_spin)
        row_n.addStretch()
        policy_box.content_layout.addLayout(row_n)
        layout.addWidget(policy_box)

        self.add_rb.toggled.connect(lambda on: self.n_variants_spin.setEnabled(on))

        # -- Normalization --------------------------------------------------
        norm_box = CollapsibleBox("Normalization")
        row_norm = QHBoxLayout()
        self.norm_group = QButtonGroup(self)
        self.norm_none_rb = QRadioButton("None")
        self.norm_l2_rb = QRadioButton("L2 (per-sample)")
        self.norm_l2_rb.setToolTip(
            "Each pixel's feature vector is divided by its own L2 norm.\n"
            "No fitted parameters needed.")
        self.norm_zscore_rb = QRadioButton("Z-score (standard deviation)")
        self.norm_zscore_rb.setToolTip(
            "Per-feature (x - mean) / std, fitted once on the training set\n"
            "and stored in the model bundle so prediction uses the exact\n"
            "same fitted mean/std.")
        self.norm_none_rb.setChecked(DEFAULT_NORMALIZATION == "none")
        self.norm_l2_rb.setChecked(DEFAULT_NORMALIZATION == "l2")
        self.norm_zscore_rb.setChecked(DEFAULT_NORMALIZATION == "zscore")
        for rb in (self.norm_none_rb, self.norm_l2_rb, self.norm_zscore_rb):
            self.norm_group.addButton(rb)
            row_norm.addWidget(rb)
        row_norm.addStretch()
        norm_box.content_layout.addLayout(row_norm)
        layout.addWidget(norm_box)

        # -- Preview ----------------------------------------------------
        preview_box = CollapsibleBox("Preview")
        row_img = QHBoxLayout()
        row_img.addWidget(QLabel("Preview image:"))
        self.preview_combo = QComboBox()
        for group_name, filepath in self._entries:
            import os as _os
            self.preview_combo.addItem(f"{group_name} — {_os.path.basename(filepath)}", filepath)
        row_img.addWidget(self.preview_combo, 1)
        preview_box.content_layout.addLayout(row_img)

        self.preview_label = QLabel("No preview available.")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumHeight(240)
        self.preview_label.setStyleSheet("background-color: #222; color: #ccc;")
        preview_box.content_layout.addWidget(self.preview_label)
        layout.addWidget(preview_box)

        self.preview_combo.currentIndexChanged.connect(self._schedule_preview)
        for w in (self.mu_jitter_spin, self.sd_jitter_spin, self.tile_size_spin, self.smoothness_spin):
            w.valueChanged.connect(self._schedule_preview)
        for w in (self.brightness_enabled_cb, self.shadow_enabled_cb,
                  self.image_level_rb, self.patch_level_rb):
            w.toggled.connect(self._schedule_preview)

        for box in (bright_box, shadow_box, policy_box, norm_box, preview_box):
            box.toggle_button.setChecked(True)

        # -- Buttons ------------------------------------------------------
        self.buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        ok_btn = self.buttonBox.button(QDialogButtonBox.Ok)
        ok_btn.setText("Continue")
        ok_btn.setDefault(True)
        layout.addWidget(self.buttonBox)

        self._on_brightness_toggled(False)

    def _on_brightness_toggled(self, enabled):
        for w in (self.image_level_rb, self.patch_level_rb):
            w.setEnabled(enabled)
        self.tile_size_spin.setEnabled(enabled and self.patch_level_rb.isChecked())
        self.mu_jitter_spin.setEnabled(enabled)
        self.sd_jitter_spin.setEnabled(enabled)

    # ------------------------------------------------------------------
    def apply_style(self):
        # Same CanoPie palette every other options dialog in this app uses.
        self.setStyleSheet("""
            QWidget { font-size: 10px; }
            QLabel { margin: 0px; padding: 0px; font-size: 12px; }
            QSpinBox, QDoubleSpinBox, QComboBox {
                padding: 0px 4px;
                min-height: 20px;
                font-size: 12px;
            }
            QPushButton {
                padding: 2px 6px;
                min-height: 20px;
                background-color: #2e7d32;
                color: white;
                font-weight: bold;
                border-radius: 3px;
                border: 1px solid #1b5e20;
            }
            QPushButton:hover { background-color: #388e3c; }
            QToolButton { padding: 0px; }
            QCheckBox, QRadioButton { font-size: 13px; }
            QCheckBox::indicator:checked, QRadioButton::indicator:checked {
                background-color: #FFD700;
                border: 1px solid #006400;
            }
            QCheckBox::indicator:unchecked, QRadioButton::indicator:unchecked {
                background-color: white;
                border: 1px solid gray;
            }
            QToolButton.collapsible-toggle {
                color: #006400;
                font-weight: bold;
                font-size: 11pt;
            }
        """)

    # ------------------------------------------------------------------
    def _schedule_preview(self, *_args):
        self._debounce.start()

    def _load_preview_source(self, filepath):
        if filepath in self._preview_img_cache:
            return self._preview_img_cache[filepath]
        try:
            img, _C = self._get_export_image(filepath)
        except Exception:
            img = None
        self._preview_img_cache[filepath] = img
        return img

    def _refresh_preview(self):
        if not self._entries:
            self.preview_label.setText("No training images available to preview.")
            return
        filepath = self.preview_combo.currentData()
        if not filepath:
            return
        img = self._load_preview_source(filepath)
        if img is None:
            self.preview_label.setText(f"Could not load preview image:\n{filepath}")
            return

        cfg = self.get_options()["augmentation"]
        try:
            preview_rng = np.random.default_rng()  # not seed-matched to training -- purely illustrative
            img_aug = ml_augmentation.augment_image_for_training(img, cfg, preview_rng)
        except Exception as e:
            self.preview_label.setText(f"Preview failed: {e}")
            return

        try:
            disp = _normalize_for_display(img_aug, input_is_rgb=True, return_bgr=True)
            if disp is None:
                raise ValueError("normalize_for_display returned None")
            if disp.ndim == 2:
                h, w = disp.shape
                qimg = QImage(np.ascontiguousarray(disp).data, w, h, w, QImage.Format_Grayscale8)
            else:
                h, w, _c = disp.shape
                qimg = QImage(np.ascontiguousarray(disp).data, w, h, 3 * w, QImage.Format_BGR888)
            pix = QPixmap.fromImage(qimg)
            avail_w = max(200, self.preview_label.width())
            pix = pix.scaledToWidth(min(avail_w, 480), Qt.SmoothTransformation)
            self.preview_label.setPixmap(pix)
        except Exception as e:
            self.preview_label.setText(f"Preview render failed: {e}")

    # ------------------------------------------------------------------
    def get_options(self):
        """Plain dict of the chosen options -- see module docstring for the
        exact shape train_models consumes."""
        mode = "none"
        if self.brightness_enabled_cb.isChecked():
            mode = "patch" if self.patch_level_rb.isChecked() else "image"

        normalization = "none"
        if self.norm_l2_rb.isChecked():
            normalization = "l2"
        elif self.norm_zscore_rb.isChecked():
            normalization = "zscore"

        shadow_enabled = bool(self.shadow_enabled_cb.isChecked())
        brightness_on = (mode != "none")

        return {
            "augmentation": {
                "enabled": brightness_on or shadow_enabled,
                "brightness": {
                    "mode": mode,
                    "tile_size": int(self.tile_size_spin.value()),
                    "mu_jitter": float(self.mu_jitter_spin.value()),
                    "sd_jitter": float(self.sd_jitter_spin.value()),
                },
                "shadow": {
                    "enabled": shadow_enabled,
                    "smoothness": int(self.smoothness_spin.value()),
                },
                "row_policy": "add" if self.add_rb.isChecked() else "replace",
                "n_variants": int(self.n_variants_spin.value()),
            },
            "normalization": normalization,
        }
