import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects
from PIL import Image

# Target directory
OUTPUT_DIR = r"c:\Users\natha\Downloads\CanoPie-main_updated\CanoPie-main\docs_assets"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Color Palette Constants (Professional Modern Technical UI Theme)
BG_DARK = "#1E1E2E"
PANEL_BG = "#2A2A3C"
PANEL_BORDER = "#45475A"
ACCENT_BLUE = "#89B4FA"
ACCENT_TEAL = "#94E2D5"
ACCENT_GREEN = "#A6E3A1"
ACCENT_YELLOW = "#F9E2AF"
ACCENT_ORANGE = "#FAB387"
ACCENT_RED = "#F38BA8"
ACCENT_PURPLE = "#CBA6F7"
TEXT_LIGHT = "#CDD6F4"
TEXT_MUTED = "#BAC2DE"

def set_fig_style(fig, ax):
    fig.patch.set_facecolor(BG_DARK)
    ax.set_facecolor(BG_DARK)
    ax.axis('off')

def add_header(ax, title, subtitle):
    ax.text(0.5, 0.96, title, color=TEXT_LIGHT, fontsize=18, fontweight='bold', ha='center', va='top')
    ax.text(0.5, 0.925, subtitle, color=ACCENT_TEAL, fontsize=11, fontweight='normal', ha='center', va='top')

# ==============================================================================
# FIG 01: Main Window Layout
# ==============================================================================
def draw_fig01():
    fig, ax = plt.subplots(figsize=(14, 9), dpi=150)
    set_fig_style(fig, ax)
    add_header(ax, "CanoPie Main Window Architecture & UI Layout", "Figure 1: Full application window components, toolbars, viewports, and status indicators")

    # Main Window Container Frame
    main_box = patches.FancyBboxPatch((0.02, 0.03), 0.96, 0.86, boxstyle="round,pad=0.01", fc=PANEL_BG, ec=PANEL_BORDER, lw=2)
    ax.add_patch(main_box)

    # 1. Menu Bar
    menu_box = patches.Rectangle((0.03, 0.84), 0.94, 0.04, fc="#181825", ec=ACCENT_BLUE, lw=1.5)
    ax.add_patch(menu_box)
    ax.text(0.04, 0.86, "File   View   Tools   Help   |   CanoPie Multi-Image Geospatial Suite v2.4", color=TEXT_LIGHT, fontsize=10, fontweight='bold', va='center')

    # 2. Main Toolbar / Header
    hdr_box = patches.Rectangle((0.03, 0.78), 0.94, 0.05, fc="#313244", ec=PANEL_BORDER, lw=1)
    ax.add_patch(hdr_box)
    ax.text(0.04, 0.805, "[+] New Project   [O] Open Folder   [S] Save Project   |   Active Project: Site_A_2026.canopie", color=ACCENT_YELLOW, fontsize=10, va='center')

    # 3. Project Tab Bar & Tab Toolbar
    tabbar_box = patches.Rectangle((0.03, 0.72), 0.94, 0.05, fc="#45475A", ec=ACCENT_TEAL, lw=1.5)
    ax.add_patch(tabbar_box)
    ax.text(0.04, 0.745, "Tabs: [ Tab 1: plot01.tif (Active) ]  [ Tab 2: plot02.tif ]  [ Tab 3: plot03.tif ]   |   22-Control Action Bar", color=TEXT_LIGHT, fontsize=10, va='center')

    # 4. Central Graphics Viewport (QGraphicsView / QGraphicsScene)
    view_box = patches.Rectangle((0.03, 0.10), 0.70, 0.60, fc="#11111B", ec=ACCENT_BLUE, lw=2)
    ax.add_patch(view_box)
    ax.text(0.38, 0.67, "Central Graphics Viewport (QGraphicsView)", color=ACCENT_BLUE, fontsize=12, fontweight='bold', ha='center')

    # Simulated Raster Image Canvas inside Viewport
    img_box = patches.Rectangle((0.08, 0.15), 0.60, 0.48, fc="#18231c", ec="#315c38", lw=1.5)
    ax.add_patch(img_box)
    ax.text(0.38, 0.39, "Base Pyramidal Image Raster Canvas\n[RGB / Multispectral Layer]\n1024 x 1024 px", color="#A6E3A1", fontsize=11, ha='center', va='center')

    # Simulated Polygon Annotations on Viewport
    poly = patches.Polygon([[0.15, 0.25], [0.35, 0.22], [0.42, 0.45], [0.22, 0.50]], closed=True, fc='#F9E2AF33', ec=ACCENT_YELLOW, lw=2)
    ax.add_patch(poly)
    ax.text(0.28, 0.35, "Poly #1 (Canopy A)", color=ACCENT_YELLOW, fontsize=9, fontweight='bold')
    for px, py in [[0.15, 0.25], [0.35, 0.22], [0.42, 0.45], [0.22, 0.50]]:
        ax.add_patch(patches.Circle((px, py), 0.008, fc=ACCENT_RED, ec=TEXT_LIGHT, lw=1))

    # Zoom / Pan Overlay Indicator
    ax.add_patch(patches.Rectangle((0.04, 0.11), 0.18, 0.05, fc="#181825CC", ec=ACCENT_PURPLE, lw=1))
    ax.text(0.05, 0.135, "Zoom: 125% | Pan: OK", color=ACCENT_PURPLE, fontsize=9, va='center')

    # 5. Right Sidebar (Polygon List & Layer Manager)
    side_box = patches.Rectangle((0.74, 0.10), 0.23, 0.62, fc="#1E1E2E", ec=PANEL_BORDER, lw=1.5)
    ax.add_patch(side_box)
    ax.text(0.855, 0.69, "Polygon List & Layers", color=TEXT_LIGHT, fontsize=11, fontweight='bold', ha='center')
    
    # Sidebar items
    items = [("Group: Trees_Plot1", ACCENT_GREEN), ("- Canopy_01 (4 pts)", ACCENT_YELLOW), ("- Canopy_02 (6 pts)", ACCENT_TEAL), ("Group: Soil_Background", ACCENT_ORANGE), ("- Ground_01 (3 pts)", ACCENT_PURPLE)]
    for i, (txt, clr) in enumerate(items):
        ax.add_patch(patches.Rectangle((0.75, 0.62 - i*0.06), 0.21, 0.05, fc="#313244", ec=clr, lw=1))
        ax.text(0.76, 0.645 - i*0.06, txt, color=TEXT_LIGHT, fontsize=8.5, va='center')

    # 6. Status Bar
    status_box = patches.Rectangle((0.03, 0.04), 0.94, 0.05, fc="#11111B", ec=ACCENT_GREEN, lw=1.5)
    ax.add_patch(status_box)
    ax.text(0.04, 0.065, "Status Bar | X: 452.1 px  Y: 812.4 px  |  Val: R:142 G:189 B:76 (NIR: 0.68)  |  CRS: EPSG:32616 (UTM 16N)  |  RAM: 2.1 GB  |  Idle", color=ACCENT_GREEN, fontsize=9, va='center')

    # Callout Annotations
    ax.annotate("[1] Menu Bar", xy=(0.15, 0.86), xytext=(0.01, 0.93), arrowprops=dict(arrowstyle="->", color=ACCENT_BLUE, lw=1.5), color=ACCENT_BLUE, fontsize=10, fontweight='bold')
    ax.annotate("[2] Main Toolbar", xy=(0.15, 0.805), xytext=(0.01, 0.805), arrowprops=dict(arrowstyle="->", color=ACCENT_YELLOW, lw=1.5), color=ACCENT_YELLOW, fontsize=10, fontweight='bold')
    ax.annotate("[3] Project Tab Toolbar", xy=(0.50, 0.745), xytext=(0.50, 0.88), arrowprops=dict(arrowstyle="->", color=ACCENT_TEAL, lw=1.5), color=ACCENT_TEAL, fontsize=10, fontweight='bold')
    ax.annotate("[4] Viewport Canvas", xy=(0.40, 0.45), xytext=(0.45, 0.55), arrowprops=dict(arrowstyle="->", color=ACCENT_BLUE, lw=1.5), color=ACCENT_BLUE, fontsize=10, fontweight='bold')
    ax.annotate("[5] Status Bar", xy=(0.50, 0.065), xytext=(0.50, 0.01), arrowprops=dict(arrowstyle="->", color=ACCENT_GREEN, lw=1.5), color=ACCENT_GREEN, fontsize=10, fontweight='bold')

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "fig01_main_window_layout.png")
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    print(f"Saved: {out_path} ({os.path.getsize(out_path)} bytes)")

# ==============================================================================
# FIG 02: Project Tab Toolbar (22 Controls)
# ==============================================================================
def draw_fig02():
    fig, ax = plt.subplots(figsize=(16, 10), dpi=150)
    set_fig_style(fig, ax)
    add_header(ax, "CanoPie Project Tab Toolbar — 22 Core UI Controls Annotated", "Figure 2: Complete control inventory across 6 functional groups")

    # Main Toolbar Container
    tb_box = patches.Rectangle((0.02, 0.70), 0.96, 0.18, fc=PANEL_BG, ec=ACCENT_BLUE, lw=2)
    ax.add_patch(tb_box)
    ax.text(0.03, 0.85, "Project Tab Action Toolbar (22 Action Buttons & Pickers)", color=ACCENT_BLUE, fontsize=12, fontweight='bold')

    controls = [
        # Group 1: File & Project (1-4)
        ("1", "Load Img", ACCENT_BLUE, "Load Single Image"),
        ("2", "Add Dir", ACCENT_BLUE, "Add Image Directory"),
        ("3", "Save Proj", ACCENT_BLUE, "Save Project JSON"),
        ("4", "Load Proj", ACCENT_BLUE, "Load Project File"),
        # Group 2: View & Zoom (5-8)
        ("5", "Zoom In", ACCENT_TEAL, "Zoom In Viewport"),
        ("6", "Zoom Out", ACCENT_TEAL, "Zoom Out Viewport"),
        ("7", "Fit Win", ACCENT_TEAL, "Fit Image to Window"),
        ("8", "1:1 Scale", ACCENT_TEAL, "100% Native Resolution"),
        # Group 3: Geometry & Draw (9-13)
        ("9", "Draw Poly", ACCENT_YELLOW, "Draw Polygon Tool"),
        ("10", "Draw Pt", ACCENT_YELLOW, "Draw Point Tool"),
        ("11", "Edit Vtx", ACCENT_YELLOW, "Vertex Editing Mode"),
        ("12", "Select/Move", ACCENT_YELLOW, "Select & Move Poly"),
        ("13", "Del Poly", ACCENT_RED, "Delete Selected Poly"),
        # Group 4: Layers & Groups (14-16)
        ("14", "Group Mgr", ACCENT_GREEN, "Manage Groups/Colors"),
        ("15", "Toggle Lbl", ACCENT_GREEN, "Toggle Text Labels"),
        ("16", "Toggle Clr", ACCENT_GREEN, "Toggle Poly Fill Color"),
        # Group 5: Image & Pipeline (17-19)
        ("17", "Img Editor", ACCENT_PURPLE, "Open .ax Editor Dialog"),
        ("18", "Stretch", ACCENT_PURPLE, "Stretch & LUT Dialog"),
        ("19", "Band Math", ACCENT_PURPLE, "Band Math Evaluator"),
        # Group 6: Analysis & Export (20-22)
        ("20", "ML Mgr", ACCENT_ORANGE, "ML Classifier Dialog"),
        ("21", "Poly Mgr", ACCENT_ORANGE, "Polygon Manager Dialog"),
        ("22", "Export", ACCENT_ORANGE, "CSV/EXIF Export Mgr"),
    ]

    # Render buttons in toolbar row
    cols = 11
    for i, (num, label, color, desc) in enumerate(controls):
        r = 0 if i < cols else 1
        c = i % cols
        bx = 0.03 + c * 0.084
        by = 0.78 - r * 0.065
        
        btn = patches.FancyBboxPatch((bx, by), 0.078, 0.055, boxstyle="round,pad=0.005", fc="#181825", ec=color, lw=1.5)
        ax.add_patch(btn)
        ax.text(bx + 0.039, by + 0.035, f"[{num}]", color=color, fontsize=9, fontweight='bold', ha='center')
        ax.text(bx + 0.039, by + 0.015, label, color=TEXT_LIGHT, fontsize=7.5, fontweight='bold', ha='center')

    # Legend / Annotation Key below toolbar
    key_box = patches.Rectangle((0.02, 0.04), 0.96, 0.62, fc=PANEL_BG, ec=PANEL_BORDER, lw=1.5)
    ax.add_patch(key_box)
    ax.text(0.04, 0.62, "Annotated Control Inventory & Description Key", color=ACCENT_TEAL, fontsize=12, fontweight='bold')

    groups = [
        ("File & Project Operations (1-4)", ACCENT_BLUE, controls[0:4]),
        ("View & Zoom Controls (5-8)", ACCENT_TEAL, controls[4:8]),
        ("Geometry & Polygon Editing (9-13)", ACCENT_YELLOW, controls[8:13]),
        ("Layer & Display Toggles (14-16)", ACCENT_GREEN, controls[13:16]),
        ("Image Processing & Pipeline (17-19)", ACCENT_PURPLE, controls[16:19]),
        ("Analysis & Export Engines (20-22)", ACCENT_ORANGE, controls[19:22]),
    ]

    for g_idx, (g_title, g_color, g_items) in enumerate(groups):
        col = g_idx % 2
        row = g_idx // 2
        gx = 0.04 + col * 0.47
        gy = 0.56 - row * 0.17
        
        ax.text(gx, gy, g_title, color=g_color, fontsize=10, fontweight='bold')
        for item_idx, (num, label, color, desc) in enumerate(g_items):
            iy = gy - 0.03 - item_idx * 0.025
            ax.text(gx + 0.01, iy, f"• [{num}] {label}:", color=color, fontsize=8.5, fontweight='bold')
            ax.text(gx + 0.16, iy, desc, color=TEXT_MUTED, fontsize=8.5)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "fig02_project_tab_toolbar.png")
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    print(f"Saved: {out_path} ({os.path.getsize(out_path)} bytes)")

# ==============================================================================
# FIG 03: Viewport Z-Layers Architecture
# ==============================================================================
def draw_fig03():
    fig, ax = plt.subplots(figsize=(14, 9), dpi=150)
    set_fig_style(fig, ax)
    add_header(ax, "Viewport Graphics View Architecture & Z-Stacking Layers", "Figure 3: 5-Tier Z-layer stacking order and vertex handle interactions")

    layers = [
        ("Layer 5 (Top)", "LABEL_Z = 10,000,000 (1e7)", "Text Labels, Group Tags, & Hover Callouts", ACCENT_RED, 0.70),
        ("Layer 4", "TEMP_DRAWING_Z = 1,000,000 (1e6)", "Active Drawing Rubberband & In-Progress Nodes", ACCENT_ORANGE, 0.56),
        ("Layer 3", "POLYGON_Z = 1.0", "Editable Polygon/Point Items & Interactive Handles", ACCENT_YELLOW, 0.42),
        ("Layer 2", "HIGHRES_TILE_Z = 0.5", "High-Res Dynamic Zoom Tile Overlay (Tile Pyramid)", ACCENT_TEAL, 0.28),
        ("Layer 1 (Base)", "IMAGE_Z = 0.0", "Base Pyramidal Image Raster Canvas", ACCENT_BLUE, 0.14),
    ]

    # Draw Stacked Layer Cards with 3D/Isometric Offset Look
    for title, z_val, desc, color, y_pos in layers:
        # Layer Card
        card = patches.FancyBboxPatch((0.05, y_pos), 0.55, 0.10, boxstyle="round,pad=0.01", fc="#2A2A3C", ec=color, lw=2)
        ax.add_patch(card)
        ax.text(0.07, y_pos + 0.06, title, color=color, fontsize=11, fontweight='bold')
        ax.text(0.24, y_pos + 0.06, f"[{z_val}]", color=TEXT_LIGHT, fontsize=9.5, fontweight='bold')
        ax.text(0.07, y_pos + 0.025, desc, color=TEXT_MUTED, fontsize=9)

        # Connector arrow pointing up
        if y_pos < 0.70:
            ax.annotate("", xy=(0.32, y_pos + 0.10), xytext=(0.32, y_pos + 0.14), arrowprops=dict(arrowstyle="<-", color=TEXT_MUTED, lw=1.5, ls="--"))

    # Detail Side Panel: Handles & Interactions
    detail_box = patches.Rectangle((0.64, 0.14), 0.32, 0.66, fc=PANEL_BG, ec=ACCENT_PURPLE, lw=2)
    ax.add_patch(detail_box)
    ax.text(0.80, 0.76, "Interactive Geometry Handles", color=ACCENT_PURPLE, fontsize=11, fontweight='bold', ha='center')

    # Handle 1: Vertex Handle (Square)
    ax.add_patch(patches.Rectangle((0.67, 0.63), 0.03, 0.04, fc=ACCENT_RED, ec=TEXT_LIGHT, lw=1.5))
    ax.text(0.72, 0.65, "Vertex Handle (Square)\nDrag to move individual node", color=TEXT_LIGHT, fontsize=8.5, va='center')

    # Handle 2: Selected Vertex (Circle Highlight)
    ax.add_patch(patches.Circle((0.685, 0.53), 0.02, fc=ACCENT_YELLOW, ec=ACCENT_RED, lw=2))
    ax.text(0.72, 0.53, "Active Node (Highlight Circle)\nSelected for deletion/relocation", color=TEXT_LIGHT, fontsize=8.5, va='center')

    # Handle 3: Center Drag Handle (Circle)
    ax.add_patch(patches.Circle((0.685, 0.41), 0.018, fc=ACCENT_TEAL, ec=TEXT_LIGHT, lw=1.5))
    ax.text(0.72, 0.41, "Center Drag Handle (Circle)\nDrag to move entire polygon", color=TEXT_LIGHT, fontsize=8.5, va='center')

    # Handle 4: Midpoint Edge Handle (Diamond)
    poly_dia = patches.Polygon([[0.685, 0.31], [0.70, 0.29], [0.685, 0.27], [0.67, 0.29]], closed=True, fc=ACCENT_GREEN, ec=TEXT_LIGHT, lw=1.5)
    ax.add_patch(poly_dia)
    ax.text(0.72, 0.29, "Edge Midpoint Handle (Diamond)\nClick to insert new vertex", color=TEXT_LIGHT, fontsize=8.5, va='center')

    # Render Architecture Note
    note_box = patches.Rectangle((0.67, 0.16), 0.26, 0.08, fc="#181825", ec=PANEL_BORDER, lw=1)
    ax.add_patch(note_box)
    ax.text(0.80, 0.20, "Coordinate System:\nQGraphicsScene float coords\nmapped to raster pixel grid", color=ACCENT_TEAL, fontsize=8, ha='center', va='center')

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "fig03_viewport_z_layers.png")
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    print(f"Saved: {out_path} ({os.path.getsize(out_path)} bytes)")

# ==============================================================================
# FIG 04: Stretch & Color Controls
# ==============================================================================
def draw_fig04():
    fig, ax = plt.subplots(figsize=(14, 9), dpi=150)
    set_fig_style(fig, ax)
    add_header(ax, "Stretch & Color Controls Architecture", "Figure 4: Percentile, StdDev, Min/Max methods, LUT fast-path, and NoData white rendering")

    # Left: Histogram & Stretch Thresholds Simulation
    hist_box = patches.Rectangle((0.04, 0.10), 0.52, 0.72, fc=PANEL_BG, ec=ACCENT_BLUE, lw=2)
    ax.add_patch(hist_box)
    ax.text(0.30, 0.78, "Image Pixel Intensity Histogram & Stretch Cutoffs", color=ACCENT_BLUE, fontsize=11, fontweight='bold', ha='center')

    # Draw simulated histogram bars
    x_bins = np.linspace(0.08, 0.52, 40)
    heights = np.exp(-((x_bins - 0.30) ** 2) / 0.008) * 0.35 + np.random.normal(0, 0.01, 40)
    heights = np.clip(heights, 0.01, 0.40)
    for xb, h in zip(x_bins, heights):
        ax.add_patch(patches.Rectangle((xb, 0.22), 0.009, h, fc="#45475A", ec="#585B70", lw=0.5))

    # Low / High Percentile Cutoff Lines
    ax.plot([0.16, 0.16], [0.22, 0.65], color=ACCENT_RED, lw=2, ls="--")
    ax.text(0.16, 0.67, "Low Cutoff (2.0%)\n[Min Threshold]", color=ACCENT_RED, fontsize=8.5, ha='center', fontweight='bold')

    ax.plot([0.44, 0.44], [0.22, 0.65], color=ACCENT_GREEN, lw=2, ls="--")
    ax.text(0.44, 0.67, "High Cutoff (98.0%)\n[Max Threshold]", color=ACCENT_GREEN, fontsize=8.5, ha='center', fontweight='bold')

    # Stretch Mapping Curve (Linear between cutoffs)
    ax.plot([0.08, 0.16, 0.44, 0.52], [0.22, 0.22, 0.57, 0.57], color=ACCENT_YELLOW, lw=2.5, label="LUT Output Curve")
    ax.text(0.30, 0.42, "Active Stretch Range\nMapped to 0..255 Display Range", color=ACCENT_YELLOW, fontsize=9.5, ha='center', fontweight='bold')

    # Right: Control Modes & Fast Path Engine
    ctrl_box = patches.Rectangle((0.60, 0.10), 0.36, 0.72, fc=PANEL_BG, ec=ACCENT_TEAL, lw=2)
    ax.add_patch(ctrl_box)
    ax.text(0.78, 0.78, "Stretch Engine & Controls", color=ACCENT_TEAL, fontsize=11, fontweight='bold', ha='center')

    modes = [
        ("1. Percentile Stretch", "Low % (0.1-5.0) to High % (95-99.9)", ACCENT_BLUE),
        ("2. StdDev (±σ) Stretch", "Mean ± k*σ (k = 0.5, 1.0, 2.0, 3.0)", ACCENT_TEAL),
        ("3. Linear Min/Max", "Absolute min/max pixel intensity values", ACCENT_YELLOW),
        ("4. Per-Channel Control", "Independent RGB channel min/max sliders", ACCENT_PURPLE),
    ]
    for i, (m_title, m_desc, clr) in enumerate(modes):
        card = patches.Rectangle((0.62, 0.58 - i*0.11), 0.32, 0.09, fc="#181825", ec=clr, lw=1.5)
        ax.add_patch(card)
        ax.text(0.63, 0.64 - i*0.11, m_title, color=clr, fontsize=9.5, fontweight='bold')
        ax.text(0.63, 0.60 - i*0.11, m_desc, color=TEXT_MUTED, fontsize=8)

    # LUT & NoData Technical Box
    tech_box = patches.Rectangle((0.62, 0.12), 0.32, 0.11, fc="#11111B", ec=ACCENT_ORANGE, lw=1.5)
    ax.add_patch(tech_box)
    ax.text(0.63, 0.20, "LUT Fast Path Acceleration:", color=ACCENT_ORANGE, fontsize=8.5, fontweight='bold')
    ax.text(0.63, 0.17, "• 256/4096-bin Look-Up Table mapping", color=TEXT_LIGHT, fontsize=8)
    ax.text(0.63, 0.14, "• NoData / NaN → Rendered as White (255,255,255)", color=TEXT_LIGHT, fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "fig04_stretch_controls.png")
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    print(f"Saved: {out_path} ({os.path.getsize(out_path)} bytes)")

# ==============================================================================
# FIG 05: Polygon Manager & Shapefile Importer
# ==============================================================================
def draw_fig05():
    fig, ax = plt.subplots(figsize=(14, 9), dpi=150)
    set_fig_style(fig, ax)
    add_header(ax, "Polygon Manager & Shapefile Importer Dialog Architecture", "Figure 5: Polygon table management, group hierarchy, shapefile ingestion, and CRS reprojection")

    # Left Panel: Polygon Manager Dialog Mockup
    pm_box = patches.Rectangle((0.03, 0.10), 0.45, 0.72, fc=PANEL_BG, ec=ACCENT_YELLOW, lw=2)
    ax.add_patch(pm_box)
    ax.text(0.255, 0.78, "Polygon Manager Dialog", color=ACCENT_YELLOW, fontsize=11, fontweight='bold', ha='center')

    # Table Mockup
    tbl_hdr = patches.Rectangle((0.05, 0.70), 0.41, 0.05, fc="#45475A", ec=PANEL_BORDER, lw=1)
    ax.add_patch(tbl_hdr)
    ax.text(0.06, 0.725, "Name         Group        Pts    Area (m²)   Color", color=TEXT_LIGHT, fontsize=8.5, fontweight='bold', va='center')

    rows = [
        ("Tree_01", "Canopy", "8", "45.2", ACCENT_GREEN),
        ("Tree_02", "Canopy", "12", "82.1", ACCENT_TEAL),
        ("Soil_01", "Ground", "4", "120.5", ACCENT_ORANGE),
        ("Crop_01", "Crop_Plot", "6", "64.0", ACCENT_PURPLE),
    ]
    for i, (nm, grp, pts, area, clr) in enumerate(rows):
        r_box = patches.Rectangle((0.05, 0.63 - i*0.06), 0.41, 0.05, fc="#181825", ec=PANEL_BORDER, lw=1)
        ax.add_patch(r_box)
        ax.text(0.06, 0.655 - i*0.06, f"{nm:<10} {grp:<10} {pts:<6} {area:<10}", color=TEXT_LIGHT, fontsize=8, va='center')
        ax.add_patch(patches.Rectangle((0.41, 0.64 - i*0.06), 0.03, 0.03, fc=clr, ec=TEXT_LIGHT, lw=1))

    # Action Buttons
    btns = ["Rename Group", "Change Color", "Batch Delete", "Export Selected"]
    for i, btxt in enumerate(btns):
        bx = 0.05 + (i % 2) * 0.21
        by = 0.32 - (i // 2) * 0.06
        ax.add_patch(patches.Rectangle((bx, by), 0.19, 0.045, fc="#313244", ec=ACCENT_YELLOW, lw=1))
        ax.text(bx + 0.095, by + 0.022, btxt, color=TEXT_LIGHT, fontsize=8, ha='center', va='center')

    # Right Panel: Shapefile Importer Architecture
    shp_box = patches.Rectangle((0.52, 0.10), 0.45, 0.72, fc=PANEL_BG, ec=ACCENT_TEAL, lw=2)
    ax.add_patch(shp_box)
    ax.text(0.745, 0.78, "Shapefile Importer Workflow & CRS Engine", color=ACCENT_TEAL, fontsize=11, fontweight='bold', ha='center')

    flow_steps = [
        ("1. File Ingestion", "Read .shp, .shx, .dbf geometry & attribute stream", ACCENT_BLUE),
        ("2. CRS Detection & Reprojection", "Transform EPSG / WGS84 coordinates to pixel space", ACCENT_TEAL),
        ("3. Spatial Index Matching", "KD-Tree bounds filtering against active raster frame", ACCENT_YELLOW),
        ("4. Attribute Column Mapping", "Map DBF field attributes to CanoPie polygon groups", ACCENT_PURPLE),
        ("5. Boundary Clipping & Loading", "Discard out-of-bounds polygons & populate Viewport", ACCENT_GREEN),
    ]

    for i, (s_title, s_desc, clr) in enumerate(flow_steps):
        s_card = patches.FancyBboxPatch((0.54, 0.62 - i*0.11), 0.41, 0.08, boxstyle="round,pad=0.005", fc="#181825", ec=clr, lw=1.5)
        ax.add_patch(s_card)
        ax.text(0.55, 0.67 - i*0.11, s_title, color=clr, fontsize=9, fontweight='bold')
        ax.text(0.55, 0.64 - i*0.11, s_desc, color=TEXT_MUTED, fontsize=8)

        if i < 4:
            ax.annotate("", xy=(0.745, 0.62 - i*0.11), xytext=(0.745, 0.60 - i*0.11), arrowprops=dict(arrowstyle="->", color=TEXT_MUTED, lw=1.5))

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "fig05_polygon_manager.png")
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    print(f"Saved: {out_path} ({os.path.getsize(out_path)} bytes)")

# ==============================================================================
# FIG 06: Image Editor & .ax 9-Stage Modification Pipeline
# ==============================================================================
def draw_fig06():
    fig, ax = plt.subplots(figsize=(16, 9), dpi=150)
    set_fig_style(fig, ax)
    add_header(ax, "Image Editor & .ax 9-Stage Modification Pipeline", "Figure 6: Non-destructive image modification pipeline and .ax sidecar JSON replay engine")

    stages = [
        ("Stage 1", "Registration", "pystackreg matrix\nAffine/Perspective", ACCENT_BLUE),
        ("Stage 2", "Rotation", "90°, 180°, 270°\nQuadrant rotation", ACCENT_TEAL),
        ("Stage 3", "Crop", "ROI Bounding Box\n(X, Y, W, H)", ACCENT_GREEN),
        ("Stage 4", "Hist Match", "Reference CDF\nIntensity align", ACCENT_YELLOW),
        ("Stage 5", "Resize", "Bilinear/Bicubic\nPixel rescaling", ACCENT_ORANGE),
        ("Stage 6", "Band Math", "Custom formula\n(e.g., NDVI)", ACCENT_RED),
        ("Stage 7", "NoData Mask", "NaN mapping &\nsentinel restore", ACCENT_PURPLE),
        ("Stage 8", "ML Classify", "Random Forest\npixel prediction", ACCENT_TEAL),
        ("Stage 9", "Appended Bands", "Sidecar JSON output\nnon-destructive", ACCENT_GREEN),
    ]

    # Draw Pipeline Flowchart (3x3 grid or horizontal snake flow)
    for i, (stg, name, desc, clr) in enumerate(stages):
        col = i % 3
        row = i // 3
        bx = 0.04 + col * 0.31
        by = 0.62 - row * 0.22
        
        card = patches.FancyBboxPatch((bx, by), 0.28, 0.18, boxstyle="round,pad=0.01", fc=PANEL_BG, ec=clr, lw=2)
        ax.add_patch(card)
        ax.text(bx + 0.02, by + 0.14, stg, color=clr, fontsize=9.5, fontweight='bold')
        ax.text(bx + 0.02, by + 0.10, name, color=TEXT_LIGHT, fontsize=11, fontweight='bold')
        ax.text(bx + 0.02, by + 0.04, desc, color=TEXT_MUTED, fontsize=8.5)

        # Flow connectors
        if col < 2:
            ax.annotate("", xy=(bx + 0.28, by + 0.09), xytext=(bx + 0.31, by + 0.09), arrowprops=dict(arrowstyle="->", color=clr, lw=2))
        elif row < 2:
            ax.annotate("", xy=(bx + 0.14, by), xytext=(bx + 0.14, by - 0.04), arrowprops=dict(arrowstyle="->", color=clr, lw=2))

    # Technical Footnote Box: .ax Sidecar Engine
    foot_box = patches.Rectangle((0.04, 0.04), 0.90, 0.10, fc="#11111B", ec=ACCENT_BLUE, lw=1.5)
    ax.add_patch(foot_box)
    ax.text(0.06, 0.10, ".ax Sidecar Replay Engine Architecture:", color=ACCENT_BLUE, fontsize=10, fontweight='bold')
    ax.text(0.06, 0.06, "• Raw raster image files remain 100% untouched on disk. All edits are saved as lightweight JSON parameters in `.ax` sidecars.\n• Pipeline stages execute sequentially on-demand during viewport rendering or export, ensuring lossless non-destructive workflows.", color=TEXT_LIGHT, fontsize=8.5)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "fig06_image_editor_ax.png")
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    print(f"Saved: {out_path} ({os.path.getsize(out_path)} bytes)")

# ==============================================================================
# FIG 07: Machine Learning Manager
# ==============================================================================
def draw_fig07():
    fig, ax = plt.subplots(figsize=(14, 9), dpi=150)
    set_fig_style(fig, ax)
    add_header(ax, "Machine Learning Manager Dialog Architecture", "Figure 7: Band selector, spatial patch sizes, Random Forest training, and pixel classification")

    # Left: UI Dialog Mockup
    ml_box = patches.Rectangle((0.03, 0.10), 0.45, 0.72, fc=PANEL_BG, ec=ACCENT_PURPLE, lw=2)
    ax.add_patch(ml_box)
    ax.text(0.255, 0.78, "Machine Learning Manager", color=ACCENT_PURPLE, fontsize=11, fontweight='bold', ha='center')

    # Section 1: Band Selector
    ax.text(0.05, 0.72, "1. Select Features / Spectral Bands:", color=TEXT_LIGHT, fontsize=9.5, fontweight='bold')
    bands = [("[X] Band 1 (Red)", ACCENT_RED), ("[X] Band 2 (Green)", ACCENT_GREEN), ("[X] Band 3 (Blue)", ACCENT_BLUE), ("[X] Band 4 (NIR)", ACCENT_PURPLE)]
    for i, (b_txt, clr) in enumerate(bands):
        ax.add_patch(patches.Rectangle((0.05 + (i%2)*0.20, 0.64 - (i//2)*0.045), 0.18, 0.04, fc="#181825", ec=clr, lw=1))
        ax.text(0.06 + (i%2)*0.20, 0.66 - (i//2)*0.045, b_txt, color=clr, fontsize=8, va='center')

    # Section 2: Patch Size & Parameters
    ax.text(0.05, 0.53, "2. Spatial Patch Size & Classifier:", color=TEXT_LIGHT, fontsize=9.5, fontweight='bold')
    p_box = patches.Rectangle((0.05, 0.38), 0.41, 0.13, fc="#181825", ec=PANEL_BORDER, lw=1)
    ax.add_patch(p_box)
    ax.text(0.06, 0.48, "Patch Dimensions:  [ 3x3 ]  [ 5x5 ]  [ 7x7 ]  [ 15x15 ]", color=ACCENT_YELLOW, fontsize=8.5)
    ax.text(0.06, 0.44, "Classifier Engine: Random Forest Classifier", color=TEXT_LIGHT, fontsize=8.5)
    ax.text(0.06, 0.40, "Estimators: 100 trees | Max Depth: 15 | Balanced", color=TEXT_MUTED, fontsize=8)

    # Action Triggers
    ax.add_patch(patches.Rectangle((0.05, 0.28), 0.19, 0.06, fc="#313244", ec=ACCENT_GREEN, lw=1.5))
    ax.text(0.145, 0.31, "Train Model", color=ACCENT_GREEN, fontsize=9, fontweight='bold', ha='center', va='center')

    ax.add_patch(patches.Rectangle((0.27, 0.28), 0.19, 0.06, fc="#313244", ec=ACCENT_PURPLE, lw=1.5))
    ax.text(0.365, 0.31, "Run Predict", color=ACCENT_PURPLE, fontsize=9, fontweight='bold', ha='center', va='center')

    # Right: Classification Workflow Flowchart
    wf_box = patches.Rectangle((0.52, 0.10), 0.45, 0.72, fc=PANEL_BG, ec=ACCENT_GREEN, lw=2)
    ax.add_patch(wf_box)
    ax.text(0.745, 0.78, "Pixel Classification Workflow", color=ACCENT_GREEN, fontsize=11, fontweight='bold', ha='center')

    steps = [
        ("Polygon Training Labels", "Extract spectral/spatial features from annotated canopy ROI polygons", ACCENT_YELLOW),
        ("Patch Feature Extraction", "Generate multi-band neighborhood patch tensors around ROI centroids", ACCENT_TEAL),
        ("Random Forest Training", "Train Scikit-Learn Ensemble on extracted feature vectors", ACCENT_PURPLE),
        ("Full Image Inference", "Predict class probabilities per-pixel across active scene", ACCENT_BLUE),
        ("Classification Map Output", "Color-coded thematic layer appended to Viewport & .ax sidecar", ACCENT_GREEN),
    ]

    for i, (st, sd, clr) in enumerate(steps):
        scard = patches.FancyBboxPatch((0.54, 0.62 - i*0.11), 0.41, 0.08, boxstyle="round,pad=0.005", fc="#181825", ec=clr, lw=1.5)
        ax.add_patch(scard)
        ax.text(0.55, 0.67 - i*0.11, st, color=clr, fontsize=9, fontweight='bold')
        ax.text(0.55, 0.64 - i*0.11, sd, color=TEXT_MUTED, fontsize=8)

        if i < 4:
            ax.annotate("", xy=(0.745, 0.62 - i*0.11), xytext=(0.745, 0.60 - i*0.11), arrowprops=dict(arrowstyle="->", color=TEXT_MUTED, lw=1.5))

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "fig07_ml_manager.png")
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    print(f"Saved: {out_path} ({os.path.getsize(out_path)} bytes)")

# ==============================================================================
# FIG 08: Export Manager
# ==============================================================================
def draw_fig08():
    fig, ax = plt.subplots(figsize=(14, 9), dpi=150)
    set_fig_style(fig, ax)
    add_header(ax, "Export Manager Dialog Architecture & Output Engines", "Figure 8: CSV zonal statistics export, EXIF metadata extraction, and thumbnail generator")

    engines = [
        ("CSV Statistics Export Engine", "Per-Polygon Zonal Statistics", [
            "• Calculates Mean, Median, StdDev, Min, Max, Q5-Q95",
            "• Supports custom Band Math formulas per polygon",
            "• Background thread execution (ExportWorker QThread)",
            "• Handles NoData values & NaN masking automatically"
        ], ACCENT_BLUE, 0.04),
        ("EXIF Metadata Export Engine", "Camera Tag & Telemetry Aggregation", [
            "• Extracts camera model, ISO, shutter speed, aperture",
            "• Reads GPS Latitude, Longitude, Altitude telemetry",
            "• Batch processes entire project folder into CSV format",
            "• Preserves datetime timestamps for time-series analysis"
        ], ACCENT_TEAL, 0.36),
        ("Thumbnail & Image Export Engine", "Cropped ROI & Transformed Rasters", [
            "• Generates polygon ROI crop thumbnails (PNG/JPEG)",
            "• Exports full transformed rasters with .ax pipeline applied",
            "• Custom thumbnail size & output directory configuration",
            "• High-speed multi-threaded batch rendering engine"
        ], ACCENT_GREEN, 0.68),
    ]

    for title, subtitle, points, color, x_pos in engines:
        card = patches.FancyBboxPatch((x_pos, 0.10), 0.28, 0.74, boxstyle="round,pad=0.01", fc=PANEL_BG, ec=color, lw=2)
        ax.add_patch(card)
        ax.text(x_pos + 0.02, 0.78, title, color=color, fontsize=10.5, fontweight='bold')
        ax.text(x_pos + 0.02, 0.74, subtitle, color=TEXT_MUTED, fontsize=8.5, fontweight='bold')

        p_box = patches.Rectangle((x_pos + 0.015, 0.14), 0.25, 0.56, fc="#181825", ec=PANEL_BORDER, lw=1)
        ax.add_patch(p_box)
        for i, pt in enumerate(points):
            ax.text(x_pos + 0.025, 0.64 - i*0.12, pt, color=TEXT_LIGHT, fontsize=8, va='top', wrap=True)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "fig08_export_manager.png")
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    print(f"Saved: {out_path} ({os.path.getsize(out_path)} bytes)")

# ==============================================================================
# FIG 09: Band Selector & Band Math
# ==============================================================================
def draw_fig09():
    fig, ax = plt.subplots(figsize=(14, 9), dpi=150)
    set_fig_style(fig, ax)
    add_header(ax, "Display Band Selector & Band Math Evaluation Engine", "Figure 9: RGB/Single-band selection bar, formula parsing, and fast math evaluation")

    # Top: Band Selector Bar Mockup
    bar_box = patches.Rectangle((0.04, 0.64), 0.92, 0.20, fc=PANEL_BG, ec=ACCENT_BLUE, lw=2)
    ax.add_patch(bar_box)
    ax.text(0.06, 0.80, "Display Band Selector Bar (Project Tab Control Bar)", color=ACCENT_BLUE, fontsize=11, fontweight='bold')

    # Display Mode Combos
    ax.add_patch(patches.Rectangle((0.06, 0.70), 0.24, 0.06, fc="#181825", ec=ACCENT_TEAL, lw=1))
    ax.text(0.07, 0.73, "Display Mode: [ RGB Composite  v ]", color=TEXT_LIGHT, fontsize=8.5, va='center')

    # Channel Pickers (RGB)
    ax.add_patch(patches.Rectangle((0.32, 0.70), 0.18, 0.06, fc="#181825", ec=ACCENT_RED, lw=1))
    ax.text(0.33, 0.73, "Red: [ Band 3 (NIR)  v ]", color=ACCENT_RED, fontsize=8.5, va='center')

    ax.add_patch(patches.Rectangle((0.52, 0.70), 0.18, 0.06, fc="#181825", ec=ACCENT_GREEN, lw=1))
    ax.text(0.53, 0.73, "Green: [ Band 1 (Red) v ]", color=ACCENT_GREEN, fontsize=8.5, va='center')

    ax.add_patch(patches.Rectangle((0.72, 0.70), 0.18, 0.06, fc="#181825", ec=ACCENT_BLUE, lw=1))
    ax.text(0.73, 0.73, "Blue: [ Band 2 (Grn) v ]", color=ACCENT_BLUE, fontsize=8.5, va='center')

    # Bottom Left: Band Math Formula Input
    math_box = patches.Rectangle((0.04, 0.10), 0.44, 0.50, fc=PANEL_BG, ec=ACCENT_YELLOW, lw=2)
    ax.add_patch(math_box)
    ax.text(0.26, 0.54, "Band Math Evaluator Engine", color=ACCENT_YELLOW, fontsize=11, fontweight='bold', ha='center')

    ax.text(0.06, 0.47, "Formula Input String:", color=TEXT_LIGHT, fontsize=9, fontweight='bold')
    in_box = patches.Rectangle((0.06, 0.40), 0.40, 0.05, fc="#181825", ec=ACCENT_YELLOW, lw=1.5)
    ax.add_patch(in_box)
    ax.text(0.07, 0.425, "(b3 - b1) / (b3 + b1 + 1e-6)", color=ACCENT_YELLOW, fontsize=10, fontweight='bold', va='center')

    ax.text(0.06, 0.35, "Preset Formulas:", color=TEXT_LIGHT, fontsize=8.5, fontweight='bold')
    presets = ["NDVI: (NIR-R)/(NIR+R)", "NDWI: (G-NIR)/(G+NIR)", "EVI: 2.5*(NIR-R)/(NIR+6*R-7.5*B+1)"]
    for i, pr in enumerate(presets):
        ax.text(0.07, 0.31 - i*0.04, f"• {pr}", color=TEXT_MUTED, fontsize=8)

    ax.add_patch(patches.Rectangle((0.06, 0.14), 0.18, 0.045, fc="#313244", ec=ACCENT_GREEN, lw=1))
    ax.text(0.15, 0.1625, "Evaluate & Render", color=ACCENT_GREEN, fontsize=8.5, fontweight='bold', ha='center', va='center')

    # Bottom Right: Execution Engine Architecture
    eng_box = patches.Rectangle((0.52, 0.10), 0.44, 0.50, fc=PANEL_BG, ec=ACCENT_PURPLE, lw=2)
    ax.add_patch(eng_box)
    ax.text(0.74, 0.54, "Math Processing Architecture", color=ACCENT_PURPLE, fontsize=11, fontweight='bold', ha='center')

    steps = [
        ("1. Expression Tokenizer", "Parse formula into NumExpr / AST syntax tree", ACCENT_TEAL),
        ("2. Multi-Core Evaluation", "Accelerated NumExpr / Numba parallel execution", ACCENT_PURPLE),
        ("3. NoData / NaN Safeguards", "Division-by-zero protection & NaN masking", ACCENT_ORANGE),
        ("4. Dynamic Band Appending", "Appends result array as virtual band in Viewport", ACCENT_GREEN),
    ]

    for i, (st, sd, clr) in enumerate(steps):
        scard = patches.FancyBboxPatch((0.54, 0.43 - i*0.08), 0.40, 0.06, boxstyle="round,pad=0.005", fc="#181825", ec=clr, lw=1.5)
        ax.add_patch(scard)
        ax.text(0.55, 0.46 - i*0.08, st, color=clr, fontsize=8.5, fontweight='bold')
        ax.text(0.55, 0.44 - i*0.08, sd, color=TEXT_MUTED, fontsize=7.5)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "fig09_band_selector_math.png")
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    print(f"Saved: {out_path} ({os.path.getsize(out_path)} bytes)")

def main():
    print("Generating CanoPie diagram assets...")
    draw_fig01()
    draw_fig02()
    draw_fig03()
    draw_fig04()
    draw_fig05()
    draw_fig06()
    draw_fig07()
    draw_fig08()
    draw_fig09()
    print("All diagram assets generated successfully!")

if __name__ == "__main__":
    main()
