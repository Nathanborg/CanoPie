import os

filename = r"canopie\project_tab.py"
if not os.path.exists(filename):
    print(f"File not found: {filename}")
else:
    with open(filename, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            if "class ProjectImagesExportWorker" in line or "def export_project_images" in line or "def _save_tiff" in line:
                print(f"Line {idx}: {line.strip()}")
