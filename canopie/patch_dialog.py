
import os
import sys

TARGET_FILE = r"c:\Users\natha\Downloads\CanoPie_v_1_0_0 - Copia (2) - Copy\canopie\project_tab.py"

# Unique start string (Line 11507)
START_MARKER_1 = "        # Show processing mode dialog"
START_MARKER_2 = "class ProcessingModeDialog(QDialog):"

# Unique end string (Line 11698)
END_MARKER = 'logging.info(f"[CSV export] Selected: {processing_mode} with {n_workers} workers, background={run_background}")'

NEW_CONTENT = """        # Show processing mode dialog OR use provided options
        if options and 'processing_params' in options:
            processing_mode, n_workers, run_background = options['processing_params']
            logging.info(f"[CSV export] Using batch options: {processing_mode} with {n_workers} workers, background={run_background}")
        else:
            # Use module-level dialog class
            mode_dialog = ProcessingModeDialog(self, n_polygons, n_files, available_ram_gb, n_cpu,
                                           max_file_gb, allow_parallel, recommended_threads, auto_max_in_flight)
            if mode_dialog.exec_() != QtWidgets.QDialog.Accepted:
                return None
            
            processing_mode, n_workers = mode_dialog.get_mode()
            run_background = mode_dialog.run_in_background()
            if run_background:
                # Background exports always run in safe file-batched sequential.
                processing_mode, n_workers = ("multiprocessing", 1)
            logging.info(f"[CSV export] Selected: {processing_mode} with {n_workers} workers, background={run_background}")
"""

def main():
    if not os.path.exists(TARGET_FILE):
        print(f"Error: {TARGET_FILE} not found")
        sys.exit(1)

    print(f"Reading {TARGET_FILE}...")
    try:
        with open(TARGET_FILE, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Failed to read file: {e}")
        sys.exit(1)

    start_idx = -1
    end_idx = -1

    for i, line in enumerate(lines):
        if START_MARKER_1 in line:
            # Verify next few lines to ensure we are at the definition block
            if i+3 < len(lines) and START_MARKER_2 in lines[i+3]: 
                start_idx = i
        
        if END_MARKER in line:
            end_idx = i
            if start_idx != -1 and end_idx > start_idx:
                break
    
    if start_idx == -1:
         # Try finding just class def if comment missing/changed
         for i, line in enumerate(lines):
             if START_MARKER_2 in line:
                 start_idx = i - 3 # Adjust for comments/imports
                 break

    if start_idx == -1 or end_idx == -1:
        print(f"Could not find markers. Start: {start_idx}, End: {end_idx}")
        sys.exit(1)

    print(f"Found block: {start_idx} to {end_idx}")

    # Construct new lines
    # We remove lines from start_idx to end_idx (inclusive) -> replaced by NEW_CONTENT
    # NEW_CONTENT is a single string, we need to split it or write it
    
    final_lines = lines[:start_idx]
    final_lines.append(NEW_CONTENT)
    final_lines.extend(lines[end_idx+1:])

    print(f"Writing {len(lines)} -> {len(final_lines)} lines...")
    
    out_file = TARGET_FILE + ".new"
    try:
        with open(out_file, 'w', encoding='utf-8') as f:
            f.writelines(final_lines)
        print(f"Success! Written to {out_file}")
    except Exception as e:
        print(f"Failed to write file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
