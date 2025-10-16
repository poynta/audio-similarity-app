#!/usr/bin/env python3
"""
audio_gui.py
FreeSimpleGUI GUI for Audio Similarity app (macOS)
Features:
- CLAP-based audio similarity (wav, flac, aiff, mp3)
- Non-blocking indexing with progress bar
- Pause / Continue buttons
- Shows current file being indexed
- Query section shows top similar tracks with percentage
- Clear index button at bottom with confirmation popup
- Resizable window with dynamically adjusting elements
- Double-click table row to open folder containing the track
"""

import FreeSimpleGUI as sg
from pathlib import Path
import threading
import time
import subprocess
import pickle
import numpy as np
from laion_clap import CLAP_Module
import hashlib

# ---------------- Fonts ----------------
FONT_LARGE = ("Helvetica", 14)
FONT_TABLE = ("Helvetica", 12)

# ---------------- GUI Layout ----------------
layout = [
    [sg.Text("Index Audio Folder:", font=FONT_LARGE), 
     sg.Input(key="-FOLDER-", expand_x=True, font=FONT_LARGE), 
     sg.FolderBrowse(font=FONT_LARGE), sg.Button("Index", font=FONT_LARGE)],
     
    [sg.Text("Progress:", font=FONT_LARGE), 
     sg.Text("", key="-CURRENT_FILE-", size=(50,1), expand_x=True, font=FONT_LARGE)],
    
    [sg.ProgressBar(max_value=1, orientation='h', key="-PROGRESS-", expand_x=True)],
    
    [sg.Button("Pause", font=FONT_LARGE), sg.Button("Continue", font=FONT_LARGE)],
    
    [sg.Text("Query Audio File:", font=FONT_LARGE), 
     sg.Input(key="-QUERY-", expand_x=True, font=FONT_LARGE), 
     sg.FileBrowse(font=FONT_LARGE), 
     sg.Input(default_text="3", size=(5,1), key="-K-", font=FONT_LARGE), 
     sg.Button("Query", font=FONT_LARGE)],
     
    [sg.Table(values=[], headings=["Rank", "Track", "Similarity (%)"], key="-RESULTS-", 
              expand_x=True, expand_y=True, auto_size_columns=False, col_widths=[5,40,12],
              justification='left', enable_events=True, font=FONT_TABLE)],
    
    [sg.Button("Clear Index", font=FONT_LARGE)]  # moved to bottom
]

window = sg.Window("Audio Similarity (CLAP)", layout, resizable=True, finalize=True)

# ---------------- Shared State ----------------
indexing_state = {"paused": False, "running": False, "total_files": 1, "current_index": 0, "current_file": ""}

# ---------------- Initialize CLAP Model ----------------
model = CLAP_Module(enable_fusion=False)
model.load_ckpt()

INDEX_FILE = "index_embeddings.pkl"

# ---------------- Helper Functions ----------------

def save_index(file_paths, embeddings):
    with open(INDEX_FILE, "wb") as f:
        pickle.dump({"files": file_paths, "embeddings": embeddings}, f)

def load_index():
    if Path(INDEX_FILE).exists():
        with open(INDEX_FILE, "rb") as f:
            return pickle.load(f)
    return {"files": [], "embeddings": []}

def log_results_table(results):
    """Update the results table with query results."""
    window["-RESULTS-"].update(values=results)

def query_file_gui(file_path, k):
    """Query a file and display top results in the table using CLAP embeddings."""
    index_data = load_index()
    file_paths = index_data["files"]
    embeddings = index_data["embeddings"]

    if not embeddings:
        window["-CURRENT_FILE-"].update("Index is empty. Please index first.")
        return

    query_vec = model.get_audio_embedding_from_filelist([str(file_path)], use_tensor=False)[0]
    query_vec = query_vec / np.linalg.norm(query_vec)

    similarities = [float(np.dot(query_vec, vec)) for vec in embeddings]
    similarities_percentage = [s * 100 for s in similarities]

    results = sorted(zip(file_paths, similarities_percentage), key=lambda x: x[1], reverse=True)[:k]
    table_values = [[i+1, path, f"{sim:.1f}%"] for i, (path, sim) in enumerate(results)]
    log_results_table(table_values)

# ---------------- Indexing Thread & Functions ----------------

def index_folder_thread(folder_path):
    folder = Path(folder_path)
    if not folder.exists():
        window["-CURRENT_FILE-"].update(f"Folder not found: {folder}")
        return

    files = []
    for ext in [".wav", ".flac", ".aiff", ".mp3"]:
        files.extend(folder.glob(f"*{ext}"))

    if not files:
        window["-CURRENT_FILE-"].update("No audio files found")
        return

    indexing_state["total_files"] = len(files)
    indexing_state["current_index"] = 0
    indexing_state["running"] = True

    # Load existing index if present
    index_data = load_index()
    existing_files = {f: idx for idx, f in enumerate(index_data["files"])}
    file_paths = index_data["files"].copy()
    embeddings = index_data["embeddings"].copy()

    window["-PROGRESS-"].update(current_count=0, max=indexing_state["total_files"])

    processed = 0
    skipped = 0
    failed = 0

    for i, f in enumerate(files):
        indexing_state["current_file"] = str(f.name)
        indexing_state["current_index"] = i
        window["-CURRENT_FILE-"].update(f"Processing: {f.name}")
        window["-PROGRESS-"].update(current_count=i+1)

        while indexing_state["paused"]:
            time.sleep(0.1)

        try:
            with open(f, "rb") as fh:
                f_hash = hashlib.sha1(fh.read()).hexdigest()
        except Exception:
            failed += 1
            continue

        if str(f) in existing_files:
            skipped += 1
            continue

        try:
            vec = model.get_audio_embedding_from_filelist([str(f)], use_tensor=False)[0]
            vec = vec / np.linalg.norm(vec)
            embeddings.append(vec)
            file_paths.append(str(f))
            processed += 1
        except Exception as e:
            print(f"Failed to process {f.name}: {e}")
            failed += 1

    save_index(file_paths, embeddings)
    indexing_state["running"] = False
    window["-CURRENT_FILE-"].update(f"Indexing complete: processed={processed}, skipped={skipped}, failed={failed}")
    window["-PROGRESS-"].update(current_count=indexing_state["total_files"])

def start_indexing(folder_path):
    if indexing_state["running"]:
        return
    threading.Thread(target=index_folder_thread, args=(folder_path,), daemon=True).start()

def clear_index():
    if sg.popup_yes_no("This will clear all scanned tracks. Proceed?") == "Yes":
        Path(INDEX_FILE).unlink(missing_ok=True)
        window["-RESULTS-"].update(values=[])
        window["-CURRENT_FILE-"].update("Index cleared.")

# ---------------- Event Loop ----------------

while True:
    event, values = window.read(timeout=100)
    if event == sg.WINDOW_CLOSED:
        break
    elif event == "Index":
        folder = values["-FOLDER-"]
        if folder:
            start_indexing(folder)
        else:
            window["-CURRENT_FILE-"].update("Please select a folder to index")
    elif event == "Pause":
        indexing_state["paused"] = True
    elif event == "Continue":
        indexing_state["paused"] = False
    elif event == "Query":
        file = values["-QUERY-"]
        if file:
            try:
                k = int(values["-K-"])
            except ValueError:
                k = 3
            query_file_gui(file, k)
    elif event == "Clear Index":
        threading.Thread(target=clear_index, daemon=True).start()
    elif event == "-RESULTS-":
        selected_rows = values["-RESULTS-"]
        if selected_rows:
            row_index = selected_rows[0]
            track_path_str = window["-RESULTS-"].get()[row_index][1]
            track_path = Path(track_path_str)
            if track_path.exists():
                subprocess.run(["open", track_path.parent])

window.close()
