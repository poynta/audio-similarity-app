#!/usr/bin/env python3
"""
audio_gui5.py
FreeSimpleGUI GUI for Audio Similarity app (macOS)

- Right-click on results table to "Query This Track" or "Show in Folder"
- Hidden column keeps full path for each result
- Safe row handling to avoid IndexError
"""

import os
import tempfile
import subprocess
import threading
import time
import pickle
import hashlib
from pathlib import Path

import numpy as np
import FreeSimpleGUI as sg
from laion_clap import CLAP_Module
import faiss

# ---------------- Config ----------------
FONT_LARGE = ("Helvetica", 14)
FONT_TABLE = ("Helvetica", 12)

AUDIO_EXTS = [".wav", ".flac", ".aiff", ".mp3"]
SEGMENT_SECONDS = 20
BATCH_TRACKS = 3
FAISS_INDEX_FILE = "index.faiss"
META_FILE = "index_meta.pkl"

# ---------------- GUI Layout ----------------
results_headings = ["Rank", "Track", "Similarity (%)", "FullPath"]  # last column hidden

layout = [
    [sg.Text("Index Audio Folder:", font=FONT_LARGE),
     sg.Input(key="-FOLDER-", expand_x=True, font=FONT_LARGE),
     sg.FolderBrowse(font=FONT_LARGE), sg.Button("Index", font=FONT_LARGE)],

    [sg.Text("Progress:", font=FONT_LARGE),
     sg.Text("", key="-CURRENT_FILE-", size=(60,1), expand_x=True, font=FONT_LARGE)],

    [sg.ProgressBar(max_value=1, orientation='h', key="-PROGRESS-", expand_x=True)],

    [sg.Button("Pause", font=FONT_LARGE), sg.Button("Continue", font=FONT_LARGE)],

    # Search input + listbox
    [sg.Text("Search Indexed Track:", font=FONT_LARGE)],
    [sg.Input(key="-INDEX_SEARCH_INPUT-", size=(60,1), enable_events=True, font=FONT_LARGE)],
    [sg.Listbox(values=[], key="-INDEX_MATCHES-", size=(80,6), enable_events=True, font=FONT_TABLE)],
    [sg.Button("Query Indexed", font=FONT_LARGE)],

    # External file query
    [sg.Text("Query External File:", font=FONT_LARGE),
     sg.Input(key="-QUERY-", expand_x=True, font=FONT_LARGE),
     sg.FileBrowse(font=FONT_LARGE),
     sg.Input(default_text="5", size=(5,1), key="-K-", font=FONT_LARGE), sg.Button("Query", font=FONT_LARGE)],

    [sg.Table(values=[], headings=results_headings, key="-RESULTS-",
              expand_x=True, expand_y=True, auto_size_columns=False,
              col_widths=[5,60,12,0], justification='left', enable_events=True,
              right_click_menu=["", ["Query This Track", "Show in Folder"]],
              font=FONT_TABLE)],

    [sg.Button("Clear Index", font=FONT_LARGE)]
]

window = sg.Window("Audio Similarity (CLAP + FAISS) — Searchable Index", layout, resizable=True, finalize=True)

# ---------------- Shared State ----------------
indexing_state = {"paused": False, "running": False, "total_files": 1, "current_index": 0, "current_file": ""}
embeddings_all = []  # list of numpy arrays
filenames = []       # full paths
hashes = []          # sha1 hashes
faiss_index = None

# ---------------- Initialize CLAP Model ----------------
window["-CURRENT_FILE-"].update("Loading CLAP model (this may take a moment)...")
window.refresh()
model = CLAP_Module(enable_fusion=False)
model.load_ckpt()
window["-CURRENT_FILE-"].update("CLAP model ready.")
window.refresh()
time.sleep(0.6)
window["-CURRENT_FILE-"].update("")

# ---------------- Helpers ----------------
def save_meta_and_index(filenames_list, hashes_list, embeddings_list, index):
    meta = {
        "files": filenames_list,
        "hashes": hashes_list,
        "dim": index.d if index else (embeddings_list[0].shape[0] if embeddings_list else 0),
        "embeddings": embeddings_list
    }
    with open(META_FILE, "wb") as f:
        pickle.dump(meta, f)
    if index:
        faiss.write_index(index, FAISS_INDEX_FILE)

def load_meta_and_index():
    if Path(META_FILE).exists() and Path(FAISS_INDEX_FILE).exists():
        with open(META_FILE, "rb") as f:
            meta = pickle.load(f)
        index = faiss.read_index(FAISS_INDEX_FILE)
        emb_list = meta.get("embeddings", [])
        emb_list = [np.array(e, dtype=np.float32) for e in emb_list]
        return list(meta.get("files", [])), list(meta.get("hashes", [])), emb_list, index
    return [], [], [], None

def create_faiss_index(dim):
    return faiss.IndexFlatIP(dim)

def update_index_search_list():
    short_names = [Path(p).name for p in filenames]
    window["-INDEX_MATCHES-"].update(values=short_names)

# ---------------- Audio segmentation ----------------
def get_audio_duration(path):
    try:
        out = subprocess.check_output([
            "ffprobe", "-v", "error", "-select_streams", "a:0",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(path)
        ], stderr=subprocess.DEVNULL).decode().strip()
        return float(out)
    except Exception:
        return None

def extract_segment_with_ffmpeg(src_path, start_s, duration_s, out_path):
    try:
        subprocess.run([
            "ffmpeg", "-y", "-ss", str(start_s), "-t", str(duration_s),
            "-i", str(src_path), "-ar", "16000", "-ac", "1", "-f", "wav", str(out_path)
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except Exception:
        return False

def make_segments_for_file(path):
    dur = get_audio_duration(path)
    if dur is None:
        return []
    starts = []
    for frac in (0.25, 0.5, 0.75):
        start = max(0.0, dur * frac - SEGMENT_SECONDS / 2)
        if start + SEGMENT_SECONDS > dur:
            start = max(0.0, dur - SEGMENT_SECONDS)
        starts.append(start)
    tmp_files = []
    for s in starts:
        fd, tmp = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        ok = extract_segment_with_ffmpeg(path, s, SEGMENT_SECONDS, tmp)
        if not ok:
            for t in tmp_files:
                try: os.remove(t)
                except: pass
            return []
        tmp_files.append(tmp)
    return tmp_files

def cleanup_temp_files(files):
    for f in files:
        try: os.remove(f)
        except: pass

# ---------------- Load existing meta & index ----------------
filenames, hashes, embeddings_all, faiss_index = load_meta_and_index()
if faiss_index is None:
    faiss_index = None
update_index_search_list()

# ---------------- Index building ----------------
def index_folder_thread(folder_path):
    global filenames, hashes, embeddings_all, faiss_index
    folder = Path(folder_path)
    if not folder.exists():
        window["-CURRENT_FILE-"].update(f"Folder not found: {folder}")
        return

    files = [f for ext in AUDIO_EXTS for f in sorted(folder.rglob(f"*{ext}"))]
    if not files:
        window["-CURRENT_FILE-"].update("No audio files found")
        return

    total = len(files)
    indexing_state.update({"total_files": total, "running": True, "current_index": 0})
    window["-PROGRESS-"].update(current_count=0, max=total)

    if faiss_index is None and embeddings_all:
        dim = embeddings_all[0].shape[0]
        faiss_index = create_faiss_index(dim)
        faiss_index.add(np.vstack(embeddings_all).astype(np.float32))

    processed = skipped = failed = 0
    i = 0
    while i < total:
        if indexing_state["paused"]:
            time.sleep(0.1)
            continue

        batch_files = files[i:i + BATCH_TRACKS]
        to_process = []
        for f in batch_files:
            try:
                with open(f, "rb") as fh:
                    f_hash = hashlib.sha1(fh.read()).hexdigest()
            except Exception:
                f_hash = None
            if f_hash and (str(f) in filenames or f_hash in hashes):
                skipped += 1
                window["-CURRENT_FILE-"].update(f"Skipping: {f.name}")
                window["-PROGRESS-"].update(current_count=i + 1)
                i += 1
                continue
            to_process.append((f, f_hash))
        if not to_process:
            continue

        batch_segment_paths, track_segment_map = [], []
        for f, _ in to_process:
            segs = make_segments_for_file(str(f))
            if segs:
                track_segment_map.append(segs)
                batch_segment_paths.extend(segs)
            else:
                track_segment_map.append([str(f)])
                batch_segment_paths.append(str(f))

        try:
            embeddings = model.get_audio_embedding_from_filelist(batch_segment_paths, use_tensor=False)
            idx = 0
            track_embeddings = []
            for segs in track_segment_map:
                seg_embs = []
                for _ in segs:
                    v = np.array(embeddings[idx], dtype=np.float32)
                    v /= (np.linalg.norm(v) + 1e-9)
                    seg_embs.append(v)
                    idx += 1
                avg = np.mean(seg_embs, axis=0)
                avg /= (np.linalg.norm(avg) + 1e-9)
                track_embeddings.append(avg)
        except Exception as e:
            print("Batch embedding failed:", e)
            failed += len(to_process)
            for segs in track_segment_map:
                cleanup_temp_files(segs)
            i += len(batch_files)
            continue

        for segs in track_segment_map:
            cleanup_temp_files([p for p in segs if p.endswith(".wav")])

        if faiss_index is None and track_embeddings:
            dim = track_embeddings[0].shape[0]
            faiss_index = create_faiss_index(dim)

        for (f, f_hash), emb in zip(to_process, track_embeddings):
            faiss_index.add(np.expand_dims(emb.astype(np.float32), axis=0))
            filenames.append(str(f))
            hashes.append(f_hash or "")
            embeddings_all.append(emb.astype(np.float32))
            processed += 1
            window["-CURRENT_FILE-"].update(f"Indexed: {f.name}")
            window["-PROGRESS-"].update(current_count=min(i + 1, total))
            i += 1

    save_meta_and_index(filenames, hashes, embeddings_all, faiss_index)
    update_index_search_list()
    indexing_state["running"] = False
    window["-CURRENT_FILE-"].update(f"Indexing complete: processed={processed}, skipped={skipped}, failed={failed}")
    window["-PROGRESS-"].update(current_count=total)

def start_indexing(folder_path):
    if indexing_state["running"]:
        return
    threading.Thread(target=index_folder_thread, args=(folder_path,), daemon=True).start()

# ---------------- Query helpers ----------------
def query_file_gui_external(file_path, top_k):
    if faiss_index is None or not filenames:
        window["-CURRENT_FILE-"].update("Index is empty. Please index first.")
        return

    segs = make_segments_for_file(str(file_path))
    if segs:
        emb_list = model.get_audio_embedding_from_filelist(segs, use_tensor=False)
        cleanup_temp_files(segs)
        vecs = [np.array(e, dtype=np.float32) / (np.linalg.norm(e) + 1e-9) for e in emb_list]
        query_vec = np.mean(vecs, axis=0)
    else:
        emb = model.get_audio_embedding_from_filelist([str(file_path)], use_tensor=False)[0]
        query_vec = np.array(emb, dtype=np.float32) / (np.linalg.norm(emb) + 1e-9)

    query_vec /= (np.linalg.norm(query_vec) + 1e-9)
    D, I = faiss_index.search(np.expand_dims(query_vec.astype(np.float32), axis=0), top_k)
    results = []
    for sim, idx in zip(D[0], I[0]):
        if idx < 0 or idx >= len(filenames):
            continue
        path = filenames[idx]
        results.append([len(results)+1, Path(path).name, f"{sim*100:.1f}%", path])
    window["-RESULTS-"].update(values=results)

def query_file_gui_indexed_by_idx(idx, top_k):
    """Query track in index by its index in filenames/embeddings."""
    if faiss_index is None or not filenames:
        window["-CURRENT_FILE-"].update("Index is empty.")
        return
    try:
        vec = np.empty((faiss_index.d,), dtype='float32')
        faiss_index.reconstruct(idx, vec)
        query_vec = vec / (np.linalg.norm(vec) + 1e-9)
    except Exception:
        query_vec = embeddings_all[idx] / (np.linalg.norm(embeddings_all[idx]) + 1e-9)
    D, I = faiss_index.search(np.expand_dims(query_vec.astype(np.float32), axis=0), top_k+1)
    results = []
    for sim, i_idx in zip(D[0], I[0]):
        if i_idx == idx or i_idx < 0 or i_idx >= len(filenames):
            continue
        results.append([len(results)+1, Path(filenames[i_idx]).name, f"{sim*100:.1f}%", filenames[i_idx]])
        if len(results) >= top_k:
            break
    window["-RESULTS-"].update(values=results)

def query_file_gui_indexed(selected_name, top_k):
    """Query using track name from index."""
    matched_idx = None
    for i, p in enumerate(filenames):
        if Path(p).name == selected_name:
            matched_idx = i
            break
    if matched_idx is not None:
        query_file_gui_indexed_by_idx(matched_idx, top_k)
    else:
        window["-CURRENT_FILE-"].update("Selected track not found in metadata.")

# ---------------- Clear index ----------------
def clear_index():
    global filenames, hashes, embeddings_all, faiss_index
    if sg.popup_yes_no("This will clear the FAISS index and metadata. Proceed?") == "Yes":
        try:
            Path(FAISS_INDEX_FILE).unlink(missing_ok=True)
            Path(META_FILE).unlink(missing_ok=True)
        except Exception:
            pass
        filenames, hashes, embeddings_all, faiss_index = [], [], [], None
        window["-RESULTS-"].update(values=[])
        window["-CURRENT_FILE-"].update("Index cleared.")
        update_index_search_list()

# ---------------- Event loop ----------------
while True:
    event, values = window.read(timeout=100)
    if event == sg.WINDOW_CLOSED:
        break
    elif event == "Index":
        folder = values["-FOLDER-"]
        if folder:
            start_indexing(folder)
        else:
            window["-CURRENT_FILE-"].update("Please select a folder.")
    elif event == "Pause":
        indexing_state["paused"] = True
    elif event == "Continue":
        indexing_state["paused"] = False
    elif event == "-INDEX_SEARCH_INPUT-":
        q = values.get("-INDEX_SEARCH_INPUT-", "").strip().lower()
        short_names = [Path(p).name for p in filenames]
        matches = [n for n in short_names if q in n.lower()] if q else short_names
        window["-INDEX_MATCHES-"].update(values=matches[:1000])
    elif event == "-INDEX_MATCHES-":
        sel = values.get("-INDEX_MATCHES-")
        if sel:
            window["-INDEX_SEARCH_INPUT-"].update(sel[0])
    elif event == "Query":
        file = values["-QUERY-"]
        if not file:
            continue
        try:
            k = int(values["-K-"])
        except ValueError:
            k = 5
        query_file_gui_external(file, k)
    elif event == "Query Indexed":
        sel_list = values.get("-INDEX_MATCHES-")
        sel_name = sel_list[0] if sel_list else values.get("-INDEX_SEARCH_INPUT-", "").strip()
        if not sel_name:
            window["-CURRENT_FILE-"].update("No indexed track selected/found.")
            continue
        try:
            k = int(values.get("-K-", 5))
        except ValueError:
            k = 5
        query_file_gui_indexed(sel_name, k)
    elif event == "Clear Index":
        threading.Thread(target=clear_index, daemon=True).start()
    elif event in ["Query This Track", "Show in Folder"]:
        sel_rows = values["-RESULTS-"]
        if not sel_rows:
            continue
        row_idx = sel_rows[0]
        table_data = window["-RESULTS-"].get()
        if not (0 <= row_idx < len(table_data)):
            continue
        full_path = table_data[row_idx][3]  # hidden full path
        if event == "Query This Track":
            if full_path in filenames:
                idx = filenames.index(full_path)
                try:
                    k = int(values.get("-K-", 5))
                except ValueError:
                    k = 5
                query_file_gui_indexed_by_idx(idx, k)
        elif event == "Show in Folder":
            if Path(full_path).exists():
                subprocess.run(["open", Path(full_path).parent])

window.close()
