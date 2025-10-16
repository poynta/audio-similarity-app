#!/usr/bin/env python3
"""
audio_gui.py
FreeSimpleGUI GUI for Audio Similarity app (macOS)

Features:
- CLAP-based audio similarity with segmented embeddings:
    20s at 25%, 50%, 75% of track
- Dynamic normalization per-segment, average per-track
- Small batching: 3 tracks per batch (9 segments per batch)
- FAISS nearest-neighbour index (IndexFlatIP on normalized vectors)
- Incremental indexing via SHA1 file hash
- FreeSimpleGUI UI: progress, pause/continue, clear index (confirm)
- Double-click result opens Finder and highlights that exact file
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
import faiss  # faiss-cpu

# ---------------- Config ----------------
FONT_LARGE = ("Helvetica", 14)
FONT_TABLE = ("Helvetica", 12)

AUDIO_EXTS = [".wav", ".flac", ".aiff", ".mp3"]
SEGMENT_SECONDS = 20
BATCH_TRACKS = 3
FAISS_INDEX_FILE = "index.faiss"
META_FILE = "index_meta.pkl"
CLAP_BATCH_SIZE = BATCH_TRACKS * 3  # 3 segments per track

# ---------------- GUI Layout ----------------
layout = [
    [sg.Text("Index Audio Folder:", font=FONT_LARGE),
     sg.Input(key="-FOLDER-", expand_x=True, font=FONT_LARGE),
     sg.FolderBrowse(font=FONT_LARGE), sg.Button("Index", font=FONT_LARGE)],

    [sg.Text("Progress:", font=FONT_LARGE),
     sg.Text("", key="-CURRENT_FILE-", size=(60, 1), expand_x=True, font=FONT_LARGE)],

    [sg.ProgressBar(max_value=1, orientation='h', key="-PROGRESS-", expand_x=True)],

    [sg.Button("Pause", font=FONT_LARGE), sg.Button("Continue", font=FONT_LARGE)],

    [sg.Text("Query Audio File:", font=FONT_LARGE),
     sg.Input(key="-QUERY-", expand_x=True, font=FONT_LARGE),
     sg.FileBrowse(font=FONT_LARGE),
     sg.Input(default_text="5", size=(5, 1), key="-K-", font=FONT_LARGE),
     sg.Button("Query", font=FONT_LARGE)],

    [sg.Table(values=[], headings=["Rank", "Track", "Similarity (%)"], key="-RESULTS-",
              expand_x=True, expand_y=True, auto_size_columns=False, col_widths=[5, 60, 12],
              justification='left', enable_events=True, font=FONT_TABLE)],

    [sg.Button("Clear Index", font=FONT_LARGE)]
]

window = sg.Window("Audio Similarity (CLAP + FAISS)", layout, resizable=True, finalize=True)

# ---------------- Shared State ----------------
indexing_state = {"paused": False, "running": False, "total_files": 1, "current_index": 0, "current_file": ""}

# ---------------- Initialize CLAP Model (clean, no lingering popup) ----------------
window["-CURRENT_FILE-"].update("Loading CLAP model (this may take a moment)...")
window.refresh()  # ensure text is visible immediately

try:
    model = CLAP_Module(enable_fusion=False)
    model.load_ckpt()  # download/load default pretrained checkpoint
except Exception as e:
    window["-CURRENT_FILE-"].update(f"Error loading CLAP model: {e}")
    raise

window["-CURRENT_FILE-"].update("CLAP model ready.")
window.refresh()
time.sleep(1)  # brief pause so user can see ready status
window["-CURRENT_FILE-"].update("")  # clear status

# ---------------- Helpers: Meta & FAISS ----------------
def save_meta_and_index(filenames, hashes, index):
    meta = {"files": filenames, "hashes": hashes, "dim": index.d}
    with open(META_FILE, "wb") as f:
        pickle.dump(meta, f)
    faiss.write_index(index, FAISS_INDEX_FILE)

def load_meta_and_index():
    if Path(META_FILE).exists() and Path(FAISS_INDEX_FILE).exists():
        with open(META_FILE, "rb") as f:
            meta = pickle.load(f)
        index = faiss.read_index(FAISS_INDEX_FILE)
        return meta["files"], meta["hashes"], index
    return [], [], None

def create_faiss_index(dim):
    return faiss.IndexFlatIP(dim)  # cosine similarity after normalization

# ---------------- Helpers: audio segmentation ----------------
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
                try:
                    os.remove(t)
                except:
                    pass
            return []
        tmp_files.append(tmp)
    return tmp_files

def cleanup_temp_files(files):
    for f in files:
        try:
            os.remove(f)
        except:
            pass

# ---------------- Index building ----------------
def index_folder_thread(folder_path):
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
    filenames, hashes, faiss_index = load_meta_and_index()
    if faiss_index is None:
        dim = None
    else:
        dim = faiss_index.d
    filenames, hashes = list(filenames), list(hashes)
    window["-PROGRESS-"].update(current_count=0, max=total)
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

        # Segment and embed
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
                    v /= np.linalg.norm(v) + 1e-9
                    seg_embs.append(v)
                    idx += 1
                avg = np.mean(seg_embs, axis=0)
                avg /= np.linalg.norm(avg) + 1e-9
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
        if faiss_index is None:
            dim = track_embeddings[0].shape[0]
            faiss_index = create_faiss_index(dim)
        for (f, f_hash), emb in zip(to_process, track_embeddings):
            faiss_index.add(np.expand_dims(emb.astype(np.float32), axis=0))
            filenames.append(str(f))
            hashes.append(f_hash or "")
            processed += 1
            window["-CURRENT_FILE-"].update(f"Indexed: {f.name}")
            window["-PROGRESS-"].update(current_count=min(i + 1, total))
            i += 1

    save_meta_and_index(filenames, hashes, faiss_index)
    indexing_state["running"] = False
    window["-CURRENT_FILE-"].update(f"Indexing complete: processed={processed}, skipped={skipped}, failed={failed}")
    window["-PROGRESS-"].update(current_count=total)

# ---------------- Query ----------------
def query_file_gui(file_path, top_k):
    filenames, hashes, faiss_index = load_meta_and_index()
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
    query_vec /= np.linalg.norm(query_vec) + 1e-9
    D, I = faiss_index.search(np.expand_dims(query_vec.astype(np.float32), axis=0), top_k)
    results = []
    for sim, idx in zip(D[0], I[0]):
        if idx < 0 or idx >= len(filenames):
            continue
        path = filenames[idx]
        results.append([len(results) + 1, path, f"{sim * 100:.1f}%"])
    window["-RESULTS-"].update(values=results)

# ---------------- Clear index ----------------
def clear_index():
    if sg.popup_yes_no("This will clear all indexed tracks. Proceed?") == "Yes":
        Path(FAISS_INDEX_FILE).unlink(missing_ok=True)
        Path(META_FILE).unlink(missing_ok=True)
        window["-RESULTS-"].update(values=[])
        window["-CURRENT_FILE-"].update("Index cleared.")

# ---------------- Start index thread ----------------
def start_indexing(folder_path):
    if indexing_state["running"]:
        return
    threading.Thread(target=index_folder_thread, args=(folder_path,), daemon=True).start()

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
    elif event == "Query":
        file = values["-QUERY-"]
        if not file:
            continue
        try:
            k = int(values["-K-"])
        except ValueError:
            k = 5
        query_file_gui(file, k)
    elif event == "Clear Index":
        threading.Thread(target=clear_index, daemon=True).start()
    elif event == "-RESULTS-":
        selected = values["-RESULTS-"]
        if selected:
            try:
                row_idx = selected[0]
                row = window["-RESULTS-"].get()[row_idx]
                track_path = Path(row[1])
                if track_path.exists():
                    # Reveal in Finder instead of just opening the folder
                    subprocess.run(["open", "-R", str(track_path)], check=False)
                else:
                    sg.popup_error(f"File not found:\n{track_path}")
            except Exception as e:
                sg.popup_error(f"Could not open Finder:\n{e}")

window.close()
