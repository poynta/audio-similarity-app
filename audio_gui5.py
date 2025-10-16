#!/usr/bin/env python3
"""
audio_gui5.py
FreeSimpleGUI GUI for Audio Similarity app (macOS)

- Right-click on results table to "Query This Track" or "Show in Folder"
- Hidden column keeps full path for each result
- Safe row handling to avoid IndexError
- Metadata cache now includes display names
"""

import os
import sys
import tempfile
import subprocess
import threading
import time
import pickle
import hashlib
from pathlib import Path
import queue

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

# Persistent, writable app data directory (works in PyInstaller bundles)
def get_app_data_dir() -> Path:
    base = os.environ.get("AUDIO_SIM_DATA_DIR")
    if base:
        p = Path(base)
    else:
        # macOS: ~/Library/Application Support/AudioSimilarity
        if sys.platform == "darwin":
            p = Path.home() / "Library" / "Application Support" / "AudioSimilarity"
        # Linux: ~/.local/share/AudioSimilarity
        elif sys.platform.startswith("linux"):
            p = Path.home() / ".local" / "share" / "AudioSimilarity"
        # Windows: %APPDATA%/AudioSimilarity
        elif os.name == "nt":
            p = Path(os.environ.get("APPDATA", str(Path.home() / "AppData" / "Roaming"))) / "AudioSimilarity"
        else:
            p = Path.home() / ".audio_similarity"
    p.mkdir(parents=True, exist_ok=True)
    return p

APP_DATA_DIR = get_app_data_dir()
FAISS_INDEX_FILE = APP_DATA_DIR / "index.faiss"
META_FILE = APP_DATA_DIR / "index_meta.pkl"

# ---------------- GUI Layout ----------------
results_headings = ["Rank", "Track", "Similarity (%)", "FullPath"]  # last column hidden

layout = [
    [sg.Text("Index Audio Folder:", font=FONT_LARGE),
     sg.Input(key="-FOLDER-", expand_x=True, font=FONT_LARGE),
     sg.FolderBrowse(font=FONT_LARGE), sg.Button("Index", key="-BTN-INDEX-", font=FONT_LARGE, disabled=True)],

    [sg.Text("Progress:", font=FONT_LARGE),
     sg.Text("", key="-CURRENT_FILE-", size=(60,1), expand_x=True, font=FONT_LARGE)],

    [sg.ProgressBar(max_value=1, orientation='h', key="-PROGRESS-", expand_x=True)],

    [sg.Button("Pause", key="-BTN-PAUSE-", font=FONT_LARGE, disabled=True),
     sg.Button("Continue", key="-BTN-CONTINUE-", font=FONT_LARGE, disabled=True)],

    # Search input + listbox
    [sg.Text("Search Indexed Track:", font=FONT_LARGE)],
    [sg.Input(key="-INDEX_SEARCH_INPUT-", size=(60,1), enable_events=True, font=FONT_LARGE)],
    [sg.Listbox(values=[], key="-INDEX_MATCHES-", size=(80,6), enable_events=True, font=FONT_TABLE)],
    [sg.Button("Query Indexed", key="-BTN-QUERY-IDX-", font=FONT_LARGE, disabled=True)],

    # External file query
    [sg.Text("Query External File:", font=FONT_LARGE),
     sg.Input(key="-QUERY-", expand_x=True, font=FONT_LARGE),
     sg.FileBrowse(font=FONT_LARGE),
     sg.Input(default_text="5", size=(5,1), key="-K-", font=FONT_LARGE),
     sg.Button("Query", key="-BTN-QUERY-", font=FONT_LARGE, disabled=True)],

    [sg.Table(values=[], headings=results_headings, key="-RESULTS-",
              expand_x=True, expand_y=True, auto_size_columns=False,
              col_widths=[5,60,12,0], justification='left', enable_events=True,
              right_click_menu=["", ["Query This Track", "Show in Folder"]],
              font=FONT_TABLE)],

    [sg.Button("Clear Index", key="-BTN-CLEAR-", font=FONT_LARGE, disabled=True)]
]

window = sg.Window("Audio Similarity (CLAP + FAISS) — Searchable Index", layout, resizable=True, finalize=True)

# ---------------- Shared State ----------------
indexing_state = {"paused": False, "running": False, "total_files": 1, "current_index": 0, "current_file": ""}
embeddings_all = []  # list of numpy arrays
filenames = []       # full paths
hashes = []          # sha1 hashes
display_names = []   # cached display names
faiss_index = None
model = None

# Thread-safe UI event key and queue fallback
UI_EVENT_KEY = "-THREAD-UI-"
ui_queue: "queue.Queue[tuple]" = queue.Queue()

def post_ui(action: str, **payload):
    """Post a UI action to the main thread."""
    try:
        # Prefer write_event_value when available (thread-safe)
        window.write_event_value(UI_EVENT_KEY, (action, payload))
    except Exception:
        ui_queue.put((action, payload))

# ---------------- Initialize CLAP Model (background) ----------------
def _load_model_thread():
    global model
    try:
        post_ui("status", text="Loading CLAP model (this may take a moment)...")
        m = CLAP_Module(enable_fusion=False)
        m.load_ckpt()
        model = m
        window.write_event_value("-MODEL-READY-", True)
    except Exception as e:
        window.write_event_value("-MODEL-ERROR-", str(e))

threading.Thread(target=_load_model_thread, daemon=True).start()

# ---------------- Helpers ----------------
def get_metadata_display(path):
    """Return a display-friendly name for a file path."""
    return Path(path).name

def save_meta_and_index(filenames_list, hashes_list, embeddings_list, index):
    display_list = [get_metadata_display(f) for f in filenames_list]
    meta = {
        "files": filenames_list,
        "hashes": hashes_list,
        "display_names": display_list,
        "dim": index.d if index else (embeddings_list[0].shape[0] if embeddings_list else 0),
        "embeddings": embeddings_list
    }
    with open(META_FILE, "wb") as f:
        pickle.dump(meta, f)
    if index:
        faiss.write_index(index, str(FAISS_INDEX_FILE))

def load_meta_and_index():
    if Path(META_FILE).exists() and Path(FAISS_INDEX_FILE).exists():
        with open(META_FILE, "rb") as f:
            meta = pickle.load(f)
        index = faiss.read_index(str(FAISS_INDEX_FILE))
        emb_list = meta.get("embeddings", [])
        emb_list = [np.array(e, dtype=np.float32) for e in emb_list]
        filenames_list = list(meta.get("files", []))
        hashes_list = list(meta.get("hashes", []))
        display_list = list(meta.get("display_names", []))
        # handle missing or mismatched display_names
        if not display_list or len(display_list) != len(filenames_list):
            display_list = [get_metadata_display(f) for f in filenames_list]
        return filenames_list, hashes_list, display_list, emb_list, index
    return [], [], [], [], None

def create_faiss_index(dim):
    return faiss.IndexFlatIP(dim)

def update_index_search_list():
    window["-INDEX_MATCHES-"].update(values=display_names)

def reveal_in_file_manager(path: Path):
    try:
        if sys.platform == "darwin":
            subprocess.run(["open", "-R", str(path)], check=False)
        elif sys.platform.startswith("linux"):
            subprocess.run(["xdg-open", str(path.parent)], check=False)
        elif os.name == "nt":
            subprocess.run(["explorer", "/select,", str(path)], check=False)
    except Exception:
        pass

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
filenames, hashes, display_names, embeddings_all, faiss_index = load_meta_and_index()
if faiss_index is None and embeddings_all:
    dim = embeddings_all[0].shape[0]
    faiss_index = create_faiss_index(dim)
    faiss_index.add(np.vstack(embeddings_all).astype(np.float32))
update_index_search_list()

# ---------------- Index building ----------------
def index_folder_thread(folder_path):
    global filenames, hashes, display_names, embeddings_all, faiss_index
    folder = Path(folder_path)
    if not folder.exists():
        post_ui("status", text=f"Folder not found: {folder}")
        return

    files = [f for ext in AUDIO_EXTS for f in sorted(folder.rglob(f"*{ext}"))]
    if not files:
        post_ui("status", text="No audio files found")
        return

    total = len(files)
    indexing_state.update({"total_files": total, "running": True, "current_index": 0})
    post_ui("progress", current=0, max=total)

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
                post_ui("status", text=f"Skipping: {f.name}")
                post_ui("progress", current=i + 1, max=total)
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
            display_names.append(get_metadata_display(f))
            embeddings_all.append(emb.astype(np.float32))
            processed += 1
            post_ui("status", text=f"Indexed: {f.name}")
            post_ui("progress", current=min(i + 1, total), max=total)
            i += 1

    save_meta_and_index(filenames, hashes, embeddings_all, faiss_index)
    indexing_state["running"] = False
    post_ui("rebuild_index_list")
    post_ui("status", text=f"Indexing complete: processed={processed}, skipped={skipped}, failed={failed}")
    post_ui("progress", current=total, max=total)

def start_indexing(folder_path):
    if indexing_state["running"]:
        return
    threading.Thread(target=index_folder_thread, args=(folder_path,), daemon=True).start()

# ---------------- Query helpers ----------------
def query_file_gui_external(file_path, top_k):
    """Run external query in a background thread and post results to UI."""
    def _thread():
        if faiss_index is None or not filenames:
            post_ui("status", text="Index is empty. Please index first.")
            return
        if model is None:
            post_ui("status", text="Model not ready yet.")
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
        post_ui("results", rows=results)
    threading.Thread(target=_thread, daemon=True).start()

def query_file_gui_indexed_by_idx(idx, top_k):
    def _thread():
        if faiss_index is None or not filenames:
            post_ui("status", text="Index is empty.")
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
        post_ui("results", rows=results)
    threading.Thread(target=_thread, daemon=True).start()

def query_file_gui_indexed(selected_name, top_k):
    matched_idx = None
    for i, name in enumerate(display_names):
        if name == selected_name:
            matched_idx = i
            break
    if matched_idx is not None:
        query_file_gui_indexed_by_idx(matched_idx, top_k)
    else:
        window["-CURRENT_FILE-"].update("Selected track not found in metadata.")

# ---------------- Clear index ----------------
def clear_index():
    global filenames, hashes, display_names, embeddings_all, faiss_index
    if sg.popup_yes_no("This will clear the FAISS index and metadata. Proceed?") == "Yes":
        try:
            Path(FAISS_INDEX_FILE).unlink(missing_ok=True)
            Path(META_FILE).unlink(missing_ok=True)
        except Exception:
            pass
        filenames, hashes, display_names, embeddings_all, faiss_index = [], [], [], [], None
        window["-RESULTS-"].update(values=[])
        window["-CURRENT_FILE-"].update("Index cleared.")
        update_index_search_list()

# ---------------- Event loop ----------------
while True:
    # Drain fallback UI queue if used
    try:
        while True:
            action, payload = ui_queue.get_nowait()
            window.write_event_value(UI_EVENT_KEY, (action, payload))
    except Exception:
        pass

    event, values = window.read(timeout=100)
    if event == sg.WINDOW_CLOSED:
        break
    elif event in ("Index", "-BTN-INDEX-"):
        folder = values["-FOLDER-"]
        if folder:
            start_indexing(folder)
        else:
            window["-CURRENT_FILE-"].update("Please select a folder.")
    elif event in ("Pause", "-BTN-PAUSE-"):
        indexing_state["paused"] = True
    elif event in ("Continue", "-BTN-CONTINUE-"):
        indexing_state["paused"] = False
    elif event == "-INDEX_SEARCH_INPUT-":
        q = values.get("-INDEX_SEARCH_INPUT-", "").strip().lower()
        matches = [n for n in display_names if q in n.lower()] if q else display_names
        window["-INDEX_MATCHES-"].update(values=matches[:1000])
    elif event == "-INDEX_MATCHES-":
        sel = values.get("-INDEX_MATCHES-")
        if sel:
            window["-INDEX_SEARCH_INPUT-"].update(sel[0])
    elif event in ("Query", "-BTN-QUERY-"):
        file = values["-QUERY-"]
        if not file:
            continue
        try:
            k = int(values["-K-"])
        except ValueError:
            k = 5
        query_file_gui_external(file, k)
    elif event in ("Query Indexed", "-BTN-QUERY-IDX-"):
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
    elif event in ("Clear Index", "-BTN-CLEAR-"):
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
            p = Path(full_path)
            if p.exists():
                reveal_in_file_manager(p)
    elif event == UI_EVENT_KEY:
        try:
            action, payload = values.get(UI_EVENT_KEY, (None, {}))
        except Exception:
            action, payload = (None, {})
        if action == "status":
            window["-CURRENT_FILE-"].update(payload.get("text", ""))
        elif action == "results":
            rows = payload.get("rows", [])
            window["-RESULTS-"].update(values=rows)
        elif action == "progress":
            cur = payload.get("current", 0)
            mx = payload.get("max", 1)
            window["-PROGRESS-"].update(current_count=cur, max=mx)
        elif action == "rebuild_index_list":
            update_index_search_list()
    elif event == "-MODEL-READY-":
        window["-CURRENT_FILE-"].update("CLAP model ready.")
        for key in ("-BTN-INDEX-", "-BTN-PAUSE-", "-BTN-CONTINUE-", "-BTN-QUERY-IDX-", "-BTN-QUERY-", "-BTN-CLEAR-"):
            try:
                window[key].update(disabled=False)
            except Exception:
                pass
        window.refresh()
        time.sleep(0.4)
        window["-CURRENT_FILE-"].update("")
    elif event == "-MODEL-ERROR-":
        err = values.get("-MODEL-ERROR-", "Unknown error loading model")
        sg.popup_error(f"Error loading CLAP model:\n{err}")

window.close()
