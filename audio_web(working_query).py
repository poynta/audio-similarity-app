#!/usr/bin/env python3
"""
audio_web_fixed.py — Local browser-based Audio Similarity app
Single tab: Index folder + query external files + query indexed tracks
"""

import os, tempfile, subprocess, pickle, hashlib, threading
from pathlib import Path

import numpy as np
import gradio as gr
from laion_clap import CLAP_Module
import faiss

# ---------------- Config ----------------
AUDIO_EXTS = [".wav", ".flac", ".aiff", ".mp3"]
SEGMENT_SECONDS = 20
BATCH_TRACKS = 3
FAISS_INDEX_FILE = "index.faiss"
META_FILE = "index_meta.pkl"

# ---------------- Shared state ----------------
filenames, hashes, embeddings_all, faiss_index = [], [], [], None
indexing_state = {"running": False, "paused": False}
model = None

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
        emb_list = [np.array(e, dtype=np.float32) for e in meta.get("embeddings", [])]
        return list(meta.get("files", [])), list(meta.get("hashes", [])), emb_list, index
    return [], [], [], None

def create_faiss_index(dim):
    return faiss.IndexFlatIP(dim)

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

def extract_segment(src_path, start_s, duration_s, out_path):
    try:
        subprocess.run([
            "ffmpeg", "-y", "-ss", str(start_s), "-t", str(duration_s),
            "-i", str(src_path), "-ar", "16000", "-ac", "1", "-f", "wav", str(out_path)
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except Exception:
        return False

def make_segments(path):
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
        if not extract_segment(path, s, SEGMENT_SECONDS, tmp):
            for t in tmp_files:
                os.remove(t)
            return []
        tmp_files.append(tmp)
    return tmp_files

def cleanup(files):
    for f in files:
        try: os.remove(f)
        except: pass

def init_model():
    global model
    if model is None:
        model = CLAP_Module(enable_fusion=False)
        model.load_ckpt()
    return model

# ---------------- Indexing ----------------
def index_folder(folder):
    global filenames, hashes, embeddings_all, faiss_index
    model = init_model()
    folder = Path(folder)
    if not folder.exists():
        return "❌ Folder not found.", []

    files = [f for ext in AUDIO_EXTS for f in sorted(folder.rglob(f"*{ext}"))]
    if not files:
        return "⚠️ No audio files found.", []

    total = len(files)
    processed = skipped = failed = 0

    for i, f in enumerate(files):
        with open(f, "rb") as fh:
            f_hash = hashlib.sha1(fh.read()).hexdigest()
        if f_hash in hashes or str(f) in filenames:
            skipped += 1
            continue

        segs = make_segments(str(f))
        if not segs:
            failed += 1
            continue

        try:
            embs = model.get_audio_embedding_from_filelist(segs, use_tensor=False)
            vecs = [np.array(e, dtype=np.float32) for e in embs]
            for v in vecs: v /= (np.linalg.norm(v) + 1e-9)
            avg = np.mean(vecs, axis=0)
            avg /= (np.linalg.norm(avg) + 1e-9)
            if faiss_index is None:
                faiss_index = create_faiss_index(avg.shape[0])
            faiss_index.add(np.expand_dims(avg.astype(np.float32), axis=0))
            filenames.append(str(f))
            hashes.append(f_hash)
            embeddings_all.append(avg)
            processed += 1
        except Exception as e:
            print("Embedding failed:", e)
            failed += 1
        finally:
            cleanup(segs)

    save_meta_and_index(filenames, hashes, embeddings_all, faiss_index)
    # Return dropdown choices (file names) after indexing
    return f"✅ Indexed {processed} files, skipped {skipped}, failed {failed}", [Path(p).name for p in filenames]

# ---------------- Query ----------------
def query_audio(upload_file=None, selected_track=None, top_k=5):
    global faiss_index
    if faiss_index is None or not filenames:
        return []

    model = init_model()
    query_vec = None

    # External file query
    if upload_file is not None:
        segs = make_segments(str(upload_file.name))
        if segs:
            emb_list = model.get_audio_embedding_from_filelist(segs, use_tensor=False)
            cleanup(segs)
            vecs = [np.array(e, dtype=np.float32)/(np.linalg.norm(e)+1e-9) for e in emb_list]
            query_vec = np.mean(vecs, axis=0)
        else:
            emb = model.get_audio_embedding_from_filelist([str(upload_file.name)], use_tensor=False)[0]
            query_vec = np.array(emb, dtype=np.float32)/(np.linalg.norm(emb)+1e-9)
    # Indexed track query
    elif selected_track is not None:
        idx = next((i for i,p in enumerate(filenames) if Path(p).name==selected_track), None)
        if idx is not None:
            try:
                vec = np.empty((faiss_index.d,), dtype='float32')
                faiss_index.reconstruct(idx, vec)
                query_vec = vec/(np.linalg.norm(vec)+1e-9)
            except:
                query_vec = embeddings_all[idx]/(np.linalg.norm(embeddings_all[idx])+1e-9)

    if query_vec is None:
        return []

    D, I = faiss_index.search(np.expand_dims(query_vec.astype(np.float32), axis=0), top_k)
    results = []
    for sim, idx in zip(D[0], I[0]):
        if idx < 0 or idx >= len(filenames): continue
        path = filenames[idx]
        results.append([Path(path).name, f"{sim*100:.1f}%", path])
    return results

# ---------------- Clear index ----------------
def clear_index():
    global filenames, hashes, embeddings_all, faiss_index
    filenames, hashes, embeddings_all, faiss_index = [], [], [], None
    Path(FAISS_INDEX_FILE).unlink(missing_ok=True)
    Path(META_FILE).unlink(missing_ok=True)
    return "🧹 Index cleared", []

# ---------------- Load existing index ----------------
filenames, hashes, embeddings_all, faiss_index = load_meta_and_index()
existing_choices = [Path(p).name for p in filenames]

# ---------------- Gradio UI ----------------
with gr.Blocks(title="Audio Similarity (CLAP + FAISS)") as demo:
    gr.Markdown("# 🎧 Audio Similarity App")

    with gr.Tab("Index & Query"):
        gr.Markdown("### Step 1: Index a folder of audio tracks")
        folder_in = gr.Textbox(label="Folder Path")
        index_btn = gr.Button("Start Indexing")
        index_out = gr.Textbox(label="Index Status")
        indexed_dropdown = gr.Dropdown(choices=existing_choices, label="Select Indexed Track", interactive=True)

        gr.Markdown("### Step 2: Query audio")
        file_in = gr.File(label="Upload Audio File")
        topk = gr.Slider(1, 10, value=5, step=1, label="Top K Results")
        query_btn = gr.Button("Query Similarity")
        clear_btn = gr.Button("Clear Index")
        results = gr.Dataframe(headers=["Track", "Similarity", "Full Path"], datatype=["str","str","str"])

    # ---------------- Callbacks ----------------
    def do_index(folder):
        status, names = index_folder(folder)
        return status, gr.Dropdown.update(choices=names)

    index_btn.click(do_index, inputs=folder_in, outputs=[index_out, indexed_dropdown])

    def do_query(upload_file, selected_track, top_k):
        return query_audio(upload_file, selected_track, top_k)

    query_btn.click(do_query, inputs=[file_in, indexed_dropdown, topk], outputs=results)
    clear_btn.click(clear_index, outputs=[index_out, indexed_dropdown])

# ---------------- Launch ----------------
if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
