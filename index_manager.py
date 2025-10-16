#!/usr/bin/env python3
"""
index_manager.py
Tiny FAISS index manager.
Creates an index, adds vectors, saves and loads it.
"""

import faiss
import numpy as np
from pathlib import Path
from features import extract_mfcc

INDEX_FILE = Path("faiss.index")

def create_index(dim: int):
    """
    Create a FAISS index for vectors of size dim.
    We use L2 (Euclidean) distance for similarity.
    """
    index = faiss.IndexFlatL2(dim)
    return index

def save_index(index, path: Path):
    faiss.write_index(index, str(path))
    print(f"Index saved to {path}")

def load_index(path: Path):
    if path.exists():
        index = faiss.read_index(str(path))
        print(f"Index loaded from {path}")
        return index
    else:
        print("No existing index found. Creating new one.")
        return None

def main():
    # 1. Extract MFCC from test tone
    vec = extract_mfcc("data/audio/test_tone.wav")
    if vec is None:
        print("MFCC extraction failed. Run features.py first.")
        return

    vec = vec.astype("float32")  # FAISS requires float32

    # 2. Create or load index
    dim = len(vec)
    index = load_index(INDEX_FILE)
    if index is None:
        index = create_index(dim)

    # 3. Add vector to index
    index.add(np.expand_dims(vec, axis=0))
    print(f"Index contains {index.ntotal} vector(s)")

    # 4. Save index
    save_index(index, INDEX_FILE)

if __name__ == "__main__":
    main()

