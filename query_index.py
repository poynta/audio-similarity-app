#!/usr/bin/env python3
"""
query_index.py
Tiny FAISS query example.
Finds nearest neighbors for a given audio file in the index.
"""

import numpy as np
from pathlib import Path
from features import extract_mfcc
from index_manager import load_index, INDEX_FILE

def query_audio(file_path: str, k: int = 1):
    """
    Query FAISS index for most similar tracks.
    """
    # 1. Extract MFCC
    vec = extract_mfcc(file_path)
    if vec is None:
        print("MFCC extraction failed.")
        return

    vec = vec.astype("float32")
    vec = np.expand_dims(vec, axis=0)  # shape (1, dim)

    # 2. Load index
    index = load_index(INDEX_FILE)
    if index is None:
        print("No index to query.")
        return

    # 3. Query
    distances, indices = index.search(vec, k)

    # 4. Print results
    for rank, (idx, dist) in enumerate(zip(indices[0], distances[0]), 1):
        print(f"{rank}. Vector ID {idx} — Distance: {dist:.6f}")

if __name__ == "__main__":
    test_file = "data/audio/test_tone.wav"
    if not Path(test_file).exists():
        print("No test file found. Run scanner.py --test first.")
    else:
        print(f"Querying index for {test_file} ...")
        query_audio(test_file)

