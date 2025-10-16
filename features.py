#!/usr/bin/env python3
"""
features.py
Extracts MFCC (Mel-Frequency Cepstral Coefficients) from an audio file.
Used as the 'ears' of the similarity engine.
"""

import numpy as np
import librosa

def extract_mfcc(path: str, n_mfcc: int = 20):
    """
    Load an audio file, compute MFCCs, and average across time.

    Args:
        path (str): Path to an audio file (wav, mp3, etc.)
        n_mfcc (int): Number of MFCC coefficients to compute (default 20)

    Returns:
        np.ndarray: A 1D numpy array (length = n_mfcc) representing the timbre fingerprint.
    """
    try:
        # Load the audio file
        y, sr = librosa.load(path, sr=None, mono=True)

        # Compute MFCCs (time x n_mfcc)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

        # Mean across time axis to get a compact vector
        mfcc_mean = np.mean(mfcc, axis=1)

        # Normalize (L2 norm) to make all vectors comparable in scale
        norm = np.linalg.norm(mfcc_mean)
        if norm > 0:
            mfcc_mean = mfcc_mean / norm

        return mfcc_mean

    except Exception as e:
        print(f"[Error] Could not process {path}: {e}")
        return None

if __name__ == "__main__":
    # Manual test
    import sys
    from pathlib import Path

    test_path = Path("data/audio/test_tone.wav")
    if not test_path.exists():
        print("No test file found. Run scanner.py --test first.")
        sys.exit(1)

    print(f"Extracting MFCCs from {test_path} ...")
    vec = extract_mfcc(str(test_path))
    if vec is not None:
        print("MFCC vector length:", len(vec))
        print("First few values:", np.round(vec[:5], 4))

