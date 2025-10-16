#!/usr/bin/env python3
"""
scanner.py
Tiny folder scanner for audio_sim project.

Usage:
  python scanner.py            # scan data/audio and print files
  python scanner.py --test     # create a tiny test tone if folder empty, then scan
  python scanner.py path/to/folder
"""

from pathlib import Path
import os
import argparse

# optional: used only when --test to create a tiny WAV
try:
    import numpy as np
    import soundfile as sf
except Exception:
    np = None
    sf = None

SUPPORTED_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aiff", ".aif"}

def create_test_tone(path: Path, duration=1.0, sr=22050, freq=440.0):
    if np is None or sf is None:
        raise RuntimeError("numpy and soundfile required to create test tone. pip install soundfile numpy")
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    tone = 0.2 * np.sin(2 * np.pi * freq * t).astype(np.float32)
    sf.write(str(path), tone, sr)
    print(f"Created test tone: {path}")

def scan_folder(folder: Path):
    files = []
    for root, _, filenames in os.walk(folder):
        for fn in filenames:
            if Path(fn).suffix.lower() in SUPPORTED_EXTS:
                files.append(Path(root) / fn)
    files.sort()
    return files

def main():
    p = argparse.ArgumentParser(description="Scan a folder for audio files (tiny step).")
    p.add_argument("folder", nargs="?", default="data/audio", help="Folder to scan (default: data/audio).")
    p.add_argument("--test", action="store_true", help="If folder empty, create a small test WAV first.")
    args = p.parse_args()

    folder = Path(args.folder)
    folder.mkdir(parents=True, exist_ok=True)

    files = scan_folder(folder)
    if args.test and len(files) == 0:
        # create a tiny test tone so you can see output even if you have no audio yet
        test_path = folder / "test_tone.wav"
        try:
            create_test_tone(test_path)
        except Exception as e:
            print("Could not create test tone:", e)
            print("Install dependencies with: pip install numpy soundfile")
            return
        files = scan_folder(folder)

    print(f"\nFound {len(files)} audio file(s) in: {folder.resolve()}\n")
    for f in files:
        print(" -", f.name)

if __name__ == "__main__":
    main()

