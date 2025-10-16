from pathlib import Path
import numpy as np
import pickle
from laion_clap import CLAP_Module

def index_folder_with_clap(folder_path, save_file="index_embeddings.pkl"):
    folder = Path(folder_path)
    if not folder.exists():
        print(f"Folder not found: {folder}")
        return

    # Initialize CLAP
    model = CLAP_Module(enable_fusion=False)
    model.load_ckpt()

    embeddings = []
    file_paths = []

    # Loop through audio files with multiple extensions
    audio_extensions = [".wav", ".flac", ".aiff", ".mp3"]
    for ext in audio_extensions:
        for f in folder.glob(f"*{ext}"):
            file_paths.append(str(f))
            vec = model.get_audio_embedding_from_filelist([str(f)], use_tensor=False)[0]
            vec = vec / np.linalg.norm(vec)  # Normalize to unit length
            embeddings.append(vec)

            print(f"Indexed: {f.name}")

    # Save index to disk
    with open(save_file, "wb") as f:
        pickle.dump({"files": file_paths, "embeddings": embeddings}, f)

    print(f"Index saved to {save_file}")
