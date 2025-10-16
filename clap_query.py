import pickle
import numpy as np
from laion_clap import CLAP_Module

def query_file_with_clap(query_path, index_file="index_embeddings.pkl", top_k=5):
    # Load index
    with open(index_file, "rb") as f:
        data = pickle.load(f)
    file_paths = data["files"]
    embeddings = data["embeddings"]

    # Initialize CLAP
    model = CLAP_Module(enable_fusion=False)
    model.load_ckpt()

    # Generate embedding for query file
    query_vec = model.get_audio_embedding_from_filelist([query_path], use_tensor=False)[0]
    query_vec = query_vec / np.linalg.norm(query_vec)  # Normalize

    # Compute cosine similarity
    similarities = [float(np.dot(query_vec, vec)) for vec in embeddings]
    similarities_percentage = [s * 100 for s in similarities]

    # Sort results
    results = sorted(zip(file_paths, similarities_percentage), key=lambda x: x[1], reverse=True)

    # Return top_k results
    return results[:top_k]
