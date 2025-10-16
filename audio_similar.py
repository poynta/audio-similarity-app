#!/usr/bin/env python3
"""
audio_similar.py
Terminal CLI for scanning, indexing, querying, and clearing audio similarity data.

Features:
- Metadata tracking (meta.json) with SHA1 hashes for incremental indexing
- Progress bars with Rich
- FAISS index persistence
- Color-coded query results
- Interactive folder prompt for indexing
- Safe index clearing
"""

from pathlib import Path
import argparse
import json
import hashlib
import numpy as np
from rich.progress import track
from rich.table import Table
from rich.console import Console
from rich.text import Text
from scanner import scan_folder
from features import extract_mfcc
from index_manager import create_index, load_index, save_index, INDEX_FILE

META_FILE = Path("meta.json")
console = Console()


# ---------------- Metadata Helpers ----------------

def load_meta():
    if META_FILE.exists():
        with open(META_FILE) as f:
            return json.load(f)
    return {"files": []}


def save_meta(meta):
    with open(META_FILE, "w") as f:
        json.dump(meta, f, indent=2)


# ---------------- File Hash Helper ----------------

def file_sha1(path: Path, block_size=65536):
    """Compute SHA1 hash of a file."""
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(block_size), b""):
            h.update(block)
    return h.hexdigest()


# ---------------- Indexing ----------------

def index_folder(folder: Path):
    files = scan_folder(folder)
    if not files:
        console.print(f"[yellow]No audio files found in {folder}[/yellow]")
        return

    # Load metadata and convert to dict {path: hash}
    meta = load_meta()
    indexed_files = {entry["path"]: entry["hash"] for entry in meta.get("files", [])}

    # Determine MFCC dimension from first unindexed/changed file
    for f in files:
        f_str = str(f)
        f_hash = file_sha1(f)
        if f_str not in indexed_files or indexed_files[f_str] != f_hash:
            first_vec = extract_mfcc(f_str)
            if first_vec is not None:
                dim = len(first_vec)
                break
    else:
        console.print("[green]All files already indexed and unchanged.[/green]")
        return

    index = load_index(INDEX_FILE)
    if index is None:
        index = create_index(dim)

    processed = 0
    skipped = 0
    failed = 0

    for f in track(files, description="Indexing audio..."):
        f_str = str(f)
        f_hash = file_sha1(f)

        # Skip if already indexed and unchanged
        if f_str in indexed_files and indexed_files[f_str] == f_hash:
            skipped += 1
            continue

        vec = extract_mfcc(f_str)
        if vec is not None:
            vec = vec.astype("float32")
            index.add(np.expand_dims(vec, axis=0))
            indexed_files[f_str] = f_hash
            processed += 1
        else:
            failed += 1

    # Save FAISS index and updated metadata
    save_index(index, INDEX_FILE)
    meta["files"] = [{"path": path, "hash": h} for path, h in indexed_files.items()]
    save_meta(meta)

    console.print(f"[green]Indexing complete.[/green] "
                  f"Processed: {processed}, Skipped: {skipped}, Failed: {failed}, "
                  f"Total vectors: {index.ntotal}")


# ---------------- Querying ----------------

def query_file(file_path: Path, k: int = 3):
    if not file_path.exists():
        console.print(f"[red]File not found:[/red] {file_path}")
        return

    vec = extract_mfcc(str(file_path))
    if vec is None:
        console.print("[red]Failed to extract MFCC.[/red]")
        return
    vec = vec.astype("float32")
    vec = np.expand_dims(vec, axis=0)

    index = load_index(INDEX_FILE)
    if index is None:
        console.print("[red]No index found. Run indexing first.[/red]")
        return

    distances, indices = index.search(vec, k)

    meta = load_meta()
    files_list = [entry["path"] for entry in meta.get("files", [])]

    table = Table(title=f"Top {k} matches for {file_path.name}")
    table.add_column("Rank", justify="center")
    table.add_column("File", justify="left")
    table.add_column("Distance", justify="center")

    for rank, (idx, dist) in enumerate(zip(indices[0], distances[0]), 1):
        if idx < 0 or idx >= len(files_list):
            file_name = "[red]Unknown[/red]"
        else:
            file_name = Path(files_list[idx]).name
        # Color-coded distance
        if dist < 0.05:
            color = "green"
        elif dist < 0.2:
            color = "yellow"
        else:
            color = "red"
        dist_text = Text(f"{dist:.6f}", style=color)
        table.add_row(str(rank), file_name, dist_text)

    console.print(table)


# ---------------- Clear ----------------

def clear_index():
    if INDEX_FILE.exists():
        INDEX_FILE.unlink()
        console.print(f"[yellow]Deleted FAISS index:[/yellow] {INDEX_FILE}")
    if META_FILE.exists():
        META_FILE.unlink()
        console.print(f"[yellow]Deleted metadata:[/yellow] {META_FILE}")
    console.print("[green]Index and metadata cleared. You can now run 'index' to rebuild.[/green]")


# ---------------- CLI ----------------

def main():
    parser = argparse.ArgumentParser(description="Audio similarity CLI (minimal).")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Index command
    parser_index = subparsers.add_parser("index", help="Scan folder and build index")
    parser_index.add_argument(
        "folder",
        nargs="?",          # optional
        default=None,       # prompt if not provided
        help="Folder to scan and index"
    )

    # Query command
    parser_query = subparsers.add_parser("query", help="Query an audio file against the index")
    parser_query.add_argument("file", help="Audio file to query")
    parser_query.add_argument("-k", type=int, default=3, help="Number of nearest neighbors to return")

    # Clear command
    parser_clear = subparsers.add_parser("clear", help="Delete FAISS index and metadata to start fresh")

    args = parser.parse_args()

    if args.command == "index":
        folder_path = args.folder
        if folder_path is None:
            folder_input = console.input("[cyan]Enter folder to index:[/cyan] ").strip()
            folder = Path(folder_input)
        else:
            folder = Path(folder_path)

        if not folder.exists():
            console.print(f"[red]Folder not found:[/red] {folder}")
        else:
            folder.mkdir(parents=True, exist_ok=True)
            index_folder(folder)

    elif args.command == "query":
        query_file(Path(args.file), k=args.k)

    elif args.command == "clear":
        clear_index()


if __name__ == "__main__":
    main()
