import hashlib

def file_sha1(path: Path, block_size=65536):
    """Compute SHA1 hash of a file."""
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(block_size), b""):
            h.update(block)
    return h.hexdigest()
