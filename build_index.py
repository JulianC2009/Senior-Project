import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536

# Chunking
CHUNK_SIZE = 900
OVERLAP = 120
MAX_CHUNKS_TOTAL = 6000        
MAX_CHARS_PER_DOC = 120_000     
MAX_FILES = 80               

# Fixed width arrays
MAX_CHUNK_CHARS = 900
MAX_SOURCE_CHARS = 260

# Embedding batching
BATCH_SIZE = 64

# Artifacts paths
ARTIFACTS_DIR = Path("artifacts")
CHUNK_TEXTS_PATH = ARTIFACTS_DIR / "chunk_texts.npy"
CHUNK_SOURCES_PATH = ARTIFACTS_DIR / "chunk_sources.npy"
EMBEDDINGS_PATH = ARTIFACTS_DIR / "embeddings.npy"
EMBED_TMP_PATH = ARTIFACTS_DIR / "embeddings_tmp.dat"


def _read_txt(path: Path, max_chars: int) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")[:max_chars]
    except Exception:
        return ""


def _chunk_text(text: str, chunk_size: int = 900, overlap: int = 120):
    text = (text or "").strip()
    if not text:
        return []

    if overlap >= chunk_size:
        overlap = max(0, chunk_size // 4)

    chunks = []
    start = 0
    n = len(text)

    while start < n:
        end = min(n, start + chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Stop if end hits to prevent an infinite loop
        if end >= n:
            break

        next_start = end - overlap
        if next_start <= start:
            next_start = end

        start = next_start

    return chunks


def collect_chunks(kb_root: str = "knowledge_base") -> Tuple[np.ndarray, np.ndarray]:
    kb = Path(kb_root)
    if not kb.exists():
        raise RuntimeError(f"knowledge_base folder not found: {kb_root}")

    txt_files = sorted([p for p in kb.rglob("*.txt") if p.is_file()], key=lambda p: str(p).lower())
    txt_files = txt_files[:MAX_FILES]

    print(f"Found {len(txt_files)} .txt files (capped to MAX_FILES={MAX_FILES})")

    chunk_texts: List[str] = []
    chunk_sources: List[str] = []

    for p in txt_files:
        raw = _read_txt(p, MAX_CHARS_PER_DOC)
        if not raw.strip():
            continue

        chunks = _chunk_text(raw)
        if not chunks:
            continue

        for c in chunks:
            if len(chunk_texts) >= MAX_CHUNKS_TOTAL:
                break
            chunk_texts.append(c)
            chunk_sources.append(str(p)[:MAX_SOURCE_CHARS])

        print(f"Chunked: {p.name} -> {len(chunks)} chunks (total {len(chunk_texts)})")

        if len(chunk_texts) >= MAX_CHUNKS_TOTAL:
            print("Reached MAX_CHUNKS_TOTAL cap; stopping chunk collection.")
            break

    if not chunk_texts:
        raise RuntimeError("No chunks created. Did you run build_text_cache.py to generate .txt files?")

    # Convert to fixed width numpy arrays 
    chunk_texts_np = np.array(chunk_texts, dtype=f"<U{MAX_CHUNK_CHARS}")
    chunk_sources_np = np.array(chunk_sources, dtype=f"<U{MAX_SOURCE_CHARS}")
    return chunk_texts_np, chunk_sources_np


def main():
    print("build_index.py starting...")

    # Ensure artifacts folder exists
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    chunk_texts_np, chunk_sources_np = collect_chunks("knowledge_base")
    n = chunk_texts_np.shape[0]
    print(f"\nTotal chunks to embed: {n}")

    # Save chunk arrays now
    np.save(str(CHUNK_TEXTS_PATH), chunk_texts_np)
    np.save(str(CHUNK_SOURCES_PATH), chunk_sources_np)

    # Create a disk backed embeddings array
    emb_mem = np.memmap(str(EMBED_TMP_PATH), dtype="float32", mode="w+", shape=(n, EMBED_DIM))

    # Embed in batches and write directly into memmap
    for start in range(0, n, BATCH_SIZE):
        end = min(n, start + BATCH_SIZE)
        batch = [str(x) for x in chunk_texts_np[start:end]]

        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        batch_emb = np.array([d.embedding for d in resp.data], dtype="float32")

        emb_mem[start:end, :] = batch_emb

        if (start // BATCH_SIZE) % 10 == 0:
            print(f"Embedded {end}/{n} chunks...")

    emb_mem.flush()

    # Normalize embeddings for cosine similarity
    emb_np = np.array(emb_mem, dtype="float32")
    norms = np.linalg.norm(emb_np, axis=1, keepdims=True) + 1e-12
    emb_np = emb_np / norms

    # Save final normalized embeddings.npy
    np.save(str(EMBEDDINGS_PATH), emb_np)

    try:
        os.remove(str(EMBED_TMP_PATH))
    except Exception:
        pass

    print("\nSaved:")
    print(f"- {EMBEDDINGS_PATH}")
    print(f"- {CHUNK_TEXTS_PATH}")
    print(f"- {CHUNK_SOURCES_PATH}")
    print("\nDone.")


if __name__ == "__main__":
    main()
