"""Build the FAISS index from PDFs / .txt transcripts in data/.

File naming convention: Company_Year.pdf   (e.g. TCS_2024.pdf, Infosys_2024.pdf)

Usage:
    python -m src.ingest --data data --out index
"""
import argparse
import json
import re
from pathlib import Path

import faiss
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "BAAI/bge-small-en-v1.5"


def parse_name(path: Path):
    parts = path.stem.split("_")
    return parts[0], (parts[1] if len(parts) > 1 else "NA")


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def chunk_words(text: str, size: int = 300, overlap: int = 60):
    """Split text into overlapping word windows (size words, overlap words shared)."""
    words = text.split()
    if len(words) < 25:  # skip near-empty pages (covers, blank pages)
        return []
    chunks, step = [], size - overlap
    for start in range(0, len(words), step):
        piece = words[start:start + size]
        if len(piece) < 25 and chunks:
            break
        chunks.append(" ".join(piece))
        if start + size >= len(words):
            break
    return chunks


def read_pages(path: Path):
    """Yield (page_number, text). Chunks never cross page boundaries so citations stay exact."""
    if path.suffix.lower() == ".pdf":
        with fitz.open(path) as doc:
            for i, page in enumerate(doc, start=1):
                yield i, clean(page.get_text())
    else:  # .txt transcript: treat every 400 words as a pseudo-page
        words = path.read_text(encoding="utf-8", errors="ignore").split()
        for i in range(0, len(words), 400):
            yield i // 400 + 1, " ".join(words[i:i + 400])


def build(data_dir: str, out_dir: str, size: int, overlap: int):
    files = sorted(p for p in Path(data_dir).glob("*") if p.suffix.lower() in {".pdf", ".txt"})
    if not files:
        raise SystemExit(f"No PDF/TXT files found in {data_dir}/. Add files named Company_Year.pdf")

    chunks = []
    for f in files:
        company, year = parse_name(f)
        before = len(chunks)
        for page, text in read_pages(f):
            for c in chunk_words(text, size, overlap):
                chunks.append({"id": len(chunks), "company": company, "year": year, "page": page, "text": c})
        print(f"{f.name}: {len(chunks) - before} chunks")

    print(f"Embedding {len(chunks)} chunks with {EMBED_MODEL} ...")
    model = SentenceTransformer(EMBED_MODEL)
    emb = model.encode(
        [c["text"] for c in chunks], batch_size=64, normalize_embeddings=True, show_progress_bar=True
    ).astype("float32")

    index = faiss.IndexFlatIP(emb.shape[1])  # inner product == cosine (vectors are normalized)
    index.add(emb)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out / "index.faiss"))
    (out / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False))
    print(f"Saved index ({index.ntotal} vectors, dim={emb.shape[1]}) to {out}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="index")
    ap.add_argument("--chunk-words", type=int, default=300)
    ap.add_argument("--overlap", type=int, default=60)
    a = ap.parse_args()
    build(a.data, a.out, a.chunk_words, a.overlap)
