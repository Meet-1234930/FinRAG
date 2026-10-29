"""Dense retriever (bge-small + FAISS) with optional cross-encoder reranking."""
import json
from pathlib import Path

import faiss
from sentence_transformers import CrossEncoder, SentenceTransformer

from .ingest import EMBED_MODEL

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "  # bge convention
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Retriever:
    def __init__(self, index_dir: str = "index"):
        d = Path(index_dir)
        self.index = faiss.read_index(str(d / "index.faiss"))
        self.chunks = json.loads((d / "chunks.json").read_text())
        self.embedder = SentenceTransformer(EMBED_MODEL)
        self._reranker = None
        self.companies = sorted({c["company"] for c in self.chunks})

    def detect_company(self, query: str):
        """If the question names exactly one indexed company, return it (used as a filter)."""
        found = [c for c in self.companies if c.lower() in query.lower()]
        return found[0] if len(found) == 1 else None

    def search(self, query: str, top_k: int = 5, company: str = None, rerank: bool = False, fetch_k: int = 30):
        q = self.embedder.encode([QUERY_PREFIX + query], normalize_embeddings=True).astype("float32")
        scores, ids = self.index.search(q, min(fetch_k * (3 if company else 1), self.index.ntotal))

        cands = []
        for s, i in zip(scores[0], ids[0]):
            if i < 0:
                continue
            c = self.chunks[i]
            if company and c["company"].lower() != company.lower():
                continue
            cands.append({**c, "score": float(s)})

        cands = cands[:fetch_k]
        if rerank and cands:
            if self._reranker is None:
                self._reranker = CrossEncoder(RERANK_MODEL)
            rs = self._reranker.predict([(query, c["text"]) for c in cands])
            for c, r in zip(cands, rs):
                c["score"] = float(r)
            cands.sort(key=lambda c: c["score"], reverse=True)
        return cands[:top_k]
