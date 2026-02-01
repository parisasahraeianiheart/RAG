from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
import os
import json
import numpy as np

import faiss
from sentence_transformers import SentenceTransformer


# -----------------------------
# Data structures
# -----------------------------
@dataclass
class Chunk:
    doc_id: str
    chunk_id: int
    text: str
    meta: Dict[str, Any]


# -----------------------------
# Chunking
# -----------------------------
def chunk_text(
    text: str,
    doc_id: str,
    chunk_size: int = 800,
    overlap: int = 120
) -> List[Chunk]:
    """
    Simple character-based chunking. For production, chunk by tokens/paragraphs.
    """
    text = text.strip()
    if not text:
        return []

    chunks: List[Chunk] = []
    start = 0
    cid = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunk = text[start:end]
        chunks.append(Chunk(doc_id=doc_id, chunk_id=cid, text=chunk, meta={"start": start, "end": end}))
        cid += 1
        start = end - overlap  # overlap
        if start < 0:
            start = 0
        if end == len(text):
            break
    return chunks


# -----------------------------
# Vector Store (FAISS)
# -----------------------------
class FaissVectorStore:
    """
    Stores embeddings in FAISS and metadata in a Python list (persistable).
    """
    def __init__(self, embedder: SentenceTransformer):
        self.embedder = embedder
        self.index: Optional[faiss.Index] = None
        self.chunks: List[Chunk] = []
        self.emb_dim: Optional[int] = None

    def _ensure_index(self, dim: int) -> None:
        if self.index is None:
            # Inner Product index (use with normalized vectors => cosine similarity)
            self.index = faiss.IndexFlatIP(dim)
            self.emb_dim = dim

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
        return vectors / norms

    def add_chunks(self, chunks: List[Chunk], batch_size: int = 64) -> None:
        texts = [c.text for c in chunks]
        if not texts:
            return

        # Embed
        embs = self.embedder.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True  # ensures cosine sim when using IP
        ).astype(np.float32)

        self._ensure_index(embs.shape[1])
        assert self.index is not None

        self.index.add(embs)
        self.chunks.extend(chunks)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Chunk, float]]:
        if self.index is None or not self.chunks:
            return []

        q_emb = self.embedder.encode(
            [query],
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype(np.float32)

        scores, ids = self.index.search(q_emb, top_k)
        results: List[Tuple[Chunk, float]] = []
        for idx, score in zip(ids[0], scores[0]):
            if idx == -1:
                continue
            results.append((self.chunks[int(idx)], float(score)))
        return results

    def save(self, dir_path: str) -> None:
        os.makedirs(dir_path, exist_ok=True)
        if self.index is None:
            raise ValueError("No FAISS index to save.")

        faiss.write_index(self.index, os.path.join(dir_path, "index.faiss"))
        payload = [
            {
                "doc_id": c.doc_id,
                "chunk_id": c.chunk_id,
                "text": c.text,
                "meta": c.meta,
            }
            for c in self.chunks
        ]
        with open(os.path.join(dir_path, "chunks.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        with open(os.path.join(dir_path, "meta.json"), "w", encoding="utf-8") as f:
            json.dump({"emb_dim": self.emb_dim}, f, indent=2)

    def load(self, dir_path: str) -> None:
        idx_path = os.path.join(dir_path, "index.faiss")
        chunks_path = os.path.join(dir_path, "chunks.json")
        meta_path = os.path.join(dir_path, "meta.json")

        if not (os.path.exists(idx_path) and os.path.exists(chunks_path) and os.path.exists(meta_path)):
            raise FileNotFoundError("Missing index.faiss / chunks.json / meta.json")

        self.index = faiss.read_index(idx_path)

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.emb_dim = meta.get("emb_dim")

        with open(chunks_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        self.chunks = [
            Chunk(doc_id=p["doc_id"], chunk_id=p["chunk_id"], text=p["text"], meta=p.get("meta", {}))
            for p in payload
        ]


# -----------------------------
# RAG: build prompt + generate
# -----------------------------
def build_rag_prompt(question: str, retrieved: List[Tuple[Chunk, float]]) -> str:
    """
    This is the prompt you would send to an LLM.
    Even if you don't call an LLM, it's useful to see the structure.
    """
    context_lines = []
    for chunk, score in retrieved:
        cite = f"[{chunk.doc_id}:{chunk.chunk_id}]"
        context_lines.append(f"{cite} {chunk.text}")

    context = "\n\n".join(context_lines)

    prompt = f"""
You are a helpful assistant. Answer the question using ONLY the provided context.
If the context does not contain the answer, say: "I don't know based on the provided context."

Question:
{question}

Context:
{context}

Requirements:
- Cite sources using [doc_id:chunk_id] after each claim.
- Be concise and factual.

Answer:
""".strip()

    return prompt


def offline_answer_composer(question: str, retrieved: List[Tuple[Chunk, float]]) -> str:
    """
    A simple offline 'answer' that stitches evidence (no LLM).
    Replace this with an LLM call in production.
    """
    if not retrieved:
        return "I don't know based on the provided context."

    bullets = []
    for chunk, score in retrieved[:3]:
        bullets.append(f"- [{chunk.doc_id}:{chunk.chunk_id}] {chunk.text.strip()}")

    return (
        f"Question: {question}\n\n"
        f"Top evidence:\n" + "\n".join(bullets) + "\n\n"
        "Next step: plug the prompt into your LLM of choice (OpenAI/Claude/Gemini) to generate a final grounded answer."
    )


# -----------------------------
# Example: Build index from documents
# -----------------------------
def build_demo_corpus() -> List[Dict[str, str]]:
    """
    Replace this with your real docs:
    - markdown files
    - PDFs converted to text
    - runbooks
    - security KB
    - incident reports
    """
    return [
        {
            "doc_id": "runbook_login_spike",
            "text": (
                "If suspicious logins spike, check VPN usage, IP reputation, and failed_login rates. "
                "Common root causes: credential stuffing, bot traffic, or a new client rollout. "
                "Mitigation: rate limit, step-up authentication (MFA), block known bad IPs, and alert the SOC."
            )
        },
        {
            "doc_id": "ip_reputation_policy",
            "text": (
                "IP reputation provides a risk score from 0 to 1 and a known-bad flag. "
                "High-risk IPs (>=0.7) or known-bad IPs should trigger elevated scrutiny. "
                "For investigations, hybrid retrieval (keyword + semantic) works best for matching IPs and behaviors."
            )
        },
        {
            "doc_id": "metrics_definitions",
            "text": (
                "suspicious_rate is the fraction of events flagged as suspicious (label=1). "
                "success_rate is the fraction of successful logins (success=1). "
                "Aggregate daily for monitoring and segment by risk tiers (low/medium/high)."
            )
        },
    ]


def main():
    # 1) Choose an embedding model
    # Good default: all-MiniLM-L6-v2 (fast + decent)
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # 2) Build chunks
    docs = build_demo_corpus()
    all_chunks: List[Chunk] = []
    for d in docs:
        all_chunks.extend(chunk_text(d["text"], doc_id=d["doc_id"], chunk_size=450, overlap=80))

    # 3) Build FAISS index
    store = FaissVectorStore(embedder)
    store.add_chunks(all_chunks)

    # (Optional) persist index
    store.save("rag_index")

    # 4) Query
    question = "Why would suspicious logins spike and what mitigations should we apply?"
    retrieved = store.search(question, top_k=5)

    # 5) Show retrieval
    print("\n=== Retrieved Chunks ===")
    for c, s in retrieved:
        print(f"- score={s:.3f}  [{c.doc_id}:{c.chunk_id}] {c.text[:120]}...")

    # 6) Build prompt (send to any LLM if you want)
    prompt = build_rag_prompt(question, retrieved)
    print("\n=== RAG Prompt (for an LLM) ===\n")
    print(prompt)

    # 7) Offline answer (no LLM)
    print("\n=== Offline Answer Composer (no LLM) ===\n")
    print(offline_answer_composer(question, retrieved))


if __name__ == "__main__":
    main()
