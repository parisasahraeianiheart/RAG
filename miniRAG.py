# RAG in ~40 lines: retrieve chunks -> feed them to an LLM -> answer

from openai import OpenAI

client = OpenAI()

def embed(text: str) -> list[float]:
    # Create embeddings for semantic search
    return client.embeddings.create(
        model="text-embedding-3-large",
        input=text
    ).data[0].embedding

def retrieve_top_k(query: str, chunks: list[dict], k: int = 5) -> list[dict]:
    """
    chunks = [{"id": "doc1:0", "text": "...", "embedding": [...]}, ...]
    We'll pick top-k by cosine similarity.
    """
    import numpy as np

    q = np.array(embed(query), dtype=float)
    q = q / (np.linalg.norm(q) + 1e-12)

    scored = []
    for c in chunks:
        v = np.array(c["embedding"], dtype=float)
        v = v / (np.linalg.norm(v) + 1e-12)
        score = float(q @ v)  # cosine similarity
        scored.append((score, c))

    scored.sort(reverse=True, key=lambda x: x[0])
    return [c for score, c in scored[:k]]

def rag_answer(query: str, chunks: list[dict]) -> str:
    top_chunks = retrieve_top_k(query, chunks, k=5)

    context = "\n\n".join(
        f"[{c['id']}] {c['text']}" for c in top_chunks
    )

    prompt = f"""
You are a helpful assistant.
Answer the question using ONLY the context below.
If the answer is not in the context, say: "I don't know."

Question: {query}

Context:
{context}
""".strip()

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return resp.choices[0].message.content

# Example usage:
# chunks would be built offline once: chunk docs -> embed each -> store embeddings
# print(rag_answer("Why did suspicious logins spike?", chunks))
