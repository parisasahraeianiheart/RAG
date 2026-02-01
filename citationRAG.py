from openai import OpenAI
client = OpenAI()

def rag_answer_with_citations(query: str, retrieved_chunks: list[dict]) -> dict:
    """
    retrieved_chunks = [{"id": "doc1:0", "text": "..."}, ...]
    """
    context = "\n\n".join(f"[{c['id']}] {c['text']}" for c in retrieved_chunks)

    system = """
You answer questions using the provided context.
Rules:
1) Use ONLY facts from the context.
2) If missing, say you don't know.
3) Provide citations as the chunk IDs you relied on.
Return strict JSON with keys: answer, citations.
""".strip()

    user = f"""
Question: {query}

Context:
{context}
""".strip()

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )

    import json
    return json.loads(resp.choices[0].message.content)

# Example usage:
# retrieved_chunks = retrieve_top_k(...)  # your retriever returns chunk dicts
# out = rag_answer_with_citations("What is suspicious_rate?", retrieved_chunks)
# print(out["answer"], out["citations"])
