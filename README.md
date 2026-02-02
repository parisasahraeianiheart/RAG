RAG (Retrieval-Augmented Generation)

Summary:
A complete RAG prototype that combines vector-based semantic retrieval with large language model generation. The system ingests documents, computes embeddings, builds a FAISS vector store, and uses top-k retrieval to ground LLM answers in actual evidence. It also produces structured prompts with citations, reducing hallucination and increasing trust.

Key Features & Contributions:
•	Built a vector store with FAISS for semantic retrieval.
•	Chunked corpora, computed embeddings, and retrieved relevant context.
•	Constructed grounded prompting patterns that enforced context-only responses.
•	Demonstrated retrieval, reranking, and prompt assembly for real-world Q&A.
•	Designed simple offline answer composers and showed how to plug in any LLM API.

Skills Demonstrated:
Semantic retrieval, vector search engineering, prompt engineering, grounding strategies, contextualized generation, and systems thinking.
