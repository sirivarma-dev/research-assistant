"""
The memory of the assistant.

When a paper is uploaded, its chunks are turned into numbers (embeddings) and
stored here. When you ask a question, we find the chunks whose meaning is
closest to your question and feed only those to the AI.

This uses ChromaDB's built-in embedding model, which runs locally and is FREE
(it downloads a small model the first time you run it - that needs internet once).
"""
import math

import chromadb
from chromadb.utils import embedding_functions

from config import CHROMA_DIR


def _cosine(a, b):
    """Cosine similarity between two equal-length vectors (pure Python)."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class VectorStore:
    def __init__(self, collection_name="papers"):
        # PersistentClient saves the data to disk so it survives restarts
        self.client = chromadb.PersistentClient(path=CHROMA_DIR)
        self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
        )
        # Keep the full text of the most recently uploaded paper in memory so
        # "whole document" questions (references, authors, full summary) can be
        # answered from the entire paper instead of just a few search chunks.
        self.full_text = ""

    def add_chunks(self, chunks, source):
        """Store a paper's chunks. 'source' is usually the file name."""
        ids = [f"{source}__{i}" for i in range(len(chunks))]
        metadatas = [{"source": source} for _ in chunks]
        # upsert = add, or overwrite if the same id already exists
        self.collection.upsert(documents=chunks, ids=ids, metadatas=metadatas)

    def search(self, query, n_results=4):
        """Return the chunks most relevant to the query."""
        count = self.collection.count()
        if count == 0:
            return []
        n_results = min(n_results, count)
        results = self.collection.query(query_texts=[query], n_results=n_results)
        return results.get("documents", [[]])[0]

    def count(self):
        """How many chunks are stored right now (0 means no paper yet)."""
        return self.collection.count()

    def rerank(self, query, documents, top_k=5):
        """
        Re-order `documents` (a list of strings) by how close their MEANING is
        to `query`, using the same local embedding model the vector store uses.

        This is what makes search understand a whole sentence instead of just
        matching keywords: we turn the query and every candidate into vectors and
        sort by cosine similarity. Returns a list of (index, score) pairs,
        best match first.
        """
        if not documents:
            return []
        # Embed the query and all documents in one call (first vector = query).
        vectors = self.embedding_fn([query] + list(documents))
        query_vec = vectors[0]
        scored = [
            (i, _cosine(query_vec, doc_vec))
            for i, doc_vec in enumerate(vectors[1:])
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_k]

    def set_full_text(self, text):
        """Remember the full text of the paper that was just uploaded."""
        self.full_text = text

    def get_full_text(self, max_chars=30000):
        """
        Return the whole paper's text (trimmed to fit the model's input limit).
        Used for questions that need the entire document, not just a few chunks.
        If the in-memory copy is empty (e.g. after an app restart), rebuild it
        from the stored chunks instead.
        """
        if self.full_text:
            return self.full_text[:max_chars]

        # Fallback: reassemble from stored chunks in their original order.
        data = self.collection.get()
        ids = data.get("ids", []) or []
        docs = data.get("documents", []) or []

        def chunk_index(chunk_id):
            try:
                return int(chunk_id.rsplit("__", 1)[1])
            except (IndexError, ValueError):
                return 0

        ordered = [doc for _, doc in sorted(zip(ids, docs), key=lambda p: chunk_index(p[0]))]
        return "\n".join(ordered)[:max_chars]

    def get_references_section(self, max_chars=10000):
        """
        Return just the paper's reference list. Reference sections sit at the
        very end of a paper, so we jump to the 'REFERENCES' heading instead of
        sending (and possibly truncating) the whole document.
        """
        text = self.get_full_text(max_chars=200000)  # effectively the full text

        # The heading is almost always the all-caps word "REFERENCES".
        idx = text.find("REFERENCES")
        if idx == -1:
            idx = text.lower().rfind("references")
        if idx == -1:
            return text[-max_chars:]  # no heading found: use the tail of the paper
        return text[idx:idx + max_chars]
