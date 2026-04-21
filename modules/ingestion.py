"""
Module 1: Document Ingestion & RAG
Handles text chunking, embedding, and in-memory retrieval.
"""

import time
from functools import lru_cache

import numpy as np


@lru_cache(maxsize=1)
def _get_embedding_model():
    """Load and cache embedding model once per process."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("all-MiniLM-L6-v2")


class InMemoryCollection:
    """A minimal collection API compatible with the app's usage."""

    def __init__(self):
        self._items = []

    def add(self, ids, embeddings, documents, metadatas):
        for idx, embedding, document, metadata in zip(
            ids, embeddings, documents, metadatas
        ):
            self._items.append(
                {
                    "id": idx,
                    "embedding": np.array(embedding, dtype=float),
                    "document": document,
                    "metadata": metadata,
                }
            )

    def query(self, query_embeddings, n_results=3):
        if not self._items:
            return {"documents": [[]]}

        query_vec = np.array(query_embeddings[0], dtype=float)
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return {"documents": [[]]}

        scored = []
        for item in self._items:
            item_vec = item["embedding"]
            denom = query_norm * np.linalg.norm(item_vec)
            similarity = float(np.dot(query_vec, item_vec) / denom) if denom else 0.0
            scored.append((similarity, item["document"]))

        top_docs = [doc for _, doc in sorted(scored, reverse=True)[:n_results]]
        return {"documents": [top_docs]}


def initialize_chromadb():
    """
    Initialize an in-memory collection for retrieval.
    Returns the collection instance.
    """
    return InMemoryCollection()


def chunk_text(text, chunk_size=150, overlap=30):
    """
    Split text into overlapping chunks of chunk_size words.
    Each chunk overlaps with the previous by `overlap` words.
    Skips chunks smaller than 20 words.
    Returns a list of chunk strings.
    """
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]

        # Only add chunks with at least 20 words
        if len(chunk_words) >= 20:
            chunks.append(" ".join(chunk_words))

        # Move forward by (chunk_size - overlap) so chunks overlap
        start += chunk_size - overlap

        # Safety: avoid infinite loop if overlap >= chunk_size
        if chunk_size <= overlap:
            break

    return chunks


def embed_and_store(text, collection, source_name="uploaded_doc"):
    """
    Embed text chunks and store in the in-memory collection.
    Returns the number of chunks stored.
    """
    model = _get_embedding_model()

    chunks = chunk_text(text)
    if not chunks:
        print("[Ingestion] No valid chunks found in text.")
        return 0

    stored_count = 0
    timestamp = int(time.time())

    for i, chunk in enumerate(chunks):
        try:
            embedding = model.encode(chunk).tolist()

            # Create a unique ID using source_name, index, and timestamp
            doc_id = f"{source_name}_{i}_{timestamp}"

            collection.add(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[{"source": source_name, "chunk_id": i}],
            )

            stored_count += 1
            print(f"[Ingestion] Stored chunk {i + 1}/{len(chunks)}: {chunk[:60]}...")

        except Exception as e:
            # Try with a more unique ID if duplicate error
            try:
                doc_id = f"{source_name}_{i}_{timestamp}_{stored_count}"
                embedding = model.encode(chunk).tolist()
                collection.add(
                    ids=[doc_id],
                    embeddings=[embedding],
                    documents=[chunk],
                    metadatas=[{"source": source_name, "chunk_id": i}],
                )
                stored_count += 1
                print(f"[Ingestion] Stored chunk {i + 1} with alternate ID.")
            except Exception as e2:
                print(f"[Ingestion] Failed to store chunk {i}: {e2}")

    return stored_count


def retrieve_relevant_chunks(query, collection, top_k=3):
    """
    Retrieve the most relevant text chunks for a given query.
    Returns a list of relevant text chunk strings.
    """
    try:
        model = _get_embedding_model()
        query_embedding = model.encode(query).tolist()

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

        documents = results.get("documents", [[]])[0]
        return documents if documents else []

    except Exception as e:
        print(f"[Retrieval] Error retrieving chunks: {e}")
        return []
