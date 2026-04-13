"""
Module 1: Document Ingestion & RAG
Handles ChromaDB initialization, text chunking, embedding, and retrieval.
"""

import time


def initialize_chromadb():
    """
    Initialize an in-memory ChromaDB client and return the collection.
    Uses in-memory client to avoid file permission errors.
    Returns the collection or None if error occurs.
    """
    try:
        import chromadb

        client = chromadb.Client()

        # Try to get existing collection, create if not found
        try:
            collection = client.get_collection(name="are_documents")
        except Exception:
            collection = client.create_collection(name="are_documents")

        return collection

    except Exception as e:
        print(f"[ChromaDB] Error initializing: {e}")
        return None


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
    Embed text chunks using SentenceTransformer and store in ChromaDB.
    Returns the number of chunks stored.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")

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
    Retrieve the most relevant text chunks from ChromaDB for a given query.
    Returns a list of relevant text chunk strings.
    """
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("all-MiniLM-L6-v2")
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
