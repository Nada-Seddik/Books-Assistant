"""
05_create_chroma_store.py

Creates and manages the persisted Chroma vector store. New module: the
original project never persisted anything, rebuilding an in-memory index
each run. This is required for a deployed app, and it's also what makes
multi-book support work cleanly here — each book gets its own Chroma
collection, named by its normalized book key, so books don't interfere
with each other and can be listed/reused across sessions.

build_store_for_book() is idempotent: re-indexing a book clears its old
entries first, so re-uploading updated files for the same book doesn't
leave stale duplicate chunks behind.
"""

import chromadb
from typing import Any

CHROMA_DIR = "chroma_db"

# chromadb does not export ClientAPI; use a generic Any annotation to avoid
# depending on internals of the chromadb package.
_client: Any | None = None


def get_client() -> Any:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=CHROMA_DIR)
    return _client


def list_indexed_books() -> list[str]:
    """Return the normalized names of every book already indexed."""
    return sorted(c.name for c in get_client().list_collections())


def get_or_create_collection(book_key: str):
    return get_client().get_or_create_collection(name=book_key)


def build_store_for_book(book_key: str, chunks: list[dict], embeddings: list[list[float]]) -> None:
    """(Re)build a book's collection from its chunks and matching embeddings."""
    collection = get_or_create_collection(book_key)

    existing = collection.get()
    if existing and existing.get("ids"):
        collection.delete(ids=existing["ids"])

    collection.add(
        ids=[c["chunk_id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        embeddings=embeddings,
        metadatas=[
            {"title": c["title"], "book": c["book"], "document_id": c["document_id"]}
            for c in chunks
        ],
    )
