import json
import chromadb  # type: ignore[import]
CHROMA_DIR = "chroma_db"
_client: chromadb.ClientAPI | None = None


def get_client() -> chromadb.ClientAPI:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=CHROMA_DIR)
    return _client


def list_indexed_books() -> list[str]:
    return sorted(c.name for c in get_client().list_collections())


def get_or_create_collection(book_key: str):
    return get_client().get_or_create_collection(
        name=book_key, metadata={"hnsw:space": "cosine"}
    )


def _update_metadata(book_key: str, updates: dict) -> None:
    """Merge `updates` into a collection's existing metadata, rather than
    replacing it outright. collection.modify(metadata=...) OVERWRITES the
    whole metadata dict — calling it with just {"description": ...} would
    silently wipe out "hnsw:space": "cosine" set at creation time (see
    get_or_create_collection above), quietly reverting the collection to
    Chroma's default distance space and breaking the MAX_DISTANCE threshold
    in 06_retrieve_context.py without any visible error. Every metadata
    write in this file goes through this helper so that never happens.

    "hnsw:space" specifically must also be *dropped* before calling
    modify(), not just preserved unchanged: Chroma raises an error if that
    key is present in a modify() call at all, even set to its current,
    unchanged value ("changing the distance function ... is not supported
    currently") — it isn't actually re-appliable, just permanently fixed at
    creation time, so there's nothing to "preserve" here, only to omit.
    """
    collection = get_or_create_collection(book_key)
    merged = dict(collection.metadata or {})
    merged.pop("hnsw:space", None)
    merged.update(updates)
    collection.modify(metadata=merged)


def get_description(book_key: str) -> str:
    """Return a book's stored description, or '' if none has been set yet."""
    collection = get_or_create_collection(book_key)
    return (collection.metadata or {}).get("description", "")


def set_description(book_key: str, description: str) -> None:
    """Store a short description for a book, held as collection metadata
    rather than as a chunk — so it never shows up as retrieved context."""
    _update_metadata(book_key, {"description": description})


# Retrieval methods 06_retrieve_context.py knows how to run. "embeddings" is
# always a safe fallback (it's what every collection already has vectors
# for); the others are computed on the fly from the collection's raw text.
RETRIEVAL_METHODS = ("embeddings", "tfidf", "bm25", "hybrid")
DEFAULT_RETRIEVAL_METHOD = "embeddings"


def get_retrieval_method(book_key: str) -> str:
    """Return the retrieval method chosen for this book (defaults to
    'embeddings' for books that haven't been auto-evaluated yet, e.g. ones
    indexed before this feature existed)."""
    collection = get_or_create_collection(book_key)
    return (collection.metadata or {}).get("retrieval_method", DEFAULT_RETRIEVAL_METHOD)


def set_retrieval_method(book_key: str, method: str) -> None:
    """Store which retrieval method (embeddings/tfidf/bm25/hybrid) should be
    used for this specific book. Set automatically right after indexing by
    evaluation.recommend_best_method(), and held as collection metadata —
    same reasoning as the description: it's a property of the book's
    collection, not a chunk, so it never leaks into retrieved context."""
    if method not in RETRIEVAL_METHODS:
        raise ValueError(f"Unknown retrieval method: {method!r}. Must be one of {RETRIEVAL_METHODS}.")
    _update_metadata(book_key, {"retrieval_method": method})


def get_retrieval_scores(book_key: str) -> dict:
    """Return the per-method Precision/MRR scores recorded when the
    retrieval method was last auto-picked, for optional display in the UI
    (e.g. 'Using Hybrid — MRR 0.89 vs 0.74 for Embeddings'). Returns {} if
    no evaluation has been recorded yet."""
    collection = get_or_create_collection(book_key)
    raw = (collection.metadata or {}).get("retrieval_scores", "")
    return json.loads(raw) if raw else {}


def set_retrieval_scores(book_key: str, scores: dict) -> None:
    """Chroma collection metadata values must be flat primitives (str,
    int, float, bool) — a nested dict like {"embeddings": {"precision":
    ..., "mrr": ...}, ...} isn't allowed directly, so it's stored as a
    JSON string and decoded again in get_retrieval_scores()."""
    _update_metadata(book_key, {"retrieval_scores": json.dumps(scores)})


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


def delete_book(book_key: str) -> None:
    """Remove a book's entire collection (chunks, embeddings, and description)."""
    get_client().delete_collection(name=book_key)
