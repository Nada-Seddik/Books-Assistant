import chromadb
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
    # hnsw:space="cosine" pairs with the normalized embeddings from
    # 04_vector_representation.py, so collection.query() distances are
    # always (1 - cosine_similarity), in a fixed [0, 2] range. Without this,
    # Chroma defaults to squared L2 on raw (non-unit-length) vectors, whose
    # scale depends on the embedding model and isn't safe to threshold on.
    # NOTE: this only applies to collections created from now on — any
    # collection created before this change was built with the old default
    # space and must be deleted and re-indexed for the threshold in
    # 06_retrieve_context.py to be meaningful (delete_book() + re-upload).
    return get_client().get_or_create_collection(
        name=book_key, metadata={"hnsw:space": "cosine"}
    )


def get_description(book_key: str) -> str:
    """Return a book's stored description, or '' if none has been set yet."""
    collection = get_or_create_collection(book_key)
    return (collection.metadata or {}).get("description", "")


def set_description(book_key: str, description: str) -> None:
    """Store a short description for a book, held as collection metadata
    rather than as a chunk — so it never shows up as retrieved context."""
    collection = get_or_create_collection(book_key)
    collection.modify(metadata={"description": description})


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
