"""
06_retrieve_context.py

Queries a book's Chroma collection and builds a filtered, word-budgeted,
source-labeled context package — the same shaping logic the original
project used (cap chunks per source document, stop once the word budget
is filled), now applied to Chroma's nearest-neighbor results instead of
an in-memory hybrid score.

Takes a pre-computed query embedding rather than importing
04_vector_representation itself, so this module doesn't need to import
another numbered file (whose names, starting with digits, aren't valid
Python import targets) — streamlit_app.py wires the stages together instead.
"""

RETRIEVAL_K = 5
MAX_CONTEXT_CHUNKS = 3
MAX_CHUNKS_PER_DOCUMENT = 1
WORD_BUDGET = 150


def build_context_package(
    collection,
    query: str,
    query_embedding: list[float],
    k: int = RETRIEVAL_K,
    max_context_chunks: int = MAX_CONTEXT_CHUNKS,
    max_chunks_per_document: int = MAX_CHUNKS_PER_DOCUMENT,
    word_budget: int = WORD_BUDGET,
) -> dict:
    """Retrieve top-k chunks for a query and shape them into a context package."""
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    if not documents:
        return {"query": query, "chunks": [], "context_text": "", "sources": []}

    candidates = [
        {"text": doc, "title": meta["title"], "document_id": meta["document_id"], "distance": dist}
        for doc, meta, dist in zip(documents, metadatas, distances)
    ]
    candidates.sort(key=lambda c: c["distance"])  # lower distance = more similar

    selected = []
    per_document_count: dict[str, int] = {}
    word_count = 0

    for candidate in candidates:
        if len(selected) >= max_context_chunks:
            break
        doc_count = per_document_count.get(candidate["document_id"], 0)
        if doc_count >= max_chunks_per_document:
            continue
        chunk_words = len(candidate["text"].split())
        if word_count + chunk_words > word_budget and selected:
            continue

        selected.append(candidate)
        per_document_count[candidate["document_id"]] = doc_count + 1
        word_count += chunk_words

    context_lines = [f"[Source: {c['title']}] {c['text']}" for c in selected]

    return {
        "query": query,
        "chunks": selected,
        "context_text": "\n\n".join(context_lines),
        "sources": sorted({c["title"] for c in selected}),
    }
