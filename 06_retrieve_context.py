RETRIEVAL_K = 8
MAX_CONTEXT_CHUNKS = 4
# Cosine distance is (1 - cosine_similarity), so 0 = identical, 2 = opposite.
# Chunks with distance above this are treated as "not actually relevant"
# and dropped, rather than force-fed into the prompt just because they were
# the closest of the top-k. Tune this using the evaluation notebook: look at
# the distance of true-positive matches vs. off-topic queries and pick a
# cutoff that separates them. 0.6 is a reasonable starting point for
# all-MiniLM-L6-v2 on short factual queries.
MAX_DISTANCE = 0.6
# Note: in this project, one uploaded file == one document_id. This cap
# limits how many chunks from the SAME document can appear in one
# context package. It must be >= MAX_CONTEXT_CHUNKS for single-file
# books (the common case) — otherwise, for a book with only one source
# file, this silently caps the entire context at whatever this number
# is, regardless of MAX_CONTEXT_CHUNKS. Kept as a separate setting only
# in case a book is split across several files and you want to force
# diversity across them.
MAX_CHUNKS_PER_DOCUMENT = 4
WORD_BUDGET = 250


def build_context_package(
    collection,
    query: str,
    query_embedding: list[float],
    k: int = RETRIEVAL_K,
    max_context_chunks: int = MAX_CONTEXT_CHUNKS,
    max_chunks_per_document: int = MAX_CHUNKS_PER_DOCUMENT,
    word_budget: int = WORD_BUDGET,
    max_distance: float = MAX_DISTANCE,
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

    # Drop chunks that aren't actually relevant, even if they were in the
    # top-k. Without this, a query about something outside the book still
    # retrieves its "least bad" nearest neighbors and the LLM gets fed
    # context that has nothing to do with the question.
    candidates = [c for c in candidates if c["distance"] <= max_distance]

    if not candidates:
        return {"query": query, "chunks": [], "context_text": "", "sources": []}

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
