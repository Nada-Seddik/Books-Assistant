import importlib
import os
import sys

sys.path.insert(0, os.getcwd())
evaluation = importlib.import_module("evaluation")

RETRIEVAL_K = 8
MAX_CONTEXT_CHUNKS = 4
# Cosine distance is (1 - cosine_similarity), so 0 = identical, 2 = opposite.
# Chunks with distance above this are treated as "not actually relevant"
# and dropped, rather than force-fed into the prompt just because they were
# the closest of the top-k. Tune this using the evaluation notebook: look at
# the distance of true-positive matches vs. off-topic queries and pick a
# cutoff that separates them. 0.6 is a reasonable starting point for
# all-MiniLM-L6-v2 on short factual queries.
#
# NOTE ON NON-EMBEDDING METHODS: TF-IDF, BM25, and Hybrid don't produce a
# cosine distance at all -- they produce their own, differently-scaled
# similarity scores. To let ONE threshold apply across every method, those
# methods' scores are min-max normalized to [0, 1] across the batch of
# retrieved candidates and then converted to a "distance" the same way
# (distance = 1 - normalized_score), so this constant means roughly the
# same thing ("relative to this query's other candidates, is this one
# still competitive?") everywhere, even though it isn't literally a cosine
# distance for three of the four methods.
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


def _fetch_candidates(collection, query: str, query_embedding: list[float], k: int, method: str):
    """Return a list of {"text","title","document_id","distance"} dicts for
    the given method, sorted best-first (lowest distance first). This is
    the one place that knows how to bridge each of the four retrieval
    strategies into the same shape the rest of build_context_package()
    already expects.
    """
    if method == "embeddings":
        # The fast path: Chroma's own HNSW index does the search, so we
        # never have to pull the whole book's corpus into memory just to
        # answer one question.
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        return [
            {"text": doc, "title": meta["title"], "document_id": meta["document_id"], "distance": dist}
            for doc, meta, dist in zip(documents, metadatas, distances)
        ]

    # TF-IDF / BM25 / Hybrid need the book's full raw text to score against,
    # since (unlike embeddings) there's no persistent Chroma index for
    # them. For a book-sized corpus this is cheap enough to do per-query;
    # it does mean these three methods are slower than "embeddings" as a
    # book grows into the thousands of chunks.
    data = collection.get(include=["documents", "metadatas", "embeddings"])
    corpus_ids = data["ids"]
    corpus_texts = data["documents"]
    corpus_metadatas = data["metadatas"]
    corpus_embeddings = data["embeddings"]

    id_to_meta = {cid: meta for cid, meta in zip(corpus_ids, corpus_metadatas)}
    id_to_text = {cid: text for cid, text in zip(corpus_ids, corpus_texts)}

    benchmark = evaluation.RetrievalBenchmark(corpus_ids, corpus_texts, corpus_embeddings)
    retrieved_ids, scores = benchmark.retrieve(method, query, query_embedding, k)

    # Normalize this method's raw scores (which have their own, method-
    # specific scale) into a [0, 1] "how good is this vs. the other
    # candidates in THIS result set" score, then flip it into a distance
    # so max_distance means roughly the same thing for every method.
    normalized = benchmark.minmax(scores)
    return [
        {
            "text": id_to_text[cid],
            "title": id_to_meta[cid]["title"],
            "document_id": id_to_meta[cid]["document_id"],
            "distance": 1 - norm_score,
        }
        for cid, norm_score in zip(retrieved_ids, normalized)
    ]


def build_context_package(
    collection,
    query: str,
    query_embedding: list[float],
    method: str = "embeddings",
    k: int = RETRIEVAL_K,
    max_context_chunks: int = MAX_CONTEXT_CHUNKS,
    max_chunks_per_document: int = MAX_CHUNKS_PER_DOCUMENT,
    word_budget: int = WORD_BUDGET,
    max_distance: float = MAX_DISTANCE,
) -> dict:
    """Retrieve top-k chunks for a query and shape them into a context
    package. `method` is one of "embeddings" / "tfidf" / "bm25" / "hybrid"
    -- streamlit_app.answer_question() looks this up per-book via
    05_create_chroma_store.get_retrieval_method(), which was set
    automatically by evaluation.recommend_best_method() right after the
    book was indexed."""
    candidates = _fetch_candidates(collection, query, query_embedding, k, method)

    if not candidates:
        return {"query": query, "chunks": [], "context_text": "", "sources": [], "method": method}

    candidates.sort(key=lambda c: c["distance"])  # lower distance = more similar

    # Drop chunks that aren't actually relevant, even if they were in the
    # top-k. Without this, a query about something outside the book still
    # retrieves its "least bad" nearest neighbors and the LLM gets fed
    # context that has nothing to do with the question.
    candidates = [c for c in candidates if c["distance"] <= max_distance]

    if not candidates:
        return {"query": query, "chunks": [], "context_text": "", "sources": [], "method": method}

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
        "method": method,
    }
