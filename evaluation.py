"""
evaluation.py

Two things live in this file:

1. REUSABLE FUNCTIONS (top of the file) — a RetrievalBenchmark class plus
   evaluate_method(), generate_synthetic_ground_truth(), and
   recommend_best_method(). These are imported directly by
   streamlit_app.py: right after a book is indexed, it calls
   recommend_best_method() to automatically test all four retrieval
   strategies on that specific book and store the winner (see
   05_create_chroma_store.set_retrieval_method), so 06_retrieve_context.py
   knows which method to actually use when answering real questions.

2. MANUAL, IN-DEPTH ANALYSIS (bottom, under `if __name__ == "__main__":`,
   in # %% cells) — the original hand-run notebook flow: point BOOK_KEY at
   an already-indexed book, hand-write a GROUND_TRUTH list yourself, and
   get full Precision@K/MRR charts plus the distance-distribution plot
   used to justify MAX_DISTANCE in 06_retrieve_context.py. This part is
   guarded so it does NOT run just from being imported — only the
   functions above do, keeping streamlit_app.py's import side-effect-free.
"""
import importlib
import random
import sys
import os

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi

sys.path.insert(0, os.getcwd())
chroma_store = importlib.import_module("05_create_chroma_store")
rag = importlib.import_module("07_prompting")


class RetrievalBenchmark:
    """Wraps one book's already-indexed corpus and lets you query it with
    four different retrieval strategies, purely in-memory (no writes back
    to Chroma). Used both by recommend_best_method() below (live, at
    indexing time) and by 06_retrieve_context.py (live, at answer time,
    for whichever method wasn't "embeddings")."""

    def __init__(self, corpus_ids: list[str], corpus_texts: list[str], corpus_embeddings: list[list[float]]):
        self.corpus_ids = corpus_ids
        self.corpus_texts = corpus_texts
        self.corpus_embeddings = np.array(corpus_embeddings)
        self.tfidf_vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(corpus_texts)
        tokenized_corpus = [text.lower().split() for text in corpus_texts]
        self.bm25 = BM25Okapi(tokenized_corpus)

    @staticmethod
    def minmax(scores: np.ndarray) -> np.ndarray:
        lo, hi = scores.min(), scores.max()
        return (scores - lo) / (hi - lo) if hi > lo else np.zeros_like(scores)

    def retrieve_embedding(self, query_embedding: list[float], k: int):
        sims = self.corpus_embeddings @ np.array(query_embedding)
        top_idx = np.argsort(-sims)[:k]
        return [self.corpus_ids[i] for i in top_idx], sims[top_idx]

    def retrieve_tfidf(self, query: str, k: int):
        q_vec = self.tfidf_vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.tfidf_matrix)[0]
        top_idx = np.argsort(-sims)[:k]
        return [self.corpus_ids[i] for i in top_idx], sims[top_idx]

    def retrieve_bm25(self, query: str, k: int):
        scores = self.bm25.get_scores(query.lower().split())
        top_idx = np.argsort(-scores)[:k]
        return [self.corpus_ids[i] for i in top_idx], scores[top_idx]

    def retrieve_hybrid(self, query: str, query_embedding: list[float], k: int):
        emb_scores = self.corpus_embeddings @ np.array(query_embedding)
        bm25_scores = np.array(self.bm25.get_scores(query.lower().split()))
        combined = 0.5 * self.minmax(emb_scores) + 0.5 * self.minmax(bm25_scores)
        top_idx = np.argsort(-combined)[:k]
        return [self.corpus_ids[i] for i in top_idx], combined[top_idx]

    def retrieve(self, method: str, query: str, query_embedding: list[float], k: int):
        """Single dispatch point used by both this file and
        06_retrieve_context.py, so there's exactly one place that maps a
        method name to the function that runs it."""
        if method == "embeddings":
            return self.retrieve_embedding(query_embedding, k)
        if method == "tfidf":
            return self.retrieve_tfidf(query, k)
        if method == "bm25":
            return self.retrieve_bm25(query, k)
        if method == "hybrid":
            return self.retrieve_hybrid(query, query_embedding, k)
        raise ValueError(f"Unknown retrieval method: {method!r}")


def evaluate_method(benchmark: RetrievalBenchmark, method: str, ground_truth: list[dict],
                     embed_query_fn, k: int) -> dict:
    """Precision@k and MRR for one retrieval method against one ground-truth set."""
    precisions, reciprocal_ranks = [], []
    for item in ground_truth:
        query_embedding = embed_query_fn(item["query"]) if method in ("embeddings", "hybrid") else None
        retrieved_ids, _ = benchmark.retrieve(method, item["query"], query_embedding, k)
        hit = item["expected_chunk_id"] in retrieved_ids
        precisions.append((1 / k) if hit else 0.0)
        reciprocal_ranks.append((1 / (retrieved_ids.index(item["expected_chunk_id"]) + 1)) if hit else 0.0)
    return {
        "precision": float(np.mean(precisions)) if precisions else 0.0,
        "mrr": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0,
    }


def generate_synthetic_ground_truth(corpus_ids: list[str], corpus_texts: list[str],
                                     max_questions: int = 10) -> list[dict]:
    """Auto-builds a small ground-truth set for a book with NO manual
    labeling, by asking the LLM to write one natural question per sampled
    chunk, whose answer is fully contained in that chunk (07_prompting.
    generate_eval_question). The chunk that produced a question becomes
    that question's own "expected" answer. This is what lets a brand-new
    book get evaluated automatically at indexing time, instead of requiring
    a human to hand-write 10-15 questions first, as the manual __main__
    analysis below still does for a deeper, more trustworthy write-up.

    If a book is small, every chunk gets a question. If it's large, a
    random sample of `max_questions` chunks is used, to keep indexing fast
    (each question costs one LLM call).
    """
    indices = list(range(len(corpus_ids)))
    if len(indices) > max_questions:
        random.seed(42)  # fixed seed: same book -> same sample -> reproducible method choice
        indices = sorted(random.sample(indices, max_questions))

    ground_truth = []
    for i in indices:
        try:
            question = rag.generate_eval_question(corpus_texts[i])
        except Exception:
            # One failed LLM call (rate limit, network blip) shouldn't sink
            # the whole auto-evaluation -- just skip that chunk's question.
            continue
        if question:
            ground_truth.append({"query": question.strip(), "expected_chunk_id": corpus_ids[i]})
    return ground_truth


def recommend_best_method(book_key: str, embed_query_fn, k: int = 3, max_questions: int = 10,
                           min_chunks_to_evaluate: int = 4) -> tuple[str, dict]:
    """THE AUTO-PICKER. Called by streamlit_app.index_book() right after a
    book is indexed.

    Loads the book's already-indexed chunks from Chroma, generates a small
    synthetic ground-truth set on the fly, scores all four retrieval
    strategies against it, and returns (best_method_name, all_scores) --
    ranked by MRR first (rewards finding the right chunk near the top),
    then Precision@k as a tiebreaker.

    Falls back to ("embeddings", {}) if the book is too small to evaluate
    meaningfully (fewer than `min_chunks_to_evaluate` chunks) or if no
    synthetic questions could be generated (e.g. the LLM call failed for
    every sampled chunk) -- "embeddings" is always safe, since every
    collection already has vectors for it regardless of book size.
    """
    collection = chroma_store.get_or_create_collection(book_key)
    data = collection.get(include=["documents", "embeddings"])
    corpus_ids, corpus_texts, corpus_embeddings = data["ids"], data["documents"], data["embeddings"]

    if len(corpus_ids) < min_chunks_to_evaluate:
        return "embeddings", {}

    ground_truth = generate_synthetic_ground_truth(corpus_ids, corpus_texts, max_questions=max_questions)
    if not ground_truth:
        return "embeddings", {}

    benchmark = RetrievalBenchmark(corpus_ids, corpus_texts, corpus_embeddings)
    scores = {
        method: evaluate_method(benchmark, method, ground_truth, embed_query_fn, k)
        for method in chroma_store.RETRIEVAL_METHODS
    }

    best_method = max(scores, key=lambda name: (scores[name]["mrr"], scores[name]["precision"]))
    return best_method, scores


if __name__ == "__main__":
    # %% ---- Manual, in-depth analysis (run this cell-by-cell in VS Code) ----
    # This is the original hand-run workflow, unchanged in spirit: point
    # BOOK_KEY at a book you've already indexed via the app, hand-write
    # real GROUND_TRUTH questions (use list_chunks_helper-style printing
    # below to find real chunk_ids), and get full charts. Use this for the
    # deeper, presentation-ready analysis; the live app itself uses
    # recommend_best_method() above, automatically, with no manual step.
    import matplotlib.pyplot as plt
    vector_representation = importlib.import_module("04_vector_representation")

    BOOK_KEY = "REPLACE_ME"  # e.g. "the_art_of_war" — must already be indexed via the app
    collection = chroma_store.get_or_create_collection(BOOK_KEY)
    data = collection.get(include=["documents", "embeddings", "metadatas"])

    corpus_ids = data["ids"]
    corpus_texts = data["documents"]
    corpus_embeddings = data["embeddings"]

    print(f"Loaded {len(corpus_ids)} chunks for '{BOOK_KEY}'")
    for cid, text in list(zip(corpus_ids, corpus_texts))[:5]:
        print(f"  {cid}: {text[:90]!r}...")

    # %%
    GROUND_TRUTH = [
        {"query": "REPLACE with a real question about the book", "expected_chunk_id": "REPLACE with a chunk id from corpus_ids"},
        # ... add 10-15 total
    ]

    NEGATIVE_QUERIES = [
        "What is the capital of France?",
        "How do I bake a chocolate cake?",
    ]

    # %%
    K = 5
    benchmark = RetrievalBenchmark(corpus_ids, corpus_texts, corpus_embeddings)
    method_labels = {
        "embeddings": "Embeddings (current)",
        "tfidf": "TF-IDF",
        "bm25": "BM25",
        "hybrid": "Hybrid (BM25 + Embeddings)",
    }
    results = {
        method_labels[m]: evaluate_method(benchmark, m, GROUND_TRUTH, vector_representation.embed_query, K)
        for m in chroma_store.RETRIEVAL_METHODS
    }
    for name, metrics in results.items():
        print(f"{name:28s}  Precision@{K}={metrics['precision']:.3f}   MRR={metrics['mrr']:.3f}")

    # %%
    labels = list(results.keys())
    precision_vals = [results[m]["precision"] for m in labels]
    mrr_vals = [results[m]["mrr"] for m in labels]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width/2, precision_vals, width, label=f"Precision@{K}")
    ax.bar(x + width/2, mrr_vals, width, label="MRR")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Retrieval method comparison")
    ax.legend()
    plt.tight_layout()
    plt.show()

    # %%
    in_book_distances = []
    for item in GROUND_TRUTH:
        q_emb = vector_representation.embed_query(item["query"])
        retrieved_ids, sims = benchmark.retrieve_embedding(q_emb, k=len(corpus_ids))
        if item["expected_chunk_id"] in retrieved_ids:
            idx = retrieved_ids.index(item["expected_chunk_id"])
            in_book_distances.append(1 - sims[idx])

    out_of_book_distances = []
    for query in NEGATIVE_QUERIES:
        q_emb = vector_representation.embed_query(query)
        _, sims = benchmark.retrieve_embedding(q_emb, k=1)
        out_of_book_distances.append(1 - sims[0])

    print("In-book (relevant) distances:    ", [round(d, 3) for d in in_book_distances])
    print("Out-of-book (irrelevant) distances:", [round(d, 3) for d in out_of_book_distances])

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(in_book_distances, bins=10, alpha=0.6, label="In-book queries (should be relevant)")
    ax.hist(out_of_book_distances, bins=10, alpha=0.6, label="Out-of-book queries (should be rejected)")
    ax.set_xlabel("Cosine distance (lower = more similar)")
    ax.set_ylabel("Count")
    ax.legend()
    ax.set_title("Distance distribution: relevant vs. irrelevant queries")
    plt.tight_layout()
    plt.show()
