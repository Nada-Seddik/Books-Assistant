import importlib
import sys
import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi


sys.path.insert(0, os.getcwd())
vector_representation = importlib.import_module("04_vector_representation")
chroma_store = importlib.import_module("05_create_chroma_store")

BOOK_KEY = "REPLACE_ME"  # e.g. "the_art_of_war" — must already be indexed via the app
collection = chroma_store.get_or_create_collection(BOOK_KEY)
data = collection.get(include=["documents", "embeddings", "metadatas"])

corpus_ids = data["ids"]
corpus_texts = data["documents"]
corpus_embeddings = np.array(data["embeddings"])  # already normalized (see 04_vector_representation.py)
corpus_titles = [m["title"] for m in data["metadatas"]]

print(f"Loaded {len(corpus_ids)} chunks for '{BOOK_KEY}'")
for cid, text in list(zip(corpus_ids, corpus_texts))[:5]:
    print(f"  {cid}: {text[:90]!r}...")

GROUND_TRUTH = [
    {"query": "REPLACE with a real question about the book", "expected_chunk_id": "REPLACE with a chunk id from corpus_ids"},
    # ... add 10-15 total
]

# Questions the book has no answer to — should retrieve nothing after thresholding
NEGATIVE_QUERIES = [
    "What is the capital of France?",
    "How do I bake a chocolate cake?",
]

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(corpus_texts)

tokenized_corpus = [text.lower().split() for text in corpus_texts]
bm25 = BM25Okapi(tokenized_corpus)


def _minmax(scores: np.ndarray) -> np.ndarray:
    lo, hi = scores.min(), scores.max()
    return (scores - lo) / (hi - lo) if hi > lo else np.zeros_like(scores)


def retrieve_embedding(query: str, k: int):
    q_emb = np.array(vector_representation.embed_query(query))
    sims = corpus_embeddings @ q_emb  # cosine similarity, since both are unit-normalized
    top_idx = np.argsort(-sims)[:k]
    return [corpus_ids[i] for i in top_idx], sims[top_idx]


def retrieve_tfidf(query: str, k: int):
    q_vec = tfidf_vectorizer.transform([query])
    sims = cosine_similarity(q_vec, tfidf_matrix)[0]
    top_idx = np.argsort(-sims)[:k]
    return [corpus_ids[i] for i in top_idx], sims[top_idx]


def retrieve_bm25(query: str, k: int):
    scores = bm25.get_scores(query.lower().split())
    top_idx = np.argsort(-scores)[:k]
    return [corpus_ids[i] for i in top_idx], scores[top_idx]


def retrieve_hybrid(query: str, k: int):
    q_emb = np.array(vector_representation.embed_query(query))
    emb_scores = corpus_embeddings @ q_emb
    bm25_scores = np.array(bm25.get_scores(query.lower().split()))
    combined = 0.5 * _minmax(emb_scores) + 0.5 * _minmax(bm25_scores)
    top_idx = np.argsort(-combined)[:k]
    return [corpus_ids[i] for i in top_idx], combined[top_idx]


METHODS = {
    "Embeddings (current)": retrieve_embedding,
    "TF-IDF": retrieve_tfidf,
    "BM25": retrieve_bm25,
    "Hybrid (BM25 + Embeddings)": retrieve_hybrid,
}

K = 5
def evaluate(method_fn):
    precisions, reciprocal_ranks = [], []
    for item in GROUND_TRUTH:
        retrieved_ids, _ = method_fn(item["query"], K)
        hit = item["expected_chunk_id"] in retrieved_ids
        precisions.append((1 / K) if hit else 0.0)
        if hit:
            rank = retrieved_ids.index(item["expected_chunk_id"]) + 1
            reciprocal_ranks.append(1 / rank)
        else:
            reciprocal_ranks.append(0.0)
    return {
        "Precision@{}".format(K): float(np.mean(precisions)),
        "MRR": float(np.mean(reciprocal_ranks)),
    }


results = {name: evaluate(fn) for name, fn in METHODS.items()}
for name, metrics in results.items():
    print(f"{name:28s}  Precision@{K}={metrics[f'Precision@{K}']:.3f}   MRR={metrics['MRR']:.3f}")

labels = list(results.keys())
precision_vals = [results[m][f"Precision@{K}"] for m in labels]
mrr_vals = [results[m]["MRR"] for m in labels]

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

in_book_distances = []
for item in GROUND_TRUTH:
    retrieved_ids, sims = retrieve_embedding(item["query"], k=len(corpus_ids))
    if item["expected_chunk_id"] in retrieved_ids:
        idx = retrieved_ids.index(item["expected_chunk_id"])
        in_book_distances.append(1 - sims[idx])

out_of_book_distances = []
for query in NEGATIVE_QUERIES:
    _, sims = retrieve_embedding(query, k=1)
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
