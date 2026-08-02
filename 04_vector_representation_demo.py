"""
DEMO-ONLY substitute for 04_vector_representation.py.

This sandbox environment cannot reach huggingface.co to download the real
all-MiniLM-L6-v2 model (network access here is restricted to package
registries like PyPI, not model hubs). To still demonstrate the full
pipeline and the retrieval-method comparison end-to-end, this module
implements a classic, no-internet-required dense embedding technique
(TF-IDF followed by SVD dimensionality reduction, i.e. "Latent Semantic
Analysis" / LSA) behind the exact same function names your real
04_vector_representation.py uses (embed_texts / embed_query), so nothing
else in the pipeline needs to change to run this demo.

On your own machine or on Streamlit Cloud (both of which have normal
internet access), use your real 04_vector_representation.py with
all-MiniLM-L6-v2 instead of this file.
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

_vectorizer: TfidfVectorizer | None = None
_svd: TruncatedSVD | None = None
_fitted_corpus: list[str] | None = None


def fit_demo_embeddings(corpus_texts: list[str], n_components: int = 32) -> None:
    """Must be called once with the full chunk corpus before embed_texts/
    embed_query are used, since LSA needs to be fit on the corpus first
    (unlike a pre-trained transformer model, which needs no fitting)."""
    global _vectorizer, _svd, _fitted_corpus
    n_components = min(n_components, max(1, len(corpus_texts) - 1))
    _vectorizer = TfidfVectorizer()
    tfidf_matrix = _vectorizer.fit_transform(corpus_texts)
    _svd = TruncatedSVD(n_components=n_components, random_state=42)
    _svd.fit(tfidf_matrix)
    _fitted_corpus = corpus_texts


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return vectors / norms


def embed_texts(texts: list[str]) -> list[list[float]]:
    if _vectorizer is None or _svd is None:
        raise RuntimeError("Call fit_demo_embeddings(corpus_texts) first.")
    tfidf = _vectorizer.transform(texts)
    dense = _svd.transform(tfidf)
    return _normalize(dense).tolist()


def embed_query(query: str) -> list[float]:
    return embed_texts([query])[0]


def get_embedding_model():
    return {"vectorizer": _vectorizer, "svd": _svd}
