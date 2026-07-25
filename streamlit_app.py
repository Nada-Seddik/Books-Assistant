"""
streamlit_app.py

Streamlit UI and pipeline orchestrator. Wires together every numbered
stage in order: documents -> preprocessing -> chunking -> vector
representation -> vector store -> context retrieval -> prompting.

Numbered modules (01_documents.py, etc.) can't be imported with a normal
`import 01_documents` statement, since Python identifiers can't start
with a digit — that's a language rule, not something to work around with
extra files. importlib.import_module() has no such restriction, since it
looks modules up by string name rather than parsing an identifier.
"""

# Streamlit Cloud's Linux environment ships an older system sqlite3 than
# Chroma requires (>= 3.35.0), which otherwise surfaces as a confusing
# KeyError deep inside chromadb rather than a clear version error. This
# swaps in a modern, self-contained sqlite3 build. pysqlite3-binary has
# no Windows wheels (and isn't needed there, since Windows already ships
# a new enough sqlite3), so this must run before chromadb is imported
# anywhere, and must stay optional for local Windows development.
try:
    __import__("pysqlite3")
    import sys as _sys
    _sys.modules["sqlite3"] = _sys.modules.pop("pysqlite3")
except ImportError:
    pass

import importlib
import os
import sys

import streamlit as st
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
load_dotenv()  # loads OPENROUTER_API_KEY/OPENROUTER_MODEL from a local .env, if present

documents = importlib.import_module("01_documents")
preprocessing = importlib.import_module("02_preprocessing")
chunking = importlib.import_module("03_chunking")
vector_representation = importlib.import_module("04_vector_representation")
chroma_store = importlib.import_module("05_create_chroma_store")
retrieve_context = importlib.import_module("06_retrieve_context")
rag = importlib.import_module("07_prompting")

# If no API key was set via a local .env, fall back to Streamlit secrets
# (used when deployed). Wrapped in try/except since st.secrets raises if
# no secrets.toml exists at all (e.g. running locally without one).
try:
    if not rag.OPENROUTER_API_KEY:
        rag.OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY", "")
        rag.OPENROUTER_MODEL = st.secrets.get("OPENROUTER_MODEL", rag.OPENROUTER_MODEL)
except Exception:
    pass


# ---------------------------------------------------------------------------
# Visual design: a "reading room" aesthetic — parchment background, oxblood
# and forest-green accents, a book-cover display face paired with a warm
# serif body face. The signature element is the row of colored "book
# spines" above the title, standing in for a stock icon.
# ---------------------------------------------------------------------------
LIBRARY_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=Lora:ital,wght@0,400;0,600;1,400&family=IBM+Plex+Mono:wght@500&display=swap');

:root {
    --ink: #2A2118;
    --parchment: #F1E9D8;
    --parchment-deep: #E8DCC0;
    --oxblood: #7A2E2E;
    --oxblood-dark: #5E2222;
    --forest: #3B5941;
    --gold: #B08D57;
    --plum: #5B4A6F;
}

.stApp { background-color: var(--parchment); }

/* Book-spine signature above the title */
.spine-row { display: flex; gap: 6px; margin-bottom: 0.6rem; }
.spine {
    width: 14px;
    border-radius: 4px 4px 0 0;
    box-shadow: 1px 1px 3px rgba(0,0,0,0.25);
}

.library-header h1 {
    font-family: 'Playfair Display', serif;
    font-weight: 900;
    color: var(--ink);
    margin-bottom: 0.1rem;
    letter-spacing: -0.5px;
}
.library-subtitle {
    font-family: 'Lora', serif;
    font-style: italic;
    color: var(--plum);
    font-size: 1.05rem;
    margin-top: 0;
    margin-bottom: 1.6rem;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: var(--parchment-deep);
    border-right: 1px solid rgba(0,0,0,0.08);
}
section[data-testid="stSidebar"] h2 {
    font-family: 'Playfair Display', serif;
    color: var(--ink);
}

/* Utility-face labels (small caps eyebrow style) */
.eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: var(--forest);
    margin-bottom: 0.2rem;
}

/* Buttons */
.stButton > button {
    background-color: var(--oxblood);
    color: var(--parchment);
    border: none;
    border-radius: 6px;
    font-family: 'Lora', serif;
    font-weight: 600;
    padding: 0.5rem 1.2rem;
    transition: background-color 0.15s ease;
}
.stButton > button:hover {
    background-color: var(--oxblood-dark);
    color: var(--parchment);
}
.stButton > button:disabled {
    background-color: #C9BFA8;
    color: #8A806B;
}

/* The "page" card holding the answer */
.page-card {
    background-color: #FBF7EE;
    border: 1px solid rgba(0,0,0,0.08);
    border-top: 3px solid var(--gold);
    border-radius: 4px;
    padding: 1.4rem 1.6rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    font-family: 'Lora', serif;
    color: var(--ink);
    line-height: 1.6;
    margin-top: 0.6rem;
}

/* Source bookmarks */
.bookmark-row { display: flex; flex-wrap: wrap; gap: 6px; margin: 0.4rem 0 1rem 0; }
.bookmark {
    background-color: var(--forest);
    color: var(--parchment);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    padding: 3px 10px;
    border-radius: 999px;
}

/* Empty state */
.empty-shelf {
    font-family: 'Lora', serif;
    font-style: italic;
    color: var(--plum);
    background-color: var(--parchment-deep);
    border: 1px dashed var(--gold);
    border-radius: 6px;
    padding: 1.4rem;
    text-align: center;
}
</style>
"""

# Six spines of varying height/color — evokes a shelf, not a stock icon.
SPINE_ROW_HTML = """
<div class="spine-row">
    <div class="spine" style="height:52px; background-color:#7A2E2E;"></div>
    <div class="spine" style="height:38px; background-color:#3B5941;"></div>
    <div class="spine" style="height:60px; background-color:#B08D57;"></div>
    <div class="spine" style="height:44px; background-color:#5B4A6F;"></div>
    <div class="spine" style="height:56px; background-color:#2A2118;"></div>
    <div class="spine" style="height:40px; background-color:#7A2E2E;"></div>
</div>
"""


@st.cache_resource(show_spinner=False)
def cached_embedding_model():
    return vector_representation.get_embedding_model()


@st.cache_resource(show_spinner=False)
def cached_chroma_client():
    return chroma_store.get_client()


def index_book(book_name: str, uploaded_files) -> None:
    files = [(f.name, f.getvalue()) for f in uploaded_files]
    docs = documents.load_documents(book_name, files)
    docs = preprocessing.clean_documents(docs)
    chunks = chunking.build_chunks(docs)

    cached_embedding_model()  # ensure the model is loaded/cached before use
    embeddings = vector_representation.embed_texts([c["search_text"] for c in chunks])

    book_key = documents.normalize_book_name(book_name)
    chroma_store.build_store_for_book(book_key, chunks, embeddings)


def answer_question(book_key: str, query: str) -> dict:
    cached_chroma_client()  # ensure the persistent client is initialized
    collection = chroma_store.get_or_create_collection(book_key)

    query_embedding = vector_representation.embed_query(query)
    context_package = retrieve_context.build_context_package(collection, query, query_embedding)

    prompt = rag.build_prompt(context_package, book_name=book_key)
    answer_text = rag.generate_answer(prompt)

    return {"context": context_package, "answer": answer_text}


def main() -> None:
    st.set_page_config(page_title="Book RAG Assistant", page_icon="📚", layout="centered")
    st.markdown(LIBRARY_CSS, unsafe_allow_html=True)

    st.markdown(SPINE_ROW_HTML, unsafe_allow_html=True)
    st.markdown(
        '<div class="library-header"><h1>The Reading Room</h1></div>'
        '<p class="library-subtitle">Ask your library anything — every answer, grounded in the text.</p>',
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown('<p class="eyebrow">Library Desk</p>', unsafe_allow_html=True)
        st.header("Manage Books")
        indexed_books = chroma_store.list_indexed_books()

        new_book_name = st.text_input("Book name (new or existing)")
        uploaded_files = st.file_uploader(
            "Upload .pdf or .txt file(s) for this book", accept_multiple_files=True
        )
        if st.button("Build / Update Index", disabled=not (new_book_name and uploaded_files)):
            with st.spinner(f"Shelving '{new_book_name}'..."):
                try:
                    index_book(new_book_name, uploaded_files)
                    st.success(f"'{new_book_name}' is on the shelf.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Indexing failed: {e}")

        st.divider()
        st.caption(
            f"📚 {len(indexed_books)} book(s) on the shelf" if indexed_books else "The shelf is empty."
        )

    if not indexed_books:
        st.markdown(
            '<div class="empty-shelf">The shelf is empty — add a book from the Library Desk '
            "in the sidebar to begin.</div>",
            unsafe_allow_html=True,
        )
        return

    st.markdown('<p class="eyebrow">Choose a Book</p>', unsafe_allow_html=True)
    book_key = st.selectbox("", indexed_books, label_visibility="collapsed")

    st.markdown('<p class="eyebrow">Your Question</p>', unsafe_allow_html=True)
    query = st.text_input("", placeholder="What does this book say about...", label_visibility="collapsed")

    if st.button("Ask", disabled=not query):
        with st.spinner("Turning the pages..."):
            try:
                result = answer_question(book_key, query)
            except Exception as e:
                st.error(f"Could not generate an answer: {e}")
                return

        sources = result["context"]["sources"]
        if sources:
            st.markdown('<p class="eyebrow">Sources</p>', unsafe_allow_html=True)
            tags = "".join(f'<span class="bookmark">🔖 {s}</span>' for s in sources)
            st.markdown(f'<div class="bookmark-row">{tags}</div>', unsafe_allow_html=True)

        st.markdown('<p class="eyebrow">Answer</p>', unsafe_allow_html=True)
        st.markdown(f'<div class="page-card">{result["answer"]}</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
