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
    if not getattr(rag, "OPENROUTER_API_KEY", ""):
        setattr(rag, "OPENROUTER_API_KEY", st.secrets.get("OPENROUTER_API_KEY", ""))
        setattr(
            rag,
            "OPENROUTER_MODEL",
            st.secrets.get("OPENROUTER_MODEL", getattr(rag, "OPENROUTER_MODEL", "")),
        )
except Exception:
    pass


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
    st.set_page_config(page_title="Book RAG Assistant", page_icon="📚")
    st.title("📚 Book RAG Assistant")

    with st.sidebar:
        st.header("Manage Books")
        indexed_books = chroma_store.list_indexed_books()

        new_book_name = st.text_input("Book name (new or existing)")
        uploaded_files = st.file_uploader(
            "Upload .pdf or .txt file(s) for this book", accept_multiple_files=True
        )
        if st.button("Build / Update Index", disabled=not (new_book_name and uploaded_files)):
            with st.spinner(f"Indexing '{new_book_name}'..."):
                try:
                    index_book(new_book_name, uploaded_files)
                    st.success(f"'{new_book_name}' indexed successfully.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Indexing failed: {e}")

        st.divider()
        st.caption(f"{len(indexed_books)} book(s) indexed" if indexed_books else "No books indexed yet.")

    if not indexed_books:
        st.info("Upload and index a book from the sidebar to get started.")
        return

    book_key = st.selectbox("Ask a question about:", indexed_books)
    query = st.text_input("Your question")

    if st.button("Ask", disabled=not query):
        with st.spinner("Retrieving context and generating an answer..."):
            try:
                result = answer_question(book_key, query)
            except Exception as e:
                st.error(f"Could not generate an answer: {e}")
                return

        sources = result["context"]["sources"]
        st.markdown(f"**Sources used:** {', '.join(sources) if sources else '(none found)'}")
        st.markdown("**Answer:**")
        st.write(result["answer"])


if __name__ == "__main__":
    main()
