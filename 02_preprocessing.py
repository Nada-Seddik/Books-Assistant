"""
02_preprocessing.py

Text cleanup applied to raw extracted text before chunking: strips
repeated running headers/footers, standalone page numbers, rejoins
words hyphenated across a line break, and strips raw glyph-code
artifacts that some PDFs produce.

Unchanged from the original project except for _strip_glyph_artifacts:
some PDFs (commonly ones with decorative heading fonts) lack a proper
ToUnicode map for certain characters, so pypdf falls back to emitting
raw glyph IDs like "/G53/G75/G6E" instead of the actual text ("Sun").
This is a PDF-encoding problem, not something chunking or retrieval can
work around, so it's cleaned up here, upstream of everything else.
"""

import re
from collections import Counter


def _strip_glyph_artifacts(text: str) -> str:
    """Remove runs of raw "/GXX" glyph codes left by undecodable PDF fonts.

    A single "/G4B" could theoretically be legitimate text, but three or
    more in a row is always this extraction artifact, never real prose.
    """
    return re.sub(r"(?:/G[0-9A-Fa-f]{2}){3,}", " ", text)


def _remove_repeated_lines(lines: list[str], min_repeats: int = 3) -> list[str]:
    short_lines = [line.strip() for line in lines if 0 < len(line.strip()) <= 60]
    counts = Counter(short_lines)
    repeated = {line for line, count in counts.items() if count >= min_repeats}
    return [line for line in lines if line.strip() not in repeated]


def _remove_standalone_page_numbers(lines: list[str]) -> list[str]:
    return [line for line in lines if not re.fullmatch(r"\s*\d{1,4}\s*", line)]


def _dehyphenate(text: str) -> str:
    return re.sub(r"(\w)-\n(\w)", r"\1\2", text)


def _normalize_whitespace(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_text(raw_text: str) -> str:
    """Run the full cleanup pipeline on one document's raw extracted text."""
    text = _strip_glyph_artifacts(raw_text)
    text = _dehyphenate(text)
    lines = text.split("\n")
    lines = _remove_repeated_lines(lines)
    lines = _remove_standalone_page_numbers(lines)
    text = "\n".join(lines)
    return _normalize_whitespace(text)


def clean_documents(documents: list[dict]) -> list[dict]:
    """Apply clean_text to every document's "text" field."""
    for doc in documents:
        doc["text"] = clean_text(doc["text"])
    return documents
