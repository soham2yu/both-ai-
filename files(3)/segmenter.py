"""
Sentence segmentation for student answers.

Why not just split on punctuation/newlines: OCR output (Tesseract or a
vision model) hard-wraps lines mid-sentence, breaks words across lines with
hyphens, and handwriting often has sloppy or missing punctuation. Splitting
on every newline would cut sentences in half and wreck the rubric matching
that runs on top of this.

Steps:
  1. Repair layout: de-hyphenate line-wrapped words, and join lines that are
     just wraps. A newline is KEPT as a boundary when it precedes a list
     marker ("1.", "-", "•") or is a blank line (paragraph break).
  2. Sentence-split each piece with spaCy's parser (copes with missing
     punctuation better than a regex); fall back to a punctuation regex if
     the spaCy model isn't installed.
"""

import re
from functools import lru_cache

import spacy

import config

_NLP = None
_NLP_LOADED = False


def get_nlp():
    """Lazily load spaCy once; returns None if the model isn't installed."""
    global _NLP, _NLP_LOADED
    if not _NLP_LOADED:
        _NLP_LOADED = True
        try:
            _NLP = spacy.load(config.SPACY_MODEL, disable=["ner", "lemmatizer"])
        except OSError:
            _NLP = None
    return _NLP


_LIST_MARKER = r"(?:[-•*]|\d{1,2}[.)]|[a-z][.)])"


def _repair_layout(text: str) -> list:
    """Returns a list of 'blocks' — paragraphs / list items whose internal
    line wraps have been joined into single lines."""
    text = text.replace("\r", "")
    text = re.sub(r"(?<=[A-Za-z])-\n(?=[a-z])", "", text)             # de-hyphenate wrapped words
    text = re.sub(r"\n[ \t]*\n+", "\x00", text)                          # blank line = paragraph break
    text = re.sub(rf"\n(?=[ \t]*{_LIST_MARKER}\s)", "\x00", text)      # newline before a list item = boundary
    text = re.sub(r"[ \t]*\n[ \t]*", " ", text)                          # any other newline is just a wrap
    blocks = [re.sub(r"\s+", " ", b).strip() for b in text.split("\x00")]
    # The list marker itself isn't content, and spaCy would otherwise emit
    # "1." as its own one-token sentence.
    blocks = [re.sub(rf"^{_LIST_MARKER}\s+", "", b) for b in blocks]
    return [b for b in blocks if b]


def _regex_split(block: str) -> list:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])", block)
    return [p.strip() for p in parts if p.strip()]


@lru_cache(maxsize=256)
def _segment_cached(text: str) -> tuple:
    nlp = get_nlp()
    sentences = []
    for block in _repair_layout(text):
        if nlp is not None:
            sentences.extend(s.text.strip() for s in nlp(block).sents if s.text.strip())
        else:
            sentences.extend(_regex_split(block))
    return tuple(sentences)


def segment_sentences(text: str) -> list:
    """Public entry point. Returns a fresh list each call (cached
    internally, since the matcher asks for the same text repeatedly)."""
    if not text or not text.strip():
        return []
    return list(_segment_cached(text))


if __name__ == "__main__":
    sample = ("Newtons third law states that for every action there is an\n"
              "equal and opposite reaction. Forces act on two dif-\nferent bodies\n\n"
              "Examples:\n1. A rocket expels gas\n2. Walking pushes the ground back")
    for i, s in enumerate(segment_sentences(sample), 1):
        print(i, repr(s))
