"""Step 1: chunking. Pack whole sentences up to a word limit, with sentence overlap."""
import re

from .config import CHUNK_MAX_WORDS, CHUNK_OVERLAP_SENTS


def split_sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n{2,}", text) if s.strip()]


def chunk_pieces(pieces, max_words=CHUNK_MAX_WORDS, overlap_sents=CHUNK_OVERLAP_SENTS):
    """Returns a list of {"location": ..., "text": ...}. Chunks never cross page boundaries."""
    chunks = []
    for location, text in pieces:
        cur, n = [], 0
        for s in split_sentences(text):
            w = len(s.split())
            if cur and n + w > max_words:
                chunks.append({"location": location, "text": " ".join(cur)})
                cur = cur[-overlap_sents:] if overlap_sents else []
                n = sum(len(x.split()) for x in cur)
            cur.append(s)
            n += w
        if cur:
            chunks.append({"location": location, "text": " ".join(cur)})
    return chunks
