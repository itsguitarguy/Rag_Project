"""
Step 2: embeddings. text -> dense unit vector.

LSA = TF-IDF matrix X (chunks x vocab), then SVD  X = U S Vt.
Projecting onto the top-k right singular vectors maps text into a k-dim
"concept space". To use a neural embedder instead, write a class with the same
fit(texts) / embed(text) / embed_many(texts) interface and swap it in retriever.py.
"""
import math
from collections import Counter

import numpy as np

from .config import EMBED_DIM
from .text_utils import tokenize


class LSAEmbedder:
    min_score = 0.0   # LSA only bails out when the query shares no words with the document

    def __init__(self, dim=EMBED_DIM):
        self.dim = dim

    def fit(self, texts):
        docs = [tokenize(t) for t in texts]
        df = Counter()
        for d in docs:
            df.update(set(d))
        words = sorted(df)
        self.vocab = {w: i for i, w in enumerate(words)}
        N = len(docs)
        self.idf = np.array([math.log((1 + N) / (1 + df[w])) + 1 for w in words])
        X = np.vstack([self._tfidf(d) for d in docs])
        _, S, Vt = np.linalg.svd(X, full_matrices=False)
        k = min(self.dim, len(S))
        self.V = Vt[:k].T                      # (vocab x k) projection matrix
        return self.embed_many(texts)

    def _tfidf(self, tokens):
        v = np.zeros(len(self.vocab))
        for w, c in Counter(tokens).items():
            if w in self.vocab:
                v[self.vocab[w]] = (1 + math.log(c)) * self.idf[self.vocab[w]]
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed(self, text):
        z = self._tfidf(tokenize(text)) @ self.V
        n = np.linalg.norm(z)
        return z / n if n else z               # unit length -> dot product == cosine

    def embed_many(self, texts):
        return np.vstack([self.embed(t) for t in texts])
