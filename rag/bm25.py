"""Lexical ranking: BM25 = idf * saturating term frequency * length normalisation."""
import math
from collections import Counter

import numpy as np

from .text_utils import tokenize


class BM25:
    def __init__(self, texts, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(tokenize(t)) for t in texts]
        self.len = np.array([sum(d.values()) for d in self.tf], dtype=float)
        self.avg = self.len.mean() or 1.0
        df = Counter()
        for d in self.tf:
            df.update(d.keys())
        N = len(texts)
        self.idf = {w: math.log(1 + (N - n + 0.5) / (n + 0.5)) for w, n in df.items()}

    def scores(self, query):
        out = np.zeros(len(self.tf))
        for i, d in enumerate(self.tf):
            for w in tokenize(query):
                if w in d:
                    f = d[w]
                    norm = f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg)
                    out[i] += self.idf[w] * f * (self.k1 + 1) / norm
        return out
