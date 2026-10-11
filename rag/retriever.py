"""Step 4: hybrid retrieval = BM25 (keywords) + dense vector search, merged with RRF."""
from collections import Counter

import numpy as np

from .bm25 import BM25
from .config import EMBEDDER, IVF_NPROBE, TOP_K
from .embedder import LSAEmbedder
from .index import FlatIndex, IVFIndex


def make_embedder(kind):
    if kind == "lsa":
        return LSAEmbedder()
    from .neural_embedder import NeuralEmbedder   # imported lazily: heavy dependency
    return NeuralEmbedder()


def rrf(rankings, k=60):
    """Reciprocal Rank Fusion: score(doc) = sum of 1/(k + rank) over each ranking."""
    scores = Counter()
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            scores[idx] += 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])


class Retriever:
    def __init__(self, chunks, use_ivf=False, embedder=EMBEDDER):
        self.chunks = chunks
        texts = [c["text"] for c in chunks]
        self.bm25 = BM25(texts)
        self.embedder = make_embedder(embedder)
        vecs = self.embedder.fit(texts)
        self.index = IVFIndex(vecs, nprobe=IVF_NPROBE) if use_ivf else FlatIndex(vecs)

    def retrieve(self, question, k=TOP_K, debug=False):
        lex = self.bm25.scores(question)
        dense = self.index.search(self.embedder.embed(question), k=len(self.chunks))
        top_dense = dense[0][1] if dense else 0.0

        # Bail out only when there is nothing in common with the document: no keyword
        # match AND no semantic match above the embedder's threshold. (LSA: 0.0, so this
        # is the `<= 0` check; neural: NEURAL_MIN_SCORE.) Otherwise return the best
        # chunks and let the LLM + quote verification decide.
        if lex.max() <= 0 and top_dense <= self.embedder.min_score:
            return []

        lex_rank = [int(i) for i in np.argsort(-lex) if lex[i] > 0][:k * 2]
        dense_rank = [i for i, _ in dense[:k * 2]]
        fused = rrf([lex_rank, dense_rank])[:k]

        if debug:
            dmap = dict(dense)
            for i, s in fused:
                print(f"  chunk {i:3d} | rrf={s:.4f} bm25={lex[i]:.2f} "
                      f"cos={dmap.get(i, 0):.3f} | {self.chunks[i]['location']}")
        return [self.chunks[i] for i, _ in fused]
