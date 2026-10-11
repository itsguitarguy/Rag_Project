"""Step 3: vector indexes. Both expect L2-normalised vectors (dot product == cosine)."""
import math

import numpy as np


class FlatIndex:
    """Exact search: one matrix-vector multiply over every stored vector."""

    def __init__(self, vecs):
        self.vecs = vecs

    def search(self, q, k):
        sims = self.vecs @ q
        return [(int(i), float(sims[i])) for i in np.argsort(-sims)[:k]]


class IVFIndex:
    """
    Approximate search (inverted file). k-means groups vectors into clusters; a query
    scans only the `nprobe` clusters with the closest centroids. Faster at scale,
    but may miss neighbours that live in other clusters.
    """

    def __init__(self, vecs, n_clusters=None, nprobe=2, iters=20, seed=0):
        rng = np.random.default_rng(seed)
        n = len(vecs)
        self.vecs, self.nprobe = vecs, nprobe
        k = n_clusters or max(1, int(math.sqrt(n)))
        cent = vecs[rng.choice(n, k, replace=False)].copy()
        for _ in range(iters):                              # spherical k-means
            assign = np.argmax(vecs @ cent.T, axis=1)
            for c in range(k):
                members = vecs[assign == c]
                if len(members):
                    m = members.mean(axis=0)
                    cent[c] = m / (np.linalg.norm(m) or 1.0)
        self.cent = cent
        assign = np.argmax(vecs @ cent.T, axis=1)
        self.lists = {c: np.where(assign == c)[0] for c in range(k)}

    def search(self, q, k):
        probe = np.argsort(-(self.cent @ q))[: self.nprobe]
        cand = np.concatenate([self.lists[c] for c in probe])
        if len(cand) == 0:
            return []
        sims = self.vecs[cand] @ q
        order = np.argsort(-sims)[:k]
        return [(int(cand[i]), float(sims[i])) for i in order]
