"""
Neural embedder: a pretrained sentence-transformer maps text -> dense vector, trained on
huge amounts of text, so "money back" and "refund" land close together.

Same interface as LSAEmbedder: fit(texts) / embed(text) / embed_many(texts) / min_score.
"""
import hashlib
import os

import numpy as np

from .config import CACHE_DIR, EMBED_BATCH, EMBED_MODEL, NEURAL_MIN_SCORE


class NeuralEmbedder:
    min_score = NEURAL_MIN_SCORE

    def __init__(self, model_name=EMBED_MODEL):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise SystemExit("Neural embedder needs: pip install sentence-transformers\n"
                                 "(or run with --lsa to use the offline LSA embedder)")
            print(f"Loading embedding model '{self.model_name}' (first run downloads it)...")
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _encode(self, texts):
        vecs = self.model.encode(texts, batch_size=EMBED_BATCH, normalize_embeddings=True,
                                 show_progress_bar=len(texts) > 50)
        return np.asarray(vecs, dtype=np.float32)   # unit length -> dot product == cosine

    def fit(self, texts):
        """Embed all chunks, with an on-disk cache keyed by (model, chunk texts)."""
        key = hashlib.sha256((self.model_name + "\x00" + "\x00".join(texts)).encode()).hexdigest()[:16]
        path = os.path.join(CACHE_DIR, f"{key}.npy")
        if os.path.exists(path):
            return np.load(path)
        vecs = self._encode(texts)
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.save(path, vecs)
        return vecs

    def embed(self, text):
        return self._encode([text])[0]

    def embed_many(self, texts):
        return self._encode(texts)
