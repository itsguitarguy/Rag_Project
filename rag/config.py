"""Central settings. Tune the pipeline here without touching the logic."""

# Answer returned whenever the document doesn't contain the information
NOT_FOUND = "This information is not available in the provided document."

# LLM: "gemini" or "claude"
LLM_PROVIDER = "gemini"
GEMINI_MODEL = "gemini-3.5-flash"   # if "model not found", try "gemini-3.5-flash"
MODEL = "claude-sonnet-5-5"         # used only when LLM_PROVIDER = "claude"
MAX_TOKENS = 1000

# Chunking
CHUNK_MAX_WORDS = 120      # max words per chunk (try 150-200 for long, dense PDFs)
CHUNK_OVERLAP_SENTS = 1    # sentences repeated between neighbouring chunks
TEXT_BLOCK_LINES = 40      # for .txt/.md files, location is reported as line ranges

# Retrieval
TOP_K = 5                  # chunks sent to the LLM
IVF_NPROBE = 3             # clusters scanned by the IVF index (only used with --ivf)

# Embedder: "neural" (pretrained, understands meaning/synonyms) or "lsa" (offline, keyword-ish)
EMBEDDER = "neural"
EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"   # ~420 MB, strong general-purpose model
# Alternatives:
#   "sentence-transformers/all-MiniLM-L6-v2"  (~90 MB, fastest, good quality)
#   "BAAI/bge-large-en-v1.5"                   (~1.3 GB, largest/most accurate; works best with its
#                                               query prefix "Represent this sentence for searching
#                                               relevant passages: " added to questions)
EMBED_BATCH = 32
CACHE_DIR = ".rag_cache"   # chunk embeddings are cached here so re-runs are instant
NEURAL_MIN_SCORE = 0.25    # cosine below this (and no keyword match) = "nothing relevant"
EMBED_DIM = 64             # size of the LSA embedding vectors (LSA only)
