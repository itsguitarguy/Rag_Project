"""
Document Q&A agent: answers only from your document, with verified citations.

    python main.py mydoc.pdf                    # full agent (needs ANTHROPIC_API_KEY)
    python main.py mydoc.pdf --no-llm --debug   # retrieval only, shows scores
    python main.py mydoc.pdf --ivf              # use the approximate IVF index
    python main.py mydoc.pdf --lsa              # offline LSA embedder instead of the neural one
"""
import sys
from dotenv import load_dotenv
load_dotenv()
from rag import NOT_FOUND, Retriever, ask_llm, chunk_pieces, load_document
from rag.config import EMBEDDER


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a for a in sys.argv[1:] if a.startswith("--")}
    if not args:
        sys.exit(__doc__)

    chunks = chunk_pieces(load_document(args[0]))
    if not chunks:
        sys.exit("No text could be extracted (scanned PDF? try OCR first).")
    retriever = Retriever(chunks, use_ivf="--ivf" in flags,
                          embedder="lsa" if "--lsa" in flags else EMBEDDER)
    print(f"Indexed {len(chunks)} chunks. Ask questions ('exit' to quit).\n")

    while True:
        q = input("You: ").strip()
        if q.lower() in {"exit", "quit"}:
            break
        if not q:
            continue
        hits = retriever.retrieve(q, debug="--debug" in flags)
        if not hits:
            print(f"\nAgent: {NOT_FOUND}\n")
        elif "--no-llm" in flags:
            for c in hits[:2]:
                print(f"\n[{c['location']}] {c['text'][:300]}...")
            print()
        else:
            print(f"\nAgent: {ask_llm(q, hits)}\n")


if __name__ == "__main__":
    main()
