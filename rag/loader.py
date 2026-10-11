"""Step 0: read a document into (location_label, text) pieces."""
from .config import TEXT_BLOCK_LINES


def load_document(path):
    """PDFs -> one piece per page. Text/markdown -> blocks of TEXT_BLOCK_LINES lines."""
    if path.lower().endswith(".pdf"):
        from pypdf import PdfReader
        return [(f"Page {i}", page.extract_text() or "")
                for i, page in enumerate(PdfReader(path).pages, 1)]

    with open(path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    return [(f"Lines {i + 1}-{min(i + TEXT_BLOCK_LINES, len(lines))}",
             "\n".join(lines[i:i + TEXT_BLOCK_LINES]))
            for i in range(0, len(lines), TEXT_BLOCK_LINES)]
