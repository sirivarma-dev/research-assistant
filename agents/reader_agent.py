"""
READER AGENT
Takes an uploaded document (PDF, DOCX, TXT, PPTX, or a scanned image), extracts
its text, splits it into chunks, and stores those chunks in the vector store so
the Q&A agent can use them later.
"""
from utils.document_utils import extract_text
from utils.pdf_utils import chunk_text, classify_length


class ReaderAgent:
    def __init__(self, store):
        self.store = store  # a shared VectorStore instance

    def read_pdf(self, pdf_path, source_name):
        # extract_text handles every supported format and raises a friendly
        # ValueError for unsupported / empty / scanned-without-OCR files.
        text, num_pages = extract_text(pdf_path)
        chunks = chunk_text(text)
        self.store.add_chunks(chunks, source_name)
        # Keep the full text too, for whole-document questions (references, etc.)
        self.store.set_full_text(text)
        # Work out how long the paper is (Short / Medium / Long) for the UI.
        stats = classify_length(text, num_pages)
        return len(chunks), text, stats
