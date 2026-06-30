"""
Helpers for reading a PDF and splitting its text into smaller chunks.
Smaller chunks let the assistant find the exact part of the paper that
answers a question.
"""
from pypdf import PdfReader


def extract_text_from_pdf(pdf_path):
    """
    Read every page of a PDF and return (text, number_of_pages).
    """
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text() or ""
        text += page_text + "\n"
    return text.strip(), len(reader.pages)


def classify_length(text, num_pages=0):
    """
    Decide whether a paper is Short / Medium / Long based on its word count,
    and return a small stats dict the UI can display.
    """
    words = len(text.split())
    if words < 3000:
        label, emoji = "Short", "🟢"
    elif words <= 10000:
        label, emoji = "Medium", "🟡"
    else:
        label, emoji = "Long", "🔴"

    read_min = max(1, round(words / 200))  # ~200 words per minute reading speed
    return {
        "label": label,
        "emoji": emoji,
        "words": words,
        "pages": num_pages,
        "read_min": read_min,
    }


def chunk_text(text, chunk_size=1000, overlap=200):
    """
    Break a long string into overlapping chunks.
    Overlap keeps sentences from being cut awkwardly between chunks.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks
