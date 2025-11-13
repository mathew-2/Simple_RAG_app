import fitz
from tqdm import tqdm
from spacy.lang.en import English
import re

nlp = English()
nlp.add_pipe("sentencizer")

def text_formatter(text):
    """Clean page text."""
    return text.replace("\n", " ").strip()

def load_pdf(path):
    """Extract text from each page of a PDF."""
    doc = fitz.open(path)
    pages = []

    for idx, page in tqdm(enumerate(doc), total=len(doc)):
        raw = text_formatter(page.get_text())
        pages.append({
            "page_number": idx + 1,
            "text": raw
        })
    return pages

def split_sentences(pages):
    """Split each page into sentences."""
    for item in pages:
        sents = list(nlp(item["text"]).sents)
        item["sentences"] = [str(s).strip() for s in sents]
    return pages

def chunk_sentences(pages, max_size=10):
    """Chunk sentences into fixed-size blocks."""
    chunks = []

    for item in pages:
        sents = item["sentences"]
        grouped = [sents[i:i+max_size] for i in range(0, len(sents), max_size)]

        for block in grouped:
            joined = " ".join(block)
            # Fix missing spaces between sentences
            joined = re.sub(r'\.([A-Z])', r'. \1', joined)

            chunks.append({
                "page_number": item["page_number"],
                "text": joined
            })

    return chunks
