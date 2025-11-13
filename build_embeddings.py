# build_embeddings.py

from document_processor import load_pdf, split_sentences, chunk_sentences
from embedder import compute_embeddings_csv
from config import PDF_PATH

if __name__ == "__main__":
    print("[STEP 1] Loading PDF...")
    pages = load_pdf(PDF_PATH)

    print("[STEP 2] Splitting into sentences...")
    pages = split_sentences(pages)

    print("[STEP 3] Chunking...")
    chunks = chunk_sentences(pages)

    print("[STEP 4] Computing & saving embeddings to CSV...")
    compute_embeddings_csv(chunks)

    print("\n[✓] Embeddings generated successfully! Saved to embeddings/chunks.csv")
