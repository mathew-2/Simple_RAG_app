import os
import csv
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util

MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_CSV_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "embeddings", "chunks.csv")


_model = None

def get_model():
    """Lazy load the embedding model."""
    global _model
    if _model is None:
        print(f"[INFO] Loading model: {MODEL_NAME}")
        _model = SentenceTransformer(MODEL_NAME)
        # Force model to CPU
        _model = _model.to('cpu')
    return _model


def compute_embeddings_csv(chunks):
    """Compute embeddings and save to CSV file."""
    model = get_model()
    texts = [c["text"] for c in chunks]
    
    print(f"[INFO] Encoding {len(texts)} chunks...")
    # Force CPU and convert to tensor
    vectors = model.encode(
        texts, 
        convert_to_tensor=True, 
        show_progress_bar=True,
        device='cpu'  # Force CPU
    )
    
    # Convert to numpy for CSV storage
    vectors_np = vectors.cpu().numpy()

    os.makedirs("embeddings", exist_ok=True)

    with open(EMBED_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["chunk_id", "page_number", "text", "embedding"])

        for i, (chunk, emb) in enumerate(zip(chunks, vectors_np)):
            writer.writerow([
                i,
                chunk["page_number"],
                chunk["text"],
                ",".join(map(lambda x: f"{x:.8f}", emb))
            ])

    print(f"[INFO] Embeddings saved to {EMBED_CSV_PATH}")


def load_embeddings_csv():
    """Load chunks + embeddings from CSV and return as torch tensors on CPU."""

    # Print the path being checked
    print(f"[INFO] Looking for embeddings at: {EMBED_CSV_PATH}")

    if not os.path.exists(EMBED_CSV_PATH):
        raise FileNotFoundError(
            f"Embeddings file not found at {EMBED_CSV_PATH}. "
            "Please run the embedding generation script first."
        )
    
    chunks = []
    embeddings_list = []
    # Read CSV and make sure to handle spaces in embeddings so as to seperate the different values correctly

    with open(EMBED_CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            emb_str = row["embedding"].replace(" ", ",")
            emb = np.array([float(x) for x in emb_str.split(",")])

            chunks.append({
                "chunk_id": int(row["chunk_id"]),
                "page_number": int(row["page_number"]),
                "text": row["text"],
                "embedding": emb
            })
            embeddings_list.append(emb)

    # Convert to torch tensor and FORCE CPU because often loaded on GPU by default
    embeddings_tensor = torch.tensor(
        np.array(embeddings_list), 
        dtype=torch.float32,
        device='cpu'  # Explicitly set to CPU
    )
    
    print(f"[INFO] Loaded {len(chunks)} chunks from CSV")
    print(f"[INFO] Embeddings shape: {embeddings_tensor.shape}")
    print(f"[INFO] Embeddings device: {embeddings_tensor.device}")
    
    return chunks, embeddings_tensor