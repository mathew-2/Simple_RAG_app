import torch
from sentence_transformers import SentenceTransformer, util
from time import perf_counter as timer

MODEL_NAME = "all-MiniLM-L6-v2"

_model = None

def get_model():
    """Lazy load the embedding model on CPU."""
    global _model
    if _model is None:
        print(f"[INFO] Loading model: {MODEL_NAME}")
        _model = SentenceTransformer(MODEL_NAME)
        # Force model to CPU
        _model = _model.to('cpu')
    return _model


def search(query, chunks, embeddings_tensor, top_k=5):
    """
    Perform semantic search using dot product.
    All operations forced to CPU to avoid device mismatch.
    
    Args:
        query: Search query string
        chunks: List of chunk dictionaries
        embeddings_tensor: Torch tensor of all embeddings (on CPU)
        top_k: Number of results to return
    """
    if not chunks:
        return []
    
    model = get_model()
    
    start_time = timer()
    
    query_embedding = model.encode(
        query, 
        convert_to_tensor=True,
        device='cpu'  
    )
    
    # Ensure embeddings_tensor is on CPU
    if embeddings_tensor.device.type != 'cpu':
        embeddings_tensor = embeddings_tensor.cpu()
    
    # Ensure query_embedding is on CPU
    if query_embedding.device.type != 'cpu':
        query_embedding = query_embedding.cpu()
    
    # Calculate dot product scores
    dot_scores = util.dot_score(a=query_embedding, b=embeddings_tensor)[0]
    
    end_time = timer()
    
    print(f"[INFO] Search took {end_time - start_time:.5f} seconds for {len(chunks)} chunks")
    
    # Get top k results
    top_results = torch.topk(dot_scores, k=min(top_k, len(chunks)))
    
    # Extract the matching chunks
    results = []
    for score, idx in zip(top_results.values, top_results.indices):
        chunk = chunks[idx.item()].copy()
        chunk['score'] = score.item()
        results.append(chunk)
    
    print(f"[INFO] Top score: {results[0]['score']:.4f}")
    
    return results


def pages_from_results(results):
    """Return sorted unique page numbers."""
    return sorted(list(set([r["page_number"] for r in results])))