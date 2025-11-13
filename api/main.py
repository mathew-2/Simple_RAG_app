from fastapi import FastAPI, HTTPException
from retriever import search, pages_from_results
from generator import answer_question
from embedder import load_embeddings_csv

app = FastAPI(title="Boeing 737 Manual RAG API")

# Load vectors into memory at startup
try:
    chunks, embeddings_tensor = load_embeddings_csv()
    print(f"[INFO] Loaded {len(chunks)} chunks successfully")
except Exception as e:
    print(f"[ERROR] Failed to load embeddings: {e}")
    chunks = []
    embeddings_tensor = None


@app.get("/")
def root():
    return {
        "message": "Boeing 737 Manual RAG API (Ollama Backend)",
        "chunks_loaded": len(chunks),
        "endpoint": "/ask?query=YOUR_QUESTION",
        "model": "llama3.2"
    }


@app.get("/ask")
def ask(query: str, top_k: int = 5, model: str = "llama3.2"):
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    if not chunks or embeddings_tensor is None:
        raise HTTPException(
            status_code=503, 
            detail="Embeddings not loaded. Please generate embeddings first."
        )
    
    try:
        # Search for relevant chunks
        top_chunks = search(query, chunks, embeddings_tensor, top_k=top_k)
        pages = pages_from_results(top_chunks)
        
        # Generate answer - NOTE THE CORRECT ORDER: query first, then chunks
        answer = answer_question(query, top_chunks, model=model)

        return {
            "query": query,
            "answer": answer,
            "pages": pages,
            "num_chunks_used": len(top_chunks),
            "top_scores": [f"{chunk['score']:.4f}" for chunk in top_chunks],
            "model_used": model
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)