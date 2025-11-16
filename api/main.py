from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# CRITICAL: Load .env BEFORE any other imports(This makes sure the gemini api key and paths are set and configured properly)
load_dotenv()

from api.retriever import search, pages_from_results
from api.generator import answer_question
from api.embedder import load_embeddings_csv
import os
import sys

# Add parent directory to path (this is where the .env file is located)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = FastAPI(title="Boeing 737 Manual RAG API")


# Load embedding vectors at the beginning
try:
    chunks, embeddings_tensor = load_embeddings_csv()
    print(f"[INFO] Loaded {len(chunks)} chunks successfully")
except Exception as e:
    print(f"[ERROR] Failed to load embeddings: {e}")
    chunks = []
    embeddings_tensor = None


@app.on_event("startup")
async def startup_event():
    """Load embeddings on startup"""
    global chunks, embeddings_tensor
    try:
        print("[INFO] Loading embeddings...")
        chunks, embeddings_tensor = load_embeddings_csv()
        print(f"[INFO] Successfully loaded {len(chunks)} chunks")
    except Exception as e:
        print(f"[ERROR] Failed to load embeddings: {e}")


# this is just to check if the api is online

@app.get("/")
def root():
    """API information and health check"""
    return {
        "name": "Boeing 737 Manual RAG API",
        "version": "1.0.0",
        "status": "online",
        "llm": "Google Gemini 1.5 Flash",
        "embeddings_loaded": len(chunks) > 0,
        "total_chunks": len(chunks),
        "endpoints": {
            "ask": "/ask?query=YOUR_QUESTION",
            "health": "/health",
            "docs": "/docs"
        },
        "example": "/ask?query=What is the hydraulic system?",
        "github": "https://github.com/mathew-2/Simple_RAG_app"
    }

@app.get("/health")
def health():
    """Health check endpoint"""
    gemini_configured = os.getenv("GEMINI_API_KEY") is not None
    
    return {
        "status": "healthy" if (len(chunks) > 0 and gemini_configured) else "degraded",
        "embeddings_loaded": len(chunks) > 0,
        "total_chunks": len(chunks),
        "gemini_configured": gemini_configured,
        "llm": "Gemini 1.5 Flash"
    }

# here is where the main logic happens and where we pose the user query
@app.get("/ask")
def ask(query: str, top_k: int = 5, model: str = "gemini-2.5-flash"):
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    if not chunks or embeddings_tensor is None:
        raise HTTPException(
            status_code=503, 
            detail="Embeddings not loaded. Please generate embeddings first."
        )
    
    # Check Gemini API key
    if not os.getenv("GEMINI_API_KEY"):
        raise HTTPException(
            status_code=503,
            detail="Service unavailable: Gemini API key not configured."
        )
    
    try:
        # Search for relevant chunks
        top_chunks = search(query, chunks, embeddings_tensor, top_k=top_k)
        pages = pages_from_results(top_chunks)

        if not top_chunks:
            return {
                "query": query,
                "answer": "I couldn't find any relevant information in the manual to answer your question.",
                "pages": [],
                "num_chunks_used": 0,
                "top_scores": [],
                "model_used": model
            }
        
        print(f"[INFO] Top chunk scores: {[chunk['score'] for chunk in top_chunks]}") 

        
        # Generate answer - NOTE THE CORRECT ORDER: query first, then chunks
        answer = answer_question(query, top_chunks, model=model)

        return {
            "query": query,
            "answer": answer,
            "pages": pages,
            "num_chunks_used": len(top_chunks),
            "top_scores": [f"{chunk['score']:.4f}" for chunk in top_chunks],#it is based on this the answer is given 
            "model_used": model
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)