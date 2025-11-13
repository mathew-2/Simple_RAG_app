import requests
import json

def query_ollama(prompt, model="llama3.2"):
    """Query Ollama API"""
    url = "http://localhost:11434/api/generate"
    
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,  # Low temperature for factual answers
            "top_p": 0.9,
            "num_predict": 512
        }
    }
    
    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()['response']
    except requests.exceptions.ConnectionError:
        return "Error: Ollama is not running. Start it with 'ollama serve'"
    except requests.exceptions.Timeout:
        return "Error: Request timed out. The model might be taking too long to respond."
    except Exception as e:
        return f"Error: {str(e)}"


def answer_question(query, retrieved_chunks, model="llama3.2"):
    """
    Answer question based on retrieved chunks using Ollama.
    
    Args:
        query: The user's question
        retrieved_chunks: List of dicts with 'text', 'page_number', 'score' keys
        model: Ollama model to use
    """
    
    # Validate input
    if not retrieved_chunks:
        return "I couldn't find any relevant information in the manual to answer your question."
    
    # Extract text from chunks
    if isinstance(retrieved_chunks, list):
        # Handle list of chunk dictionaries
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            if isinstance(chunk, dict):
                text = chunk.get('text', '').strip()
                page = chunk.get('page_number', '?')
                score = chunk.get('score', 0)
                
                if text:
                    context_parts.append(
                        f"[Excerpt {i} - Page {page} - Relevance: {score:.2f}]\n{text}"
                    )
            else:
                context_parts.append(str(chunk))
        
        context_text = "\n\n---\n\n".join(context_parts)
    else:
        context_text = str(retrieved_chunks)
    
    # Verify we have actual content
    if not context_text or len(context_text.strip()) < 50:
        return "Error: No valid context retrieved. Please check your retrieval system."
    
    # DEBUG: Print what we're sending to the model
    print("\n" + "=" * 70)
    print(f"[DEBUG] Query: {query}")
    print(f"[DEBUG] Context length: {len(context_text)} characters")
    print(f"[DEBUG] Number of chunks: {len(retrieved_chunks)}")
    print(f"[DEBUG] Context preview (first 300 chars):")
    print(context_text[:300] + "...")
    print("=" * 70 + "\n")
    
    # Build prompt
    prompt = f"""You are a technical assistant for Boeing 737 aircraft operations manuals.

Below are relevant excerpts from the Boeing 737 Operations Manual. Use ONLY this information to answer the question.

===== MANUAL EXCERPTS =====
{context_text}
===== END OF EXCERPTS =====

Question: {query}

Instructions:
- Answer based ONLY on the excerpts above
- Be specific and cite page numbers when mentioned
- If the excerpts mention specific switch positions or procedures, include them
- If the answer is not in the excerpts, clearly state that
- Keep your answer concise and technical

Answer:"""
    
    print(f"[INFO] Sending request to Ollama (model: {model})...")
    response = query_ollama(prompt, model)
    print(f"[INFO] Response received from Ollama\n")
    
    return response