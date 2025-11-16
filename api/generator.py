import os
import google.generativeai as genai

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    print("[INFO] Gemini API configured successfully")
else:
    print("[WARNING] GEMINI_API_KEY not set. Gemini model calls will fail.")

_model = None

def get_gemini_model(model_name="gemini-1.5-flash"):
    #load the gemini model
    global _model
    if _model is None:
        print(f"[INFO] Loading Gemini model: {model_name}")
        _model = genai.GenerativeModel(model_name)
    return _model


def answer_question(query, retrieved_chunks, model="gemini-1.5-flash"):
    """
    Answer question based on retrieved chunks (max chunks would have equivalent
      context length of 10 sentences) using Gemini.
    
    Args:
        query: The user's question
        retrieved_chunks: List of dicts with 'text', 'page_number', 'score' keys
        model: Gemini model to use
    """
    
    # Validate input
    if not retrieved_chunks:
        return "I couldn't find any relevant information in the manual to answer your question."
    
    # Extract text from chunks
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
    
    context_text = "\n\n---\n\n".join(context_parts)
    
    # Verify we have actual content
    if not context_text or len(context_text.strip()) < 50:
        return "Error: No valid context retrieved. Please check your retrieval system."
    
    # DEBUG: Print what we're sending to the model
    print("\n" + "=" * 70)
    print(f"[DEBUG] Query: {query}")
    print(f"[DEBUG] Context length: {len(context_text)} characters")
    print(f"[DEBUG] Number of chunks: {len(retrieved_chunks)}")
    print(f"[DEBUG] Using Gemini API with model: {model}")
    print(f"[DEBUG] Context preview (first 300 chars):")
    print(context_text[:300] + "...")
    print("=" * 70 + "\n")
    
    # Build prompt(this is the system prompt that guides the model's behavior)
    # (note that gemini is more strict on safety so sometimes technical content gets blocked)
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
    
    try:
        print(f"[INFO] Sending request to Gemini API (model: {model})...")
        
        # Get Gemini model
        gemini_model = get_gemini_model(model)
        
        # CORRECT safety settings format - using HarmCategory and HarmBlockThreshold
        # here changing the settings so it is less strict and allows more content through
        # Alot of times technical content gets blocked by safety filters ,So have to run it with relaxed settings
        # Also run it multiple times if blocked 
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE"
            }
        ]
        
        # Generate response with relaxed safety settings
        response = gemini_model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,
                top_p=0.9,
                max_output_tokens=2000,
            ),
            safety_settings=safety_settings
        )
        
        # Handle different response scenarios
        if not response.candidates:
            print("[ERROR] No candidates in response")
            return "Error: No response generated. The content may have been filtered."
        
        candidate = response.candidates[0]
        
        # Check finish reason
        finish_reason = candidate.finish_reason
        print(f"[DEBUG] Finish reason: {finish_reason}")
        
        # Finish reason values:
        # 1 = STOP (natural completion) 
        # 2 = MAX_TOKENS (hit token limit)
        # 3 = SAFETY (blocked by safety)
        # 4 = RECITATION (blocked by recitation)
        # 5 = OTHER
        
        if finish_reason == 3:  # SAFETY
            print("[WARNING] Response blocked by safety filters")
            if hasattr(candidate, 'safety_ratings'):
                print(f"[DEBUG] Safety ratings: {candidate.safety_ratings}")
            return "The response was blocked by content safety filters. This appears to be a false positive for technical aviation content. Please try rephrasing your question."
        
        if finish_reason == 4:  # RECITATION
            print("[WARNING] Response blocked due to recitation")
            return "The response was blocked due to potential copyright concerns."
        
        if finish_reason == 2:  # MAX_TOKENS
            print("[WARNING] Response truncated due to token limit")
        
        # Try to get the text
        if hasattr(response, 'text') and response.text:
            print(f"[INFO] Response received from Gemini ({len(response.text)} chars)\n")
            return response.text.strip()
        elif candidate.content and candidate.content.parts:
            # Manually extract text from parts
            text_parts = []
            for part in candidate.content.parts:
                if hasattr(part, 'text'):
                    text_parts.append(part.text)
            
            if text_parts:
                result = ''.join(text_parts).strip()
                print(f"[INFO] Response extracted from parts ({len(result)} chars)\n")
                return result
        
        print("[ERROR] No text in response")
        print(f"[DEBUG] Response object: {response}")
        return "Error: Unable to extract text from response. Please try again."
        
    except Exception as e:
        error_msg = str(e)
        print(f"[ERROR] Gemini API error: {error_msg}")
        
        # Provide helpful error messages
        if "API_KEY" in error_msg.upper():
            return "Error: Invalid API key configuration."
        elif "QUOTA" in error_msg.upper() or "RATE_LIMIT" in error_msg.upper():
            return "Error: API quota exceeded. Please try again later."
        elif "INVALID_ARGUMENT" in error_msg:
            return f"Error: Invalid request - {error_msg}"
        else:
            return f"Error generating answer: {error_msg}"