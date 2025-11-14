# Boeing 737 Manual RAG API 

A Retrieval-Augmented Generation (RAG) service for answering questions about Boeing 737 technical manuals using Google Gemini and FastAPI.

The system extracts text from the manual, chunks it semantically, embeds the chunks, retrieves the most relevant parts, and uses Google Gemini to generate accurate, grounded answers.

---


## Repository Structure

```
simple_rag/
├── api/
│   ├── __init__.py
│   ├── main.py
│   ├── embedder.py
│   ├── retriever.py
│   ├── generator.py
│   ├── config.py
│   └── document_processor.py
├── embeddings/
│   └── chunks.csv          
├── data/
│   └── manual.pdf          
├── modal_ollama.py
├── build_embeddings.py
├── test_csv.py
├── requirements.txt
├── .env
├── .gitignore
├── README.md
└── .env                    
```

---

## Setup

### Prerequisites

- Python 3.8+
- Google Gemini API key


### 1. Install dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Configure environment variables

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Add your Gemini API key to `.env`:
   ```
   GEMINI_API_KEY=your_api_key_here
   ```

### 3. Prepare your manual

Place your Boeing 737 manual PDF in the `data/` directory:
```
data/manual.pdf
```

### 4. Generate embeddings (first-time setup)

Run the following Python script to process the manual and generate embeddings:

```python
from document_processor import load_pdf, split_sentences, chunk_sentences
from embedder import compute_embeddings_csv
from config import PDF_PATH

# Process the PDF
pages = load_pdf(PDF_PATH)
pages = split_sentences(pages)
chunks = chunk_sentences(pages)

# Generate embeddings
compute_embeddings_csv(chunks)
```

This generates `embeddings/chunks.csv` containing all embedded text chunks.

---

##  API Usage

### Available Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information and health check |
| `/health` | GET | Service health status |
| `/ask` | GET | Ask a question about the manual |
| `/docs` | GET | Interactive API documentation |

---

### Query the API

#### Using cURL

**Basic query:**
```bash
curl "http://localhost:8000/ask?query=What+is+the+hydraulic+system"
```

**With custom parameters:**
```bash
curl "http://localhost:8000/ask?query=What+is+the+autopilot+system&top_k=10"
```


#### Using Postman

1. **Open Postman** and create a new request

2. **Set request type to GET**

3. **Enter URL:**
```
   http://localhost:8000/ask
```

4. **Add Query Parameters:**
   - Click on **"Params"** tab
   - Add parameters:

   | KEY | VALUE |
   |-----|-------|
   | query | What is the hydraulic system? |
   | top_k | 5 |

5. **Click "Send"**



## Usage

### Start the API server

```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`
So one can run the get query on postman to check and see if the query is giving the right value (json value or not)

### API Endpoints

#### `GET /query`

Query the Boeing 737 manual with a question.

**Request body:**
```json
{
  "question": "What is the maximum takeoff weight?"
}
```

**Response:**
```json
{
  "query": "What is the hydraulic system?",
  "answer": "Based on the manual excerpts from pages 15 and 23, the hydraulic system provides power to...",
  "pages": [15, 23, 34],
  "num_chunks_used": 5,
  "top_scores": ["0.7234", "0.6891", "0.6542", "0.6234", "0.6012"],
  "model": "gemini-1.5-flash"
}
```

### Example using Python

```python
import requests

response = requests.post(
    "http://localhost:8000/query",
    json={"question": "What is the maximum takeoff weight?"}
)

data = response.json()
print(f"Answer: {data['answer']}")
print(f"Pages: {data['pages']}")
```

---

## Configuration

Edit `config.py` to customize:

- **PDF_PATH**: Path to your manual
- **CHUNK_SIZE**: Number of sentences per chunk
- **TOP_K**: Number of chunks to retrieve
- **MODEL_NAME**: Gemini model to use

---

## How It Works

1. **Document Processing**: PDF is loaded and split into sentences using spaCy
2. **Chunking**: Sentences are grouped into semantic chunks
3. **Embedding**: Each chunk is embedded using `sentence-transformers/all-MiniLM-L6-v2`
4. **Storage**: Embeddings are saved to CSV for easy versioning
5. **Query Processing**: User questions are embedded and matched against chunks
6. **Retrieval**: Top-K most relevant chunks are retrieved
7. **Generation**: Gemini generates an answer based on retrieved context
8. **Response**: Answer and source pages are returned

---

## Dependencies

- `fastapi` – Web framework
- `uvicorn` – ASGI server
- `pdfplumber` – PDF text extraction
- `spacy` – Sentence tokenization
- `sentence-transformers` – Embeddings
- `google-generativeai` – Gemini integration
- `pandas` – CSV handling
- `python-dotenv` – Environment configuration

---

