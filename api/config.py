# config.py

import os
from dotenv import load_dotenv

load_dotenv()

# # Get the project root directory (parent of api/)
# PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Get the project root directory (parent of api/)
PROJECT_ROOT = Path(__file__).parent.parent

# Load .env from project root
env

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PDF_PATH = os.getenv("PDF_PATH", os.path.join(PROJECT_ROOT, "data", "manual.pdf"))

EMBED_CSV_PATH = os.getenv("EMBED_CSV_PATH", os.path.join(PROJECT_ROOT, "embeddings", "chunks.csv"))

# OLLAMA_URL = os.getenv("OLLAMA_URL", "https://mathewmanoj13--ollama-rag-fastapi-app.modal.run")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

if not GEMINI_API_KEY:
    print("[WARNING] GEMINI_API_KEY not set. Gemini model calls will fail.")
    print("[INFO] Set it in a .env file or your environment variables.")

# Validate paths exist
if not os.path.exists(PDF_PATH):
    print(f"[WARNING] PDF file not found at {PDF_PATH}")

if not os.path.exists(EMBED_CSV_PATH):
    print(f"[WARNING] Embeddings file not found at {EMBED_CSV_PATH}")
    print(f"[INFO] Run: python scripts/setup.py")

# Print loaded config (for debugging)
if __name__ == "__main__":
    print("Configuration:")
    print(f"  PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"  PDF_PATH: {PDF_PATH}")
    print(f"  EMBED_CSV_PATH: {EMBED_CSV_PATH}")
    # print(f"  OLLAMA_URL: {OLLAMA_URL}")
    print(f"  GEMINI_API_KEY: {'Set' if GEMINI_API_KEY else 'Not set'}")