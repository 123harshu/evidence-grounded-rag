import os
from pydantic import BaseModel

class Settings(BaseModel):
    CHROMA_PATH: str = os.getenv("CHROMA_PATH", "./data/chroma_db")
    COLLECTION_NAME: str = "ai_research_docs"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "ollama")
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = "llama-3.1-8b-instant"

    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 100
    TOP_K: int = 4

settings = Settings()
