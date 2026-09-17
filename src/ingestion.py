import os
import uuid
import re
from typing import List, Dict, Any
from pypdf import PdfReader
import chromadb
from sentence_transformers import SentenceTransformer
from src.config import settings

class DocumentIngestionEngine:
    def __init__(self):
        self.chroma_client = chromadb.PersistentClient(path=settings.CHROMA_PATH)
        self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
        self.collection = self.chroma_client.get_or_create_collection(name=settings.COLLECTION_NAME)

    def _clean_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _chunk_text(self, text: str) -> List[str]:
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = start + settings.CHUNK_SIZE
            chunk = text[start:end]
            if chunk:
                chunks.append(chunk)
            start += (settings.CHUNK_SIZE - settings.CHUNK_OVERLAP)
        return chunks

    def process_file(self, file_path: str, filename: str) -> int:
        pages_data = []
        
        if filename.endswith(".pdf"):
            reader = PdfReader(file_path)
            for page_idx, page in enumerate(reader.pages):
                raw_text = page.extract_text() or ""
                pages_data.append((page_idx + 1, self._clean_text(raw_text)))
        elif filename.endswith((".txt", ".md")):
            with open(file_path, "r", encoding="utf-8") as f:
                pages_data.append((1, self._clean_text(f.read())))
        else:
            raise ValueError("Unsupported file format.")

        documents, metadatas, ids = [], [], []

        for page_num, text in pages_data:
            if not text:
                continue
            chunks = self._chunk_text(text)
            for c_idx, chunk in enumerate(chunks):
                chunk_id = f"{filename}_p{page_num}_c{c_idx}_{str(uuid.uuid4())[:8]}"
                documents.append(chunk)
                metadatas.append({
                    "source": filename,
                    "page": page_num,
                    "chunk_id": f"c{c_idx}",
                    "full_ref": f"{filename} (Page {page_num}, Chunk c{c_idx})"
                })
                ids.append(chunk_id)

        if documents:
            embeddings = self.embedding_model.encode(documents).tolist()
            self.collection.add(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )

        return len(documents)

    def get_all_documents(self) -> Dict[str, Any]:
        return self.collection.get()
