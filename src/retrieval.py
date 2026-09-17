from typing import List, Dict, Any
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import chromadb
from src.config import settings

class HybridRetrievalEngine:
    def __init__(self):
        self.chroma_client = chromadb.PersistentClient(path=settings.CHROMA_PATH)
        self.collection = self.chroma_client.get_or_create_collection(name=settings.COLLECTION_NAME)
        self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)

    def _get_bm25_scores(self, query: str, docs: List[str]) -> List[float]:
        tokenized_corpus = [doc.lower().split(" ") for doc in docs]
        tokenized_query = query.lower().split(" ")
        bm25 = BM25Okapi(tokenized_corpus)
        return bm25.get_scores(tokenized_query).tolist()

    def search(self, query: str, top_k: int = settings.TOP_K) -> List[Dict[str, Any]]:
        all_records = self.collection.get()
        if not all_records or not all_records["documents"]:
            return []

        docs = all_records["documents"]
        metadatas = all_records["metadatas"]

        query_vec = self.embedding_model.encode([query]).tolist()
        dense_results = self.collection.query(query_embeddings=query_vec, n_results=min(top_k * 2, len(docs)))
        
        dense_hits = []
        if dense_results["documents"]:
            for d, m, dist in zip(dense_results["documents"][0], dense_results["metadatas"][0], dense_results["distances"][0]):
                dense_hits.append({"content": d, "metadata": m, "score": 1.0 / (1.0 + dist)})

        bm25_scores = self._get_bm25_scores(query, docs)
        top_bm25_indices = np.argsort(bm25_scores)[::-1][:top_k * 2]
        
        sparse_hits = []
        for idx in top_bm25_indices:
            if bm25_scores[idx] > 0:
                sparse_hits.append({"content": docs[idx], "metadata": metadatas[idx], "score": float(bm25_scores[idx])})

        rrf_scores = {}
        k_const = 60
        
        for rank, hit in enumerate(dense_hits):
            cid = hit["metadata"]["full_ref"] + hit["content"][:30]
            rrf_scores[cid] = rrf_scores.get(cid, {"hit": hit, "rrf": 0.0})
            rrf_scores[cid]["rrf"] += 1.0 / (k_const + rank + 1)

        for rank, hit in enumerate(sparse_hits):
            cid = hit["metadata"]["full_ref"] + hit["content"][:30]
            rrf_scores[cid] = rrf_scores.get(cid, {"hit": hit, "rrf": 0.0})
            rrf_scores[cid]["rrf"] += 1.0 / (k_const + rank + 1)

        sorted_results = sorted(rrf_scores.values(), key=lambda x: x["rrf"], reverse=True)
        
        final_passages = []
        for item in sorted_results[:top_k]:
            hit = item["hit"]
            final_passages.append({
                "content": hit["content"],
                "source": hit["metadata"]["source"],
                "page": hit["metadata"]["page"],
                "chunk_id": hit["metadata"]["chunk_id"],
                "full_ref": hit["metadata"]["full_ref"],
                "rrf_score": round(item["rrf"], 4)
            })

        return final_passages
