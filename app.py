from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import shutil
import os
import time
from src.ingestion import DocumentIngestionEngine
from src.retrieval import HybridRetrievalEngine
from src.generation import GroundedGenerator
from src.evaluation import EvaluationHarness

app = FastAPI(title="Evidence-Grounded AI Research Assistant API")

ingestion_engine = DocumentIngestionEngine()
retrieval_engine = HybridRetrievalEngine()
generation_engine = GroundedGenerator()
eval_harness = EvaluationHarness()

os.makedirs("./data/docs", exist_ok=True)

class QueryRequest(BaseModel):
    question: str
    top_k: int = 4

@app.post("/ingest")
async def ingest_document(file: UploadFile = File(...)):
    file_path = f"./data/docs/{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        num_chunks = ingestion_engine.process_file(file_path, file.filename)
        return {"filename": file.filename, "status": "Success", "chunks_created": num_chunks}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/query")
async def query_rag(req: QueryRequest):
    t0 = time.time()
    passages = retrieval_engine.search(req.question, top_k=req.top_k)
    response = generation_engine.generate_answer(req.question, passages)
    latency = time.time() - t0
    
    return {
        "question": req.question,
        "answer": response["answer"],
        "citations": response["citations"],
        "passages": passages,
        "latency_sec": round(latency, 3),
        "grounded": response["grounded"]
    }

@app.get("/evaluate")
async def run_eval():
    return eval_harness.run_suite()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
