import time
import json
from typing import List, Dict, Any
from src.retrieval import HybridRetrievalEngine
from src.generation import GroundedGenerator

class EvaluationHarness:
    def __init__(self):
        self.retriever = HybridRetrievalEngine()
        self.generator = GroundedGenerator()

    def run_suite(self, eval_file_path: str = "./data/eval_dataset.json") -> Dict[str, Any]:
        with open(eval_file_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)

        results = []
        total_latency = 0.0
        hits = 0
        correct_refusals = 0
        total_unanswerable = 0

        for item in dataset:
            q_id = item["id"]
            question = item["question"]
            category = item["category"]
            expected_refusal = item.get("should_refuse", False)

            t0 = time.time()
            passages = self.retriever.search(question, top_k=4)
            gen_output = self.generator.generate_answer(question, passages)
            latency = time.time() - t0
            total_latency += latency

            ans = gen_output["answer"]
            is_refusal = "Insufficient evidence" in ans

            if category == "answerable" and len(passages) > 0:
                hits += 1

            if expected_refusal:
                total_unanswerable += 1
                if is_refusal:
                    correct_refusals += 1

            status_color = "green"
            if expected_refusal and not is_refusal:
                status_color = "red"
            elif not expected_refusal and is_refusal:
                status_color = "amber"

            results.append({
                "id": q_id,
                "category": category,
                "question": question,
                "answer": ans,
                "latency_sec": round(latency, 3),
                "retrieved_count": len(passages),
                "is_refusal": is_refusal,
                "status": status_color
            })

        summary = {
            "total_queries": len(dataset),
            "avg_latency_sec": round(total_latency / len(dataset), 3),
            "hit_rate": round(hits / max(1, len([d for d in dataset if d["category"] == "answerable"])), 2),
            "refusal_accuracy": round(correct_refusals / max(1, total_unanswerable), 2),
            "detailed_results": results
        }

        return summary
