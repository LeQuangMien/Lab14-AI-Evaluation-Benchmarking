import re
from typing import Dict, List


class RetrievalEvaluator:
    def calculate_hit_rate(self, expected_ids: List[str], retrieved_ids: List[str], top_k: int = 3) -> float:
        if not expected_ids:
            return 1.0 if not retrieved_ids else 0.0
        top_retrieved = retrieved_ids[:top_k]
        return 1.0 if any(doc_id in top_retrieved for doc_id in expected_ids) else 0.0

    def calculate_mrr(self, expected_ids: List[str], retrieved_ids: List[str]) -> float:
        if not expected_ids:
            return 1.0 if not retrieved_ids else 0.0
        for index, doc_id in enumerate(retrieved_ids):
            if doc_id in expected_ids:
                return 1.0 / (index + 1)
        return 0.0

    def _token_set(self, text: str) -> set:
        return {token for token in re.findall(r"[a-zA-Z0-9-]+", text.lower()) if len(token) > 2}

    def _overlap_score(self, expected: str, actual: str) -> float:
        expected_tokens = self._token_set(expected)
        actual_tokens = self._token_set(actual)
        if not expected_tokens:
            return 1.0
        return min(1.0, len(expected_tokens & actual_tokens) / max(1, len(expected_tokens) * 0.55))

    async def score(self, case: Dict, response: Dict) -> Dict:
        expected_ids = case.get("expected_retrieval_ids", [])
        retrieved_ids = response.get("retrieved_ids", [])
        hit_rate = self.calculate_hit_rate(expected_ids, retrieved_ids)
        mrr = self.calculate_mrr(expected_ids, retrieved_ids)
        relevancy = self._overlap_score(case["question"], response["answer"])
        faithfulness = 1.0 if hit_rate == 1.0 else 0.35
        if not expected_ids and "do not have enough support" in response["answer"].lower():
            faithfulness = 1.0
            relevancy = max(relevancy, 0.85)

        return {
            "faithfulness": round(faithfulness, 3),
            "relevancy": round(relevancy, 3),
            "retrieval": {
                "hit_rate": hit_rate,
                "mrr": round(mrr, 3),
                "expected_ids": expected_ids,
                "retrieved_ids": retrieved_ids,
            },
        }

    async def evaluate_batch(self, rows: List[Dict]) -> Dict:
        if not rows:
            return {"avg_hit_rate": 0.0, "avg_mrr": 0.0}
        return {
            "avg_hit_rate": sum(row["ragas"]["retrieval"]["hit_rate"] for row in rows) / len(rows),
            "avg_mrr": sum(row["ragas"]["retrieval"]["mrr"] for row in rows) / len(rows),
        }
