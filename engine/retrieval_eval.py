from __future__ import annotations

from collections import defaultdict
from typing import Dict, List


class RetrievalEvaluator:
    def calculate_hit_rate(
        self, expected_ids: List[str], retrieved_ids: List[str], top_k: int = 3
    ) -> float:
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

    def answer_relevancy(self, expected_answer: str, actual_answer: str) -> float:
        expected_terms = {term.lower() for term in expected_answer.split() if len(term) > 3}
        actual_terms = {term.lower() for term in actual_answer.split() if len(term) > 3}
        if not expected_terms:
            return 0.0
        return round(len(expected_terms & actual_terms) / len(expected_terms), 4)

    def faithfulness_proxy(self, contexts: List[str], answer: str) -> float:
        if not contexts or not answer.strip():
            return 0.0
        context_terms = {term.lower() for ctx in contexts for term in ctx.split() if len(term) > 3}
        answer_terms = {term.lower() for term in answer.split() if len(term) > 3}
        if not answer_terms:
            return 0.0
        return round(len(answer_terms & context_terms) / len(answer_terms), 4)

    async def score(self, case: Dict, response: Dict, top_k: int = 3) -> Dict:
        expected_ids = case.get("expected_retrieval_ids", [])
        retrieved_ids = response.get("retrieved_ids", [])
        hit_rate = self.calculate_hit_rate(expected_ids, retrieved_ids, top_k)
        mrr = self.calculate_mrr(expected_ids, retrieved_ids)
        return {
            "faithfulness": self.faithfulness_proxy(response.get("contexts", []), response.get("answer", "")),
            "relevancy": self.answer_relevancy(case.get("expected_answer", ""), response.get("answer", "")),
            "retrieval": {
                "hit_rate": hit_rate,
                "mrr": mrr,
                "top_k": top_k,
                "expected_ids": expected_ids,
                "retrieved_ids": retrieved_ids,
            },
        }

    def aggregate(self, results: List[Dict]) -> Dict:
        if not results:
            return {"avg_hit_rate": 0.0, "avg_mrr": 0.0, "by_type": {}, "by_difficulty": {}}

        def avg(items: List[float]) -> float:
            return round(sum(items) / len(items), 4) if items else 0.0

        by_type = defaultdict(lambda: {"hit_rate": [], "mrr": [], "count": 0})
        by_difficulty = defaultdict(lambda: {"hit_rate": [], "mrr": [], "count": 0})

        hit_rates = []
        mrrs = []
        for result in results:
            retrieval = result["ragas"]["retrieval"]
            case_meta = result.get("metadata", {})
            hit_rates.append(retrieval["hit_rate"])
            mrrs.append(retrieval["mrr"])

            case_type = case_meta.get("type", "unknown")
            difficulty = case_meta.get("difficulty", "unknown")
            by_type[case_type]["hit_rate"].append(retrieval["hit_rate"])
            by_type[case_type]["mrr"].append(retrieval["mrr"])
            by_type[case_type]["count"] += 1
            by_difficulty[difficulty]["hit_rate"].append(retrieval["hit_rate"])
            by_difficulty[difficulty]["mrr"].append(retrieval["mrr"])
            by_difficulty[difficulty]["count"] += 1

        def serialize(grouped: Dict) -> Dict:
            return {
                key: {
                    "count": value["count"],
                    "hit_rate": avg(value["hit_rate"]),
                    "mrr": avg(value["mrr"]),
                }
                for key, value in grouped.items()
            }

        return {
            "avg_hit_rate": avg(hit_rates),
            "avg_mrr": avg(mrrs),
            "by_type": serialize(by_type),
            "by_difficulty": serialize(by_difficulty),
        }
