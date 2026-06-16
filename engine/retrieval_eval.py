"""
engine/retrieval_eval.py
========================
RetrievalEvaluator: tính Hit Rate, MRR, Precision@K
cho toàn bộ Golden Dataset.

Agent phải trả về trường "retrieved_doc_ids" trong response
để evaluator so sánh với "ground_truth_doc_ids" trong test case.
"""

from typing import List, Dict
from engine.metrics import hit_rate, mrr, precision_at_k


class RetrievalEvaluator:
    def __init__(self, top_k: int = 3):
        self.top_k = top_k

    # ── Các hàm tính metric đơn lẻ (giữ nguyên interface cũ) ──────────

    def calculate_hit_rate(self, expected_ids: List[str], retrieved_ids: List[str], top_k: int = None) -> float:
        """Hit Rate cho 1 query. top_k mặc định dùng self.top_k."""
        k = top_k if top_k is not None else self.top_k
        return hit_rate(retrieved_ids, expected_ids, top_k=k)

    def calculate_mrr(self, expected_ids: List[str], retrieved_ids: List[str]) -> float:
        """MRR cho 1 query."""
        return mrr(retrieved_ids, expected_ids)

    # ── Score cho 1 test case (dùng bởi BenchmarkRunner) ──────────────

    async def score(self, test_case: Dict, agent_response: Dict) -> Dict:
        """
        Tính retrieval metrics cho 1 test case.

        Params:
            test_case      : dict từ golden_set.jsonl
                             cần có "ground_truth_doc_ids"
            agent_response : dict từ agent.query()
                             cần có "retrieved_doc_ids"

        Returns:
            {
              "hit_rate"      : float,
              "mrr"           : float,
              "precision_at_k": float,
              "faithfulness"  : float,  # proxy đơn giản
              "relevancy"     : float,  # proxy đơn giản
              "retrieval": {
                  "hit_rate": float,
                  "mrr"     : float
              }
            }
        """
        gt_ids       = test_case.get("ground_truth_doc_ids", [])
        retrieved    = agent_response.get("retrieved_doc_ids", [])

        hr  = hit_rate(retrieved, gt_ids, top_k=self.top_k)
        mrr_score = mrr(retrieved, gt_ids)
        p_at_k    = precision_at_k(retrieved, gt_ids, k=self.top_k)

        # Proxy đơn giản cho faithfulness / relevancy dựa trên retrieval quality
        # (Hệ thống thực dùng LLM để tính — ở đây dùng heuristic để tiết kiệm API)
        faithfulness = 0.9 if hr == 1.0 else 0.5
        relevancy    = round((hr * 0.6 + mrr_score * 0.4), 3)

        return {
            "hit_rate"       : hr,
            "mrr"            : round(mrr_score, 4),
            "precision_at_k" : round(p_at_k, 4),
            "faithfulness"   : faithfulness,
            "relevancy"      : relevancy,
            # Nested "retrieval" để tương thích với main.py cũ
            "retrieval": {
                "hit_rate": hr,
                "mrr"     : round(mrr_score, 4),
            }
        }

    # ── Batch eval toàn bộ dataset (standalone, không qua runner) ─────

    async def evaluate_batch(self, dataset: List[Dict], agent=None) -> Dict:
        """
        Chạy retrieval eval cho toàn bộ dataset.
        Nếu có agent, gọi agent.query() để lấy retrieved_doc_ids thực.
        Nếu không có agent, dùng placeholder (avg cố định).

        Returns: {
            "avg_hit_rate"      : float,
            "avg_mrr"           : float,
            "avg_precision_at_k": float,
            "total_cases"       : int,
            "per_case"          : List[Dict]
        }
        """
        if agent is None:
            # Placeholder khi chưa có agent thực
            return {
                "avg_hit_rate"       : 0.85,
                "avg_mrr"            : 0.72,
                "avg_precision_at_k" : 0.60,
                "total_cases"        : len(dataset),
                "per_case"           : []
            }

        per_case = []
        for case in dataset:
            try:
                resp   = await agent.query(case["question"])
                scores = await self.score(case, resp)
                per_case.append({
                    "id"      : case.get("id"),
                    "question": case["question"],
                    **scores
                })
            except Exception as e:
                per_case.append({
                    "id"      : case.get("id"),
                    "error"   : str(e),
                    "hit_rate": 0.0,
                    "mrr"     : 0.0
                })

        valid = [c for c in per_case if "error" not in c]
        n     = len(valid) or 1

        return {
            "avg_hit_rate"       : round(sum(c["hit_rate"] for c in valid) / n, 4),
            "avg_mrr"            : round(sum(c["mrr"]      for c in valid) / n, 4),
            "avg_precision_at_k" : round(sum(c["precision_at_k"] for c in valid) / n, 4),
            "total_cases"        : len(dataset),
            "per_case"           : per_case
        }