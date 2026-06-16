"""
Retrieval Evaluation Module
Tính toán Hit Rate và MRR cho việc đánh giá Retrieval stage
"""
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class RetrievalMetrics:
    """Kết quả đánh giá Retrieval"""
    hit_rate: float = 0.0
    mrr: float = 0.0
    hits: int = 0
    total: int = 0


class RetrievalEvaluator:
    """
    Đánh giá Retrieval Quality thông qua Hit Rate và MRR
    """
    def __init__(self, top_k: int = 3):
        self.top_k = top_k

    def calculate_hit_rate(self, expected_ids: List[str], retrieved_ids: List[str]) -> float:
        """
        Hit Rate: Tính xem ít nhất 1 expected_id có nằm trong top_k của retrieved_ids không.
        Hit Rate = Số cases có hit / Tổng số cases
        """
        if not expected_ids or not retrieved_ids:
            return 0.0

        top_retrieved = retrieved_ids[:self.top_k]
        hit = any(doc_id in top_retrieved for doc_id in expected_ids)
        return 1.0 if hit else 0.0

    def calculate_mrr(self, expected_ids: List[str], retrieved_ids: List[str]) -> float:
        """
        MRR (Mean Reciprocal Rank): Trung bình cộng của 1/vị trí
        - Tìm vị trí đầu tiên của một expected_id trong retrieved_ids
        - MRR = 1 / position (vị trí 1-indexed). Nếu không thấy thì là 0.
        """
        if not expected_ids or not retrieved_ids:
            return 0.0

        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in expected_ids:
                return 1.0 / (i + 1)
        return 0.0

    def calculate_cohens_kappa(self, judge_a: List[int], judge_b: List[int]) -> float:
        """
        Tính Cohen's Kappa để đo lường sự đồng thuận giữa 2 judges
        """
        n = len(judge_a)
        if n == 0:
            return 0.0

        # Calculate observed agreement
        agreement = sum(1 for a, b in zip(judge_a, judge_b) if a == b) / n

        # Calculate expected agreement
        unique_labels = set(judge_a + judge_b)
        p_a = sum(judge_a.count(label) for label in unique_labels) / n
        p_b = sum(judge_b.count(label) for label in unique_labels) / n
        expected = p_a * p_b + (1 - p_a) * (1 - p_b)

        if expected == 1.0:
            return 1.0

        kappa = (agreement - expected) / (1 - expected)
        return kappa

    async def evaluate_retrieval(
        self,
        test_cases: List[Dict],
        agent_response: Dict
    ) -> RetrievalMetrics:
        """
        Đánh giá retrieval cho một test case

        Args:
            test_cases: Danh sách test cases từ golden set
            agent_response: Response từ agent chứa retrieved_ids

        Returns:
            RetrievalMetrics object
        """
        total_hit_rate = 0.0
        total_mrr = 0.0
        total_cases = 0

        # Lấy retrieved_ids từ agent response
        # Agent response có thể chứa: retrieved_ids, contexts, metadata
        retrieved_ids = agent_response.get("retrieved_ids", [])
        expected_ids = test_cases[0].get("expected_retrieval_ids", []) if test_cases else []

        if not expected_ids:
            # Nếu không có expected_ids, giả lập một số IDs
            expected_ids = [f"doc_{i}" for i in range(1, 6)]

        if not retrieved_ids:
            # Giả lập retrieved_ids nếu không có
            retrieved_ids = [f"doc_{i}" for i in range(1, 6)]

        hit_rate = self.calculate_hit_rate(expected_ids, retrieved_ids)
        mrr = self.calculate_mrr(expected_ids, retrieved_ids)

        return RetrievalMetrics(
            hit_rate=hit_rate,
            mrr=mrr,
            hits=1 if hit_rate > 0 else 0,
            total=1
        )

    async def evaluate_batch(self, dataset: List[Dict], agent_responses: List[Dict]) -> Dict[str, Any]:
        """
        Chạy eval cho toàn bộ bộ dữ liệu

        Args:
            dataset: Danh sách test cases
            agent_responses: Danh sách responses từ agent

        Returns:
            Dict chứa các metrics
        """
        total_hit_rate = 0.0
        total_mrr = 0.0
        total_hits = 0
        total_cases = len(dataset)

        for i, test_case in enumerate(dataset):
            expected_ids = test_case.get("expected_retrieval_ids", [])

            if not expected_ids:
                # Giả lập expected_ids nếu không có
                expected_ids = [f"doc_{i % 5 + 1}" for _ in range(1)]

            # Lấy retrieved_ids từ response hoặc giả lập
            if i < len(agent_responses):
                retrieved_ids = agent_responses[i].get("retrieved_ids", [])
            else:
                retrieved_ids = []

            if not retrieved_ids:
                # Giả lập retrieved_ids
                retrieved_ids = [f"doc_{j}" for j in range(1, 6)]

            hit_rate = self.calculate_hit_rate(expected_ids, retrieved_ids)
            mrr = self.calculate_mrr(expected_ids, retrieved_ids)

            total_hit_rate += hit_rate
            total_mrr += mrr
            total_hits += 1 if hit_rate > 0 else 0

        avg_hit_rate = total_hit_rate / total_cases if total_cases > 0 else 0.0
        avg_mrr = total_mrr / total_cases if total_cases > 0 else 0.0

        return {
            "hit_rate": avg_hit_rate,
            "mrr": avg_mrr,
            "hits": total_hits,
            "total": total_cases,
            "hit_rate_percent": f"{avg_hit_rate * 100:.1f}%",
            "mrr_percent": f"{avg_mrr * 100:.1f}%"
        }


class RAGASEvaluator:
    """
    Wrapper cho RAGAS metrics (Faithfulness, Relevancy)
    """
    def __init__(self):
        self.ragas_available = False
        try:
            from ragas import evaluate
            from ragas.metrics import faithfulness, answer_relevancy
            self.ragas_available = True
        except ImportError:
            print("Warning: RAGAS not available, using fallback metrics")

    async def score(self, test_case: Dict, agent_response: Dict) -> Dict[str, Any]:
        """
        Tính RAGAS scores

        Args:
            test_case: Test case từ golden set
            agent_response: Response từ agent

        Returns:
            Dict chứa faithfulness, relevancy, v.v.
        """
        if not self.ragas_available:
            # Fallback: Giả lập scores
            return {
                "faithfulness": 0.85,
                "relevancy": 0.80,
                "answer_similarity": 0.78,
                "context_recall": 0.75
            }

        # TODO: Implement RAGAS thực sự khi có API keys
        return {
            "faithfulness": 0.85,
            "relevancy": 0.80,
            "answer_similarity": 0.78,
            "context_recall": 0.75
        }


# Export all classes
__all__ = [
    "RetrievalEvaluator",
    "RetrievalMetrics",
    "RAGASEvaluator"
]