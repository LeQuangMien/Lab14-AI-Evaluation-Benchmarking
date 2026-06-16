"""
engine/metrics.py
=================
Các hàm tính Retrieval Metrics thuần (không gọi API).
Dùng bởi RetrievalEvaluator trong retrieval_eval.py.
"""

from typing import List


def hit_rate(retrieved_ids: List[str], ground_truth_ids: List[str], top_k: int = 3) -> float:
    """
    Hit Rate: 1.0 nếu ít nhất 1 doc trong ground_truth_ids
    xuất hiện trong top_k kết quả retrieve đầu tiên.

    Ví dụ:
        retrieved_ids    = ["doc_003", "doc_001", "doc_007"]
        ground_truth_ids = ["doc_001"]
        top_k            = 3
        → 1.0  (doc_001 nằm ở vị trí 2, trong top 3)
    """
    if not ground_truth_ids:
        return 1.0  # Không có ground truth → không thể sai
    top_k_retrieved = retrieved_ids[:top_k]
    return 1.0 if any(doc_id in top_k_retrieved for doc_id in ground_truth_ids) else 0.0


def mrr(retrieved_ids: List[str], ground_truth_ids: List[str]) -> float:
    """
    Mean Reciprocal Rank: 1 / rank của ground truth doc đầu tiên tìm thấy.
    Nếu không tìm thấy → 0.0

    Ví dụ:
        retrieved_ids    = ["doc_003", "doc_001", "doc_007"]
        ground_truth_ids = ["doc_001"]
        → rank = 2 → MRR = 1/2 = 0.5

    Tại sao MRR khắt khe hơn Hit Rate:
        Hit Rate chỉ hỏi "có tìm thấy không?" (binary)
        MRR hỏi "tìm thấy ở vị trí thứ mấy?" (rank-aware)
        Doc đúng ở vị trí 1 → MRR = 1.0
        Doc đúng ở vị trí 5 → MRR = 0.2 (dù Hit Rate vẫn = 1.0)
    """
    if not ground_truth_ids:
        return 1.0
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in ground_truth_ids:
            return 1.0 / rank
    return 0.0


def precision_at_k(retrieved_ids: List[str], ground_truth_ids: List[str], k: int = 3) -> float:
    """
    Precision@K: Tỷ lệ doc liên quan trong top-K kết quả.

    Ví dụ:
        retrieved_ids    = ["doc_001", "doc_003", "doc_999"]
        ground_truth_ids = ["doc_001", "doc_003"]
        k = 3
        → 2/3 ≈ 0.667
    """
    if not ground_truth_ids or k == 0:
        return 0.0
    top_k = retrieved_ids[:k]
    relevant_found = sum(1 for doc_id in top_k if doc_id in ground_truth_ids)
    return relevant_found / k