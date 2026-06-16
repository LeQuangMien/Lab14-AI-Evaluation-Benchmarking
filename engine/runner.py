"""
engine/runner.py
================
BenchmarkRunner: chạy toàn bộ benchmark pipeline song song.
- Gọi Agent → lấy answer + retrieved_doc_ids
- Chạy RetrievalEvaluator (Hit Rate, MRR)
- Chạy LLMJudge (Multi-Judge, Agreement Rate)
- Theo dõi latency và ước tính cost
"""

import asyncio
import time
from typing import List, Dict

# ── Ước tính cost theo OpenRouter pricing (USD/1M tokens) ──────────────
# Dùng để báo cáo "cost per eval" — một trong các Expert Tips
COST_PER_1M_INPUT  = 0.14    
COST_PER_1M_OUTPUT = 0.28


class BenchmarkRunner:
    def __init__(self, agent, evaluator, judge, concurrency: int = 5):
        """
        agent      : instance của MainAgent
        evaluator  : instance của RetrievalEvaluator
        judge      : instance của LLMJudge
        concurrency: số test case chạy song song cùng lúc (tránh rate-limit)
        """
        self.agent       = agent
        self.evaluator   = evaluator
        self.judge       = judge
        self.concurrency = concurrency

    async def run_single_test(self, test_case: Dict, semaphore: asyncio.Semaphore) -> Dict:
        """
        Chạy 1 test case qua toàn bộ pipeline:
            Agent → RetrievalEval → MultiJudge

        Tương thích với cả field "ground_truth" và "expected_answer"
        (field trong golden_set.jsonl là "ground_truth").
        """
        async with semaphore:
            start_time = time.perf_counter()

            # ── 1. Gọi Agent ──────────────────────────────────────────
            try:
                response = await self.agent.query(test_case["question"])
            except Exception as e:
                return self._error_result(test_case, f"Agent error: {e}")

            latency = time.perf_counter() - start_time

            # Lấy ground truth (hỗ trợ cả 2 tên field)
            ground_truth = (
                test_case.get("ground_truth")
                or test_case.get("expected_answer")
                or ""
            )

            # ── 2. Retrieval Evaluation ───────────────────────────────
            try:
                ragas_scores = await self.evaluator.score(test_case, response)
            except Exception as e:
                ragas_scores = {
                    "hit_rate": 0.0, "mrr": 0.0,
                    "faithfulness": 0.0, "relevancy": 0.0,
                    "retrieval": {"hit_rate": 0.0, "mrr": 0.0},
                    "error": str(e)
                }

            # ── 3. Multi-Judge Evaluation ─────────────────────────────
            try:
                judge_result = await self.judge.evaluate_multi_judge(
                    test_case["question"],
                    response.get("answer", ""),
                    ground_truth,
                )
            except Exception as e:
                judge_result = {
                    "final_score"   : 0.0,
                    "agreement_rate": 0.0,
                    "conflict"      : True,
                    "error"         : str(e),
                }

            # ── 4. Ước tính cost (token-based) ────────────────────────
            tokens_used = response.get("metadata", {}).get("tokens_used", 150)
            cost_usd    = (tokens_used / 1_000_000) * (COST_PER_1M_INPUT + COST_PER_1M_OUTPUT)

            # ── 5. Tổng hợp kết quả ───────────────────────────────────
            final_score  = judge_result.get("final_score", 0)
            status       = "pass" if final_score >= 3 else "fail"

            return {
                "id"            : test_case.get("id", "unknown"),
                "category"      : test_case.get("category", "unknown"),
                "difficulty"    : test_case.get("difficulty", "unknown"),
                "question"      : test_case["question"],
                "ground_truth"  : ground_truth,
                "agent_response": response.get("answer", ""),
                "latency_s"     : round(latency, 3),
                "cost_usd"      : round(cost_usd, 6),
                "tokens_used"   : tokens_used,
                "ragas"         : ragas_scores,
                "judge"         : judge_result,
                "status"        : status,
            }

    async def run_all(self, dataset: List[Dict], batch_size: int = None) -> List[Dict]:
        """
        Chạy toàn bộ dataset song song với Semaphore kiểm soát concurrency.
        batch_size giữ để tương thích interface cũ (dùng self.concurrency thay thế).
        """
        concurrency = batch_size or self.concurrency
        semaphore   = asyncio.Semaphore(concurrency)

        print(f"  ▶ Chạy {len(dataset)} cases (concurrency={concurrency})...")
        t0      = time.perf_counter()
        tasks   = [self.run_single_test(case, semaphore) for case in dataset]
        results = await asyncio.gather(*tasks)
        elapsed = time.perf_counter() - t0

        # ── In tóm tắt nhanh ──────────────────────────────────────────
        passed     = sum(1 for r in results if r.get("status") == "pass")
        total_cost = sum(r.get("cost_usd", 0) for r in results)
        avg_lat    = sum(r.get("latency_s", 0) for r in results) / len(results)

        print(f"  ✅ Hoàn thành {len(results)} cases trong {elapsed:.1f}s")
        print(f"     Pass/Fail: {passed}/{len(results) - passed}")
        print(f"     Avg latency: {avg_lat:.2f}s/case")
        print(f"     Total cost: ${total_cost:.4f} USD")

        return list(results)

    @staticmethod
    def _error_result(test_case: Dict, error_msg: str) -> Dict:
        return {
            "id"            : test_case.get("id", "unknown"),
            "category"      : test_case.get("category", "unknown"),
            "difficulty"    : test_case.get("difficulty", "unknown"),
            "question"      : test_case.get("question", ""),
            "ground_truth"  : test_case.get("ground_truth", ""),
            "agent_response": "",
            "latency_s"     : 0.0,
            "cost_usd"      : 0.0,
            "tokens_used"   : 0,
            "ragas"         : {"hit_rate": 0.0, "mrr": 0.0,
                               "retrieval": {"hit_rate": 0.0, "mrr": 0.0}},
            "judge"         : {"final_score": 0.0, "agreement_rate": 0.0},
            "status"        : "error",
            "error"         : error_msg,
        }