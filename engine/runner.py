"""
Benchmark Runner
Chạy benchmark với async processing, tối ưu hiệu năng và cost reporting
"""
import asyncio
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class BenchmarkConfig:
    """Cấu hình cho benchmark"""
    batch_size: int = 10  # Số lượng concurrent requests
    max_concurrent: int = 5  # Giới hạn concurrent
    timeout: int = 60  # Timeout per test case (seconds)
    verbose: bool = True


@dataclass
class CostReport:
    """Báo cáo chi phí"""
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    estimated_cost: float = 0.0
    currency: str = "USD"


@dataclass
class BenchmarkResult:
    """Kết quả của một test case"""
    test_case_id: str
    question: str
    agent_response: str
    latency: float
    status: str
    ragas: Dict[str, Any] = field(default_factory=dict)
    retrieval: Dict[str, Any] = field(default_factory=dict)
    judge: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


class BenchmarkRunner:
    """
    Async Benchmark Runner với:
    - Batch processing với asyncio.gather
    - Semaphore để tránh rate limit
    - Cost & Token tracking
    - Progress reporting
    """
    def __init__(
        self,
        agent,
        evaluator,
        judge,
        config: Optional[BenchmarkConfig] = None
    ):
        self.agent = agent
        self.evaluator = evaluator
        self.judge = judge
        self.config = config or BenchmarkConfig()

        # Semaphore để giới hạn concurrent requests
        self.semaphore = asyncio.Semaphore(self.config.max_concurrent)

        # Cost tracking
        self.cost_report = CostReport()
        self.total_requests = 0

    async def run_single_test(self, test_case: Dict) -> BenchmarkResult:
        """
        Chạy một test case với timeout và error handling
        """
        async with self.semaphore:  # Giới hạn concurrent
            test_id = test_case.get("id", "unknown")
            question = test_case.get("question", "")
            expected_answer = test_case.get("expected_answer", "")
            expected_retrieval_ids = test_case.get("expected_retrieval_ids", [])

            start_time = time.perf_counter()
            error = None

            try:
                # 1. Gọi Agent với timeout
                response = await asyncio.wait_for(
                    self.agent.query(question),
                    timeout=self.config.timeout
                )

                # 2. Chạy Retrieval Evaluation
                retrieval_scores = await self.evaluator.evaluate_retrieval(
                    [test_case],
                    response
                )

                # 3. Chạy RAGAS metrics
                ragas_scores = await self.evaluator.score(test_case, response)

                # 4. Chạy Multi-Judge
                judge_result = await self.judge.evaluate_multi_judge(
                    question,
                    response.get("answer", ""),
                    expected_answer
                )

                # Tính cost nếu có token usage
                if "metadata" in response:
                    meta = response["metadata"]
                    tokens = meta.get("tokens_used", 0)
                    self.cost_report.total_tokens += tokens

                # Determine status
                final_score = judge_result.final_score
                status = "pass" if final_score >= 3.0 else "fail"
                latency = time.perf_counter() - start_time

                return BenchmarkResult(
                    test_case_id=test_id,
                    question=question,
                    agent_response=response.get("answer", ""),
                    latency=latency,
                    status=status,
                    ragas=ragas_scores,
                    retrieval={
                        "hit_rate": retrieval_scores.hit_rate,
                        "mrr": retrieval_scores.mrr
                    },
                    judge={
                        "final_score": final_score,
                        "agreement_rate": judge_result.agreement_rate,
                        "individual_scores": judge_result.individual_scores,
                        "conflict_detected": judge_result.conflict_detected,
                        "reasoning": judge_result.reasoning[:200] if judge_result.reasoning else ""
                    }
                )

            except asyncio.TimeoutError:
                latency = time.perf_counter() - start_time
                error = "Timeout"
                return BenchmarkResult(
                    test_case_id=test_id,
                    question=question,
                    agent_response="",
                    latency=latency,
                    status="error",
                    error=error
                )

            except Exception as e:
                latency = time.perf_counter() - start_time
                error = str(e)
                return BenchmarkResult(
                    test_case_id=test_id,
                    question=question,
                    agent_response="",
                    latency=latency,
                    status="error",
                    error=error
                )

    async def run_all(
        self,
        dataset: List[Dict],
        version: str = "Agent_v1"
    ) -> Dict[str, Any]:
        """
        Chạy benchmark cho toàn bộ dataset với batch processing

        Args:
            dataset: Danh sách test cases
            version: Phiên bản agent

        Returns:
            Dict chứa results và summary
        """
        print(f"\n{'='*60}")
        print(f"Running Benchmark: {version}")
        print(f"Total cases: {len(dataset)}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Max concurrent: {self.config.max_concurrent}")
        print(f"{'='*60}\n")

        start_time = time.perf_counter()
        results = []

        # Process in batches
        total_batches = (len(dataset) + self.config.batch_size - 1) // self.config.batch_size

        for batch_idx in range(total_batches):
            start = batch_idx * self.config.batch_size
            end = min(start + self.config.batch_size, len(dataset))
            batch = dataset[start:end]

            if self.config.verbose:
                print(f"Batch {batch_idx + 1}/{total_batches}: {len(batch)} cases")

            # Chạy batch với asyncio.gather
            tasks = [self.run_single_test(case) for case in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            for result in batch_results:
                if isinstance(result, Exception):
                    # Handle exception
                    error_result = BenchmarkResult(
                        test_case_id="error",
                        question="",
                        agent_response="",
                        latency=0,
                        status="error",
                        error=str(result)
                    )
                    results.append(error_result)
                else:
                    results.append(result)

            # Update progress
            self.total_requests += len(batch)
            if self.config.verbose:
                completed = len(results)
                print(f"  Progress: {completed}/{len(dataset)} cases ({completed*100//len(dataset)}%)")

        total_time = time.perf_counter() - start_time

        # Tính statistics
        passed = sum(1 for r in results if r.status == "pass")
        failed = sum(1 for r in results if r.status == "fail")
        errors = sum(1 for r in results if r.status == "error")

        avg_latency = sum(r.latency for r in results if r.status != "error") / len(results) if results else 0

        # Retrieval metrics
        hit_rates = [r.retrieval.get("hit_rate", 0) for r in results if r.retrieval]
        mrrs = [r.retrieval.get("mrr", 0) for r in results if r.retrieval]
        avg_hit_rate = sum(hit_rates) / len(hit_rates) if hit_rates else 0
        avg_mrr = sum(mrrs) / len(mrrs) if mrrs else 0

        # Judge metrics
        judge_scores = [r.judge.get("final_score", 0) for r in results if r.judge]
        agreement_rates = [r.judge.get("agreement_rate", 0) for r in results if r.judge]
        avg_judge_score = sum(judge_scores) / len(judge_scores) if judge_scores else 0
        avg_agreement = sum(agreement_rates) / len(agreement_rates) if agreement_rates else 0

        # Estimate cost (giả lập nếu không có API)
        # GPT-4o: ~$0.03/1k tokens, Claude: ~$0.015/1k tokens
        estimated_cost = (self.cost_report.total_tokens / 1000) * 0.02
        self.cost_report.estimated_cost = estimated_cost

        # Print summary
        print(f"\n{'='*60}")
        print(f"Benchmark Complete: {version}")
        print(f"{'='*60}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Avg latency: {avg_latency:.3f}s/case")
        print(f"Pass rate: {passed}/{len(results)} ({passed*100//len(results)}%)")
        print(f"\n--- Metrics ---")
        print(f"Hit Rate: {avg_hit_rate*100:.1f}%")
        print(f"MRR: {avg_mrr*100:.1f}%")
        print(f"Avg Judge Score: {avg_judge_score:.2f}/5.0")
        print(f"Agreement Rate: {avg_agreement*100:.1f}%")
        print(f"\n--- Cost ---")
        print(f"Total tokens: {self.cost_report.total_tokens}")
        print(f"Estimated cost: ${estimated_cost:.4f}")
        print(f"{'='*60}\n")

        # Format results for output
        results_data = []
        for r in results:
            results_data.append({
                "test_case_id": r.test_case_id,
                "question": r.question[:100] + "..." if len(r.question) > 100 else r.question,
                "agent_response": r.agent_response[:200] + "..." if len(r.agent_response) > 200 else r.agent_response,
                "latency": round(r.latency, 3),
                "status": r.status,
                "ragas": r.ragas,
                "retrieval": r.retrieval,
                "judge": r.judge,
                "error": r.error
            })

        return {
            "results": results_data,
            "summary": {
                "version": version,
                "total_cases": len(results),
                "passed": passed,
                "failed": failed,
                "errors": errors,
                "total_time_seconds": round(total_time, 2),
                "avg_latency_seconds": round(avg_latency, 3),
                "metrics": {
                    "hit_rate": round(avg_hit_rate, 3),
                    "mrr": round(avg_mrr, 3),
                    "avg_judge_score": round(avg_judge_score, 2),
                    "agreement_rate": round(avg_agreement, 2)
                },
                "cost": {
                    "total_tokens": self.cost_report.total_tokens,
                    "estimated_cost_usd": round(estimated_cost, 4)
                }
            }
        }


class AsyncBatchRunner:
    """
    Alternative runner với chunking strategy khác
    """
    def __init__(self, runner: BenchmarkRunner):
        self.runner = runner

    async def run_with_chunking(
        self,
        dataset: List[Dict],
        chunk_size: int = 10
    ) -> List[Dict]:
        """Chạy với chunking để tối ưu memory"""
        all_results = []
        for i in range(0, len(dataset), chunk_size):
            chunk = dataset[i:i+chunk_size]
            result = await self.runner.run_all(chunk)
            all_results.extend(result.get("results", []))
        return all_results


__all__ = [
    "BenchmarkRunner",
    "BenchmarkConfig",
    "BenchmarkResult",
    "CostReport",
    "AsyncBatchRunner"
]
