"""
Main Entry Point cho AI Evaluation Benchmark
Chạy benchmark cho V1 vs V2 và quyết định Release/Block
"""
import asyncio
import json
import os
import sys
import io

# Fix UTF-8 encoding for Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import time
from engine.runner import BenchmarkRunner, BenchmarkConfig
from engine.retrieval_eval import RetrievalEvaluator, RAGASEvaluator
from engine.llm_judge import MultiModelJudge
from agent.main_agent import MainAgent


class RegressionGate:
    """
    Regression Release Gate với logic tự động
    Quyết định Release hoặc Block dựa trên các metrics
    """
    def __init__(self, thresholds: dict = None):
        # Default thresholds
        self.thresholds = thresholds or {
            "min_hit_rate": 0.70,  # Tối thiểu 70% hit rate
            "min_mrr": 0.60,       # Tối thiểu 60% MRR
            "min_judge_score": 3.0,  # Tối thiểu 3.0/5.0
            "min_agreement": 0.70,  # Tối thiểu 70% agreement
            "max_regression": 0.05,  # Cho phép tối đa 5% regression
            "min_improvement": 0.02  # Cần cải thiện tối thiểu 2% để approve
        }

    def evaluate(self, v1_summary: dict, v2_summary: dict) -> dict:
        """
        Đánh giá V2 so với V1 và quyết định Release/Block

        Args:
            v1_summary: Kết quả benchmark V1
            v2_summary: Kết quả benchmark V2

        Returns:
            Dict chứa quyết định và chi tiết
        """
        v1_metrics = v1_summary.get("metrics", {})
        v2_metrics = v2_summary.get("metrics", {})

        # Tính delta
        delta_hit_rate = v2_metrics.get("hit_rate", 0) - v1_metrics.get("hit_rate", 0)
        delta_mrr = v2_metrics.get("mrr", 0) - v1_metrics.get("mrr", 0)
        delta_judge = v2_metrics.get("avg_judge_score", 0) - v1_metrics.get("avg_judge_score", 0)
        delta_agreement = v2_metrics.get("agreement_rate", 0) - v1_metrics.get("agreement_rate", 0)

        # Check các ngưỡng
        passes_hit_rate = v2_metrics.get("hit_rate", 0) >= self.thresholds["min_hit_rate"]
        passes_mrr = v2_metrics.get("mrr", 0) >= self.thresholds["min_mrr"]
        passes_judge = v2_metrics.get("avg_judge_score", 0) >= self.thresholds["min_judge_score"]
        passes_agreement = v2_metrics.get("agreement_rate", 0) >= self.thresholds["min_agreement"]

        # Check regression (V2 không được tệ hơn V1 quá nhiều)
        has_regression = (
            delta_hit_rate < -self.thresholds["max_regression"] or
            delta_mrr < -self.thresholds["max_regression"] or
            delta_judge < -self.thresholds["max_regression"]
        )

        # Check improvement (V2 cần cải thiện đáng kể)
        has_improvement = (
            delta_hit_rate >= self.thresholds["min_improvement"] or
            delta_mrr >= self.thresholds["min_improvement"] or
            delta_judge >= self.thresholds["min_improvement"]
        )

        # Quyết định
        if has_regression:
            decision = "BLOCK"
            reason = "Performance regression detected"
        elif not all([passes_hit_rate, passes_mrr, passes_judge, passes_agreement]):
            decision = "BLOCK"
            reason = "Below minimum thresholds"
        elif not has_improvement:
            decision = "CONDITIONAL"
            reason = "No significant improvement over V1"
        else:
            decision = "APPROVE"
            reason = "All metrics passed"

        return {
            "decision": decision,
            "reason": reason,
            "thresholds": self.thresholds,
            "v1_metrics": v1_metrics,
            "v2_metrics": v2_metrics,
            "deltas": {
                "hit_rate": round(delta_hit_rate, 3),
                "mrr": round(delta_mrr, 3),
                "judge_score": round(delta_judge, 2),
                "agreement": round(delta_agreement, 2)
            },
            "checks": {
                "passes_hit_rate": passes_hit_rate,
                "passes_mrr": passes_mrr,
                "passes_judge": passes_judge,
                "passes_agreement": passes_agreement,
                "has_regression": has_regression,
                "has_improvement": has_improvement
            }
        }

    def print_report(self, result: dict):
        """In báo cáo Regression"""
        print("\n" + "="*60)
        print("REGRESSION ANALYSIS REPORT")
        print("="*60)

        print(f"\nDecision: {result['decision']}")
        print(f"Reason: {result['reason']}")

        print(f"\n--- Thresholds ---")
        for key, value in result['thresholds'].items():
            print(f"  {key}: {value}")

        print(f"\n--- V1 Metrics ---")
        for key, value in result['v1_metrics'].items():
            print(f"  {key}: {value}")

        print(f"\n--- V2 Metrics ---")
        for key, value in result['v2_metrics'].items():
            print(f"  {key}: {value}")

        print(f"\n--- Deltas (V2 - V1) ---")
        for key, value in result['deltas'].items():
            sign = "+" if value >= 0 else ""
            print(f"  {key}: {sign}{value}")

        print(f"\n--- Checks ---")
        for key, value in result['checks'].items():
            status = "PASS" if value else "FAIL"
            print(f"  {key}: {status}")

        print("="*60 + "\n")


async def run_benchmark_for_version(
    version: str,
    dataset: list,
    config: BenchmarkConfig
) -> dict:
    """
    Chạy benchmark cho một phiên bản Agent
    """
    print(f"\n{'#'*60}")
    print(f"# Running Benchmark for {version}")
    print(f"{'#'*60}\n")

    # Initialize components
    agent = MainAgent()  # Hoặc Agent thực tế
    retrieval_eval = RetrievalEvaluator()
    ragas_eval = RAGASEvaluator()

    # Tạo combined evaluator
    class CombinedEvaluator:
        def __init__(self, retrieval, ragas):
            self.retrieval = retrieval
            self.ragas = ragas

        async def evaluate_retrieval(self, test_cases, response):
            return await self.retrieval.evaluate_retrieval(test_cases, response)

        async def score(self, test_case, response):
            return await self.ragas.score(test_case, response)

    evaluator = CombinedEvaluator(retrieval_eval, ragas_eval)
    judge = MultiModelJudge()

    # Run benchmark
    runner = BenchmarkRunner(agent, evaluator, judge, config)
    result = await runner.run_all(dataset, version)

    return result


async def main():
    """Main entry point"""
    print("AI Evaluation Benchmark - Starting...")

    # Load dataset
    if not os.path.exists("data/golden_set.jsonl"):
        print("Error: data/golden_set.jsonl not found!")
        print("Please run: python data/synthetic_gen.py")
        return

    with open("data/golden_set.jsonl", "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f if line.strip()]

    if not dataset:
        print("Error: No test cases found in golden_set.jsonl")
        return

    print(f"Loaded {len(dataset)} test cases")

    # Config cho benchmark
    config = BenchmarkConfig(
        batch_size=10,
        max_concurrent=5,
        timeout=30,
        verbose=True
    )

    # Run V1 (Baseline)
    print("\n" + "="*60)
    print("PHASE 1: Running V1 (Baseline)")
    print("="*60)

    v1_result = await run_benchmark_for_version("Agent_V1_Base", dataset, config)
    v1_summary = v1_result.get("summary", {})

    # Simulate V2 (hoặc chạy agent thực tế)
    # Trong thực tế, bạn sẽ chạy với agent mới hơn
    print("\n" + "="*60)
    print("PHASE 2: Running V2 (Optimized)")
    print("="*60)

    v2_result = await run_benchmark_for_version("Agent_V2_Optimized", dataset, config)
    v2_summary = v2_result.get("summary", {})

    # Regression Analysis
    gate = RegressionGate()
    regression_result = gate.evaluate(v1_summary, v2_summary)
    gate.print_report(regression_result)

    # Save reports
    os.makedirs("reports", exist_ok=True)

    # Summary report
    summary_report = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_cases": len(dataset),
            "v1_version": "Agent_V1_Base",
            "v2_version": "Agent_V2_Optimized"
        },
        "metrics": {
            "hit_rate": v2_summary.get("metrics", {}).get("hit_rate", 0),
            "mrr": v2_summary.get("metrics", {}).get("mrr", 0),
            "avg_judge_score": v2_summary.get("metrics", {}).get("avg_judge_score", 0),
            "agreement_rate": v2_summary.get("metrics", {}).get("agreement_rate", 0),
            "avg_latency": v2_summary.get("avg_latency_seconds", 0),
            "total_time": v2_summary.get("total_time_seconds", 0)
        },
        "regression": regression_result,
        "cost": v2_summary.get("cost", {})
    }

    with open("reports/summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_report, f, ensure_ascii=False, indent=2)

    # Detailed results
    with open("reports/benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(v2_result.get("results", []), f, ensure_ascii=False, indent=2)

    print("Reports saved to reports/")

    # Final decision
    print("\n" + "="*60)
    print("FINAL DECISION")
    print("="*60)
    print(f"{regression_result['decision']}: {regression_result['reason']}")
    print("="*60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())