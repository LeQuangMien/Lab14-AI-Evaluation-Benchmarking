import asyncio
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from agent.main_agent import AgentConfig, MainAgent
from engine.llm_judge import LLMJudge
from engine.retrieval_eval import RetrievalEvaluator
from engine.runner import BenchmarkRunner


DATASET_PATH = Path("data/golden_set.jsonl")
REPORTS_DIR = Path("reports")
FAILURE_ANALYSIS_PATH = Path("analysis/failure_analysis.md")


def load_dataset() -> List[Dict]:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            "Missing data/golden_set.jsonl. Run `python data/synthetic_gen.py` first."
        )
    with DATASET_PATH.open("r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f if line.strip()]
    if len(dataset) < 50:
        raise ValueError(f"Golden dataset must contain at least 50 cases, found {len(dataset)}.")
    required = {"id", "question", "expected_answer", "expected_retrieval_ids", "context", "metadata"}
    for case in dataset:
        missing = required - set(case)
        if missing:
            raise ValueError(f"Case {case.get('id', '<unknown>')} missing fields: {sorted(missing)}")
    return dataset


def average(values: List[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def summarize_results(
    version: str, results: List[Dict], evaluator: RetrievalEvaluator, judge: LLMJudge
) -> Dict:
    total = len(results)
    pass_count = sum(1 for result in results if result["status"] == "pass")
    retrieval_aggregate = evaluator.aggregate(results)
    failed_retrieval = [r for r in results if r["ragas"]["retrieval"]["hit_rate"] == 0]
    failed_answers = [r for r in results if r["status"] == "fail"]
    failed_both = [
        r
        for r in results
        if r["status"] == "fail" and r["ragas"]["retrieval"]["hit_rate"] == 0
    ]

    agent_provider = os.getenv("AGENT_PROVIDER", "").strip().lower()
    if not agent_provider:
        agent_provider = "openrouter" if os.getenv("OPENROUTER_API_KEY") else "openai"

    return {
        "metadata": {
            "version": version,
            "total": total,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "judge_models": list(results[0]["judge"]["individual_scores"].keys()) if results else [],
            "agent_model": (
                os.getenv("ANTHROPIC_AGENT_MODEL")
                if agent_provider == "anthropic"
                else None
            )
            or (
                os.getenv("OPENROUTER_AGENT_MODEL")
                or os.getenv("OPENAI_AGENT_MODEL")
                or "openrouter/auto"
            ),
            "agent_provider": agent_provider,
        },
        "metrics": {
            "avg_score": average([r["judge"]["final_score"] for r in results]),
            "pass_rate": round(pass_count / total, 4) if total else 0.0,
            "hit_rate": retrieval_aggregate["avg_hit_rate"],
            "mrr": retrieval_aggregate["avg_mrr"],
            "agreement_rate": average([r["judge"]["agreement_rate"] for r in results]),
            "cohens_kappa": judge.cohen_kappa(results),
            "avg_latency_sec": average([r["latency"] for r in results]),
            "p95_latency_sec": sorted([r["latency"] for r in results])[int(total * 0.95) - 1]
            if total
            else 0.0,
            "total_cost_usd": round(sum(r["cost"]["total_cost_usd"] for r in results), 8),
            "avg_cost_per_case_usd": round(
                sum(r["cost"]["total_cost_usd"] for r in results) / total, 8
            )
            if total
            else 0.0,
            "total_tokens": sum(r["cost"]["total_tokens"] for r in results),
            "retrieval_answer_relationship": {
                "failed_retrieval_cases": len(failed_retrieval),
                "failed_answer_cases": len(failed_answers),
                "failed_both_cases": len(failed_both),
                "answer_failures_with_retrieval_miss_rate": round(
                    len(failed_both) / len(failed_answers), 4
                )
                if failed_answers
                else 0.0,
            },
        },
        "breakdowns": {
            "retrieval_by_type": retrieval_aggregate["by_type"],
            "retrieval_by_difficulty": retrieval_aggregate["by_difficulty"],
        },
    }


def regression_gate(v1: Dict, v2: Dict) -> Dict:
    v1m = v1["metrics"]
    v2m = v2["metrics"]
    cost_growth = (
        (v2m["total_cost_usd"] - v1m["total_cost_usd"]) / v1m["total_cost_usd"]
        if v1m["total_cost_usd"]
        else 0.0
    )
    checks = {
        "avg_score_not_regressed": v2m["avg_score"] >= v1m["avg_score"],
        "hit_rate_stable": v2m["hit_rate"] >= v1m["hit_rate"] - 0.02,
        "cost_growth_within_30_percent": cost_growth <= 0.30,
        "pass_rate_not_regressed": v2m["pass_rate"] >= v1m["pass_rate"],
    }
    return {
        "decision": "APPROVE" if all(checks.values()) else "BLOCK_RELEASE",
        "checks": checks,
        "deltas": {
            "avg_score": round(v2m["avg_score"] - v1m["avg_score"], 4),
            "hit_rate": round(v2m["hit_rate"] - v1m["hit_rate"], 4),
            "mrr": round(v2m["mrr"] - v1m["mrr"], 4),
            "pass_rate": round(v2m["pass_rate"] - v1m["pass_rate"], 4),
            "cost_growth_rate": round(cost_growth, 4),
        },
    }


async def run_benchmark(version: str, config: AgentConfig, dataset: List[Dict]) -> Tuple[List[Dict], Dict]:
    evaluator = RetrievalEvaluator()
    judge = LLMJudge()
    runner = BenchmarkRunner(
        MainAgent(config),
        evaluator,
        judge,
        concurrency=int(os.getenv("EVAL_CONCURRENCY", "3")),
    )
    print(f"Running benchmark for {version} with {len(dataset)} cases...")
    results = await runner.run_all(dataset)
    summary = summarize_results(version, results, evaluator, judge)
    return results, summary


def cluster_failures(results: List[Dict]) -> Dict[str, int]:
    counter: Counter[str] = Counter()
    for result in results:
        if result["status"] == "fail":
            for tag in result["judge"].get("error_tags", ["unknown"]):
                counter[tag] += 1
    return dict(counter)


def root_cause_for_case(result: Dict) -> str:
    if result["ragas"]["retrieval"]["hit_rate"] == 0:
        return "Retrieval/chunking mismatch: expected document was not retrieved in top-k."
    tags = set(result["judge"].get("error_tags", []))
    if "hallucination" in tags:
        return "Prompting/grounding: answer added unsupported information beyond retrieved context."
    if "incomplete" in tags:
        return "Generation prompt: answer did not cover all expected facts."
    if "safety" in tags:
        return "Safety instruction: model did not resist adversarial user instruction."
    return "Judging/generation quality: low score without a retrieval miss."


def write_failure_analysis(results: List[Dict], summary: Dict) -> None:
    failures = [result for result in results if result["status"] == "fail"]
    worst_cases = sorted(results, key=lambda item: item["judge"]["final_score"])[:3]
    clusters = cluster_failures(results)

    cluster_rows = "\n".join(
        f"| {tag} | {count} | {tag.replace('_', ' ').title()} detected by multi-judge consensus |"
        for tag, count in sorted(clusters.items())
    )
    if not cluster_rows:
        cluster_rows = "| none | 0 | No failed cases detected |"

    why_sections = []
    for index, result in enumerate(worst_cases, start=1):
        why_sections.append(
            f"""### Case #{index}: {result['case_id']}
1. **Symptom:** Judge score {result['judge']['final_score']} for question: {result['test_case']}
2. **Why 1:** Agent response did not fully match the expected answer.
3. **Why 2:** Retrieval hit rate was {result['ragas']['retrieval']['hit_rate']} and MRR was {result['ragas']['retrieval']['mrr']}.
4. **Why 3:** Judge tags were {', '.join(result['judge'].get('error_tags', []))}.
5. **Why 4:** The case type was {result.get('metadata', {}).get('type', 'unknown')}, which stresses a known weak point.
6. **Root Cause:** {root_cause_for_case(result)}
"""
        )

    content = f"""# Báo cáo Phân tích Thất bại (Failure Analysis Report)

## 1. Tổng quan Benchmark
- **Tổng số cases:** {summary['metadata']['total']}
- **Tỉ lệ Pass/Fail:** {sum(1 for r in results if r['status'] == 'pass')}/{sum(1 for r in results if r['status'] == 'fail')}
- **Faithfulness proxy trung bình:** {average([r['ragas']['faithfulness'] for r in results])}
- **Relevancy proxy trung bình:** {average([r['ragas']['relevancy'] for r in results])}
- **Hit Rate trung bình:** {summary['metrics']['hit_rate']}
- **MRR trung bình:** {summary['metrics']['mrr']}
- **Điểm Multi-Judge trung bình:** {summary['metrics']['avg_score']} / 5.0
- **Agreement Rate:** {summary['metrics']['agreement_rate']}
- **Cohen's Kappa:** {summary['metrics']['cohens_kappa']}

## 2. Phân nhóm lỗi (Failure Clustering)
| Nhóm lỗi | Số lượng | Nguyên nhân dự kiến |
|----------|----------|---------------------|
{cluster_rows}

## 3. Phân tích 5 Whys (3 case tệ nhất)
{chr(10).join(why_sections)}
## 4. Kế hoạch cải tiến (Action Plan)
- [ ] Tăng chất lượng retrieval bằng semantic embedding hoặc reranking cho hard cases.
- [ ] Tách chunk theo chủ đề thay vì fixed corpus đoạn dài.
- [ ] Siết system prompt cho out-of-context, prompt injection và ambiguous questions.
- [ ] Thêm calibration set để đo position bias và judge disagreement định kỳ.
- [ ] Theo dõi cost/token theo loại case để giảm chi phí eval mà không giảm độ chính xác.
"""
    FAILURE_ANALYSIS_PATH.write_text(content, encoding="utf-8")


async def main() -> None:
    dataset = load_dataset()
    v1_results, v1_summary = await run_benchmark(
        "Agent_V1_Base",
        AgentConfig(version="Agent_V1_Base", top_k=2, min_score=0.15, strict_grounding=False),
        dataset,
    )
    v2_results, v2_summary = await run_benchmark(
        "Agent_V2_Optimized",
        AgentConfig(version="Agent_V2_Optimized", top_k=4, min_score=0.05, strict_grounding=True),
        dataset,
    )
    gate = regression_gate(v1_summary, v2_summary)
    v2_summary["regression"] = {
        "baseline": v1_summary,
        "candidate": {
            "version": v2_summary["metadata"]["version"],
            "metrics": v2_summary["metrics"],
        },
        "delta": gate["deltas"],
    }
    v2_summary["release_gate"] = gate

    REPORTS_DIR.mkdir(exist_ok=True)
    with (REPORTS_DIR / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(v2_summary, f, ensure_ascii=False, indent=2)
    with (REPORTS_DIR / "benchmark_results.json").open("w", encoding="utf-8") as f:
        json.dump(
            {"baseline": v1_results, "candidate": v2_results},
            f,
            ensure_ascii=False,
            indent=2,
        )
    write_failure_analysis(v2_results, v2_summary)

    print("\n--- REGRESSION SUMMARY ---")
    print(f"V1 avg score: {v1_summary['metrics']['avg_score']}")
    print(f"V2 avg score: {v2_summary['metrics']['avg_score']}")
    print(f"Release gate: {gate['decision']}")
    print("Reports written to reports/summary.json and reports/benchmark_results.json")


if __name__ == "__main__":
    asyncio.run(main())
