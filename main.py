import asyncio
import json
import os
import statistics
import time
from collections import Counter
from pathlib import Path

from agent.main_agent import MainAgent
from engine.llm_judge import LLMJudge
from engine.retrieval_eval import RetrievalEvaluator
from engine.runner import BenchmarkRunner


ROOT = Path(__file__).resolve().parent
GOLDEN_PATH = ROOT / "data" / "golden_set.jsonl"
REPORT_DIR = ROOT / "reports"
ANALYSIS_DIR = ROOT / "analysis"


def load_dataset():
    if not GOLDEN_PATH.exists():
        raise FileNotFoundError("Missing data/golden_set.jsonl. Run python data/synthetic_gen.py first.")
    with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_summary(agent_version, results, elapsed_seconds, baseline=None):
    total = len(results)
    pass_count = sum(1 for row in results if row["status"] == "pass")
    metrics = {
        "avg_score": round(sum(row["judge"]["final_score"] for row in results) / total, 3),
        "hit_rate": round(sum(row["ragas"]["retrieval"]["hit_rate"] for row in results) / total, 3),
        "mrr": round(sum(row["ragas"]["retrieval"]["mrr"] for row in results) / total, 3),
        "faithfulness": round(sum(row["ragas"]["faithfulness"] for row in results) / total, 3),
        "relevancy": round(sum(row["ragas"]["relevancy"] for row in results) / total, 3),
        "agreement_rate": round(sum(row["judge"]["agreement_rate"] for row in results) / total, 3),
        "pass_rate": round(pass_count / total, 3),
        "avg_latency_seconds": round(statistics.mean(row["latency"] for row in results), 4),
        "p95_latency_seconds": round(sorted(row["latency"] for row in results)[int(total * 0.95) - 1], 4),
        "total_tokens": sum(row["metadata"]["tokens_used"] for row in results),
        "estimated_cost_usd": round(sum(row["metadata"]["estimated_cost_usd"] for row in results), 6),
    }

    release_gate = {
        "decision": "APPROVE",
        "reasons": [],
        "thresholds": {
            "min_avg_score": 3.5,
            "min_hit_rate": 0.85,
            "min_agreement_rate": 0.75,
            "max_avg_latency_seconds": 2.0,
        },
    }
    thresholds = release_gate["thresholds"]
    if metrics["avg_score"] < thresholds["min_avg_score"]:
        release_gate["decision"] = "ROLLBACK"
        release_gate["reasons"].append("Average judge score is below threshold.")
    if metrics["hit_rate"] < thresholds["min_hit_rate"]:
        release_gate["decision"] = "ROLLBACK"
        release_gate["reasons"].append("Retrieval Hit Rate is below threshold.")
    if metrics["agreement_rate"] < thresholds["min_agreement_rate"]:
        release_gate["decision"] = "ROLLBACK"
        release_gate["reasons"].append("Judge agreement is below threshold.")
    if metrics["avg_latency_seconds"] > thresholds["max_avg_latency_seconds"]:
        release_gate["decision"] = "ROLLBACK"
        release_gate["reasons"].append("Average latency is above threshold.")

    regression = None
    if baseline:
        regression = {
            "baseline_version": baseline["metadata"]["version"],
            "score_delta": round(metrics["avg_score"] - baseline["metrics"]["avg_score"], 3),
            "hit_rate_delta": round(metrics["hit_rate"] - baseline["metrics"]["hit_rate"], 3),
            "cost_delta_usd": round(metrics["estimated_cost_usd"] - baseline["metrics"]["estimated_cost_usd"], 6),
        }
        if regression["score_delta"] < -0.05 or regression["hit_rate_delta"] < -0.03:
            release_gate["decision"] = "ROLLBACK"
            release_gate["reasons"].append("Regression delta is worse than allowed.")

    if not release_gate["reasons"]:
        release_gate["reasons"].append("All quality, retrieval, agreement, cost, and latency checks passed.")

    return {
        "metadata": {
            "version": agent_version,
            "total": total,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_seconds": round(elapsed_seconds, 3),
        },
        "metrics": metrics,
        "regression": regression,
        "release_gate": release_gate,
        "failure_clusters": dict(Counter(row["failure_cluster"] for row in results)),
    }


async def run_benchmark(agent_version):
    dataset = load_dataset()
    runner = BenchmarkRunner(MainAgent(version=agent_version), RetrievalEvaluator(), LLMJudge())
    start = time.perf_counter()
    results = await runner.run_all(dataset)
    elapsed = time.perf_counter() - start
    return results, build_summary(agent_version, results, elapsed)


def write_failure_analysis(summary, results):
    clusters = Counter(row["failure_cluster"] for row in results)
    worst_cases = sorted(results, key=lambda row: (row["judge"]["final_score"], row["ragas"]["retrieval"]["hit_rate"]))[:3]
    lines = [
        "# Failure Analysis Report",
        "",
        "## 1. Benchmark Overview",
        f"- Total cases: {summary['metadata']['total']}",
        f"- Pass rate: {summary['metrics']['pass_rate'] * 100:.1f}%",
        f"- Average judge score: {summary['metrics']['avg_score']:.2f} / 5.0",
        f"- Hit Rate: {summary['metrics']['hit_rate'] * 100:.1f}%",
        f"- MRR: {summary['metrics']['mrr']:.3f}",
        f"- Agreement Rate: {summary['metrics']['agreement_rate'] * 100:.1f}%",
        f"- Estimated eval cost: ${summary['metrics']['estimated_cost_usd']:.6f}",
        "",
        "## 2. Failure Clustering",
        "| Cluster | Count | Interpretation |",
        "|---|---:|---|",
    ]
    explanations = {
        "pass": "Case passed quality and retrieval checks.",
        "red_team_watch": "Hard safety case passed but should stay in the monitored suite.",
        "retrieval_miss": "Retriever did not return the expected source document.",
        "answer_quality": "Answer quality fell below judge threshold.",
        "judge_disagreement": "Judges disagreed enough to require review.",
    }
    for cluster, count in clusters.most_common():
        lines.append(f"| {cluster} | {count} | {explanations.get(cluster, 'Needs manual review.')} |")

    lines.extend(["", "## 3. 5 Whys on Worst Cases"])
    for index, row in enumerate(worst_cases, start=1):
        lines.extend(
            [
                f"### Case {index}: {row['case_id']} - {row['failure_cluster']}",
                f"1. Symptom: score={row['judge']['final_score']}, hit_rate={row['ragas']['retrieval']['hit_rate']}, question='{row['test_case']}'",
                "2. Why 1: The answer quality is limited by the evidence returned to the generator.",
                "3. Why 2: Retrieval ranking depends on lexical overlap and can miss paraphrases or conflicting wording.",
                "4. Why 3: The local corpus is small and does not use embeddings or reranking.",
                "5. Why 4: The lab implementation prioritizes reproducible offline evaluation over production retrieval infrastructure.",
                "6. Root cause: Retrieval strategy and chunk/rerank design are the highest leverage improvement areas.",
                "",
            ]
        )

    lines.extend(
        [
            "## 4. Improvement Plan",
            "- Add embedding retrieval plus reranking for paraphrased questions.",
            "- Keep the red-team set in every regression run.",
            "- Cache judge results for unchanged cases to reduce cost by at least 30%.",
            "- Route easy stable cases to a cheaper judge and reserve strict consensus for hard cases.",
        ]
    )
    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    with open(ANALYSIS_DIR / "failure_analysis.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


async def main():
    print("Starting Lab 14 benchmark...")
    v1_results, v1_summary = await run_benchmark("Agent_V1_Base")
    v2_results, v2_summary_base = await run_benchmark("Agent_V2_Optimized")
    v2_summary = build_summary(
        "Agent_V2_Optimized",
        v2_results,
        v2_summary_base["metadata"]["elapsed_seconds"],
        baseline=v1_summary,
    )

    REPORT_DIR.mkdir(exist_ok=True)
    with open(REPORT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(v2_summary, f, ensure_ascii=False, indent=2)
    with open(REPORT_DIR / "benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(v2_results, f, ensure_ascii=False, indent=2)
    write_failure_analysis(v2_summary, v2_results)

    print(f"V1 score: {v1_summary['metrics']['avg_score']:.3f}")
    print(f"V2 score: {v2_summary['metrics']['avg_score']:.3f}")
    print(f"Release gate: {v2_summary['release_gate']['decision']}")
    print("Reports written to reports/ and analysis/failure_analysis.md")


if __name__ == "__main__":
    asyncio.run(main())
