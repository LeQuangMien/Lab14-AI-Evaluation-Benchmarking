import asyncio
import time
from typing import Dict, List


class BenchmarkRunner:
    def __init__(self, agent, evaluator, judge):
        self.agent = agent
        self.evaluator = evaluator
        self.judge = judge

    def _failure_cluster(self, test_case: Dict, response: Dict, ragas_scores: Dict, judge_result: Dict) -> str:
        if ragas_scores["retrieval"]["hit_rate"] == 0:
            return "retrieval_miss"
        if judge_result["final_score"] < 3:
            return "answer_quality"
        if judge_result["agreement_rate"] < 0.75:
            return "judge_disagreement"
        if test_case.get("metadata", {}).get("type") in {"prompt-injection", "out-of-context"}:
            return "red_team_watch"
        return "pass"

    async def run_single_test(self, test_case: Dict) -> Dict:
        start_time = time.perf_counter()
        response = await self.agent.query(test_case["question"])
        latency = time.perf_counter() - start_time

        ragas_scores = await self.evaluator.score(test_case, response)
        judge_result = await self.judge.evaluate_multi_judge(
            test_case["question"],
            response["answer"],
            test_case["expected_answer"],
        )
        cluster = self._failure_cluster(test_case, response, ragas_scores, judge_result)

        return {
            "case_id": test_case.get("id"),
            "test_case": test_case["question"],
            "expected_answer": test_case["expected_answer"],
            "agent_response": response["answer"],
            "latency": round(latency, 4),
            "ragas": ragas_scores,
            "judge": judge_result,
            "metadata": {
                **test_case.get("metadata", {}),
                "tokens_used": response.get("metadata", {}).get("tokens_used", 0),
                "estimated_cost_usd": response.get("metadata", {}).get("estimated_cost_usd", 0.0),
            },
            "failure_cluster": cluster,
            "status": "fail" if cluster not in {"pass", "red_team_watch"} or judge_result["final_score"] < 3 else "pass",
        }

    async def run_all(self, dataset: List[Dict], batch_size: int = 10) -> List[Dict]:
        results = []
        for index in range(0, len(dataset), batch_size):
            batch = dataset[index:index + batch_size]
            batch_results = await asyncio.gather(*(self.run_single_test(case) for case in batch))
            results.extend(batch_results)
        return results
