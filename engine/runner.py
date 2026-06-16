import asyncio
import time
from typing import Dict, List


class BenchmarkRunner:
    def __init__(self, agent, evaluator, judge, concurrency: int = 5):
        self.agent = agent
        self.evaluator = evaluator
        self.judge = judge
        self.semaphore = asyncio.Semaphore(concurrency)

    async def run_single_test(self, test_case: Dict) -> Dict:
        async with self.semaphore:
            start_time = time.perf_counter()
            response = await self.agent.query(test_case["question"])
            ragas_scores = await self.evaluator.score(test_case, response)
            judge_result = await self.judge.evaluate_multi_judge(
                test_case["question"],
                response["answer"],
                test_case["expected_answer"],
            )
            latency = time.perf_counter() - start_time

            agent_cost = response.get("metadata", {}).get("cost_usd", 0.0)
            judge_cost = judge_result.get("usage", {}).get("cost_usd", 0.0)
            agent_tokens = response.get("metadata", {}).get("tokens_used", 0)
            judge_tokens = judge_result.get("usage", {}).get("total_tokens", 0)

            return {
                "case_id": test_case.get("id"),
                "test_case": test_case["question"],
                "expected_answer": test_case["expected_answer"],
                "expected_retrieval_ids": test_case.get("expected_retrieval_ids", []),
                "agent_response": response["answer"],
                "contexts": response.get("contexts", []),
                "retrieved_ids": response.get("retrieved_ids", []),
                "latency": round(latency, 4),
                "ragas": ragas_scores,
                "judge": judge_result,
                "cost": {
                    "agent_cost_usd": agent_cost,
                    "judge_cost_usd": judge_cost,
                    "total_cost_usd": round(agent_cost + judge_cost, 8),
                    "agent_tokens": agent_tokens,
                    "judge_tokens": judge_tokens,
                    "total_tokens": agent_tokens + judge_tokens,
                },
                "metadata": test_case.get("metadata", {}),
                "status": "pass" if judge_result["passed"] else "fail",
            }

    async def run_all(self, dataset: List[Dict]) -> List[Dict]:
        tasks = [self.run_single_test(case) for case in dataset]
        return await asyncio.gather(*tasks)
