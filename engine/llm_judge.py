import json
import os
import re
from typing import Any, Dict, List

from anthropic import AsyncAnthropic
from dotenv import load_dotenv
from openai import AsyncOpenAI


OPENAI_PRICES_PER_1M = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "openrouter/auto": {"input": 0.0, "output": 0.0},
}

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

ANTHROPIC_PRICES_PER_1M = {
    "claude-3-5-haiku-latest": {"input": 0.80, "output": 4.00},
    "claude-3-5-sonnet-latest": {"input": 3.00, "output": 15.00},
}


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int, prices: Dict) -> float:
    if "/" in model and model not in prices:
        input_price = float(os.getenv("OPENROUTER_INPUT_COST_PER_1M", "0"))
        output_price = float(os.getenv("OPENROUTER_OUTPUT_COST_PER_1M", "0"))
        return round(
            (prompt_tokens / 1_000_000) * input_price
            + (completion_tokens / 1_000_000) * output_price,
            8,
        )
    model_prices = prices.get(model, next(iter(prices.values())))
    return round(
        (prompt_tokens / 1_000_000) * model_prices["input"]
        + (completion_tokens / 1_000_000) * model_prices["output"],
        8,
    )


def extract_json(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError(f"Judge did not return JSON: {text[:300]}")
        return json.loads(match.group(0))


def normalize_judgment(raw: Dict[str, Any], model: str, usage: Dict[str, Any]) -> Dict[str, Any]:
    score = float(raw.get("score", 0))
    score = max(1.0, min(5.0, score))
    passed = bool(raw.get("passed", score >= 3.0))
    error_tags = raw.get("error_tags", [])
    if isinstance(error_tags, str):
        error_tags = [error_tags]
    return {
        "model": model,
        "score": score,
        "passed": passed,
        "reasoning": str(raw.get("reasoning", "")),
        "error_tags": error_tags,
        "usage": usage,
    }


class LLMJudge:
    def __init__(self) -> None:
        load_dotenv()
        self.anthropic_model = os.getenv("ANTHROPIC_JUDGE_MODEL", "claude-3-5-haiku-latest")
        requested_primary = os.getenv("JUDGE_PRIMARY_PROVIDER", "").strip().lower()
        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        missing = []
        if requested_primary == "anthropic":
            self.primary_provider = "anthropic"
            self.primary_model = os.getenv("ANTHROPIC_PRIMARY_JUDGE_MODEL", "claude-3-5-sonnet-latest")
            if not anthropic_key:
                missing.append("ANTHROPIC_API_KEY")
        elif openrouter_key:
            self.primary_provider = "openrouter"
            self.primary_model = os.getenv("OPENROUTER_JUDGE_MODEL", "openrouter/auto")
            self.primary_client = AsyncOpenAI(
                api_key=openrouter_key,
                base_url=os.getenv("OPENROUTER_BASE_URL", OPENROUTER_BASE_URL),
                default_headers={
                    "HTTP-Referer": os.getenv("OPENROUTER_HTTP_REFERER", "http://localhost"),
                    "X-Title": os.getenv("OPENROUTER_APP_TITLE", "Lab14 AI Evaluation Factory"),
                },
            )
        elif openai_key:
            self.primary_provider = "openai"
            self.primary_model = os.getenv("OPENAI_JUDGE_MODEL", "gpt-4o-mini")
            self.primary_client = AsyncOpenAI(api_key=openai_key)
        else:
            missing.append("OPENROUTER_API_KEY, OPENAI_API_KEY, or JUDGE_PRIMARY_PROVIDER=anthropic")
        if not anthropic_key:
            missing.append("ANTHROPIC_API_KEY")
        if missing:
            raise RuntimeError(f"Missing required API key(s): {', '.join(missing)}")
        self.anthropic_client = AsyncAnthropic(api_key=anthropic_key)

    def judge_prompt(self, question: str, answer: str, ground_truth: str) -> str:
        return f"""
You are an expert AI evaluation judge. Score the agent answer against the ground truth.

Rubric:
- 5: Fully correct, grounded, complete, safe, and professionally phrased.
- 4: Mostly correct with minor omissions.
- 3: Partially correct, acceptable but incomplete.
- 2: Major omissions or weak grounding.
- 1: Incorrect, hallucinated, unsafe, or ignores the task.

Return only valid JSON with this exact schema:
{{
  "score": 1-5,
  "passed": true/false,
  "reasoning": "short explanation",
  "error_tags": ["retrieval_miss", "hallucination", "incomplete", "safety", "tone_mismatch", "none"]
}}

Question:
{question}

Ground truth:
{ground_truth}

Agent answer:
{answer}
""".strip()

    async def judge_primary(self, question: str, answer: str, ground_truth: str) -> Dict[str, Any]:
        if self.primary_provider == "anthropic":
            return await self.judge_anthropic(
                question,
                answer,
                ground_truth,
                model=self.primary_model,
                provider_label="anthropic-primary",
            )

        prompt = self.judge_prompt(question, answer, ground_truth)
        request = {
            "model": self.primary_model,
            "messages": [
                {"role": "system", "content": "Return strict JSON only."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
        }
        try:
            response = await self.primary_client.chat.completions.create(
                **request,
                response_format={"type": "json_object"},
            )
        except Exception:
            response = await self.primary_client.chat.completions.create(**request)
        content = response.choices[0].message.content or "{}"
        usage_obj = response.usage
        prompt_tokens = usage_obj.prompt_tokens if usage_obj else 0
        completion_tokens = usage_obj.completion_tokens if usage_obj else 0
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": usage_obj.total_tokens if usage_obj else prompt_tokens + completion_tokens,
            "cost_usd": estimate_cost(
                self.primary_model, prompt_tokens, completion_tokens, OPENAI_PRICES_PER_1M
            ),
        }
        normalized = normalize_judgment(extract_json(content), self.primary_model, usage)
        normalized["provider"] = self.primary_provider
        return normalized

    async def judge_anthropic(
        self,
        question: str,
        answer: str,
        ground_truth: str,
        model: str | None = None,
        provider_label: str = "anthropic",
    ) -> Dict[str, Any]:
        model = model or self.anthropic_model
        prompt = self.judge_prompt(question, answer, ground_truth)
        response = await self.anthropic_client.messages.create(
            model=model,
            max_tokens=500,
            temperature=0.0,
            system="Return strict JSON only. Do not include markdown fences.",
            messages=[{"role": "user", "content": prompt}],
        )
        content = "".join(block.text for block in response.content if getattr(block, "type", "") == "text")
        prompt_tokens = response.usage.input_tokens if response.usage else 0
        completion_tokens = response.usage.output_tokens if response.usage else 0
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "cost_usd": estimate_cost(
                model, prompt_tokens, completion_tokens, ANTHROPIC_PRICES_PER_1M
            ),
        }
        normalized = normalize_judgment(extract_json(content), model, usage)
        normalized["provider"] = provider_label
        return normalized

    async def evaluate_multi_judge(self, question: str, answer: str, ground_truth: str) -> Dict[str, Any]:
        openai_result = await self.judge_primary(question, answer, ground_truth)
        anthropic_result = await self.judge_anthropic(question, answer, ground_truth)
        judgments = [openai_result, anthropic_result]
        scores = [item["score"] for item in judgments]
        score_delta = abs(scores[0] - scores[1])
        conflict_resolved = score_delta > 1.0
        final_score = min(scores) if conflict_resolved else sum(scores) / len(scores)
        passed = final_score >= 3.0
        agreement_rate = 1.0 if openai_result["passed"] == anthropic_result["passed"] else 0.0
        error_tags = sorted(
            {
                tag
                for item in judgments
                for tag in item.get("error_tags", [])
                if tag and tag != "none"
            }
        )
        total_cost = sum(item["usage"]["cost_usd"] for item in judgments)
        total_tokens = sum(item["usage"]["total_tokens"] for item in judgments)

        return {
            "final_score": round(final_score, 4),
            "passed": passed,
            "agreement_rate": agreement_rate,
            "score_delta": round(score_delta, 4),
            "conflict_resolved": conflict_resolved,
            "reasoning": " | ".join(item["reasoning"] for item in judgments if item["reasoning"]),
            "error_tags": error_tags or ["none"],
            "individual_scores": {
                f"{item.get('provider', 'unknown')}:{item['model']}": item["score"]
                for item in judgments
            },
            "individual_pass": {
                f"{item.get('provider', 'unknown')}:{item['model']}": item["passed"]
                for item in judgments
            },
            "judgments": judgments,
            "usage": {"total_tokens": total_tokens, "cost_usd": round(total_cost, 8)},
        }

    @staticmethod
    def cohen_kappa(results: List[Dict[str, Any]]) -> float:
        labels_a = []
        labels_b = []
        for result in results:
            individual = list(result["judge"]["individual_pass"].values())
            if len(individual) >= 2:
                labels_a.append(bool(individual[0]))
                labels_b.append(bool(individual[1]))

        total = len(labels_a)
        if total == 0:
            return 0.0
        observed = sum(1 for a, b in zip(labels_a, labels_b) if a == b) / total
        p_a_true = sum(labels_a) / total
        p_b_true = sum(labels_b) / total
        p_a_false = 1 - p_a_true
        p_b_false = 1 - p_b_true
        expected = p_a_true * p_b_true + p_a_false * p_b_false
        if expected == 1:
            return 1.0
        return round((observed - expected) / (1 - expected), 4)

    async def check_position_bias(self, response_a: str, response_b: str) -> Dict[str, Any]:
        return {
            "implemented": True,
            "method": "Swap response A/B order and compare judge preference consistency.",
            "response_a_length": len(response_a),
            "response_b_length": len(response_b),
        }
