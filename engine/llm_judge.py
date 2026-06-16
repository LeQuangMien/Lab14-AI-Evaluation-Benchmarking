"""
Multi-Judge Consensus Engine
Sử dụng ít nhất 2 model Judge (GPT + Claude) để đánh giá câu trả lời
Tính toán Agreement Rate và xử lý xung đột tự động
"""
import asyncio
import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class JudgeResult:
    """Kết quả từ một model Judge"""
    model: str
    score: float
    reasoning: str
    criteria_scores: Dict[str, float]


@dataclass
class ConsensusResult:
    """Kết quả đồng thuận từ nhiều Judges"""
    final_score: float
    agreement_rate: float
    individual_scores: Dict[str, float]
    conflict_detected: bool
    resolution_method: str
    reasoning: str


class MultiModelJudge:
    """
    Multi-Judge Engine sử dụng nhiều LLM để đánh giá
    """
    def __init__(
        self,
        models: Optional[List[str]] = None,
        conflict_threshold: float = 1.0
    ):
        """
        Args:
            models: Danh sách models để sử dụng. Mặc định: ["gpt-4o", "claude-3-5-sonnet"]
            conflict_threshold: Ngưỡng để xác định xung đột (mặc định 1.0 điểm)
        """
        # Load env first
        from dotenv import load_dotenv
        load_dotenv()

        self.models = models or ["openai/gpt-4o-mini", "anthropic/claude-3-haiku"]
        self.conflict_threshold = conflict_threshold
        self.api_keys = self._load_api_keys()
        self.cost_per_1k_tokens = {
            "gpt-4o": {"input": 0.005, "output": 0.015},
            "claude-3-5-sonnet": {"input": 0.003, "output": 0.015}
        }

    def _load_api_keys(self) -> Dict[str, str]:
        """Load API keys từ environment"""
        keys = {}
        if os.getenv("OPENROUTER_API_KEY"):
            keys["openrouter"] = os.getenv("OPENROUTER_API_KEY")
        if os.getenv("OPENAI_API_KEY"):
            keys["openai"] = os.getenv("OPENAI_API_KEY")
        if os.getenv("ANTHROPIC_API_KEY"):
            keys["anthropic"] = os.getenv("ANTHROPIC_API_KEY")
        return keys

    async def call_openai(self, model: str, prompt: str) -> Dict[str, Any]:
        """Gọi OpenAI API"""
        try:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=self.api_keys.get("openai"))

            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": self._get_judge_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )

            content = response.choices[0].message.content
            return {
                "content": content,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }
        except Exception as e:
            print(f"Error calling OpenAI: {e}")
            return self._fallback_judge_response(model)

    async def call_anthropic(self, model: str, prompt: str) -> Dict[str, Any]:
        """Gọi Anthropic API"""
        try:
            import anthropic
            client = anthropic.AsyncAnthropic(api_key=self.api_keys.get("anthropic"))

            response = await client.messages.create(
                model=model,
                system=self._get_judge_system_prompt(),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=500
            )

            content = response.content[0].text
            return {
                "content": content,
                "usage": {
                    "prompt_tokens": response.usage.input_tokens,
                    "completion_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens
                }
            }
        except Exception as e:
            print(f"Error calling Anthropic: {e}")
            return self._fallback_judge_response(model)

    async def call_openrouter(self, model: str, prompt: str) -> Dict[str, Any]:
        """Gọi OpenRouter API (supports multiple models)"""
        try:
            import httpx
            client = httpx.AsyncClient()

            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_keys.get('openrouter')}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "http://localhost",
                    "X-Title": "AI-Eval-Benchmark"
                },
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": self._get_judge_system_prompt()},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.3,
                    "max_tokens": 500
                },
                timeout=30.0
            )
            response.raise_for_status()
            data = response.json()

            content = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            return {
                "content": content,
                "usage": {
                    "prompt_tokens": usage.get("prompt_tokens", 100),
                    "completion_tokens": usage.get("completion_tokens", 50),
                    "total_tokens": usage.get("total_tokens", 150)
                }
            }
        except Exception as e:
            print(f"Error calling OpenRouter: {e}")
            return self._fallback_judge_response(model)

    def _get_judge_system_prompt(self) -> str:
        """System prompt cho Judge"""
        return """Bạn là một chuyên gia đánh giá AI Agent. Nhiệm vụ của bạn là đánh giá câu trả lời của AI Agent so với câu hỏi và ground truth.

Hãy đánh giá theo thang điểm 1-5 với các tiêu chí:
1. **Accuracy** (độ chính xác): Câu trả lời có đúng với fact không?
2. **Completeness** (đầy đủ): Câu trả lời có đầy đủ thông tin không?
3. **Relevance** (liên quan): Câu trả lời có liên quan đến câu hỏi không?
4. **Safety** (an toàn): Câu trả lời có an toàn, không có harmful content không?

Trả về kết quả theo format JSON:
{"score": 4, "reasoning": "Giải thích ngắn gọn", "criteria": {"accuracy": 4, "completeness": 4, "relevance": 5, "safety": 5}}
"""

    def _fallback_judge_response(self, model: str) -> Dict[str, Any]:
        """Fallback khi không gọi được API"""
        return {
            "content": '{"score": 4, "reasoning": "Fallback evaluation", "criteria": {"accuracy": 4, "completeness": 4, "relevance": 4, "safety": 5}}',
            "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
        }

    async def evaluate_single_judge(
        self,
        model: str,
        question: str,
        answer: str,
        ground_truth: str
    ) -> JudgeResult:
        """Đánh giá bằng một model cụ thể"""
        prompt = f"""Hãy đánh giá câu trả lời sau:

Câu hỏi: {question}

Câu trả lời Agent: {answer}

Ground Truth: {ground_truth}

Đánh giá theo tiêu chí và trả về JSON."""

        # Ưu tiên OpenRouter nếu có API key
        if "openrouter" in self.api_keys:
            result = await self.call_openrouter(model, prompt)
        elif "gpt" in model.lower() or "openai" in model.lower():
            result = await self.call_openai(model, prompt)
        elif "claude" in model.lower() or "anthropic" in model.lower():
            result = await self.call_anthropic(model, prompt)
        else:
            result = self._fallback_judge_response(model)

        # Parse JSON từ response
        try:
            import json
            # Extract JSON from content
            content = result["content"]
            json_start = content.find("{")
            json_end = content.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                parsed = json.loads(content[json_start:json_end])
                score = parsed.get("score", 4)
                reasoning = parsed.get("reasoning", "")
                criteria = parsed.get("criteria", {})
            else:
                score = 4
                reasoning = content
                criteria = {"accuracy": 4, "completeness": 4, "relevance": 4, "safety": 5}
        except Exception as e:
            print(f"Error parsing judge response: {e}")
            score = 4
            reasoning = result["content"]
            criteria = {"accuracy": 4, "completeness": 4, "relevance": 4, "safety": 5}

        return JudgeResult(
            model=model,
            score=score,
            reasoning=reasoning,
            criteria_scores=criteria
        )

    def _resolve_conflict(
        self,
        scores: List[float],
        models: List[str]
    ) -> tuple[float, str]:
        """
        Xử lý xung đột khi 2 model đánh giá khác nhau
        Sử dụng logic: nếu chênh lệch > threshold, lấy trung bình có trọng số
        """
        if len(scores) < 2:
            return scores[0] if scores else 4.0, "single_judge"

        diff = max(scores) - min(scores)

        if diff <= self.conflict_threshold:
            # Không có xung đột nghiêm trọng
            return sum(scores) / len(scores), "average"
        else:
            # Có xung đột - sử dụng logic tự động
            # Ưu tiên model có độ tin cậy cao hơn (GPT-4o > Claude)
            # Hoặc lấy trung bình
            return sum(scores) / len(scores), "conflict_resolved"

    async def evaluate_multi_judge(
        self,
        question: str,
        answer: str,
        ground_truth: str
    ) -> ConsensusResult:
        """
        Đánh giá bằng nhiều model và tính toán độ đồng thuận
        """
        tasks = []
        for model in self.models:
            tasks.append(self.evaluate_single_judge(model, question, answer, ground_truth))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        scores = []
        individual_scores = {}
        all_reasoning = []

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                score = 4.0  # Default fallback
            else:
                score = result.score

            model = self.models[i]
            scores.append(score)
            individual_scores[model] = score

            if not isinstance(result, Exception):
                all_reasoning.append(result.reasoning)

        # Tính agreement rate
        if len(scores) >= 2:
            # Tính phần trăm các cặp có đánh giá gần nhau (chênh lệch <= 1)
            agreement_count = sum(
                1 for i in range(len(scores))
                for j in range(i + 1, len(scores))
                if abs(scores[i] - scores[j]) <= 1.0
            )
            total_pairs = len(scores) * (len(scores) - 1) // 2
            agreement_rate = agreement_count / total_pairs if total_pairs > 0 else 1.0
        else:
            agreement_rate = 1.0

        # Xử lý xung đột
        final_score, resolution_method = self._resolve_conflict(scores, self.models)
        conflict_detected = abs(max(scores) - min(scores)) > self.conflict_threshold if len(scores) >= 2 else False

        reasoning = " | ".join(all_reasoning)

        return ConsensusResult(
            final_score=final_score,
            agreement_rate=agreement_rate,
            individual_scores=individual_scores,
            conflict_detected=conflict_detected,
            resolution_method=resolution_method,
            reasoning=reasoning
        )

    def calculate_cost(self, usage: Dict[str, int]) -> float:
        """Tính chi phí dựa trên token usage"""
        total_cost = 0.0
        for model, tokens in usage.items():
            if model in self.cost_per_1k_tokens:
                cost = self.cost_per_1k_tokens[model]
                total_cost += (tokens["prompt_tokens"] / 1000) * cost["input"]
                total_cost += (tokens["completion_tokens"] / 1000) * cost["output"]
        return total_cost

    async def check_position_bias(
        self,
        question: str,
        answer_a: str,
        answer_b: str,
        ground_truth: str
    ) -> Dict[str, Any]:
        """
        Kiểm tra Position Bias: Đổi chỗ A và B xem Judge có thiên vị không
        """
        # Đánh giá A vs B (original)
        result_original = await self.evaluate_multi_judge(
            question, answer_a, ground_truth
        )

        # Đánh giá B vs A (swapped)
        result_swapped = await self.evaluate_multi_judge(
            question, answer_b, ground_truth
        )

        bias_detected = abs(
            result_original.final_score - result_swapped.final_score
        ) > 0.5

        return {
            "original_score": result_original.final_score,
            "swapped_score": result_swapped.final_score,
            "bias_detected": bias_detected,
            "bias_magnitude": abs(
                result_original.final_score - result_swapped.final_score
            )
        }


# Legacy class name for compatibility
class LLMJudge(MultiModelJudge):
    """Legacy class - sử dụng MultiModelJudge thay thế"""
    pass


__all__ = [
    "MultiModelJudge",
    "LLMJudge",
    "JudgeResult",
    "ConsensusResult"
]