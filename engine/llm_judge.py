"""
engine/llm_judge.py
===================
Multi-Judge Consensus Engine dùng OpenRouter.
- Judge A: google/gemma-3-12b-it:free
- Judge B: meta-llama/llama-3.1-8b-instruct:free
- Tính Agreement Rate + xử lý xung đột tự động
- Position Bias check (đổi chỗ A/B)

Cần biến môi trường: OPENROUTER_API_KEY
"""

import asyncio
import json
import os
from typing import Dict, Any

from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

# ── 2 model Judge khác nhau  ─────────────
JUDGE_A_MODEL = "openai/gpt-oss-20b:free"
JUDGE_B_MODEL = "google/gemma-4-31b-it:free"  

# Ngưỡng xung đột: nếu 2 judge lệch nhau > CONFLICT_THRESHOLD điểm → cần xử lý
CONFLICT_THRESHOLD = 1

_client = AsyncOpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    default_headers={
        "HTTP-Referer": "https://github.com/LeQuangMien/Lab14-AI-Evaluation-Benchmarking",
        "X-Title": "Lab14 Multi-Judge",
    }
)

# ── Rubrics chi tiết ────────────────────────────────────────────────────
RUBRICS = {
    "accuracy": (
        "5 - Hoàn toàn chính xác, khớp ground truth về nội dung và chi tiết.\n"
        "4 - Phần lớn chính xác, thiếu 1-2 chi tiết nhỏ.\n"
        "3 - Đúng ý chính nhưng thiếu/sai một số thông tin quan trọng.\n"
        "2 - Chỉ đúng một phần, sai nhiều điểm quan trọng.\n"
        "1 - Hoàn toàn sai hoặc không liên quan."
    ),
    "professionalism": (
        "5 - Ngôn ngữ rõ ràng, chuyên nghiệp, cấu trúc tốt.\n"
        "3 - Chấp nhận được nhưng có thể cải thiện.\n"
        "1 - Ngôn ngữ tối nghĩa, thiếu chuyên nghiệp."
    ),
}

JUDGE_PROMPT_TEMPLATE = """\
Bạn là một AI Judge chuyên nghiệp đánh giá chất lượng câu trả lời.

## Câu hỏi
{question}

## Ground Truth (câu trả lời chuẩn)
{ground_truth}

## Câu trả lời cần chấm
{answer}

## Rubric đánh giá Accuracy (1-5)
{accuracy_rubric}

## Yêu cầu
Chấm điểm tổng thể từ 1-5 dựa trên Accuracy là tiêu chí chính (70%) và Professionalism (30%).
Trả lời JSON duy nhất, KHÔNG có backtick, KHÔNG có text thừa:
{{"score": <số nguyên 1-5>, "reasoning": "<giải thích ngắn gọn 1-2 câu>"}}"""


async def _call_judge(model: str, question: str, answer: str, ground_truth: str) -> Dict:
    """Gọi 1 model judge, trả về {"score": int, "reasoning": str, "model": str}."""
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        question=question,
        ground_truth=ground_truth,
        answer=answer,
        accuracy_rubric=RUBRICS["accuracy"],
    )
    for attempt in range(3):
        try:
            resp = await _client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,   # Judge phải deterministic
                max_tokens=200,
            )
            content = resp.choices[0].message.content
            if content is None:
                raise ValueError("content=None")

            # Làm sạch và parse JSON
            content = content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            data = json.loads(content.strip())

            score = int(data.get("score", 3))
            score = max(1, min(5, score))   # Clamp về [1, 5]

            return {
                "score"    : score,
                "reasoning": data.get("reasoning", ""),
                "model"    : model,
            }
        except Exception as e:
            if attempt == 2:
                # Fallback sau 3 lần thất bại
                return {"score": 3, "reasoning": f"Judge lỗi: {e}", "model": model}
            await asyncio.sleep(2 * (attempt + 1))


class LLMJudge:
    def __init__(self, model: str = JUDGE_A_MODEL):
        """
        model: model mặc định (dùng khi gọi single-judge).
        Multi-judge luôn dùng JUDGE_A_MODEL + JUDGE_B_MODEL.
        """
        self.model   = model
        self.rubrics = RUBRICS

    async def evaluate_multi_judge(
        self, question: str, answer: str, ground_truth: str
    ) -> Dict[str, Any]:
        """
        Gọi 2 judge song song, tính Agreement Rate, xử lý xung đột.

        Agreement Rate:
            1.0  → 2 judge đồng ý (lệch ≤ CONFLICT_THRESHOLD)
            0.5  → xung đột nhẹ (lệch = 2)
            0.0  → xung đột nặng (lệch ≥ 3) → dùng điểm thấp hơn (conservative)

        Returns:
            {
              "final_score"       : float,
              "agreement_rate"    : float,
              "conflict"          : bool,
              "resolution"        : str,
              "individual_scores" : {"judge_a": int, "judge_b": int},
              "reasoning"         : str,
            }
        """
        # Gọi 2 judge song song
        result_a, result_b = await asyncio.gather(
            _call_judge(JUDGE_A_MODEL, question, answer, ground_truth),
            _call_judge(JUDGE_B_MODEL, question, answer, ground_truth),
        )

        score_a = result_a["score"]
        score_b = result_b["score"]
        delta   = abs(score_a - score_b)

        # ── Logic xử lý xung đột ──────────────────────────────────────
        if delta <= CONFLICT_THRESHOLD:
            # Đồng thuận: trung bình cộng
            final_score    = (score_a + score_b) / 2
            agreement_rate = 1.0
            conflict       = False
            resolution     = "consensus_average"
        elif delta == 2:
            # Xung đột nhẹ: lấy điểm thấp hơn (conservative)
            final_score    = min(score_a, score_b)
            agreement_rate = 0.5
            conflict       = True
            resolution     = "conflict_minor_take_lower"
        else:
            # Xung đột nặng (delta ≥ 3): lấy điểm thấp hơn + đánh dấu cần review
            final_score    = min(score_a, score_b)
            agreement_rate = 0.0
            conflict       = True
            resolution     = "conflict_major_needs_review"

        reasoning = (
            f"[{JUDGE_A_MODEL.split('/')[1]}] Score={score_a}: {result_a['reasoning']} | "
            f"[{JUDGE_B_MODEL.split('/')[1]}] Score={score_b}: {result_b['reasoning']}"
        )

        return {
            "final_score"       : final_score,
            "agreement_rate"    : agreement_rate,
            "conflict"          : conflict,
            "resolution"        : resolution,
            "individual_scores" : {
                "judge_a": score_a,
                "judge_b": score_b,
            },
            "reasoning"         : reasoning,
        }

    async def check_position_bias(self, response_a: str, response_b: str) -> Dict:
        """
        Position Bias Test: đổi chỗ A/B, kiểm tra Judge có cho điểm nhất quán không.

        Nếu Judge luôn ưu tiên vị trí đầu tiên → có position bias.
        Agreement sau đổi chỗ < 0.7 → cảnh báo bias.

        Returns:
            {
              "bias_detected" : bool,
              "score_order_ab": float,
              "score_order_ba": float,
              "delta"         : float,
            }
        """
        # Gọi judge với thứ tự gốc
        prompt_ab = (
            f"Câu trả lời A: {response_a}\nCâu trả lời B: {response_b}\n"
            "Câu trả lời nào tốt hơn? Chỉ trả lời JSON: "
            '{{"better": "A" hoặc "B", "score_a": 1-5, "score_b": 1-5}}'
        )
        prompt_ba = (
            f"Câu trả lời A: {response_b}\nCâu trả lời B: {response_a}\n"
            "Câu trả lời nào tốt hơn? Chỉ trả lời JSON: "
            '{{"better": "A" hoặc "B", "score_a": 1-5, "score_b": 1-5}}'
        )

        async def _score(prompt):
            try:
                resp = await _client.chat.completions.create(
                    model=JUDGE_A_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=100,
                )
                content = resp.choices[0].message.content or "{}"
                data    = json.loads(content.strip())
                return data.get("score_a", 3), data.get("score_b", 3)
            except Exception:
                return 3, 3

        (sa_ab, sb_ab), (sa_ba, sb_ba) = await asyncio.gather(
            _score(prompt_ab), _score(prompt_ba)
        )

        # response_a score ở 2 lần đánh giá (lần 2 đổi chỗ nên đọc sb_ba)
        score_order_ab = sa_ab
        score_order_ba = sb_ba   # Vị trí của response_a khi bị đổi sang B
        delta = abs(score_order_ab - score_order_ba)

        return {
            "bias_detected" : delta >= 2,
            "score_order_ab": score_order_ab,
            "score_order_ba": score_order_ba,
            "delta"         : delta,
            "note"          : "delta≥2 → có position bias" if delta >= 2 else "OK",
        }