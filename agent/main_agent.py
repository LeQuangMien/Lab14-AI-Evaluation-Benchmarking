"""
agent/main_agent.py
===================
MainAgent: RAG Agent dùng OpenRouter cho generation.
- Retrieval: Mock vector DB khớp với MOCK_DOCS trong synthetic_gen.py
  (Trong hệ thống thực: thay bằng FAISS/Chroma lookup thực sự)
- Generation: Gọi LLM qua OpenRouter sinh câu trả lời từ context
"""

import asyncio
import os
from typing import Dict, List

from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

# Model dùng cho generation (khác với judge để tiết kiệm quota)
AGENT_MODEL = "deepseek/deepseek-v4-flash"

_client = AsyncOpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    default_headers={
        "HTTP-Referer": "https://github.com/LeQuangMien/Lab14-AI-Evaluation-Benchmarking",
        "X-Title": "Lab14 MainAgent",
    }
)

# ── Mock Vector DB ──────────────────────────────────────────────────────
# Phải khớp hoàn toàn với MOCK_DOCS trong data/synthetic_gen.py
# để Hit Rate / MRR có ý nghĩa thực sự.
MOCK_VECTOR_DB: Dict[str, str] = {
    "doc_001": "RAG (Retrieval-Augmented Generation) là kiến trúc kết hợp retrieval và generation. Thay vì chỉ dựa vào tham số mô hình, RAG tra cứu tài liệu liên quan từ vector database trước khi sinh câu trả lời.",
    "doc_002": "Vector Database lưu trữ embedding của văn bản dưới dạng vector số thực. Các hệ thống phổ biến gồm FAISS (Meta), Chroma, Pinecone, Weaviate. Tìm kiếm dựa trên cosine similarity hoặc dot product.",
    "doc_003": "Chunking strategy ảnh hưởng lớn đến chất lượng retrieval. Các chiến lược gồm: Fixed-size chunking, Recursive character splitting, Semantic chunking. Chunk quá lớn làm loãng thông tin; chunk quá nhỏ mất context.",
    "doc_004": "Embedding model chuyển đổi văn bản thành vector. Các model phổ biến: text-embedding-ada-002 (OpenAI), all-MiniLM-L6-v2 (Sentence Transformers), E5-large. Chất lượng embedding quyết định độ chính xác retrieval.",
    "doc_005": "Hit Rate là tỷ lệ queries có ít nhất 1 tài liệu liên quan trong kết quả retrieve. MRR (Mean Reciprocal Rank) đo vị trí trung bình của tài liệu đúng đầu tiên trong danh sách kết quả.",
    "doc_006": "Hallucination trong LLM xảy ra khi mô hình tạo ra thông tin không có trong context. RAG giảm hallucination bằng cách neo câu trả lời vào tài liệu thực. Faithfulness metric đo tỷ lệ câu trả lời có căn cứ trong context.",
    "doc_007": "Fine-tuning điều chỉnh tham số mô hình trên dataset chuyên biệt. Khác với RAG, fine-tuning baked kiến thức vào weights. LoRA và QLoRA là kỹ thuật fine-tuning hiệu quả về bộ nhớ.",
    "doc_008": "LLM Evaluation framework gồm các phương pháp: RAGAS (đánh giá pipeline RAG), G-Eval (dùng GPT làm judge), BERTScore (đo semantic similarity), ROUGE (đo overlap n-gram). Mỗi metric phản ánh một khía cạnh khác nhau.",
    "doc_009": "Prompt Engineering là kỹ thuật thiết kế input cho LLM. Các kỹ thuật gồm: Zero-shot, Few-shot, Chain-of-Thought (CoT), ReAct. Prompt tốt giúp mô hình reasoning chính xác hơn mà không cần fine-tuning.",
    "doc_010": "Agentic AI là hệ thống AI có khả năng tự lập kế hoạch và thực thi nhiều bước để hoàn thành mục tiêu. ReAct (Reasoning + Acting) kết hợp reasoning chain với tool calls. Multi-agent system phân chia nhiệm vụ giữa nhiều agent chuyên biệt.",
}

# ── Keyword → doc mapping (mock retrieval logic) ────────────────────────
# Trong hệ thống thực: thay bằng embedding similarity search
KEYWORD_TO_DOCS: List[tuple] = [
    (["rag", "retrieval-augmented", "retrieval augmented"],          ["doc_001"]),
    (["vector database", "vector db", "faiss", "chroma", "pinecone", "weaviate", "cosine"], ["doc_002"]),
    (["chunk", "chunking", "split", "phân đoạn"],                    ["doc_003"]),
    (["embedding", "vector hóa", "sentence transformer", "e5"],      ["doc_004"]),
    (["hit rate", "mrr", "reciprocal rank", "recall"],               ["doc_005"]),
    (["hallucination", "ảo giác", "faithfulness", "căn cứ"],         ["doc_006"]),
    (["fine-tuning", "fine tuning", "lora", "qlora", "finetune"],    ["doc_007"]),
    (["ragas", "evaluation", "đánh giá", "bertscore", "rouge", "benchmark"], ["doc_008"]),
    (["prompt", "few-shot", "zero-shot", "chain-of-thought", "cot", "react", "prompt engineering"], ["doc_009"]),
    (["agent", "agentic", "multi-agent", "tool call", "lập kế hoạch"], ["doc_010"]),
]


def _mock_retrieve(question: str, top_k: int = 3) -> List[str]:
    """
    Mock retrieval: tìm doc_ids liên quan dựa trên keyword matching.
    Trả về list doc_ids theo thứ tự độ liên quan (doc liên quan nhất đầu).

    Trong hệ thống thực: thay hàm này bằng:
        query_embedding = embed_model.encode(question)
        results = vector_db.similarity_search(query_embedding, k=top_k)
        return [r.doc_id for r in results]
    """
    q_lower   = question.lower()
    scored    = []   # (score, doc_id)

    for keywords, doc_ids in KEYWORD_TO_DOCS:
        score = sum(1 for kw in keywords if kw in q_lower)
        if score > 0:
            for doc_id in doc_ids:
                scored.append((score, doc_id))

    # Sắp xếp theo score giảm dần, dedup
    seen    = set()
    ordered = []
    for score, doc_id in sorted(scored, key=lambda x: -x[0]):
        if doc_id not in seen:
            seen.add(doc_id)
            ordered.append(doc_id)
        if len(ordered) >= top_k:
            break

    # Fallback: nếu không match gì → trả doc_001 (về RAG)
    if not ordered:
        ordered = ["doc_001"]

    return ordered


class MainAgent:
    """
    RAG Agent với Mock Vector DB.
    Interface: agent.query(question) → Dict
    """

    def __init__(self, name: str = "SupportAgent-v1", top_k: int = 3):
        self.name  = name
        self.top_k = top_k

    async def query(self, question: str) -> Dict:
        """
        RAG pipeline:
        1. Retrieve: tìm top_k docs liên quan
        2. Build context từ docs
        3. Generate: gọi LLM sinh câu trả lời có căn cứ

        Returns:
            {
              "answer"           : str,
              "retrieved_doc_ids": List[str],   # Dùng cho Hit Rate / MRR
              "contexts"         : List[str],   # Nội dung chunks
              "metadata"         : Dict,
            }
        """
        # ── Bước 1: Retrieve ──────────────────────────────────────────
        retrieved_ids = _mock_retrieve(question, top_k=self.top_k)
        contexts      = [MOCK_VECTOR_DB[doc_id] for doc_id in retrieved_ids
                         if doc_id in MOCK_VECTOR_DB]
        context_text  = "\n\n".join(
            f"[{doc_id}] {text}" for doc_id, text in zip(retrieved_ids, contexts)
        )

        # ── Bước 2: Generate ──────────────────────────────────────────
        system_prompt = (
            "Bạn là trợ lý AI chuyên về Machine Learning và RAG systems. "
            "Hãy trả lời câu hỏi DỰA TRÊN context được cung cấp. "
            "Nếu context không đủ thông tin, hãy nói rõ. "
            "Trả lời ngắn gọn, chính xác, bằng tiếng Việt."
        )
        user_prompt = (
            f"## Context\n{context_text}\n\n"
            f"## Câu hỏi\n{question}\n\n"
            "## Câu trả lời"
        )

        tokens_used = 0
        try:
            resp = await _client.chat.completions.create(
                model=AGENT_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=400,
            )
            answer      = resp.choices[0].message.content or "Không thể sinh câu trả lời."
            tokens_used = getattr(resp.usage, "total_tokens", 150) if resp.usage else 150
        except Exception as e:
            answer      = f"[LLM error] {e} — Fallback: {contexts[0][:200] if contexts else 'No context'}"
            tokens_used = 0

        return {
            "answer"           : answer.strip(),
            "retrieved_doc_ids": retrieved_ids,   # ← Field quan trọng cho Retrieval Eval
            "contexts"         : contexts,
            "metadata"         : {
                "model"      : AGENT_MODEL,
                "tokens_used": tokens_used,
                "sources"    : retrieved_ids,
                "agent_name" : self.name,
            },
        }


# ── Quick test khi chạy trực tiếp ─────────────────────────────────────
if __name__ == "__main__":
    async def _test():
        agent = MainAgent()
        questions = [
            "RAG là gì và nó hoạt động như thế nào?",
            "Tại sao chunking strategy quan trọng?",
            "So sánh Fine-tuning và RAG?",
        ]
        for q in questions:
            print(f"\n❓ {q}")
            resp = await agent.query(q)
            print(f"📄 Retrieved: {resp['retrieved_doc_ids']}")
            print(f"💬 Answer: {resp['answer'][:150]}...")

    asyncio.run(_test())