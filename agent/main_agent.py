"""
Main Agent - Support Agent với RAG architecture
Sinh viên nên thay thế phần này bằng Agent thực tế đã phát triển ở các buổi trước
"""
import asyncio
import random
from typing import List, Dict, Optional


class MainAgent:
    """
    Agent mẫu sử dụng kiến trúc RAG đơn giản.
    Giả lập retrieval và generation để test benchmark system.
    """
    def __init__(self):
        self.name = "SupportAgent-v1"
        # Giả lập document store
        self.documents = [
            {"id": "doc_1", "content": "Chính sách hoàn tiền: 14 ngày"},
            {"id": "doc_2", "content": "Cài đặt SSL: 4 bước"},
            {"id": "doc_3", "content": "Liên hệ: support@company.com"},
            {"id": "doc_4", "content": "Bảo hành: 12 tháng"},
            {"id": "doc_5", "content": "Đổi mật khẩu: Settings > Security"},
            {"id": "doc_6", "content": "Tạo tài khoản: Sign Up + OTP"},
            {"id": "doc_7", "content": "Phí ship: Miễn phí >500k"},
            {"id": "doc_8", "content": "Giao hàng: 2-5 ngày"},
            {"id": "doc_9", "content": "Bảo mật: Không chia sẻ"},
            {"id": "doc_10", "content": "API: request + API key"},
        ]

    async def retrieve(self, query: str) -> List[Dict]:
        """
        Retrieval stage: Tìm kiếm documents liên quan
        Giả lập bằng keyword matching đơn giản
        """
        # Giả lập retrieval - trong thực tế sẽ dùng Vector DB
        query_lower = query.lower()

        # Simple keyword matching
        relevant_docs = []
        for doc in self.documents:
            # Check if any keyword matches
            keywords = doc["content"].lower().split()
            if any(kw in query_lower for kw in keywords):
                relevant_docs.append(doc)
            elif any(kw in doc["content"].lower() for kw in query_lower.split()):
                relevant_docs.append(doc)

        # Nếu không có relevant docs, trả về random
        if not relevant_docs:
            relevant_docs = random.sample(self.documents, min(3, len(self.documents)))

        # Simulate retrieval với ranking (có thứ tự)
        retrieved_ids = [doc["id"] for doc in relevant_docs[:3]]
        return retrieved_ids, relevant_docs[:3]

    async def generate(self, query: str, contexts: List[Dict]) -> str:
        """
        Generation stage: Sinh câu trả lời từ contexts
        Giả lập bằng template
        """
        # Giả lập độ trễ LLM
        await asyncio.sleep(0.1)

        # Tạo response từ context
        if contexts:
            context_text = " | ".join([c["content"] for c in contexts])
            return f"Dựa trên thông tin: {context_text}. Trả lời câu hỏi '{query}': [Câu trả lời từ context]"
        else:
            return f"Tôi không tìm thấy thông tin liên quan để trả lời câu hỏi '{query}'."

    async def query(self, question: str) -> Dict:
        """
        Main query pipeline:
        1. Retrieval: Tìm kiếm context liên quan
        2. Generation: Gọi LLM để sinh câu trả lời
        """
        # 1. Retrieval
        retrieved_ids, contexts = await self.retrieve(question)

        # 2. Generation
        answer = await self.generate(question, contexts)

        return {
            "answer": answer,
            "contexts": [c["content"] for c in contexts],
            "retrieved_ids": retrieved_ids,
            "metadata": {
                "model": "gpt-4o-mini",
                "tokens_used": 150,
                "sources": [c["id"] for c in contexts]
            }
        }


if __name__ == "__main__":
    agent = MainAgent()

    async def test():
        resp = await agent.query("Làm thế nào để đổi mật khẩu?")
        print(resp)

    asyncio.run(test())