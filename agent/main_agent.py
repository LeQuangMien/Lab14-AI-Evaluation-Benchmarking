import asyncio
import json
import math
import re
from pathlib import Path
from typing import Dict, List


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "does", "for",
    "from", "how", "if", "in", "is", "it", "of", "or", "should", "the",
    "to", "what", "when", "where", "why", "with", "v1", "v2",
}


class MainAgent:
    """A lightweight local RAG agent used for repeatable lab benchmarking."""

    def __init__(self, version: str = "Agent_V2_Optimized", top_k: int = 3):
        self.name = version
        self.top_k = top_k
        self.knowledge_base = self._load_knowledge_base()

    def _load_knowledge_base(self) -> List[Dict]:
        kb_path = Path(__file__).resolve().parents[1] / "data" / "knowledge_base.json"
        if kb_path.exists():
            with open(kb_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return [
            {
                "id": "DOC-FALLBACK-001",
                "title": "Fallback Evaluation Note",
                "text": "Run python data/synthetic_gen.py to create the full knowledge base and golden dataset.",
                "keywords": ["evaluation", "dataset"],
            }
        ]

    def _tokenize(self, text: str) -> List[str]:
        tokens = re.findall(r"[a-zA-Z0-9-]+", text.lower())
        return [token for token in tokens if token not in STOPWORDS and len(token) > 1]

    def _score_document(self, query_tokens: List[str], doc: Dict) -> float:
        doc_text = f"{doc['title']} {doc['text']} {' '.join(doc.get('keywords', []))}"
        doc_tokens = self._tokenize(doc_text)
        if not query_tokens or not doc_tokens:
            return 0.0
        overlap = set(query_tokens) & set(doc_tokens)
        title_bonus = sum(1 for token in query_tokens if token in self._tokenize(doc["title"])) * 0.8
        keyword_bonus = sum(1 for token in query_tokens if token in self._tokenize(" ".join(doc.get("keywords", [])))) * 0.6
        return len(overlap) + title_bonus + keyword_bonus

    def _retrieve(self, question: str) -> List[Dict]:
        query_tokens = self._tokenize(question)
        scored = [
            {**doc, "score": self._score_document(query_tokens, doc)}
            for doc in self.knowledge_base
        ]
        scored.sort(key=lambda item: item["score"], reverse=True)
        return [doc for doc in scored[: self.top_k] if doc["score"] > 0]

    def _build_answer(self, question: str, docs: List[Dict]) -> str:
        if not docs:
            return (
                "I do not have enough support in the provided documents to answer this. "
                "This should be treated as an out-of-context or unsupported request."
            )

        lead = docs[0]
        evidence = " ".join(doc["text"] for doc in docs[:2])
        if "ignore" in question.lower() or "invent" in question.lower():
            return (
                "I will not ignore the benchmark context or invent unsupported facts. "
                f"Grounded answer from {lead['id']}: {evidence}"
            )
        return f"Grounded answer from {lead['id']}: {evidence}"

    async def query(self, question: str) -> Dict:
        await asyncio.sleep(0.02)
        docs = self._retrieve(question)
        answer = self._build_answer(question, docs)
        estimated_tokens = max(40, math.ceil((len(question) + len(answer)) / 4))
        cost_per_1k = 0.00015 if "V2" in self.name else 0.00020

        return {
            "answer": answer,
            "contexts": [doc["text"] for doc in docs],
            "retrieved_ids": [doc["id"] for doc in docs],
            "metadata": {
                "model": "local-rag-simulator",
                "agent_version": self.name,
                "tokens_used": estimated_tokens,
                "estimated_cost_usd": round((estimated_tokens / 1000) * cost_per_1k, 6),
                "sources": [doc["title"] for doc in docs],
            },
        }


if __name__ == "__main__":
    async def test():
        agent = MainAgent()
        print(await agent.query("What does Retrieval Metrics say about MRR?"))

    asyncio.run(test())
