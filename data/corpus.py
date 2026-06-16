"""Local benchmark corpus shared by the dataset generator and RAG agent."""

from __future__ import annotations

from typing import Dict, List


DOCUMENTS: List[Dict[str, str]] = [
    {
        "doc_id": "eval_factory_overview",
        "title": "Evaluation Factory Overview",
        "text": (
            "AI Evaluation Factory is an automated benchmark system for measuring an AI agent. "
            "It evaluates retrieval quality before generation quality, records measurable metrics, "
            "and compares agent versions through regression testing."
        ),
    },
    {
        "doc_id": "retrieval_metrics",
        "title": "Retrieval Metrics",
        "text": (
            "Hit Rate@k is 1 when at least one ground-truth document id appears in the top-k retrieved "
            "documents. Mean Reciprocal Rank is 1 divided by the first rank position of a relevant "
            "document, and is 0 when no relevant document is retrieved."
        ),
    },
    {
        "doc_id": "golden_dataset",
        "title": "Golden Dataset Requirements",
        "text": (
            "The golden dataset must contain at least 50 test cases. Every case should include a question, "
            "an expected answer, and ground-truth retrieval document ids so retrieval metrics can be computed."
        ),
    },
    {
        "doc_id": "multi_judge",
        "title": "Multi-Judge Consensus",
        "text": (
            "A reliable evaluation system should use at least two judge models. The system records individual "
            "judge scores, computes agreement rate, and resolves conflicts automatically when the scores differ."
        ),
    },
    {
        "doc_id": "cohens_kappa",
        "title": "Cohen Kappa",
        "text": (
            "Cohen's Kappa measures agreement between two raters after accounting for chance agreement. "
            "For benchmark pass/fail labels, kappa is computed from both judges' binary decisions."
        ),
    },
    {
        "doc_id": "release_gate",
        "title": "Regression Release Gate",
        "text": (
            "A release gate compares version two against version one. The release should be approved only when "
            "quality does not regress, retrieval remains stable, pass rate does not fall, and cost stays within budget."
        ),
    },
    {
        "doc_id": "async_performance",
        "title": "Async Performance",
        "text": (
            "The benchmark runner should execute cases asynchronously with a concurrency limit. For 50 cases, "
            "a production-quality evaluation pipeline should complete quickly and report latency, tokens, and cost."
        ),
    },
    {
        "doc_id": "failure_analysis",
        "title": "Failure Analysis",
        "text": (
            "Failure analysis groups errors such as hallucination, incomplete answer, retrieval miss, safety issue, "
            "and tone mismatch. A 5 Whys investigation should identify whether the root cause is ingestion, chunking, "
            "retrieval, prompting, or judging."
        ),
    },
    {
        "doc_id": "hard_cases",
        "title": "Hard Case Design",
        "text": (
            "Hard benchmark cases include prompt injection, goal hijacking, out-of-context questions, ambiguous questions, "
            "conflicting information, multi-turn carry-over, correction, latency stress, and cost efficiency tests."
        ),
    },
    {
        "doc_id": "safety_policy",
        "title": "Safety and Grounding Policy",
        "text": (
            "The agent must answer only from retrieved context. If the context is insufficient, it should say that it "
            "does not know and ask for clarification instead of inventing unsupported facts."
        ),
    },
    {
        "doc_id": "position_bias",
        "title": "Position Bias",
        "text": (
            "Position bias happens when a judge favors the first or second answer because of presentation order. "
            "A bias check swaps answer order and compares whether the judgment changes."
        ),
    },
    {
        "doc_id": "cost_policy",
        "title": "Cost Policy",
        "text": (
            "Evaluation cost should be tracked per case and in aggregate. Token usage includes prompt tokens, completion "
            "tokens, and total tokens. A practical optimization target is reducing evaluation cost by 30 percent without "
            "reducing measurement quality."
        ),
    },
]


def get_corpus() -> List[Dict[str, str]]:
    return DOCUMENTS.copy()


def get_document(doc_id: str) -> Dict[str, str]:
    for doc in DOCUMENTS:
        if doc["doc_id"] == doc_id:
            return doc
    raise KeyError(f"Unknown document id: {doc_id}")
