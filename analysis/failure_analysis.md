# Failure Analysis Report

## 1. Benchmark Overview
- Total cases: 55
- Pass rate: 96.4%
- Average judge score: 4.73 / 5.0
- Hit Rate: 98.2%
- MRR: 0.894
- Agreement Rate: 100.0%
- Estimated eval cost: $0.000943

## 2. Failure Clustering
| Cluster | Count | Interpretation |
|---|---:|---|
| pass | 51 | Case passed quality and retrieval checks. |
| red_team_watch | 2 | Hard safety case passed but should stay in the monitored suite. |
| answer_quality | 1 | Answer quality fell below judge threshold. |
| retrieval_miss | 1 | Retriever did not return the expected source document. |

## 3. 5 Whys on Worst Cases
### Case 1: CASE-049 - answer_quality
1. Symptom: score=1.75, hit_rate=1.0, question='If a case fails around table, what root cause should the team inspect?'
2. Why 1: The answer quality is limited by the evidence returned to the generator.
3. Why 2: Retrieval ranking depends on lexical overlap and can miss paraphrases or conflicting wording.
4. Why 3: The local corpus is small and does not use embeddings or reranking.
5. Why 4: The lab implementation prioritizes reproducible offline evaluation over production retrieval infrastructure.
6. Root cause: Retrieval strategy and chunk/rerank design are the highest leverage improvement areas.

### Case 2: CASE-054 - retrieval_miss
1. Symptom: score=3.0, hit_rate=0.0, question='Why can a good answer still receive a low benchmark score?'
2. Why 1: The answer quality is limited by the evidence returned to the generator.
3. Why 2: Retrieval ranking depends on lexical overlap and can miss paraphrases or conflicting wording.
4. Why 3: The local corpus is small and does not use embeddings or reranking.
5. Why 4: The lab implementation prioritizes reproducible offline evaluation over production retrieval infrastructure.
6. Root cause: Retrieval strategy and chunk/rerank design are the highest leverage improvement areas.

### Case 3: CASE-004 - pass
1. Symptom: score=3.0, hit_rate=1.0, question='If a case fails around release, what root cause should the team inspect?'
2. Why 1: The answer quality is limited by the evidence returned to the generator.
3. Why 2: Retrieval ranking depends on lexical overlap and can miss paraphrases or conflicting wording.
4. Why 3: The local corpus is small and does not use embeddings or reranking.
5. Why 4: The lab implementation prioritizes reproducible offline evaluation over production retrieval infrastructure.
6. Root cause: Retrieval strategy and chunk/rerank design are the highest leverage improvement areas.

## 4. Improvement Plan
- Add embedding retrieval plus reranking for paraphrased questions.
- Keep the red-team set in every regression run.
- Cache judge results for unchanged cases to reduce cost by at least 30%.
- Route easy stable cases to a cheaper judge and reserve strict consensus for hard cases.
