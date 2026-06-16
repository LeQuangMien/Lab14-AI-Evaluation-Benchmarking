# Tiêu Chí Chấm Điểm Expert Level - Lab Day 14

Bài lab được đánh giá trên thang 100 điểm, gồm 60 điểm nhóm và 40 điểm cá nhân.

## 1. Điểm Nhóm - 60 Điểm

| Hạng mục | Tiêu chí | Điểm |
|---|---|---:|
| Retrieval Evaluation | Tính Hit Rate và MRR cho tối thiểu 50 cases; giải thích quan hệ giữa retrieval quality và answer quality. | 10 |
| Dataset & SDG | Golden dataset 50+ cases có expected answer, context, difficulty, case type và ground-truth document IDs; có red-team/hard cases. | 10 |
| Multi-Judge Consensus | Có ít nhất 2 judge, individual scores, agreement rate và logic xử lý xung đột. | 15 |
| Regression Testing | So sánh Agent V1 và Agent V2; có release gate tự động dựa trên quality threshold. | 10 |
| Performance & Cost | Runner async, chạy nhanh dưới 2 phút cho 50+ cases; báo cáo latency, token usage và estimated cost. | 10 |
| Failure Analysis | Có failure clustering và phân tích 5 Whys chỉ ra root cause ở retrieval, chunking, ingestion hoặc prompting. | 5 |

## 2. Điểm Cá Nhân - 40 Điểm

| Hạng mục | Tiêu chí | Điểm |
|---|---|---:|
| Engineering Contribution | Nêu rõ đóng góp vào module phức tạp như async runner, metrics, judge hoặc regression gate. | 15 |
| Technical Depth | Giải thích MRR, agreement rate/Cohen's Kappa, position bias và trade-off cost-quality. | 15 |
| Problem Solving | Trình bày vấn đề phát sinh và cách giải quyết trong quá trình hoàn thiện hệ thống. | 10 |

## 3. Đối Chiếu Với Project Hiện Tại

| Yêu cầu | Trạng thái | File liên quan |
|---|---|---|
| 50+ golden cases | Hoàn thành: 55 cases | `data/synthetic_gen.py`, `data/golden_set.jsonl` |
| Ground-truth retrieval IDs | Hoàn thành | `expected_retrieval_ids` trong dataset |
| Hit Rate và MRR | Hoàn thành | `engine/retrieval_eval.py`, `reports/summary.json` |
| Multi-judge consensus | Hoàn thành | `engine/llm_judge.py` |
| Regression V1 vs V2 | Hoàn thành | `main.py`, `reports/summary.json` |
| Release gate | Hoàn thành | `main.py` |
| Async runner | Hoàn thành | `engine/runner.py` |
| Cost/token/latency report | Hoàn thành | `reports/summary.json`, `reports/benchmark_results.json` |
| Failure analysis + 5 Whys | Hoàn thành | `analysis/failure_analysis.md` |
| Reflection cá nhân | Hoàn thành | `analysis/reflections/reflection_Trần Đức Tâm_2A202600803.md` |

## 4. Điểm Liệt Cần Tránh
- Không có retrieval metrics thì điểm nhóm bị giới hạn mạnh.
- Chỉ dùng một judge đơn lẻ thì không đạt yêu cầu expert-level.
- Thiếu `reports/summary.json`, `reports/benchmark_results.json` hoặc `analysis/failure_analysis.md` sẽ làm script kiểm tra không đạt.
- Không chạy `python check_lab.py` trước khi nộp dễ bị lỗi định dạng không đáng có.
