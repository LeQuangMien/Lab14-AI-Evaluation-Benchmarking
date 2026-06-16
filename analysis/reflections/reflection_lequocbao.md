# Individual Reflection - Lab 14 AI Evaluation Factory

## 👤 Thông tin cá nhân
- **Họ tên:** Lê Quốc Bảo
- **Vai trò:** Retrieval + Judge 

---

## 1. Engineering Contribution (15 điểm)

### Các module đã implement:

#### a) Retrieval Evaluation (`engine/retrieval_eval.py`)
- ✅ Implement `calculate_hit_rate()` - tính Hit Rate với top-k retrieval
- ✅ Implement `calculate_mrr()` - tính Mean Reciprocal Rank  
- ✅ Implement `calculate_cohens_kappa()` - đo lường inter-judge agreement
- ✅ Implement `evaluate_batch()` - chạy eval cho toàn bộ dataset

#### b) Multi-Judge Engine (`engine/llm_judge.py`)
- ✅ Triển khai với 2 models qua OpenRouter: GPT-4o-mini + Claude-3-haiku
- ✅ Implement `evaluate_multi_judge()` - gọi song song 2 models
- ✅ Implement `_resolve_conflict()` - xử lý khi 2 judges đánh giá khác nhau
- ✅ Implement `call_openrouter()` - API wrapper cho OpenRouter
- ✅ Implement `check_position_bias()` - phát hiện position bias trong Judge

#### c) Async Benchmark Runner (`engine/runner.py`)
- ✅ Implement `BenchmarkRunner` với batch processing
- ✅ Implement semaphore để tránh rate limit
- ✅ Implement cost tracking (token usage + estimated cost)
- ✅ Optimize: V1 chạy 50 cases trong 53.12s, V2 chạy 50 cases trong 52.20s, đạt yêu cầu < 2 phút cho 50 cases

#### d) Regression Gate (`main.py`)
- ✅ Implement `RegressionGate` với các thresholds:
  - min_hit_rate: 0.70
  - min_mrr: 0.60
  - min_judge_score: 3.0
  - min_agreement: 0.70
  - max_regression: 0.05

---

## 2. Technical Depth (15 điểm)

### Các khái niệm đã hiểu và giải thích được:

#### a) MRR (Mean Reciprocal Rank)
- **Định nghĩa:** Trung bình cộng của 1/rank đầu tiên tìm thấy relevant document
- **Công thức:** MRR = (1/N) * Σ(1/rank_i)
- **Ý nghĩa:** Đo lường "nhanh chóng" tìm thấy thông tin đúng
- **Trong benchmark:** MRR V1 = 62.7%, V2 = 62.0%

#### b) Cohen's Kappa
- **Định nghĩa:** Đo lường sự đồng thuận giữa 2 judges, loại bỏ random agreement
- **Công thức:** κ = (P_o - P_e) / (1 - P_e)
- **Interpretation:**
  - 0.0-0.2: Slight
  - 0.2-0.4: Fair
  - 0.4-0.6: Moderate
  - 0.6-0.8: Substantial
  - 0.8-1.0: Almost perfect

#### c) Position Bias
- **Định nghĩa:** Xu hướng của Judge prefer vị trí đầu tiên (A) hơn vị trí thứ hai (B)
- **Cách phát hiện:** Đổi chỗ A và B, nếu scores khác nhau → có position bias
- **Trong benchmark:** Đã implement hàm check_position_bias()

### Trade-off Chi phí vs Chất lượng:

| Strategy | Cost | Quality | Notes |
|----------|------|---------|-------|
| Single Judge GPT-4o | $ | ★★★★ | Fast, reliable |
| Multi-Judge (GPT+Claude) via OpenRouter | $$ | ★★★★★ | Best accuracy, thực sự đánh giá khác nhau |
| RAGAS only | $ | ★★★ | Fast, no LLM call |
| Full Multi-Judge + RAGAS | $$$$ | ★★★★★ | Most comprehensive |

**Kết quả thực tế:** Agreement rate V1 = 62%, V2 = 64%. Hai judge có khác biệt đánh giá, nên cần calibration thêm để đạt ngưỡng 70%.

---

## 3. Problem Solving (10 điểm)

### Các vấn đề đã gặp và giải quyết:

#### Problem 1: Lỗi Unicode Encoding trên Windows
- **Vấn đề:** Khi chạy main.py, Windows báo lỗi `UnicodeEncodeError`
- **Giải quyết:** Thêm UTF-8 encoding fix trong main.py

#### Problem 2: API Keys không load được
- **Vấn đề:** Environment variables không được load trong class
- **Giải quyết:** Thêm `load_dotenv()` trong constructor của MultiModelJudge

#### Problem 3: OpenRouter cần thêm headers
- **Vấn đề:** OpenRouter từ chối request thiếu HTTP-Referer và X-Title
- **Giải quyết:** Thêm các headers bắt buộc vào call_openrouter()

#### Problem 4: Rate Limiting từ API
- **Vấn đề:** Khi chạy nhiều concurrent requests, bị rate limit
- **Giải quyết:** Sử dụng `asyncio.Semaphore` để giới hạn số lượng concurrent requests

---

## 4. Đóng góp chính

- Implement RetrievalEvaluator với Hit Rate và MRR.
- Implement Multi-Judge với OpenRouter, GPT-4o-mini và Claude-3-haiku.
- Implement Async Benchmark Runner với batch processing, semaphore và cost tracking.
- Implement Regression Release Gate trong `main.py`.
- Cập nhật `failure_analysis.md` theo kết quả benchmark mới nhất và phân tích 5 Whys.
- Fix schema `summary.json` để tương thích với `check_lab.py`.

### Git Evidence

```text
9b99c4b - feat: Complete Lab 14 AI Evaluation Factory with Multi-Judge, Retrieval Eval, and Regression Testing
```

Commit này bao gồm các phần chính của bài lab: Retrieval Evaluation, Multi-Judge Consensus, Async Benchmark Runner, Regression Testing và generated reports.

---

## 5. Kết quả Benchmark Thực tế

| Metric | V1 (Baseline) | V2 (Optimized) | Delta |
|--------|---------------|----------------|-------|
| Hit Rate | 72% | 74% | +2% |
| MRR | 62.7% | 62.0% | -0.7% |
| Judge Score | 3.15 | 3.10 | -0.05 |
| Agreement | 62% | 64% | +2% |
| Pass Rate | 66% | 66% | +0% |
| Time | 53.12s | 52.20s | -0.92s |

**Final Decision:** BLOCK - Agreement rate 64% thấp hơn ngưỡng 70%

---

## 6. Tự đánh giá

| Hạng mục | Điểm tự đánh giá | Lý do |
|----------|-----------------|-------|
| Engineering Contribution | 14/15 | Đã implement đầy đủ các module phức tạp |
| Technical Depth | 14/15 | Hiểu và giải thích được MRR, Kappa, Position Bias |
| Problem Solving | 9/10 | Đã giải quyết 4 vấn đề chính |

**Tổng tự đánh giá: 37/40**

---

*Reflection submitted: 2026-06-16*