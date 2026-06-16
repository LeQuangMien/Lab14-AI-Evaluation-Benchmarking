# Individual Reflection - Trần Đức Tâm - 2A202600803

## 1. Engineering Contribution
Trong bài lab này, em phụ trách hoàn thiện pipeline đánh giá tự động cho AI Agent theo hướng có thể chạy lại nhiều lần và tạo báo cáo ổn định. Các phần chính gồm:

- Thiết kế golden dataset 55 cases với `expected_retrieval_ids`, difficulty và case type.
- Xây dựng agent RAG mô phỏng local, có retrieval, `retrieved_ids`, context, token usage và estimated cost.
- Hoàn thiện evaluator tính Hit Rate, MRR, faithfulness và relevancy.
- Triển khai multi-judge consensus offline gồm accuracy judge và risk judge, có agreement rate và conflict resolution.
- Bổ sung regression release gate so sánh Agent V1 và Agent V2 dựa trên quality, retrieval, latency, agreement và cost.
- Tạo `reports/summary.json`, `reports/benchmark_results.json` và `analysis/failure_analysis.md`.

## 2. Technical Depth
**Hit Rate** đo việc ít nhất một tài liệu đúng có xuất hiện trong top-k kết quả retrieval hay không. Chỉ số này giúp biết retrieval có lấy đúng nguồn trước khi đánh giá câu trả lời.

**MRR** đo vị trí của tài liệu đúng đầu tiên trong danh sách retrieval. Nếu tài liệu đúng đứng top 1 thì MRR = 1.0; đứng top 2 thì MRR = 0.5. MRR giúp phân biệt hệ thống lấy đúng nhưng xếp thấp với hệ thống lấy đúng ngay từ đầu.

**Agreement Rate** cho biết mức độ đồng thuận giữa các judge. Nếu hai judge lệch nhiều, kết quả không nên được tin tuyệt đối mà cần dùng logic hòa giải hoặc review thủ công.

**Position Bias** là hiện tượng judge ưu tiên câu trả lời A/B chỉ vì vị trí hiển thị. Trong bài này, judge deterministic dựa trên token coverage nên không phụ thuộc thứ tự, nhưng ở hệ thống thật cần kiểm tra bằng cách hoán đổi vị trí response.

**Trade-off cost và quality:** Chấm toàn bộ cases bằng judge mạnh nhất cho kết quả ổn hơn nhưng tốn chi phí. Cách tối ưu là cache kết quả judge cho cases không đổi, dùng judge rẻ cho easy cases, và giữ multi-judge nghiêm ngặt cho hard/red-team cases.

## 3. Problem Solving
Khó khăn chính là tạo một hệ thống chạy được offline nhưng vẫn thể hiện đúng yêu cầu expert-level. Em giải quyết bằng cách mô phỏng RAG và multi-judge theo rubric rõ ràng, không phụ thuộc API key. Nhờ vậy benchmark có thể chạy nhanh, có số liệu thật, và vẫn kiểm tra được các thành phần quan trọng như retrieval metrics, judge agreement, regression gate, failure clustering và 5 Whys.

## 4. Kết quả cá nhân
Sau khi chạy benchmark, hệ thống đạt 55 test cases, Hit Rate 98.2%, MRR 0.894, Agreement Rate 100%, Pass Rate 96.4% và Release Gate là `APPROVE`. Các lỗi còn lại chủ yếu đến từ retrieval lexical đơn giản, là hướng cải tiến tiếp theo bằng embedding retrieval và reranking.
