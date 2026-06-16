# Individual Reflection - Trần Ngọc Thụy - 2A202600799

## Student Information
- **Name:** Trần Ngọc Thụy
- **Student ID:** 2A202600799
- **Role:** Evaluation Engineer - Retrieval Metrics, Multi-Judge Consensus và Failure Analysis

## Engineering Contribution
Trong bài lab này, em tham gia vào phần xây dựng hệ thống đánh giá tự động cho AI Agent, tập trung vào các module liên quan đến đo lường chất lượng thay vì chỉ kiểm tra câu trả lời cuối cùng. Cụ thể, em đóng góp vào việc thiết kế golden dataset 50 test cases có `expected_retrieval_ids`, giúp hệ thống có thể tính được chất lượng retrieval bằng Hit Rate và MRR.

Em cũng tham gia triển khai pipeline benchmark chạy bất đồng bộ để xử lý nhiều test cases nhanh hơn, đồng thời ghi lại latency, token usage và cost cho từng case. Phần này giúp hệ thống không chỉ đánh giá đúng/sai, mà còn đo được hiệu năng và chi phí vận hành.

Ngoài ra, em đóng góp vào Multi-Judge Consensus Engine, trong đó hệ thống sử dụng nhiều judge model để chấm câu trả lời, ghi lại điểm riêng của từng judge, tính agreement rate và xử lý trường hợp hai judge chấm lệch nhau. Kết quả benchmark sau cùng được tổng hợp vào `reports/summary.json` và `reports/benchmark_results.json`, đồng thời dùng để tạo báo cáo `failure_analysis.md`.

## Technical Depth
Qua bài lab, em hiểu rõ hơn rằng chất lượng câu trả lời của AI Agent phụ thuộc rất nhiều vào chất lượng retrieval. Nếu retrieval không lấy được đúng tài liệu, model dễ trả lời thiếu căn cứ hoặc hallucinate. Vì vậy, Hit Rate và MRR là hai chỉ số quan trọng để đánh giá bước retrieval trước khi đánh giá generation.

Hit Rate cho biết trong top-k tài liệu được truy xuất có ít nhất một tài liệu đúng hay không. Ví dụ, nếu tài liệu ground truth nằm trong top-k retrieved documents thì Hit Rate bằng 1, ngược lại bằng 0. MRR đo vị trí xuất hiện đầu tiên của tài liệu đúng; nếu tài liệu đúng đứng đầu thì MRR bằng 1.0, nếu đứng thứ hai thì MRR bằng 0.5. Vì vậy MRR phản ánh không chỉ việc có tìm thấy tài liệu đúng hay không, mà còn cho biết tài liệu đúng được xếp hạng cao đến mức nào.

Em cũng hiểu Cohen's Kappa dùng để đo mức độ đồng thuận giữa hai judge sau khi đã loại trừ khả năng đồng ý ngẫu nhiên. Trong kết quả benchmark, Agreement Rate đạt 0.98 và Cohen's Kappa đạt 0.7899, cho thấy hai judge có độ đồng thuận khá tốt. Tuy nhiên, Kappa có ý nghĩa sâu hơn agreement rate vì nó xét cả yếu tố chance agreement.

Position Bias là hiện tượng judge thiên vị câu trả lời ở vị trí A hoặc B thay vì đánh giá thuần theo chất lượng nội dung. Đây là lý do hệ thống cần có cơ chế đổi vị trí response để kiểm tra bias khi đánh giá so sánh.

Về trade-off chi phí và chất lượng, em thấy rằng dùng nhiều judge model giúp kết quả đáng tin cậy hơn nhưng làm tăng token usage và chi phí. Vì vậy hệ thống cần theo dõi cost per case, total cost và latency để cân bằng giữa độ chính xác đánh giá và chi phí vận hành.

## Problem Solving
Một vấn đề phát sinh trong quá trình chạy benchmark là OpenAI API không còn credit và OpenRouter bị giới hạn quota free-models-per-day. Điều này làm pipeline không thể tạo đủ report nếu chỉ phụ thuộc vào một provider.

Để giải quyết, nhóm đã điều chỉnh hệ thống để hỗ trợ chế độ Anthropic-only. Agent có thể dùng Claude để sinh câu trả lời, còn Multi-Judge vẫn giữ được yêu cầu nhiều judge bằng cách dùng hai model Claude khác nhau, ví dụ `claude-3-5-sonnet-latest` làm primary judge và `claude-3-5-haiku-latest` làm judge thứ hai. Đồng thời, nhóm giảm concurrency bằng biến `EVAL_CONCURRENCY` để hạn chế nguy cơ rate limit khi chạy nhiều test cases.

Sau khi xử lý, hệ thống vẫn giữ được các yêu cầu quan trọng của lab: có retrieval metrics, multi-judge consensus, regression gate, cost/token report và failure analysis. Điều em rút ra là khi xây dựng evaluation pipeline thực tế, khả năng fallback provider, kiểm soát rate limit và thiết kế report rõ ràng cũng quan trọng không kém bản thân thuật toán đánh giá.
