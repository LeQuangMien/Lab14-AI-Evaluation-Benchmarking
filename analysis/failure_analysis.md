# Báo cáo Phân tích Thất bại (Failure Analysis Report)

## 1. Tổng quan Benchmark
- **Tổng số cases:** 50
- **Tỉ lệ Pass/Fail:** 45/5
- **Điểm RAGAS trung bình:**
    - Faithfulness: 0.90
    - Relevancy: 0.98
- **Điểm LLM-Judge trung bình:** 4.27 / 5.0

## 2. Phân nhóm lỗi (Failure Clustering)
| Nhóm lỗi | Số lượng | Triệu chứng | Root cause thường gặp |
|----------|----------|-------------|----------------------|
| Wrong Answer | 3 | Trả lời sai sự thật — chọn sai kỹ thuật, sai metric | Retrieval miss, prompt ambiguous |
| Hallucination | 1 | Bịa thông tin không có trong context (liệt kê RAGAS/BERTScore thay vì Hit Rate/MRR) | Faithfulness guardrail yếu |
| Refusal | 1 | Từ chối trả lời — agent nói "không có thông tin trong context" dù context có đủ | Guardrails quá chặt / prompt quá phòng thủ |

## 3. Phân tích 5 Whys (Chọn 3 case tệ nhất)

### Case #1: Cosine similarity vs Dot Product — Wrong Answer (score 1.0)
 

1. **Symptom:** Agent nói "context không có thông tin" → không trả lời được câu hỏi tại sao cosine similarity được ưu tiên hơn dot product

2. **Why 1:** Agent không tìm thấy lý do so sánh trong context

3. **Why 2:** doc_002 chỉ ghi "tìm kiếm dựa trên cosine similarity hoặc dot product" — không giải thích tại sao

4. **Why 3:** Retriever lấy đúng doc_002 (Hit Rate = 1.0) nhưng doc không chứa kiến thức cần thiết

5. **Why 4:** Ingestion pipeline chỉ lấy định nghĩa bề mặt, không bổ sung nội dung giải thích (explanatory content)

6. **Root Cause: Knowledge Base gap ở Ingestion pipeline** — doc_002 mô tả *sự tồn tại* của cosine similarity nhưng không giải thích *lý do* chọn nó. Vấn đề thật không phải prompt hay model, mà là **data pipeline** thiếu nội dung explanatory. Fix đúng chỗ: bổ sung nội dung vào doc_002 sẽ giải quyết hàng loạt failures tương tự.

### Case #2: Metric đánh giá retrieval trong Agentic AI — Hallucination (score 1.0)
 

1. **Symptom:** Agent liệt kê RAGAS, BERTScore, G-Eval thay vì Hit Rate và MRR

2. **Why 1:** Answer không dựa trên đúng document — lấy thông tin từ doc_008 thay vì doc_005

3. **Why 2:** Retriever ưu tiên doc_008 (LLM Evaluation) vì keyword "đánh giá" + "metric" match mạnh hơn doc_005 (Retrieval Metrics)

4. **Why 3:** doc_005 và doc_008 chưa được index với metadata phân biệt "retrieval metric" vs "generation metric"

5. **Why 4:** Ingestion pipeline không có bước tagging/taxonomy — mọi doc đều flat, không có label loại metric

6. **Root Cause: Ingestion pipeline không có metadata taxonomy** — keyword matching không phân biệt được "metric đo retrieval" (Hit Rate, MRR) và "metric đo generation" (RAGAS, BERTScore). Thêm tag `metric_type: retrieval` vào doc_005 và `metric_type: generation` vào doc_008 ở bước ingestion sẽ giải quyết disambiguation failure này.
 
---
 
### Case #3: Prompt Engineering vs RAG — Wrong Answer (score 1.0)
 

1. **Symptom:** Agent trả lời RAG trong khi đáp án đúng là Prompt Engineering (CoT/ReAct)

2. **Why 1:** Answer không dựa trên đúng document — chọn doc_001 (RAG) thay vì doc_009 (Prompt Engineering)

3. **Why 2:** Retriever không lấy được doc_009 vì câu hỏi dùng từ "retrieval" khiến RAG match cao hơn

4. **Why 3:** doc_001 và doc_009 chưa có cross-reference — không có thông tin so sánh "RAG vs Prompt Engineering khi không muốn thay đổi tham số"

5. **Why 4:** Ingestion pipeline chunk từng doc độc lập, không tạo "comparison chunks" nối các khái niệm liên quan

6. **Root Cause: Chunking strategy thiếu relational context** — mỗi doc là island độc lập. Câu hỏi phân biệt RAG vs Prompt Engineering cần một chunk kết nối cả hai khái niệm, nhưng pipeline hiện tại không tạo ra loại chunk này. Fix: thêm synthetic comparison chunks ở bước ingestion (ví dụ: "RAG vs Prompt Engineering vs Fine-tuning: khi nào dùng cái nào").

## 4. Kế hoạch cải tiến (Action Plan)
- [x] Thay đổi Chunking strategy từ Fixed-size sang Semantic Chunking.
- [x] Cập nhật System Prompt để nhấn mạnh vào việc "Chỉ trả lời dựa trên context".
- [x] Thêm bước Reranking vào Pipeline.
