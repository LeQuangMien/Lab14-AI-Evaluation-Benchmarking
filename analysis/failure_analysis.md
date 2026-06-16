# Báo cáo Phân tích Thất bại (Failure Analysis Report)

## 1. Tổng quan Benchmark
- **Tổng số cases:** 50
- **Tỉ lệ Pass/Fail:** 48/2
- **Faithfulness proxy trung bình:** 0.1552
- **Relevancy proxy trung bình:** 0.4144
- **Hit Rate trung bình:** 0.94
- **MRR trung bình:** 0.91
- **Điểm Multi-Judge trung bình:** 4.78 / 5.0
- **Agreement Rate:** 0.98
- **Cohen's Kappa:** 0.7899

## 2. Phân nhóm lỗi (Failure Clustering)
| Nhóm lỗi | Số lượng | Nguyên nhân dự kiến |
|----------|----------|---------------------|
| incomplete | 2 | Incomplete detected by multi-judge consensus |
| retrieval_miss | 2 | Retrieval Miss detected by multi-judge consensus |
| tone_mismatch | 1 | Tone Mismatch detected by multi-judge consensus |

## 3. Phân tích 5 Whys (3 case tệ nhất)
### Case #1: hard_43_position_bias
1. **Symptom:** Judge score 1.0 for question: Vì sao cần đổi vị trí response A/B khi đánh giá?
2. **Why 1:** Agent response did not fully match the expected answer.
3. **Why 2:** Retrieval hit rate was 0.0 and MRR was 0.0.
4. **Why 3:** Judge tags were incomplete, retrieval_miss.
5. **Why 4:** The case type was position-bias, which stresses a known weak point.
6. **Root Cause:** Retrieval/chunking mismatch: expected document was not retrieved in top-k.

### Case #2: hard_50_multi_turn_carry_over
1. **Symptom:** Judge score 2.5 for question: Sau câu trả lời trước, hãy nói tiếp 'nó' đo cái gì trong Evaluation Factory?
2. **Why 1:** Agent response did not fully match the expected answer.
3. **Why 2:** Retrieval hit rate was 1.0 and MRR was 1.0.
4. **Why 3:** Judge tags were incomplete, retrieval_miss, tone_mismatch.
5. **Why 4:** The case type was multi-turn-carry-over, which stresses a known weak point.
6. **Root Cause:** Generation prompt: answer did not cover all expected facts.

### Case #3: hard_41_conflicting_judge
1. **Symptom:** Judge score 3.0 for question: Nếu một judge cho 5 điểm và judge còn lại cho 2 điểm thì hệ thống phải làm gì?
2. **Why 1:** Agent response did not fully match the expected answer.
3. **Why 2:** Retrieval hit rate was 1.0 and MRR was 1.0.
4. **Why 3:** Judge tags were incomplete, retrieval_miss.
5. **Why 4:** The case type was conflicting-judge, which stresses a known weak point.
6. **Root Cause:** Generation prompt: answer did not cover all expected facts.

## 4. Kế hoạch cải tiến (Action Plan)
- [ ] Tăng chất lượng retrieval bằng semantic embedding hoặc reranking cho hard cases.
- [ ] Tách chunk theo chủ đề thay vì fixed corpus đoạn dài.
- [ ] Siết system prompt cho out-of-context, prompt injection và ambiguous questions.
- [ ] Thêm calibration set để đo position bias và judge disagreement định kỳ.
- [ ] Theo dõi cost/token theo loại case để giảm chi phí eval mà không giảm độ chính xác.
