# Failure Cluster Analysis — Phase A

**Sinh viên:** Mai Quốc Hiệu  
**Ngày:** 2026-08-26

---

## 1. Aggregate RAGAS Scores theo Distribution

| Metric | factual | multi_hop | adversarial |
|---|---|---|---|
| faithfulness | 0.8250 | 0.4133 | 0.7000 |
| answer_relevancy | 0.7035 | 0.6276 | 0.5187 |
| context_precision | 0.8958 | 0.9417 | 0.9333 |
| context_recall | 0.8250 | 0.7708 | 0.6167 |
| **avg_score** | **0.8123** | **0.6883** | **0.6922** |

---

## 2. Bottom 10 Questions

| Rank | Distribution | Question | avg_score | worst_metric |
|---|---|---|---|---|
| 1 | factual | Bảo hiểm sức khỏe PVI có hạn mức bao nhiêu cho nhân viên? | 0.1250 | faithfulness |
| 2 | multi_hop | So sánh yêu cầu mật khẩu giữa policy v1.0 và v2.0 về độ dài tối thiểu, thời hạn đổi và MFA. | 0.1250 | faithfulness |
| 3 | factual | Thông tin lương thuộc cấp độ phân loại dữ liệu nào? | 0.2500 | faithfulness |
| 4 | multi_hop | Nhân viên cấp Lead nhận tổng phụ cấp hàng tháng (ăn trưa + điện thoại) là bao nhiêu? | 0.3333 | faithfulness |
| 5 | adversarial | Nhân viên được nghỉ bao nhiêu ngày phép năm? | 0.3333 | faithfulness |
| 6 | multi_hop | Nhân viên Manager có thâm niên 12 năm: tổng phụ cấp hàng tháng và số ngày phép năm theo v2024 là bao nhiêu? | 0.3333 | faithfulness |
| 7 | multi_hop | Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào? | 0.3750 | faithfulness |
| 8 | adversarial | Khi phát hiện malware trên máy tính công ty, nhân viên có nên tự xử lý không? | 0.4167 | faithfulness |
| 9 | adversarial | Nhân viên Manager có thể dùng VPN cá nhân (như NordVPN) khi WFH để tăng bảo mật thêm không? | 0.4167 | faithfulness |
| 10 | factual | Nam nhân viên được nghỉ bao nhiêu ngày khi vợ sinh con? | 0.5000 | faithfulness |

---

## 3. Failure Cluster Matrix

*(Mỗi ô = số câu có worst_metric = row, thuộc distribution = col)*

| worst_metric | factual | multi_hop | adversarial | Total |
|---|---|---|---|---|
| faithfulness | 3 | 15 | 3 | 21 |
| answer_relevancy | 12 | 2 | 1 | 15 |
| context_precision | 3 | 0 | 1 | 4 |
| context_recall | 2 | 3 | 5 | 10 |
| **Total** | 20 | 20 | 10 | 50 |

---

## 4. Dominant Failure Analysis

**Dominant distribution:** multi_hop / factual  
**Dominant metric:** faithfulness (21/50 câu có faithfulness là metric kém nhất)

**Lý do phân tích:**

1. Trên tập **multi_hop**, có đến 15/20 câu có metric yếu nhất là `faithfulness` (điểm trung bình chỉ đạt 0.4133). Lý do là các câu hỏi multi-hop yêu cầu liên kết thông tin phân tán qua nhiều chunks tài liệu (ví dụ: công thức tính ngày phép thâm niên từ tài liệu Policy kết hợp với bảng phụ cấp từ tài liệu Quy chế lương). Khi context chứa nhiều thông tin ghép nối, LLM có xu hướng tự tính toán sai hoặc suy diễn ngoài context được cung cấp (hallucination).
2. Trên tập **factual**, điểm yếu chủ yếu nằm ở `answer_relevancy` (12 câu), do prompt trả lời quá ngắn gọn hoặc chỉ trích dẫn văn bản thô mà không giải thích đúng trọng tâm câu hỏi.
3. Nhìn chung trong corpus HR Policy tiếng Việt, `faithfulness` là điểm yếu lớn nhất do sự phức tạp của các điều khoản chuyển tiếp giữa các phiên bản quy định.

---

## 5. Suggested Fixes

| Metric yếu | Root cause | Suggested fix |
|---|---|---|
| faithfulness | LLM tự suy luận / hallucination khi ghép nối nhiều dữ liệu | Thắt chặt system prompt ("Chỉ trả lời dựa trên context, không tự tính toán nếu không có công thức rõ ràng"), hạ temperature = 0.0 |
| context_recall | Thiếu chunks liên quan trong các câu hỏi điều kiện phức tạp | Tăng top_k retrieve ban đầu (k=10), áp dụng Chunk Summarization + HyQA enrichment |
| context_precision | Noise từ các chunks tài liệu version cũ (v2023 vs v2024) | Áp dụng metadata filter theo trường `version`/`effective_date` và Cross-Encoder reranking |
| answer_relevancy | Câu trả lời không bám sát format yêu cầu của user | Chuẩn hóa prompt template theo định dạng CoT (Chain-of-Thought) ngắn trước khi đưa ra câu trả lời chốt |

---

## 6. Nhận xét về Adversarial Distribution

- Tập **adversarial** có `avg_score` đạt 0.6922, thấp hơn tập factual (0.8123) nhưng nhỉnh hơn multi_hop (0.6883).
- **Version conflicts:** Pipeline RAG rất dễ bị nhầm lẫn giữa chính sách v2023 (12 ngày phép) và v2024 (15 ngày phép) khi retrieve cả 2 tài liệu nếu không có bộ lọc metadata theo ngày hiệu lực. Ví dụ rõ nét nhất là câu hỏi #41 (*"Nhân viên được nghỉ bao nhiêu ngày phép năm?"*) - LLM trích dẫn nhầm thông tin của v2023 dẫn đến điểm faithfulness chỉ đạt 0.3333.
- **Các câu trong Bottom 10 rơi vào adversarial:**
  - Câu #41 (Version conflict về số ngày phép).
  - Câu #47 (Negation trap về tự xử lý malware: chính sách yêu cầu cô lập máy và báo IT, không được tự xử lý).
  - Câu #50 (Policy contradiction trap về việc dùng VPN cá nhân: chính sách nghiêm cấm dù câu hỏi mang tính gợi ý hợp lý).
  - Cả 3 câu này đều thất bại vì LLM bị đánh lừa bởi bẫy ngữ nghĩa hoặc version cũ, dẫn đến vi phạm tính trung thực (faithfulness) so với chính sách mới nhất.
