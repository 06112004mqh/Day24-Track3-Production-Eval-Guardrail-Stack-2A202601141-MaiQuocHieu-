# LLM Judge Bias Report — Phase B

**Sinh viên:** Mai Quốc Hiệu  
**Ngày:** 2026-08-26  
**Judge model:** gpt-4o-mini

---

## 1. Pairwise Judge Results

*(Chạy pairwise_judge() trên các cặp answers)*

| # | Question (tóm tắt) | Winner | Reasoning tóm tắt |
|---|---|---|---|
| 1 | Nghỉ phép khi kết hôn | B | Answer B đầy đủ hơn khi nêu rõ không bị trừ vào phép năm. |
| 2 | Mua thiết bị 55 triệu | B | Answer B chính xác vì >50 triệu cần CEO duyệt (A nói Giám đốc phòng ban là sai). |
| 3 | Thưởng Tết tối thiểu ≥6 tháng | B | Answer B đầy đủ hơn khi bổ sung quy định pro-rata cho nhân viên <6 tháng. |
| 4 | Senior 9 năm thâm niên | tie | Cả 2 câu đều tính đúng (18 ngày phép, lương 20-35tr), B chi tiết hơn về cách phân tích. |
| 5 | Hoàn trả khóa học 25 triệu sau 8 tháng | B | Answer B trích dẫn rõ cam kết tối thiểu 1 năm và hoàn trả 100% chi phí. |
| 6 | Tạm ứng 8 triệu quá hạn 30 ngày | B | Answer B nêu đủ 2 cấp duyệt (Trưởng phòng + Kế toán trưởng) và tính đúng 80.000đ phạt. |
| 7 | Manager 12 năm thâm niên | B | Answer B diễn giải chi tiết từng khoản phụ cấp và công thức tính thâm niên. |
| 8 | Số ngày phép năm (v2024 vs v2023) | B | Answer B cập nhật theo chính sách v2024 (15 ngày), chỉ rõ v2023 (12 ngày) đã hết hạn. |
| 9 | Thử việc có được nghỉ phép năm | tie | Cả 2 câu đều nêu đúng không được nghỉ phép năm và phải xin nghỉ không lương. |
| 10 | Dùng VPN cá nhân khi WFH | B | Answer B đúng quy định cấm VPN cá nhân (v1.3), Answer A nói được là vi phạm chính sách. |

---

## 2. Swap-and-Average Results

*(Chạy swap_and_average() trên cùng các cặp)*

| # | Pass 1 Winner | Pass 2 Winner | Final | Position Consistent? |
|---|---|---|---|---|
| 1 | B | B | B | True |
| 2 | B | B | B | True |
| 3 | B | B | B | True |
| 4 | B | tie | tie | False |
| 5 | B | B | B | True |
| 6 | B | B | B | True |
| 7 | B | B | B | True |
| 8 | B | B | B | True |
| 9 | B | A | tie | False |
| 10 | B | B | B | True |

**Position bias rate:** 20.0% (= 2 cases NOT consistent / 10 tổng số)

---

## 3. Cohen's κ Analysis

**Human labels:** `human_labels_10q.json` (10 câu: 6 label=1, 4 label=0)  
**Judge labels:** [1, 0, 0, 1, 0, 0, 0, 0, 1, 0]

| Question ID | Human Label | Judge Label | Agree? |
|---|---|---|---|
| 1 | 1 | 1 | Yes |
| 5 | 0 | 0 | Yes |
| 12 | 1 | 0 | No (Judge khắt khe hơn khi Answer A thiếu chi tiết pro-rata) |
| 21 | 1 | 1 | Yes |
| 23 | 1 | 0 | No (Judge ưu tiên câu có giải thích căn cứ cam kết 1 năm) |
| 29 | 0 | 0 | Yes |
| 33 | 1 | 0 | No (Judge trừ điểm vì thiếu tách chi tiết từng khoản tiền) |
| 41 | 0 | 0 | Yes |
| 46 | 1 | 1 | Yes |
| 50 | 0 | 0 | Yes |

**Cohen's κ:** 0.4444  
**Interpretation:** Moderate Agreement (Độ đồng thuận mức độ vừa phải giữa LLM Judge và chuyên gia con người).

---

## 4. Verbosity Bias

Trong các case có winner rõ ràng (không phải tie):
- A thắng + A dài hơn B: 0 / 8 cases
- B thắng + B dài hơn A: 8 / 8 cases  
- **Verbosity bias rate:** 100.0%

**Kết luận:** LLM Judge có xu hướng rất mạnh (100% trong tập thử nghiệm) ưu tiên câu trả lời dài hơn, có giải thích bổ sung và cấu trúc chi tiết hơn. Đây là một vấn đề nghiêm trọng trong đánh giá tự động (Verbosity Bias / Length Bias), vì một câu trả lời ngắn gọn, trực diện và chính xác có thể bị đánh giá thấp hơn một câu trả lời dài dòng chứa thông tin ngoài lề không được hỏi.

---

## 5. Nhận xét chung

1. **Về độ tin cậy của Judge (κ = 0.4444):** LLM Judge có khả năng phát hiện tốt các lỗi sai nghiêm trọng về mặt chính sách (như câu 5, 29, 41, 50). Tuy nhiên, Judge có xu hướng khắt khe hơn con người rất nhiều đối với các câu trả lời ngắn gọn nhưng đúng, dẫn đến việc gán nhãn 0 cho những câu mà con người chấm 1.
2. **Về Position Bias (20%):** Mức độ position bias nằm dưới ngưỡng cảnh báo 30%, cho thấy model `gpt-4o-mini` tương đối ổn định với vị trí.
3. **Hiệu quả của kỹ thuật Swap-and-Average:** Swap-and-average giúp loại bỏ hoàn toàn các phán quyết không nhất quán (chuyển thành `tie`), giúp ngăn chặn sai số do ngẫu nhiên khi câu trả lời xuất hiện ở vị trí đầu tiên.
4. **Áp dụng trong môi trường Production:**
   - Không nên sử dụng LLM Judge đơn lẻ mà nên kết hợp song song với Rule-based Checks và Ground-Truth Matching (như RAGAS).
   - Luôn bắt buộc bật chế độ Swap-and-Average cho tất cả các batch evaluation quan trọng.
   - Thêm ràng buộc rõ ràng trong System Prompt của Judge để phạt các câu trả lời dài dòng không cần thiết, cân bằng lại Verbosity Bias.
