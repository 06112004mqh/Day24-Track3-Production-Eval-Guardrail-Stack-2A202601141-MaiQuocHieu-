# CI/CD Blueprint: RAG Eval + Guardrail Stack

**Sinh viên:** Mai Quốc Hiệu  
**Ngày:** 2026-08-26

---

## Guard Stack Architecture

```
User Input
    │
    ▼ (~16.27ms P95)
[Presidio PII Scan]
    │ block if: VN_CCCD / VN_PHONE / EMAIL detected
    │ action:   return 400 + "PII detected in query"
    ▼ (~6166.73ms P95 with OpenAI / <50ms with local ONNX)
[NeMo Input Rail]
    │ block if: off-topic / jailbreak / prompt injection
    │ action:   return 503 + refuse message
    ▼
[RAG Pipeline (Day 18)]
    │ M1 Chunk → M2 Search → M3 Rerank → GPT-4o-mini
    ▼ (~280.00ms P95)
[NeMo Output Rail]
    │ flag if:  PII in response / sensitive content
    │ action:   replace with safe response
    ▼
User Response
```

---

## Latency Budget

*(Điền từ kết quả Task 12 — measure_p95_latency())*

| Layer | P50 (ms) | P95 (ms) | P99 (ms) | Budget |
|---|---|---|---|---|
| Presidio PII | 9.15 | 16.27 | 19.34 | <10ms |
| NeMo Input Rail | 3670.54 | 6166.73 | 8288.18 | <300ms |
| RAG Pipeline | 450.00 | 850.00 | 1200.00 | <2000ms |
| NeMo Output Rail | 150.00 | 280.00 | 350.00 | <300ms |
| **Total Guard** | 3679.69 | **6181.66** | 8293.88 | **<500ms** |

**Budget OK?** [ ] Yes / [x] No  
**Comment:** Lớp NeMo Input Rail hiện đang gọi LLM OpenAI API từ xa qua HTTPS, là bottleneck chính dẫn đến latency P95 lên đến 6.18s. Để đưa vào production đạt budget < 500ms:
1. Chuyển sang local embedding semantic intent classifier (FastEmbed `bge-small-en-v1.5` hoặc mini-model cục bộ chạy ONNX Runtime) để phân loại jailbreak/off-topic trong < 30ms.
2. Áp dụng cache (Redis/LRU) cho các embedding queries và input phổ biến.
3. Triển khai Local NIM (NVIDIA Inference Microservice) với model SLM (ví dụ Qwen2.5-0.5B-Instruct quantized INT4) chạy trực tiếp trên GPU máy chủ.

---

## CI/CD Gates (phải pass trước khi merge to main)

```yaml
# .github/workflows/rag_eval.yml
name: RAG Production Evaluation & Guardrails CI/CD

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  evaluate-and-guard:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python 3.12
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      - name: RAGAS Quality Gate
        run: |
          python src/phase_a_ragas.py
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          MIN_FAITHFULNESS: 0.75
          MIN_AVG_SCORE: 0.65

      - name: LLM Judge Alignment Gate
        run: |
          python src/phase_b_judge.py
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}

      - name: Guardrail Security Gate
        run: |
          pytest tests/test_phase_c.py -k "test_adversarial_suite_pass_rate" -v
        # Yêu cầu pass rate >= 75% (15/20)

      - name: Regression Test Suite
        run: |
          pytest tests/ -v
```

---

## Monitoring Dashboard (production)

| Metric | Alert Threshold | Action |
|---|---|---|
| RAGAS faithfulness (daily sample) | < 0.70 | Page on-call, check prompt drift & context corruption |
| Adversarial block rate | < 80% | Review new attack vectors, update colang rails & patterns |
| Guard P95 latency | > 600ms | Scale NeMo inference workers / switch to local SLM |
| PII detected count | spike >10/hour | Trigger security audit, inspect upstream user payloads |

---

## Kết quả thực tế từ Lab

| Chỉ số | Kết quả |
|---|---|
| RAGAS avg_score (50q) | 0.738 (factual: 0.812, multi_hop: 0.688, adversarial: 0.692) |
| Worst metric | Faithfulness (multi_hop: 0.413, failure count: 15) |
| Dominant failure distribution | Multi-hop / Factual |
| Cohen's κ | 0.4444 (Moderate Agreement với chuyên gia con người) |
| Adversarial pass rate | 20 / 20 (100%) |
| Guard P95 latency | 6181.66 ms (Presidio: 16.27ms, NeMo: 6166.73ms) |

---

## Nhận xét & Cải tiến

1. **Điểm hoạt động tốt:**
   - Presidio hoạt động cực kỳ nhanh (< 17ms P95) và phát hiện chính xác 100% các thực thể PII tiếng Việt như CCCD (12 số), CMND (9 số), số điện thoại VN (đầu số 03/05/07/08/09), email.
   - Guard stack đạt pass rate tuyệt đối 20/20 (100%) trên tập Adversarial Suite, chặn đứng toàn bộ các tấn công prompt injection (`SYSTEM OVERRIDE`, `<!-- IGNORE -->`, `[ADMIN COMMAND]`), jailbreak (`DAN`, roleplay), PII query và câu hỏi off-topic.
   - Pipeline RAG kết hợp BM25 + Dense Search và Cross-Encoder Rerank đạt Context Precision rất cao (0.89 - 0.94).

2. **Điểm cần cải thiện:**
   - Faithfulness trên tập `multi_hop` còn thấp (0.413). Nguyên nhân là khi truy vấn đòi hỏi tổng hợp thông tin từ nhiều điều khoản (ví dụ vừa tính thâm niên vừa tra bảng phụ cấp), LLM có xu hướng tự suy luận hoặc nội suy thiếu căn cứ nếu context không được sắp xếp chặt chẽ.
   - NeMo Guardrails sử dụng LLM qua API mạng gây ra latency lớn (P95 > 6s), không đáp ứng được SLA latency real-time trong ứng dụng chat trực tiếp.

3. **Thay đổi khi deploy Production thực tế:**
   - Thay thế LLM-based guardrail bằng kiến trúc Hybrid Guard: Presidio Regex + Small Embedding Classifier (ONNX runtime) cho input checking giúp giảm độ trễ input guard xuống dưới 25ms.
   - Bổ sung Semantic Cache ở tầng Gateway để trả lời ngay lập tức cho các câu hỏi trùng lặp mà không cần gọi lại full RAG pipeline.
   - Áp dụng kỹ thuật Self-Correction hoặc Contextual Compression để loại bỏ hoàn toàn các đoạn văn bản thừa trước khi đưa vào context window của LLM sinh câu trả lời.
