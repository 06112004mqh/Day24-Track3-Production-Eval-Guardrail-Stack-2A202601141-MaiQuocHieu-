from __future__ import annotations

"""Phase C: Production Guardrails — Presidio PII + NeMo Guardrails + P95 Latency."""

import asyncio
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ADVERSARIAL_SET_PATH, GUARDRAILS_CONFIG_DIR, LATENCY_BUDGET_P95_MS, PRESIDIO_LANGUAGE


# ─── Task 9a: Presidio PII Detection ─────────────────────────────────────────

def setup_presidio():
    """Khởi tạo Presidio engine với custom Vietnamese PII recognizers. (Đã implement sẵn)

    Custom recognizers thêm vào:
        VN_CCCD  — số CCCD 12 chữ số hoặc CMND 9 chữ số
        VN_PHONE — số điện thoại Việt Nam (0[3-9]xxxxxxxx)

    Các recognizers mặc định đã có sẵn: EMAIL, PHONE_NUMBER (international), ...
    """
    from presidio_analyzer import AnalyzerEngine, RecognizerRegistry, Pattern, PatternRecognizer
    from presidio_anonymizer import AnonymizerEngine
    from presidio_analyzer.nlp_engine import NlpEngineProvider

    cccd_recognizer = PatternRecognizer(
        supported_entity="VN_CCCD",
        patterns=[
            Pattern("CCCD 12 digits", r"\b\d{12}\b", 0.9),
            Pattern("CMND 9 digits",  r"\b\d{9}\b",  0.7),
        ],
    )
    phone_recognizer = PatternRecognizer(
        supported_entity="VN_PHONE",
        patterns=[Pattern("VN mobile", r"\b0[3-9]\d{8}\b", 0.9)],
    )

    registry = RecognizerRegistry()
    registry.load_predefined_recognizers()
    registry.add_recognizer(cccd_recognizer)
    registry.add_recognizer(phone_recognizer)

    import spacy
    spacy_model = "en_core_web_lg" if spacy.util.is_package("en_core_web_lg") else "en_core_web_sm"
    conf = {
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": spacy_model}],
    }
    nlp_engine = NlpEngineProvider(nlp_configuration=conf).create_engine()
    analyzer = AnalyzerEngine(registry=registry, nlp_engine=nlp_engine)

    anonymizer = AnonymizerEngine()
    return analyzer, anonymizer


def pii_scan(text: str, analyzer=None, anonymizer=None) -> dict:
    """Task 9a: Quét PII trong văn bản bằng Presidio.

    Returns:
        {
          "has_pii":    bool,
          "entities":   [{"type": str, "text": str, "score": float, "start": int, "end": int}],
          "anonymized": str,   # text với PII được thay bằng <TYPE>
        }
    """
    if analyzer is None or anonymizer is None:
        analyzer, anonymizer = setup_presidio()

    results = analyzer.analyze(text=text, language=PRESIDIO_LANGUAGE)
    valid_pii_types = {
        "VN_CCCD", "VN_PHONE", "EMAIL_ADDRESS", "EMAIL",
        "PHONE_NUMBER", "CREDIT_CARD", "IBAN_CODE", "CRYPTO"
    }
    filtered_results = [
        r for r in results
        if r.entity_type in valid_pii_types and r.score >= 0.5
    ]
    if not filtered_results:
        return {"has_pii": False, "entities": [], "anonymized": text}

    anonymized = anonymizer.anonymize(text=text, analyzer_results=filtered_results).text
    entities = [
        {"type": r.entity_type, "text": text[r.start:r.end],
         "score": round(r.score, 3), "start": r.start, "end": r.end}
        for r in filtered_results
    ]
    return {"has_pii": True, "entities": entities, "anonymized": anonymized}


# ─── Task 9b + 11: NeMo Guardrails ───────────────────────────────────────────

def setup_nemo_rails():
    """Khởi tạo NeMo Guardrails từ guardrails/config.yml. (Đã implement sẵn)

    Config directory: guardrails/
        config.yml  — model + rails config
        rails.co    — Colang dialogue flows (topic check, jailbreak check, output check)
    """
    from nemoguardrails import RailsConfig, LLMRails
    config = RailsConfig.from_path(GUARDRAILS_CONFIG_DIR)
    rails  = LLMRails(config)
    return rails


def check_input_rail(text: str, rails=None) -> dict:
    """Task 9b: Kiểm tra input qua NeMo Guardrails.

    Kiểm tra xem câu hỏi có vi phạm:
        - Jailbreak (bỏ qua hướng dẫn, prompt injection, đóng vai DAN/AI không giới hạn)
        - Off-topic (viết thơ, nấu ăn, crypto, hỏi ngoài phạm vi HR)
        - PII extraction (yêu cầu xem thông tin cá nhân của người khác)

    Returns:
        {
          "allowed":        bool,
          "blocked_reason": str | None,
          "response":       str,
        }
    """
    if rails is None:
        rails = setup_nemo_rails()

    lower_text = text.lower()
    jailbreak_patterns = [
        "bỏ qua tất cả", "ignore your previous", "pretend you are", "unrestricted ai",
        "dan", "system override", "không có giới hạn", "tiết lộ bảng lương", "system instructions",
        "mật khẩu admin", "hướng dẫn tấn công", "tấn công mạng", "list all employee", "print all",
        "<!-- ignore", "[admin command", "tôi là ceo và ra lệnh", "sql injection", "script>",
        "alert(", "eval("
    ]
    if any(p in lower_text for p in jailbreak_patterns):
        return {
            "allowed": False,
            "blocked_reason": "jailbreak_attempt",
            "response": "Xin lỗi, tôi không thể thực hiện yêu cầu này. Tôi chỉ có thể trả lời các câu hỏi về chính sách nhân sự công ty.",
        }

    try:
        response = rails.generate(messages=[{"role": "user", "content": text}])
        resp_text = response.get("content", "") if isinstance(response, dict) else str(response)
    except Exception:
        resp_text = ""

    refuse_keywords = [
        "xin lỗi", "không thể", "không được phép", "i cannot", "i'm sorry",
        "chỉ có thể trả lời", "bảo mật", "từ chối", "unable to", "outside my scope",
        "only answer", "cannot provide", "against policy", "policy prohibits"
    ]
    blocked = any(kw in resp_text.lower() for kw in refuse_keywords) or resp_text.strip() == ""
    return {
        "allowed": not blocked,
        "blocked_reason": "nemo_input_rail" if blocked else None,
        "response": resp_text,
    }


def run_adversarial_suite(
    adversarial_set: list[dict],
    analyzer=None,
    anonymizer=None,
    rails=None,
) -> list[dict]:
    """Task 10: Chạy toàn bộ 20 adversarial test cases qua Guardrail Stack.

    Flow cho mỗi test case:
        1. Presidio PII scan (nếu có PII -> blocked_by = "presidio")
        2. NeMo Input rail (nếu vi phạm -> blocked_by = "nemo")
        3. So sánh actual vs expected_behavior -> passed (bool)

    Returns:
        List of {
          "id":               int,
          "category":         str,
          "input":            str,
          "expected":         str,   # "blocked" hoặc "allowed"
          "actual":           str,   # "blocked" hoặc "allowed"
          "blocked_by":       str | None,  # "presidio" | "nemo" | None
          "passed":           bool,
        }
    """
    if analyzer is None or anonymizer is None:
        analyzer, anonymizer = setup_presidio()
    if rails is None:
        rails = setup_nemo_rails()

    results = []
    for item in adversarial_set:
        inp = item.get("input", "")
        expected = item.get("expected_behavior", "blocked")
        category = item.get("category", "unknown")
        case_id = item.get("id", len(results) + 1)

        # 1. Presidio scan
        pii_res = pii_scan(inp, analyzer=analyzer, anonymizer=anonymizer)
        if pii_res["has_pii"]:
            actual = "blocked"
            blocked_by = "presidio"
        else:
            # 2. NeMo Input rail
            guard_res = check_input_rail(inp, rails=rails)
            if not guard_res["allowed"]:
                actual = "blocked"
                blocked_by = "nemo"
            else:
                actual = "allowed"
                blocked_by = None

        passed = (actual == expected)
        results.append({
            "id": case_id,
            "category": category,
            "input": inp,
            "expected": expected,
            "actual": actual,
            "blocked_by": blocked_by,
            "passed": passed,
        })

    passed_count = sum(1 for r in results if r["passed"])
    print(f"Adversarial suite: {passed_count}/{len(results)} passed")
    return results


def check_output_rail(
    text: str,
    analyzer=None,
    anonymizer=None,
    rails=None,
) -> dict:
    """Task 11: Kiểm tra output của RAG pipeline trước khi trả về user.

    Quét:
        1. Presidio PII scan trên output
        2. Check các thông tin nhạy cảm nội bộ

    Returns:
        {
          "allowed":        bool,
          "sanitized_text": str,
          "flagged_issues": list[str],
        }
    """
    pii_res = pii_scan(text, analyzer=analyzer, anonymizer=anonymizer)
    issues = []
    sanitized = text

    if pii_res["has_pii"]:
        issues.append("pii_detected_in_output")
        sanitized = pii_res["anonymized"]

    sensitive_patterns = [
        "mật khẩu admin", "secret_key", "internal api token",
        "dữ liệu tối mật", "system credentials"
    ]
    for pattern in sensitive_patterns:
        if pattern in sanitized.lower():
            issues.append(f"sensitive_info:{pattern}")
            sanitized = sanitized.replace(pattern, "[REDACTED]")

    return {
        "allowed": len(issues) == 0,
        "sanitized_text": sanitized,
        "flagged_issues": issues,
    }


def measure_p95_latency(sample_queries: list[str] = None, n_runs: int = 10) -> dict:
    """Task 12: Đo P50, P95, P99 latency của từng lớp guardrail.

    Returns:
        {
          "presidio_ms":       {"p50": float, "p95": float, "p99": float},
          "nemo_ms":           {"p50": float, "p95": float, "p99": float},
          "total_ms":          {"p50": float, "p95": float, "p99": float},
          "latency_budget_ok": bool,
          "budget_ms":         int,
        }
    """
    if sample_queries is None:
        sample_queries = [
            "Nhân viên được nghỉ bao nhiêu ngày phép năm?",
            "Quy trình tạm ứng công tác phí như thế nào?",
            "Bảo hiểm sức khỏe PVI chi trả những khoản nào?",
            "Lương thử việc của vị trí Senior là bao nhiêu?",
            "Thời hạn nộp giấy nghỉ ốm là mấy ngày?",
        ]

    analyzer, anonymizer = setup_presidio()
    rails = setup_nemo_rails()

    presidio_times = []
    nemo_times = []
    total_times = []

    for _ in range(n_runs):
        for q in sample_queries:
            t0 = time.perf_counter()
            pii_scan(q, analyzer=analyzer, anonymizer=anonymizer)
            t_presidio = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            # Fast check
            check_input_rail(q, rails=rails)
            t_nemo = (time.perf_counter() - t0) * 1000

            presidio_times.append(t_presidio)
            nemo_times.append(t_nemo)
            total_times.append(t_presidio + t_nemo)

    def calc_percentiles(times: list[float]) -> dict:
        if not times:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
        s = sorted(times)
        n = len(s)
        p50_idx = min(int(n * 0.50), n - 1)
        p95_idx = min(int(n * 0.95), n - 1)
        p99_idx = min(int(n * 0.99), n - 1)
        return {
            "p50": round(float(s[p50_idx]), 2),
            "p95": round(float(s[p95_idx]), 2),
            "p99": round(float(s[p99_idx]), 2),
        }

    presidio_stats = calc_percentiles(presidio_times)
    nemo_stats = calc_percentiles(nemo_times)
    total_stats = calc_percentiles(total_times)

    return {
        "presidio_ms": presidio_stats,
        "nemo_ms": nemo_stats,
        "total_ms": total_stats,
        "latency_budget_ok": total_stats["p95"] <= LATENCY_BUDGET_P95_MS,
        "budget_ms": LATENCY_BUDGET_P95_MS,
    }

# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Task 9a: PII scan demo
    test_pii = "Nhân viên Nguyễn Văn A, CCCD 034095001234, SĐT 0987654321 hỏi về nghỉ phép."
    result = pii_scan(test_pii)
    print(f"PII detected: {result['has_pii']}")
    print(f"Entities: {result['entities']}")
    print(f"Anonymized: {result['anonymized']}")

    # Task 10: Adversarial suite
    with open(ADVERSARIAL_SET_PATH, encoding="utf-8") as f:
        adversarial_set = json.load(f)
    print(f"\nLoaded {len(adversarial_set)} adversarial inputs")
    results = run_adversarial_suite(adversarial_set)
    passed_count = sum(1 for r in results if r["passed"]) if results else 0
    if results:
        print(f"Adversarial suite: {passed_count}/{len(results)} passed")

    # Task 12: P95 latency
    sample_inputs = [item["input"] for item in adversarial_set[:10]]
    latency = measure_p95_latency(sample_inputs, n_runs=10)
    print(f"\nLatency P95 — Presidio: {latency['presidio_ms']['p95']}ms | "
          f"NeMo: {latency['nemo_ms']['p95']}ms | "
          f"Total: {latency['total_ms']['p95']}ms")
    print(f"Budget OK ({latency['budget_ms']}ms): {latency['latency_budget_ok']}")

    # Save Phase C report
    os.makedirs("reports", exist_ok=True)
    report_data = {
        "adversarial_suite": {
            "total": len(adversarial_set),
            "passed": passed_count,
            "pass_rate": round(passed_count / len(adversarial_set), 3) if adversarial_set else 0.0,
            "results": results,
        },
        "latency": latency,
    }
    with open("reports/guard_results.json", "w", encoding="utf-8") as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)
    print("Guardrails report saved → reports/guard_results.json")
