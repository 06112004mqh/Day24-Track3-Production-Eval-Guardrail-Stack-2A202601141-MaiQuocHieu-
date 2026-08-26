from __future__ import annotations

"""Phase B: LLM-as-Judge — pairwise, swap-and-average, Cohen κ, bias analysis."""

import json
import os
import sys
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY, JUDGE_MODEL, HUMAN_LABELS_PATH


@dataclass
class JudgeResult:
    question: str
    answer_a: str
    answer_b: str
    winner_pass1: str       # "A" | "B" | "tie"  (original order)
    winner_pass2: str       # "A" | "B" | "tie"  (after swap, ALREADY converted back)
    final_winner: str       # consensus after swap-and-average
    reasoning_pass1: str
    reasoning_pass2: str
    position_consistent: bool  # True if both passes agree on same answer
    scores_pass1: dict = field(default_factory=dict)  # {"A": float, "B": float}
    scores_pass2: dict = field(default_factory=dict)


# ─── Task 5: Pairwise Judge ───────────────────────────────────────────────────

def pairwise_judge(question: str, answer_a: str, answer_b: str) -> dict:
    """Task 5: Gọi LLM để chọn answer tốt hơn (A hoặc B) theo 3 tiêu chí.

    Tiêu chí đánh giá:
        - Độ chính xác (accuracy): có khớp với thực tế chính sách không?
        - Độ đầy đủ (completeness): có trả lời đủ câu hỏi không?
        - Tính súc tích (conciseness): có thừa / thiếu thông tin không?

    Returns:
        {"winner": "A"|"B"|"tie", "reasoning": str, "scores": {"A": float, "B": float}}
    """
    prompt_template = f"""Bạn là một chuyên gia đánh giá chất lượng câu trả lời của hệ thống RAG nội bộ về chính sách nhân sự.

Câu hỏi: {question}

Answer A:
{answer_a}

Answer B:
{answer_b}

Đánh giá dựa trên 3 tiêu chí:
1. Độ chính xác (accuracy): Thông tin có đúng với chính sách mới nhất và không bịa đặt không?
2. Độ đầy đủ (completeness): Có trả lời toàn diện các vế của câu hỏi không?
3. Tính súc tích (conciseness): Trả lời trọng tâm, không thừa thãi lan man.

Quy tắc:
- Chọn winner là "A" nếu A tốt hơn rõ ràng.
- Chọn winner là "B" nếu B tốt hơn rõ ràng.
- Chọn winner là "tie" nếu 2 câu trả lời tương đương hoặc cùng sai/thiếu.

Chỉ trả về JSON với format chính xác sau:
{{
  "winner": "A" | "B" | "tie",
  "reasoning": "giải thích ngắn gọn lý do đánh giá",
  "scores": {{
    "A": 0.0 đến 1.0,
    "B": 0.0 đến 1.0
  }}
}}"""

    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": "Bạn là chuyên gia đánh giá RAG. Luôn chỉ trả lời định dạng JSON."},
                {"role": "user",   "content": prompt_template},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        parsed = json.loads(resp.choices[0].message.content)
        winner = parsed.get("winner", "tie")
        if winner not in {"A", "B", "tie"}:
            winner = "tie"
        scores = parsed.get("scores", {"A": 0.5, "B": 0.5})
        # Normalize scores
        scores = {k: float(v) for k, v in scores.items() if k in {"A", "B"}}
        if "A" not in scores: scores["A"] = 0.5
        if "B" not in scores: scores["B"] = 0.5
        reasoning = parsed.get("reasoning", "")
        if not reasoning:
            reasoning = f"Answer {winner} được chọn dựa trên các tiêu chí chính sách."
        return {"winner": winner, "reasoning": reasoning, "scores": scores}
    except Exception as e:
        print(f"  ⚠️ Pairwise judge error: {e}")
        return {"winner": "tie", "reasoning": f"Judge fallback: {e}", "scores": {"A": 0.5, "B": 0.5}}


# ─── Task 6: Swap-and-Average ─────────────────────────────────────────────────

def swap_and_average(question: str, answer_a: str, answer_b: str) -> JudgeResult:
    """Task 6: Chạy pairwise 2 lần (hoán đổi thứ tự), lấy kết quả nhất quán.

    Lý do: LLM thường có position bias (ưu tiên answer xuất hiện trước).
    Bằng cách swap, ta phát hiện và giảm bias này.

    Logic:
        Pass 1: judge(q, A, B) → winner_1 (trong không gian A/B)
        Pass 2: judge(q, B, A) → winner_2_raw (trong không gian B/A)
        Convert: nếu winner_2_raw="A" thì thực ra là B (vì đã swap)
        Final:   nếu winner_1 == winner_2 → final = winner_1
                 nếu khác nhau → final = "tie"
    """
    pass1 = pairwise_judge(question, answer_a, answer_b)
    pass2_raw = pairwise_judge(question, answer_b, answer_a)  # SWAP!

    # Convert pass2 back to original A/B space
    swap_map = {"A": "B", "B": "A", "tie": "tie"}
    winner_pass2 = swap_map.get(pass2_raw.get("winner", "tie"), "tie")

    # Average: consensus only if both agree
    if pass1["winner"] == winner_pass2:
        final = pass1["winner"]
    else:
        final = "tie"  # disagreement = position bias

    position_consistent = (pass1["winner"] == winner_pass2)

    s1 = pass1.get("scores", {"A": 0.5, "B": 0.5})
    s2_raw = pass2_raw.get("scores", {"A": 0.5, "B": 0.5})
    scores_pass2 = {
        "A": s2_raw.get("B", 0.5),
        "B": s2_raw.get("A", 0.5),
    }

    return JudgeResult(
        question=question,
        answer_a=answer_a,
        answer_b=answer_b,
        winner_pass1=pass1["winner"],
        winner_pass2=winner_pass2,
        final_winner=final,
        reasoning_pass1=pass1.get("reasoning", ""),
        reasoning_pass2=pass2_raw.get("reasoning", ""),
        position_consistent=position_consistent,
        scores_pass1=s1,
        scores_pass2=scores_pass2,
    )


# ─── Task 7: Cohen's κ ────────────────────────────────────────────────────────

def cohen_kappa(judge_labels: list[int], human_labels: list[int]) -> float:
    """Task 7: Tính Cohen's κ giữa LLM judge và human labels.

    Args:
        judge_labels:  nhãn từ LLM judge (0 = bad answer, 1 = good answer)
        human_labels:  nhãn từ human_labels_10q.json

    Returns:
        κ ∈ [-1, 1]
    """
    if not judge_labels or not human_labels or len(judge_labels) != len(human_labels):
        return 0.0

    n = len(judge_labels)
    p_o = sum(1 for j, h in zip(judge_labels, human_labels) if j == h) / n
    p_e = (
        (judge_labels.count(1) / n) * (human_labels.count(1) / n) +
        (judge_labels.count(0) / n) * (human_labels.count(0) / n)
    )
    if abs(1.0 - p_e) < 1e-9:
        return 1.0 if p_o == 1.0 else 0.0

    kappa = (p_o - p_e) / (1.0 - p_e)
    return round(float(kappa), 4)


# ─── Task 8: Bias Report ──────────────────────────────────────────────────────

def bias_report(judge_results: list[JudgeResult]) -> dict:
    """Task 8: Đo lường position bias và verbosity bias.

    Position bias: LLM chọn answer theo vị trí (A hay B) thay vì chất lượng.
        → Đo bằng % cases where position_consistent = False

    Verbosity bias: LLM ưu tiên answer dài hơn dù không chính xác hơn.
        → Đo bằng: trong các case A thắng, A có dài hơn B không? Tương tự cho B.

    Returns:
        {
          "total_judged": int,
          "position_bias_rate": float,
          "position_bias_count": int,
          "verbosity_bias": float,
          "verbosity_details": {
            "a_wins_a_longer": int,
            "b_wins_b_longer": int,
            "total_decisive": int,
          },
          "interpretation": str,
        }
    """
    total = len(judge_results)
    if total == 0:
        return {
            "total_judged": 0,
            "position_bias_rate": 0.0,
            "position_bias_count": 0,
            "verbosity_bias": 0.0,
            "verbosity_details": {
                "a_wins_a_longer": 0,
                "b_wins_b_longer": 0,
                "total_decisive": 0,
            },
            "interpretation": "Chưa có dữ liệu đánh giá.",
        }

    position_bias_count = sum(1 for r in judge_results if not r.position_consistent)
    position_bias_rate  = position_bias_count / total

    a_wins_a_longer = sum(
        1 for r in judge_results
        if r.final_winner == "A" and len(r.answer_a) > len(r.answer_b)
    )
    b_wins_b_longer = sum(
        1 for r in judge_results
        if r.final_winner == "B" and len(r.answer_b) > len(r.answer_a)
    )
    decisive = sum(1 for r in judge_results if r.final_winner in {"A", "B"})
    verbosity_bias = (a_wins_a_longer + b_wins_b_longer) / decisive if decisive > 0 else 0.0

    interpretation = (
        "Position bias cao (>30%) — nên dùng swap-and-average để đảm bảo độ tin cậy."
        if position_bias_rate > 0.3 else "Position bias thấp (<=30%) — judge hoạt động ổn định và nhất quán."
    )
    return {
        "total_judged": total,
        "position_bias_rate": round(position_bias_rate, 3),
        "position_bias_count": position_bias_count,
        "verbosity_bias": round(verbosity_bias, 3),
        "verbosity_details": {
            "a_wins_a_longer": a_wins_a_longer,
            "b_wins_b_longer": b_wins_b_longer,
            "total_decisive": decisive,
        },
        "interpretation": interpretation,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs("reports", exist_ok=True)
    print("=" * 60)
    print("PHASE B: LLM-as-Judge & Bias Evaluation")
    print("=" * 60)

    # 1. Load human test set and 50q test set
    with open(HUMAN_LABELS_PATH, encoding="utf-8") as f:
        human_data = json.load(f)
    human_labels = [item["human_label"] for item in human_data]

    gt_map = {}
    if os.path.exists("test_set_50q.json"):
        with open("test_set_50q.json", encoding="utf-8") as f:
            tset = json.load(f)
            gt_map = {item["id"]: item.get("ground_truth", "") for item in tset}

    print(f"Loaded {len(human_data)} human annotated questions.")

    # 2. Run swap-and-average on human questions & pairwise benchmarks
    judge_results: list[JudgeResult] = []
    judge_labels: list[int] = []

    for item in human_data:
        qid = item["question_id"]
        q = item["question"]
        ans_a = item["model_answer"]
        gt = gt_map.get(qid, "Thông tin chuẩn theo chính sách nhân sự công ty.")
        # Judge model_answer vs alternative
        jr = swap_and_average(q, ans_a, gt)
        judge_results.append(jr)

        # Determine judge label (1 if model answer A won or tied with GT, 0 if GT clearly won over A)
        # If A was judged better or equal to ground truth, model answer is correct (1), else 0
        if jr.final_winner in {"A", "tie"} or jr.scores_pass1.get("A", 0) >= 0.7:
            j_label = 1
        else:
            j_label = 0
        judge_labels.append(j_label)

    # 3. Cohen's Kappa
    kappa = cohen_kappa(judge_labels, human_labels)
    print(f"\n✓ Cohen's κ: {kappa:.4f}")

    # 4. Bias Report
    bias = bias_report(judge_results)
    print(f"✓ Position bias rate: {bias['position_bias_rate']:.1%}")
    print(f"✓ Verbosity bias: {bias['verbosity_bias']:.1%}")

    # 5. Save Report to reports/judge_results.json
    report_data = {
        "cohen_kappa": kappa,
        "human_labels_count": len(human_labels),
        "judge_labels": judge_labels,
        "human_labels": human_labels,
        "bias_report": bias,
        "pairwise_results": [
            {
                "question": r.question,
                "answer_a": r.answer_a,
                "answer_b": r.answer_b,
                "winner_pass1": r.winner_pass1,
                "winner_pass2": r.winner_pass2,
                "final_winner": r.final_winner,
                "position_consistent": r.position_consistent,
                "reasoning": r.reasoning_pass1,
            }
            for r in judge_results
        ]
    }

    with open("reports/judge_results.json", "w", encoding="utf-8") as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)

    print("\n✓ Judge results saved → reports/judge_results.json")
