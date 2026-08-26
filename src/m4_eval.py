from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    from config import OPENAI_API_KEY
    if OPENAI_API_KEY:
        try:
            from ragas import evaluate
            from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
            from datasets import Dataset
            import pandas as pd

            dataset = Dataset.from_dict({
                "question": questions,
                "answer": answers,
                "contexts": contexts,
                "ground_truth": ground_truths,
            })
            result = evaluate(dataset, metrics=[faithfulness, answer_relevancy,
                                                context_precision, context_recall])
            df = result.to_pandas()

            per_question = []
            for _, row in df.iterrows():
                f_val = float(row.get("faithfulness", 0.0)) if not pd.isna(row.get("faithfulness", 0.0)) else 0.0
                ar_val = float(row.get("answer_relevancy", 0.0)) if not pd.isna(row.get("answer_relevancy", 0.0)) else 0.0
                cp_val = float(row.get("context_precision", 0.0)) if not pd.isna(row.get("context_precision", 0.0)) else 0.0
                cr_val = float(row.get("context_recall", 0.0)) if not pd.isna(row.get("context_recall", 0.0)) else 0.0

                per_question.append(EvalResult(
                    question=row["question"],
                    answer=row["answer"],
                    contexts=row["contexts"],
                    ground_truth=row["ground_truth"],
                    faithfulness=f_val,
                    answer_relevancy=ar_val,
                    context_precision=cp_val,
                    context_recall=cr_val
                ))

            res_dict = {}
            for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
                if metric in result:
                    res_dict[metric] = float(result[metric])
                elif metric in df:
                    res_dict[metric] = float(df[metric].mean(skipna=True))
                else:
                    res_dict[metric] = 0.0

            res_dict["per_question"] = per_question
            return res_dict
        except Exception as e:
            print(f"  ⚠️  RAGAS evaluation failed: {e}")

    return {"faithfulness": 0.0, "answer_relevancy": 0.0,
            "context_precision": 0.0, "context_recall": 0.0, "per_question": []}


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    diagnostic_tree = {
        "faithfulness": ("LLM hallucinating", "Tighten prompt, lower temperature, enforce context-only answering"),
        "context_recall": ("Missing relevant chunks", "Improve chunking strategy or add hybrid BM25 + dense search"),
        "context_precision": ("Too many irrelevant chunks", "Add cross-encoder reranking or metadata filter"),
        "answer_relevancy": ("Answer doesn't match question", "Improve LLM answer prompt template"),
    }
    if not eval_results:
        return []

    analyzed = []
    for item in eval_results:
        scores = {
            "faithfulness": item.faithfulness,
            "answer_relevancy": item.answer_relevancy,
            "context_precision": item.context_precision,
            "context_recall": item.context_recall,
        }
        avg_score = sum(scores.values()) / 4.0
        worst_metric = min(scores.keys(), key=lambda k: scores[k])
        diagnosis, suggested_fix = diagnostic_tree.get(
            worst_metric, ("Unknown issue", "Review pipeline step")
        )

        analyzed.append({
            "question": item.question,
            "answer": item.answer,
            "ground_truth": item.ground_truth,
            "contexts": item.contexts,
            "score": avg_score,
            "worst_metric": worst_metric,
            "worst_metric_score": scores[worst_metric],
            "diagnosis": diagnosis,
            "suggested_fix": suggested_fix,
        })

    analyzed.sort(key=lambda x: x["score"])
    return analyzed[:bottom_n]


def save_report(results: dict, failures: list[dict], path: str = "ragas_report.json"):
    """Save evaluation report to JSON."""
    def _convert(obj):
        try:
            import numpy as np
            if isinstance(obj, (np.ndarray, list, tuple)):
                return [_convert(x) for x in obj]
            if isinstance(obj, (np.generic, np.number)):
                return obj.item()
        except ImportError:
            pass

        if isinstance(obj, (list, tuple)):
            return [_convert(x) for x in obj]
        if isinstance(obj, dict):
            return {str(k): _convert(v) for k, v in obj.items()}
        if isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        return str(obj)

    report = {
        "aggregate": _convert({k: float(v) for k, v in results.items() if k != "per_question"}),
        "num_questions": len(results.get("per_question", [])),
        "failures": _convert(failures),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
