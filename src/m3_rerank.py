from __future__ import annotations

"""Module 3: Reranking — Cross-encoder top-20 → top-3 + latency benchmark."""

import os, sys, time
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RERANK_TOP_K


@dataclass
class RerankResult:
    text: str
    original_score: float
    rerank_score: float
    metadata: dict
    rank: int


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name)
            except Exception:
                self._model = None
        return self._model

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        """Rerank documents: top-20 → top-k."""
        if not documents:
            return []

        from numpy import dot
        from numpy.linalg import norm
        from config import OPENAI_API_KEY, EMBEDDING_MODEL

        scored = []
        if OPENAI_API_KEY:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=OPENAI_API_KEY)
                texts = [query] + [doc["text"] for doc in documents]
                resp = client.embeddings.create(input=texts, model=EMBEDDING_MODEL)
                q_emb = resp.data[0].embedding
                doc_embs = [item.embedding for item in resp.data[1:]]

                for doc, emb in zip(documents, doc_embs):
                    sim = float(dot(q_emb, emb) / (norm(q_emb) * norm(emb) + 1e-9))
                    scored.append((sim, doc))
            except Exception as e:
                print(f"  ⚠️ OpenAI rerank failed: {e}")

        if not scored:
            try:
                model = self._load_model()
                if model is not None:
                    pairs = [(query, doc["text"]) for doc in documents]
                    scores = model.predict(pairs)
                    if isinstance(scores, (int, float)):
                        scores = [scores]
                    scored = list(zip(scores, documents))
            except Exception:
                pass

        if not scored:
            q_words = set(query.lower().split())
            for doc in documents:
                d_words = set(doc["text"].lower().split())
                score = len(q_words.intersection(d_words)) / max(len(q_words), 1)
                scored.append((score, doc))

        scored.sort(key=lambda x: x[0], reverse=True)

        return [
            RerankResult(
                text=doc["text"],
                original_score=float(doc.get("score", 0.0)),
                rerank_score=float(score),
                metadata=doc.get("metadata", {}),
                rank=i
            )
            for i, (score, doc) in enumerate(scored[:top_k])
        ]


class FlashrankReranker:
    """Lightweight alternative (<5ms). Optional."""
    def __init__(self):
        self._model = None

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        """Optional Flashrank implementation."""
        return []


def benchmark_reranker(reranker, query: str, documents: list[dict], n_runs: int = 5) -> dict:
    """Benchmark latency over n_runs. (Đã implement sẵn)"""
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        reranker.rerank(query, documents)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    return {"avg_ms": sum(times) / len(times), "min_ms": min(times), "max_ms": max(times)}


if __name__ == "__main__":
    query = "Nhân viên được nghỉ phép bao nhiêu ngày?"
    docs = [
        {"text": "Nhân viên được nghỉ 12 ngày/năm.", "score": 0.8, "metadata": {}},
        {"text": "Mật khẩu thay đổi mỗi 90 ngày.", "score": 0.7, "metadata": {}},
        {"text": "Thời gian thử việc là 60 ngày.", "score": 0.75, "metadata": {}},
    ]
    reranker = CrossEncoderReranker()
    for r in reranker.rerank(query, docs):
        print(f"[{r.rank}] {r.rerank_score:.4f} | {r.text}")
