import time
from typing import Any
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from common import (
    call_openrouter,
    chunk_text,
    cosine_similarity,
    get_sample_documents,
    make_stable_id,
    simple_embedding,
)




def build_chunks(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items = []
    for doc in docs:
        for idx, ch in enumerate(chunk_text(doc["text"])):
            items.append(
                {
                    "id": make_stable_id(doc["source"], doc["page"], idx, ch),
                    "text": ch,
                    "embedding": simple_embedding(ch),
                    "metadata": {
                        "source": doc["source"],
                        "page": doc["page"],
                        "department": doc["department"],
                        "year": doc["year"],
                    },
                }
            )
    return items




class HybridRetriever:
    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        self.chunks = chunks
        self.tokenized_corpus = [c["text"].lower().split() for c in chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


    def bm25_search(self, query: str, top_k: int = 50, pool: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
        target = pool if pool is not None else self.chunks
        corpus = [c["text"].lower().split() for c in target]
        bm25 = BM25Okapi(corpus) if pool is not None else self.bm25
        scores = bm25.get_scores(query.lower().split())
        ranked = sorted(zip(target, scores), key=lambda x: x[1], reverse=True)
        return [{**item, "score": float(sc)} for item, sc in ranked[:top_k]]


    def vector_search(self, query: str, top_k: int = 50, pool: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
        target = pool if pool is not None else self.chunks
        q_emb = simple_embedding(query)
        scored = [{**item, "score": cosine_similarity(q_emb, item["embedding"])} for item in target]
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]


    @staticmethod
    def rrf_merge(rankings: list[list[dict[str, Any]]], top_k: int = 50, k_const: int = 60) -> list[dict[str, Any]]:
        """Reciprocal Rank Fusion: 1 / (60 + rank) using 1-based rank."""
        rrf_scores: dict[str, float] = {}
        by_id: dict[str, dict[str, Any]] = {}
        for ranking in rankings:
            for rank_idx, item in enumerate(ranking, start=1):
                cid = item["id"]
                by_id[cid] = item
                rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (k_const + rank_idx))


        merged = [{**by_id[cid], "rrf_score": sc} for cid, sc in rrf_scores.items()]
        merged.sort(key=lambda x: x["rrf_score"], reverse=True)
        return merged[:top_k]


    def hybrid_search(self, query: str, top_k: int = 50, pool: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
        b_res = self.bm25_search(query, top_k=top_k, pool=pool)
        v_res = self.vector_search(query, top_k=top_k, pool=pool)
        return self.rrf_merge([b_res, v_res], top_k=top_k)


    def rerank(self, query: str, candidates: list[dict[str, Any]], top_k: int = 5) -> list[dict[str, Any]]:
        if not candidates:
            return []
        pairs = [(query, c["text"]) for c in candidates]
        scores = self.cross_encoder.predict(pairs)
        reranked = [{**c, "ce_score": float(sc)} for c, sc in zip(candidates, scores)]
        reranked.sort(key=lambda x: x["ce_score"], reverse=True)
        return reranked[:top_k]


    @staticmethod
    def rewrite_query(query: str) -> list[str]:
        prompt = (
            "Generate 3 alternative search phrasings for the following company policy question. "
            "Return only the 3 phrasings, one per line, without numbering.\n"
            f"Question: {query}"
        )
        raw = call_openrouter(prompt=prompt, temperature=0.3)
        lines = [line.strip("-•* \t") for line in raw.splitlines() if line.strip()]
        return lines[:3] if len(lines) >= 3 else [query, query, query]


    def query_rewrite_search(self, query: str, top_k: int = 3) -> tuple[list[dict[str, Any]], float]:
        t0 = time.perf_counter()
        rewrites = self.rewrite_query(query)
        rankings = [self.hybrid_search(q, top_k=20) for q in rewrites]
        merged = self.rrf_merge(rankings, top_k=top_k)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return merged, elapsed_ms


    def filter_metadata(self, department: str | None = None, year: int | None = None) -> list[dict[str, Any]]:
        out = self.chunks
        if department is not None:
            out = [c for c in out if c["metadata"]["department"] == department]
        if year is not None:
            out = [c for c in out if c["metadata"]["year"] == year]
        return out




def hit_at_k(results: list[dict[str, Any]], expected_source: str, k: int = 3) -> float:
    top_sources = [r["metadata"]["source"] for r in results[:k]]
    return 1.0 if expected_source in top_sources else 0.0




if __name__ == "__main__":
    docs = get_sample_documents()
    chunks = build_chunks(docs)
    retriever = HybridRetriever(chunks)


    # 12 questions: 6 exact-term, 6 meaning-based
    questions_12 = [
        ("casual leaves per year", "leave_policy.pdf", True),
        ("sick leave medical certificate 3 consecutive days", "leave_policy.pdf", True),
        ("work-from-home days per week", "wfh_policy.pdf", True),
        ("GST invoices 15 days travel", "reimbursement_policy.pdf", True),
        ("Multi-factor authentication 90 days", "it_security_policy.pdf", True),
        ("6-month probation period", "probation_policy.pdf", True),
        ("Can I take a few days off for personal errands?", "leave_policy.pdf", False),
        ("What happens if I fall ill for four days straight?", "leave_policy.pdf", False),
        ("How flexible is remote work from home?", "wfh_policy.pdf", False),
        ("How do I claim taxi expenses after a business trip?", "reimbursement_policy.pdf", False),
        ("What login security rules protect my work account?", "it_security_policy.pdf", False),
        ("Can I put internal company files into a free AI chatbot?", "code_of_conduct.pdf", False),
    ]


    bm25_hits, vec_hits, hyb_hits, hyb_rerank_hits = [], [], [], []
    for q, exp_src, _ in questions_12:
        b = retriever.bm25_search(q, top_k=3)
        v = retriever.vector_search(q, top_k=3)
        h = retriever.hybrid_search(q, top_k=50)
        hr = retriever.rerank(q, h, top_k=5)


        bm25_hits.append(hit_at_k(b, exp_src, 3))
        vec_hits.append(hit_at_k(v, exp_src, 3))
        hyb_hits.append(hit_at_k(h, exp_src, 3))
        hyb_rerank_hits.append(hit_at_k(hr, exp_src, 3))


    print(
        f"Hit@3 -> BM25: {sum(bm25_hits)/len(bm25_hits):.2f} | "
        f"Vector: {sum(vec_hits)/len(vec_hits):.2f} | "
        f"Hybrid: {sum(hyb_hits)/len(hyb_hits):.2f} | "
        f"Hybrid+Rerank: {sum(hyb_rerank_hits)/len(hyb_rerank_hits):.2f}"
    )


    # Print rank of correct chunk before and after reranking (top 50 -> top 5) for 10 questions
    print("\n--- Rerank Rank Shift (10 questions) ---")
    for q, exp_src, _ in questions_12[:10]:
        top50 = retriever.hybrid_search(q, top_k=50)
        before_rank = next((i + 1 for i, c in enumerate(top50) if c["metadata"]["source"] == exp_src), None)
        top5 = retriever.rerank(q, top50, top_k=5)
        after_rank = next((i + 1 for i, c in enumerate(top5) if c["metadata"]["source"] == exp_src), None)
        print(f"Q: '{q}' | Before: {before_rank} -> After: {after_rank}")


    # Query rewriting hit@3 and latency comparison
    t_base = time.perf_counter()
    base_hits = [hit_at_k(retriever.hybrid_search(q, top_k=3), s, 3) for q, s, _ in questions_12]
    base_ms = ((time.perf_counter() - t_base) / len(questions_12)) * 1000


    rw_hits, rw_times = [], []
    for q, s, _ in questions_12:
        res, ms = retriever.query_rewrite_search(q, top_k=3)
        rw_hits.append(hit_at_k(res, s, 3))
        rw_times.append(ms)
    print(f"\nQuery Rewrite -> Hit@3: {sum(rw_hits)/len(rw_hits):.2f} (avg {sum(rw_times)/len(rw_times):.1f} ms) vs Base Hybrid Hit@3: {sum(base_hits)/len(base_hits):.2f} (avg {base_ms:.1f} ms)")


    # Metadata filtering removing outdated 2023 document
    unfiltered_cab = retriever.hybrid_search("Local cab travel reimbursement cap", top_k=3)
    filtered_pool = retriever.filter_metadata(department="Finance", year=2026)
    filtered_cab = retriever.hybrid_search("Local cab travel reimbursement cap", top_k=3, pool=filtered_pool)
    print("\n--- Metadata Filter Check ---")
    print("Unfiltered top sources:", [c["metadata"]["source"] for c in unfiltered_cab])
    print("Filtered (Finance, 2026) top sources:", [c["metadata"]["source"] for c in filtered_cab])