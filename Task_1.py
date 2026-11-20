from typing import Any
from common import (
    call_openrouter,
    chunk_text,
    cosine_similarity,
    estimate_tokens,
    get_sample_documents,
    make_stable_id,
    simple_embedding,
)


FALLBACK_MESSAGE = "I don't have information on that"
STRICT_PROMPT_TEMPLATE = """Answer ONLY using the context below. If the answer is not explicitly in the context, reply with exactly: "{fallback}"


Context:
{context}


Question: {question}
Answer:"""




class VectorStore:
    def __init__(self) -> None:
        self.records: dict[str, dict[str, Any]] = {}


    def upsert(self, item: dict[str, Any]) -> None:
        # Stable ID ensures idempotency when re-indexed
        self.records[item["id"]] = item




def index_documents(docs: list[dict[str, Any]], store: VectorStore) -> int:
    """Load -> parse -> chunk -> embed -> store with metadata and stable IDs."""
    count = 0
    for doc in docs:
        chunks = chunk_text(doc["text"])
        for idx, ch in enumerate(chunks):
            rec_id = make_stable_id(doc["source"], doc["page"], idx, ch)
            record = {
                "id": rec_id,
                "text": ch,
                "embedding": simple_embedding(ch),
                "metadata": {
                    "source": doc["source"],
                    "page": doc["page"],
                    "department": doc["department"],
                    "year": doc["year"],
                },
            }
            store.upsert(record)
            count += 1
    return count




def retrieve(store: VectorStore, question: str, k: int = 4) -> list[dict[str, Any]]:
    q_emb = simple_embedding(question)
    scored = []
    for rec in store.records.values():
        score = cosine_similarity(q_emb, rec["embedding"])
        scored.append({**rec, "score": score})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:k]




def fit_context(chunks: list[dict[str, Any]], budget: int) -> list[dict[str, Any]]:
    """Keep chunks that fit within the token budget."""
    selected = []
    used_tokens = 0
    for ch in chunks:
        tok = estimate_tokens(ch["text"])
        if used_tokens + tok <= budget:
            selected.append(ch)
            used_tokens += tok
        else:
            break
    return selected




def answer(
    store: VectorStore,
    question: str,
    k: int = 4,
    score_threshold: float = 0.15,
    token_budget: int = 400,
) -> str:
    """Embed -> retrieve -> budget -> fallback check -> prompt -> OpenRouter LLM."""
    retrieved = retrieve(store, question, k=k)
    if not retrieved or retrieved[0]["score"] < score_threshold:
        return FALLBACK_MESSAGE


    fitted = fit_context(retrieved, budget=token_budget)
    if not fitted:
        return FALLBACK_MESSAGE


    context_str = "\n".join(f"- {c['text']}" for c in fitted)
    prompt = STRICT_PROMPT_TEMPLATE.format(
        fallback=FALLBACK_MESSAGE,
        context=context_str,
        question=question,
    )
    return call_openrouter(prompt=prompt, temperature=0.0)




if __name__ == "__main__":
    store = VectorStore()
    docs = get_sample_documents()


    # Run twice to verify idempotency
    first_run_count = index_documents(docs, store)
    unique_after_first = len(store.records)
    second_run_count = index_documents(docs, store)
    unique_after_second = len(store.records)
    print(f"Idempotency check: run1={first_run_count} (store={unique_after_first}), run2={second_run_count} (store={unique_after_second})")


    # Test k = 2, 5, 10 with fit_context
    sample_q = "How many casual leaves per year?"
    for k_val in [2, 5, 10]:
        raw_chunks = retrieve(store, sample_q, k=k_val)
        budgeted = fit_context(raw_chunks, budget=60)
        print(f"k={k_val}: retrieved={len(raw_chunks)}, after fit_context(budget=60)={len(budgeted)}")


    # 6 answerable + 4 unanswerable questions
    test_questions = [
        ("How many casual leaves per year?", True),
        ("How many sick leaves per year are allowed?", True),
        ("How many work-from-home days are allowed per week?", True),
        ("What is the daily local cab reimbursement cap?", True),
        ("How often must passwords be rotated?", True),
        ("How long is the probation period for new hires?", True),
        ("Who won the world cup?", False),
        ("What is the company stock price today?", False),
        ("How many paid maternity leaves are given in 2028?", False),
        ("What is the canteen menu for Friday?", False),
    ]


    for q, _ in test_questions:
        ans = answer(store, q, k=4, score_threshold=0.15)
        print(f"Q: {q} -> {ans}")