import os
import re
from typing import Any
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from common import call_openrouter, get_sample_documents


try:
    from Task_2 import HybridRetriever, build_chunks
except ImportError:
    from Task_2 import HybridRetriever, build_chunks


CITATION_PATTERN = re.compile(r"\[Source:\s*([^,\]]+),\sPage\s(\d+)\]", re.IGNORECASE)


PROMPT_V1_BASIC = """Answer the question below.
Question: {question}
Answer:"""


PROMPT_V2_ONLY_CONTEXT = """Answer ONLY using the provided context. Do not use outside knowledge.
Context:
{context}


Question: {question}
Answer:"""


PROMPT_V3_STRICT_CITE = """Answer ONLY using the provided context.
Add a citation like right after each factual claim.
If the answer is not explicitly supported by the context, say "I don't have information on that" and do not guess.
Context:
{context}


Question: {question}
Answer:"""




def format_context_with_citations(chunks: list[dict[str, Any]]) -> str:
    lines = []
    for c in chunks:
        src = c["metadata"]["source"]
        page = c["metadata"]["page"]
        lines.append(f" {c['text']}")
    return "\n".join(lines)




def parse_and_validate_citations(answer_text: str, retrieved_chunks: list[dict[str, Any]]) -> dict[str, Any]:
    """Parse citations and flag any not in retrieved chunks."""
    valid_pairs = {
        (c["metadata"]["source"].strip().lower(), int(c["metadata"]["page"]))
        for c in retrieved_chunks
    }
    found = CITATION_PATTERN.findall(answer_text)
    parsed = [{"source": src.strip(), "page": int(page)} for src, page in found]
    invalid = [
        item for item in parsed
        if (item["source"].lower(), item["page"]) not in valid_pairs
    ]
    return {
        "citations": parsed,
        "invalid_citations": invalid,
        "has_invalid": len(invalid) > 0,
    }




def evaluate_prompts(retriever: HybridRetriever) -> dict[str, int]:
    """Compare v1, v2, v3 on 10 questions (7 answerable, 3 unanswerable) and count hallucinations."""
    questions_10 = [
        ("How many casual leaves per year?", True),
        ("How many sick leaves are allowed?", True),
        ("How many WFH days per week are permitted?", True),
        ("What is the cab reimbursement cap?", True),
        ("How often should passwords be rotated?", True),
        ("How long is probation?", True),
        ("Can company data be uploaded to public tools?", True),
        ("What is the company gratuity payout formula?", False),
        ("Who won the cricket world cup?", False),
        ("What is the office parking fee in Bengaluru?", False),
    ]


    prompts = {
        "v1": PROMPT_V1_BASIC,
        "v2": PROMPT_V2_ONLY_CONTEXT,
        "v3": PROMPT_V3_STRICT_CITE,
    }
    hallucination_counts = {"v1": 0, "v2": 0, "v3": 0}


    for q, is_answerable in questions_10:
        ctx_chunks = retriever.hybrid_search(q, top_k=3)
        ctx_str = format_context_with_citations(ctx_chunks)


        for name, tmpl in prompts.items():
            filled = tmpl.format(context=ctx_str, question=q) if "{context}" in tmpl else tmpl.format(question=q)
            ans = call_openrouter(prompt=filled, temperature=0.0)
            refused = (
                "don't have information" in ans.lower()
                or "not mentioned" in ans.lower()
                or "no information" in ans.lower()
            )
            cit_check = parse_and_validate_citations(ans, ctx_chunks)


            is_hallucinated = ((not is_answerable) and not refused) or cit_check["has_invalid"]
            if is_hallucinated:
                hallucination_counts[name] += 1


    return hallucination_counts




def manual_faithfulness_score(claims: list[str], supported_flags: list[bool]) -> float:
    """Manual faithfulness: fraction of claims supported by retrieved context."""
    if not claims:
        return 1.0
    return sum(1 for flag in supported_flags if flag) / len(claims)




app = FastAPI(title="Policy Q&A Assistant API")
API_KEY = os.getenv("API_KEY", "dev-secret-key")
_docs = get_sample_documents()
_retriever = HybridRetriever(build_chunks(_docs))




class AskRequest(BaseModel):
    question: str
    k: int = 4




@app.post("/ask")
def ask_endpoint(req: AskRequest, x_api_key: str | None = Header(default=None)) -> dict[str, Any]:
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


    chunks = _retriever.hybrid_search(req.question, top_k=req.k)
    if not chunks:
        raise HTTPException(status_code=404, detail="No relevant policy chunks found for your query.")


    context_str = format_context_with_citations(chunks)
    prompt = PROMPT_V3_STRICT_CITE.format(context=context_str, question=req.question)
    answer_text = call_openrouter(prompt=prompt, temperature=0.0)
    validation = parse_and_validate_citations(answer_text, chunks)


    sources = [
        {"source": c["metadata"]["source"], "page": c["metadata"]["page"], "snippet": c["text"]}
        for c in chunks
    ]
    return {
        "answer": answer_text,
        "sources": sources,
        "citation_validation": validation,
    }




if __name__ == "__main__":
    counts = evaluate_prompts(_retriever)
    print(f"Prompt v1 hallucinations: {counts['v1']} | v2: {counts['v2']} | v3: {counts['v3']}")


    five_sample_evaluations = [
        (["Employees get 12 casual leaves per year."], [True]),
        (["Sick leave is 7 days per year.", "Medical certificate is needed after 3 days."], [True, True]),
        (["Hybrid work allows 2 WFH days per week."], [True]),
        (["Cab travel is capped at Rs 1500 per day."], [True]),
        (["Passwords must be rotated every 90 days."], [True]),
    ]
    scores = [manual_faithfulness_score(claims, flags) for claims, flags in five_sample_evaluations]
    print(f"Manual faithfulness across 5 answers: {sum(scores)/len(scores):.2f}")