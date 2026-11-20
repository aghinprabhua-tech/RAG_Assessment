import json
import asyncio
import math
import re
from pathlib import Path
from typing import Any
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    pipeline,
)
from ragas.llms.base import InstructorBaseRagasLLM
from ragas.embeddings import HuggingFaceEmbeddings as RagasHuggingFaceEmbeddings
from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)
from common import get_sample_documents
from Task_2 import HybridRetriever, build_chunks
from Task_3 import (
    PROMPT_V3_STRICT_CITE,
    format_context_with_citations,
)


BASE_DIR = Path(__file__).resolve().parent
GOLDEN_PATH = BASE_DIR / "golden.jsonl"
LLM_MODEL = "google/flan-t5-base"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"




class LocalRagasLLM(InstructorBaseRagasLLM):
    def __init__(self, hf_pipeline):
        self.hf_pipeline = hf_pipeline


    def generate(self, prompt: str, response_model: Any):
        schema = response_model.model_json_schema()


        instruction = (
            "Return only a valid JSON object. "
            "Do not include explanations or markdown.\n\n"
            "The JSON must follow this schema:\n"
            + json.dumps(schema)
            + "\n\nEvaluation prompt:\n"
            + str(prompt)
        )
        output = ""


        for attempt in range(2):
            result = self.hf_pipeline(
                instruction,
                max_new_tokens=300,
                do_sample=False,
                truncation=True,
            )
            output = result[0]["generated_text"].strip()


            # Remove markdown code fences if present
            output = (
                re.sub(
                    r"(?:json)?",
                    "",
                    output,
                    flags=re.IGNORECASE,
                )
                .replace("", "")
                .strip()
            )


            # Extract JSON object
            match = re.search(r"\{.*\}", output, re.DOTALL)
            if match:
                try:
                    return response_model.model_validate_json(match.group(0))
                except Exception:
                    pass
            # Retry with a simpler instruction
            instruction = (
                "Return a valid JSON object only. "
                "Follow this schema exactly:\n"
                + json.dumps(schema)
                + "\nPrevious output:\n"
                + output
            )
        raise ValueError(
            "Local FLAN-T5 could not produce valid structured JSON. "
            f"Last output: {output}"
        )


    async def agenerate(self, prompt: str, response_model: Any):
        return await asyncio.to_thread(
            self.generate,
            prompt,
            response_model,
        )




def get_ragas_models() -> tuple[Any, Any, Any]:
    print("Loading local Hugging Face models...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)


    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL)
    model_pipeline = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if device == "cuda" else -1,
        max_new_tokens=180,
        do_sample=False,
        truncation=True,
    )


    # Used for normal RAG answer generation
    local_llm = model_pipeline


    # Used by RAGAS Collections
    evaluator_llm = LocalRagasLLM(model_pipeline)


    # Modern local RAGAS embeddings
    evaluator_embeddings = RagasHuggingFaceEmbeddings(
        model=EMBEDDING_MODEL,
        device=device,
        normalize_embeddings=True,
    )


    print("Local models loaded successfully.")


    return local_llm, evaluator_llm, evaluator_embeddings




def create_golden_dataset(
    path: Path = GOLDEN_PATH,
) -> list[dict[str, Any]]:


    rows = [
        {
            "id": 1,
            "type": "simple",
            "question": "How many casual leaves per year?",
            "ground_truth": "Employees receive 12 casual leaves per year.",
            "source": "leave_policy.pdf",
            "page": 1,
        },
        {
            "id": 2,
            "type": "simple",
            "question": "How many sick leaves are allowed per year?",
            "ground_truth": "Sick leave of up to 7 days per year is available.",
            "source": "leave_policy.pdf",
            "page": 2,
        },
        {
            "id": 3,
            "type": "simple",
            "question": "How many work-from-home days are allowed per week?",
            "ground_truth": "Up to 2 work-from-home days per week with manager approval.",
            "source": "wfh_policy.pdf",
            "page": 1,
        },
        {
            "id": 4,
            "type": "simple",
            "question": "What are the core collaboration hours?",
            "ground_truth": "Core collaboration hours are 10:00 AM to 4:00 PM IST.",
            "source": "wfh_policy.pdf",
            "page": 1,
        },
        {
            "id": 5,
            "type": "simple",
            "question": "Within how many days must travel reimbursements be submitted?",
            "ground_truth": "Within 15 days of trip completion with GST invoices.",
            "source": "reimbursement_policy.pdf",
            "page": 1,
        },
        {
            "id": 6,
            "type": "simple",
            "question": "How often must passwords be rotated?",
            "ground_truth": "Passwords must be rotated every 90 days.",
            "source": "it_security_policy.pdf",
            "page": 1,
        },
        {
            "id": 7,
            "type": "simple",
            "question": "How long is the probation period for new full-time hires?",
            "ground_truth": "New full-time hires serve a 6-month probation period.",
            "source": "probation_policy.pdf",
            "page": 1,
        },
        {
            "id": 8,
            "type": "simple",
            "question": "What happens if confidential data is uploaded to public tools?",
            "ground_truth": "It violates code of conduct and may lead to disciplinary action.",
            "source": "code_of_conduct.pdf",
            "page": 1,
        },
        # 4 Multi-chunk questions
        {
            "id": 9,
            "type": "multi_chunk",
            "question": "Summarize casual leave and sick leave rules.",
            "ground_truth": "12 casual leaves and up to 7 sick leaves per year (certificate needed after 3 days).",
            "source": "leave_policy.pdf",
            "page": 1,
        },
        {
            "id": 10,
            "type": "multi_chunk",
            "question": "What are the WFH days and core hours?",
            "ground_truth": "Up to 2 WFH days per week and core hours from 10:00 AM to 4:00 PM IST.",
            "source": "wfh_policy.pdf",
            "page": 1,
        },
        {
            "id": 11,
            "type": "multi_chunk",
            "question": "What are the travel claim timeline and cab cap?",
            "ground_truth": "Submit within 15 days with GST invoices; cab travel is capped at Rs 1500 per day.",
            "source": "reimbursement_policy.pdf",
            "page": 1,
        },
        {
            "id": 12,
            "type": "multi_chunk",
            "question": "What are the key security and data confidentiality rules?",
            "ground_truth": "MFA is required, passwords rotate every 90 days, and confidential data cannot go to public tools.",
            "source": "it_security_policy.pdf",
            "page": 1,
        },
        # 3 Unanswerable questions
        {
            "id": 13,
            "type": "unanswerable",
            "question": "What is the annual gym membership allowance?",
            "ground_truth": "I don't have information on that.",
            "source": None,
            "page": None,
        },
        {
            "id": 14,
            "type": "unanswerable",
            "question": "How many days of paternity leave are provided?",
            "ground_truth": "I don't have information on that.",
            "source": None,
            "page": None,
        },
        {
            "id": 15,
            "type": "unanswerable",
            "question": "What is the reimbursement limit for home broadband?",
            "ground_truth": "I don't have information on that.",
            "source": None,
            "page": None,
        },
    ]


    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row) + "\n")
    print("Golden dataset created:", path)
    return rows




def generate_local_answer(
    local_llm: Any,
    question: str,
    context: str,
) -> str:
    prompt = PROMPT_V3_STRICT_CITE.format(
        context=context,
        question=question,
    )
    result = local_llm(
        prompt,
        max_new_tokens=180,
        do_sample=False,
        truncation=True,
    )
    return str(result[0]["generated_text"]).strip()




async def calculate_ragas_scores(
    results: list[dict[str, Any]],
    ragas_llm: Any,
    ragas_embeddings: Any,
) -> dict[str, float]:


    metrics = {
        "faithfulness": Faithfulness(llm=ragas_llm),
        "answer_relevancy": AnswerRelevancy(
            llm=ragas_llm,
            embeddings=ragas_embeddings,
            strictness=1,
        ),
        "context_precision": ContextPrecision(llm=ragas_llm),
        "context_recall": ContextRecall(llm=ragas_llm),
    }
    scores = {name: [] for name in metrics}


    for item in results:
        print("Evaluating question:", item["id"])
        metric_inputs = {
            "faithfulness": {
                "user_input": item["question"],
                "response": item["answer"],
                "retrieved_contexts": item["contexts"],
            },
            "answer_relevancy": {
                "user_input": item["question"],
                "response": item["answer"],
            },
            "context_precision": {
                "user_input": item["question"],
                "reference": item["ground_truth"],
                "retrieved_contexts": item["contexts"],
            },
            "context_recall": {
                "user_input": item["question"],
                "reference": item["ground_truth"],
                "retrieved_contexts": item["contexts"],
            },
        }


        item_scores = {}
        for name, metric in metrics.items():
            try:
                result = await metric.ascore(**metric_inputs[name])
                value = float(result.value)


                if math.isfinite(value):
                    scores[name].append(value)
                    item_scores[name] = value
            except Exception as error:
                print(f"{name} failed for question " f"{item['id']}: {error}")


        item["_ragas_scores"] = item_scores
    final_scores = {}
    for name, values in scores.items():
        final_scores[name] = sum(values) / len(values) if values else 0.0
        if not values:
            print(f"Warning: No valid scores for {name}")
    return final_scores




def run_pipeline_on_golden(
    retriever: HybridRetriever,
    golden_rows: list[dict[str, Any]],
    local_llm: Any,
    ragas_llm: Any,
    ragas_embeddings: Any,
    use_rerank: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, float]]:


    per_q_results = []


    for item in golden_rows:
        question = item["question"]
        print("\nProcessing:", question)
        candidates = retriever.hybrid_search(
            question,
            top_k=20 if use_rerank else 4,
        )
        if use_rerank:
            top_chunks = retriever.rerank(
                question,
                candidates,
                top_k=4,
            )
        else:
            top_chunks = candidates[:4]
        contexts = [chunk["text"] for chunk in top_chunks]
        formatted_context = format_context_with_citations(top_chunks)
        answer = generate_local_answer(
            local_llm,
            question,
            formatted_context,
        )
        per_q_results.append(
            {
                "id": item["id"],
                "type": item["type"],
                "question": question,
                "answer": answer,
                "contexts": contexts,
                "ground_truth": item["ground_truth"],
            }
        )
        print("Answer:", answer)
    print("\nCalculating RAGAS Collections metrics...")
    metrics_dict = asyncio.run(
        calculate_ragas_scores(
            per_q_results,
            ragas_llm,
            ragas_embeddings,
        )
    )
    return per_q_results, metrics_dict




def classify_worst_3(
    per_q_results: list[dict[str, Any]],
) -> list[dict[str, str]]:


    scored = [item for item in per_q_results if item.get("_ragas_scores")]
    scored.sort(
        key=lambda item: (
            sum(item["_ragas_scores"].values()) / len(item["_ragas_scores"])
        )
    )


    candidates = scored[:3]
    classified = []


    for item in candidates:
        answer = item["answer"].lower()
        refused = (
            "don't have information" in answer
            or "no information" in answer
            or "not provided" in answer
            or "don't know" in answer
        )


        if item["type"] == "unanswerable" and not refused:
            failure_type = "generation_failure"
        elif not item["contexts"]:
            failure_type = "retrieval_failure"
        else:
            failure_type = "retrieval_or_generation_failure"


        classified.append(
            {
                "id": str(item["id"]),
                "question": item["question"],
                "failure_type": failure_type,
            }
        )
    return classified




def print_business_summary(
    before_metrics: dict[str, float],
    after_metrics: dict[str, float],
) -> None:


    print("\n--- Business Summary ---")
    print(
        f"Cross-encoder reranking changed faithfulness from "
        f"{before_metrics['faithfulness']:.2f} to "
        f"{after_metrics['faithfulness']:.2f}. "
        f"Context precision changed from "
        f"{before_metrics['context_precision']:.2f} to "
        f"{after_metrics['context_precision']:.2f}. "
        f"These metrics help assess whether employees receive "
        f"reliable answers based on HR, travel, WFH and IT policies. "
        f"Unsupported questions should be refused clearly to reduce "
        f"inaccurate responses and compliance risks."
    )




if __name__ == "__main__":
    print("\nCreating golden dataset...")
    golden = create_golden_dataset()


    print("\nLoading documents...")


    docs = get_sample_documents()


    print("\nInitializing hybrid retriever...")


    retriever = HybridRetriever(build_chunks(docs))


    print("\nLoading local models...")
    local_llm, ragas_llm, ragas_embeddings = get_ragas_models()
    print("\n========== BASELINE EVALUATION ==========")


    before_items, before_scores = run_pipeline_on_golden(
        retriever,
        golden,
        local_llm,
        ragas_llm,
        ragas_embeddings,
        use_rerank=False,
    )
    worst_3 = classify_worst_3(before_items)


    print("\n3 Worst Questions Classification:")


    for item in worst_3:
        print(item)
    print("\n========== AFTER RERANKING ==========")
    after_items, after_scores = run_pipeline_on_golden(
        retriever,
        golden,
        local_llm,
        ragas_llm,
        ragas_embeddings,
        use_rerank=True,
    )


    print("\n========== METRIC COMPARISON ==========")
    print("| Metric | Before | After |")
    print("|---|---:|---:|")
    for metric in [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    ]:
        print(
            f"| {metric} | "
            f"{before_scores[metric]:.2f} | "
            f"{after_scores[metric]:.2f} |"
        )
    print_business_summary(
        before_scores,
        after_scores,
    )