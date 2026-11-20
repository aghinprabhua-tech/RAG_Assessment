import hashlib
import os
from typing import Any
import httpx
import numpy as np
from dotenv import load_dotenv


load_dotenv()


OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_CHAT_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"




def call_openrouter(
    prompt: str,
    system_prompt: str = "You are a helpful company policy assistant.",
    model: str = DEFAULT_CHAT_MODEL,
    temperature: float = 0.0,
) -> str:
    """Call OpenRouter chat completions API."""
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY is not set in environment variables.")


    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
    }
    with httpx.Client(timeout=30.0) as client:
        resp = client.post(OPENROUTER_URL, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()




def simple_embedding(text: str, dim: int = 128) -> np.ndarray:
    """Deterministic local embedding vector for lightweight offline/test runs."""
    vec = np.zeros(dim, dtype=np.float32)
    for token in text.lower().split():
        digest = hashlib.md5(token.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:2], "little") % dim
        sign = 1.0 if digest[2] % 2 == 0 else -1.0
        vec[idx] += sign
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec




def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)




def make_stable_id(source: str, page: int, chunk_idx: int, text: str) -> str:
    raw = f"{source}|p{page}|c{chunk_idx}|{text[:50]}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]




def chunk_text(text: str, chunk_size: int = 180, overlap: int = 30) -> list[str]:
    words = text.split()
    if not words:
        return []
    chunks = []
    step = max(1, chunk_size - overlap)
    for start in range(0, len(words), step):
        part = " ".join(words[start : start + chunk_size])
        if part:
            chunks.append(part)
        if start + chunk_size >= len(words):
            break
    return chunks




def estimate_tokens(text: str) -> int:
    return max(1, int(len(text.split()) * 1.3))




def get_sample_documents() -> list[dict[str, Any]]:
    """At least 8 company policy documents with metadata."""
    return [
        {
            "source": "leave_policy.pdf",
            "page": 1,
            "department": "HR",
            "year": 2026,
            "text": "Employees receive 12 casual leaves per year and 12 earned leaves annually. Unused casual leaves lapse at year-end.",
        },
        {
            "source": "leave_policy.pdf",
            "page": 2,
            "department": "HR",
            "year": 2026,
            "text": "Sick leave of up to 7 days per year is available. A medical certificate is required for sick leave exceeding 3 consecutive days.",
        },
        {
            "source": "wfh_policy.pdf",
            "page": 1,
            "department": "HR",
            "year": 2026,
            "text": "Hybrid work allows up to 2 work-from-home days per week with manager approval. Core collaboration hours are 10:00 AM to 4:00 PM IST.",
        },
        {
            "source": "reimbursement_policy.pdf",
            "page": 1,
            "department": "Finance",
            "year": 2026,
            "text": "Travel reimbursements must be submitted within 15 days of trip completion with GST invoices. Local cab travel is capped at Rs 1500 per day.",
        },
        {
            "source": "reimbursement_policy_old.pdf",
            "page": 1,
            "department": "Finance",
            "year": 2023,
            "text": "Outdated 2023 rule: Local cab travel reimbursement was capped at Rs 800 per day and required physical vouchers only.",
        },
        {
            "source": "it_security_policy.pdf",
            "page": 1,
            "department": "IT",
            "year": 2026,
            "text": "Multi-factor authentication is mandatory for all company accounts. Passwords must be rotated every 90 days and never shared.",
        },
        {
            "source": "probation_policy.pdf",
            "page": 1,
            "department": "HR",
            "year": 2026,
            "text": "New full-time hires serve a 6-month probation period. Performance review happens at the end of month 3 and month 6.",
        },
        {
            "source": "code_of_conduct.pdf",
            "page": 1,
            "department": "Legal",
            "year": 2026,
            "text": "Confidential company data must not be uploaded to unapproved public tools. Breach of data confidentiality may lead to disciplinary action.",
        },
    ]