import os
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class AccessDenied(Exception):
    pass

EMBED_MODEL = "text-embedding-3-small"
TOP_K = 4

MAX_CONTEXT_CHARS = 7000
MAX_CHUNK_CHARS_USED = 1000


# Patient name map
_PATIENT_NAME_MAP = {
    "patient_001": ["Maria Santos"],
    "patient_002": ["James OBrien", "James O'Brien"],
    "patient_003": ["Priya Kapoor"],
    "patient_004": ["Robert Chen"],
}


# Artifact paths use absolute paths
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
_ARTIFACTS_DIR = _PROJECT_ROOT / "artifacts"

_EMBEDDINGS_PATH = _ARTIFACTS_DIR / "embeddings.npy"
_CHUNK_TEXTS_PATH = _ARTIFACTS_DIR / "chunk_texts.npy"
_CHUNK_SOURCES_PATH = _ARTIFACTS_DIR / "chunk_sources.npy"


# Normalization helper for safer name matching
def _norm(s: str) -> str:
    s = (s or "").lower().replace("’", "'")
    s = re.sub(r"[^a-z0-9\s']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# Internal access enforcement function
def enforce_doctor_access(role: str, patient_id: str, consent: bool) -> None:
    if not patient_id:
        raise AccessDenied("Access denied: missing patient identity binding (patient_id).")

    if role not in ["doctor", "admin"]:
        raise AccessDenied("Access denied: only doctors can access doctor notes.")

    if not bool(consent):
        raise AccessDenied("Access denied: patient consent is required for doctor notes.")


def _get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found. Ensure your .env contains it.")
    return OpenAI(api_key=api_key)


# Loads the global RAG artifacts
def _load_rag_arrays() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not _EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(f"Missing embeddings file: {_EMBEDDINGS_PATH}")
    if not _CHUNK_TEXTS_PATH.exists():
        raise FileNotFoundError(f"Missing chunk texts file: {_CHUNK_TEXTS_PATH}")
    if not _CHUNK_SOURCES_PATH.exists():
        raise FileNotFoundError(f"Missing chunk sources file: {_CHUNK_SOURCES_PATH}")

    embeddings = np.load(str(_EMBEDDINGS_PATH), mmap_mode="r")
    chunk_texts = np.load(str(_CHUNK_TEXTS_PATH), mmap_mode="r")
    chunk_sources = np.load(str(_CHUNK_SOURCES_PATH), mmap_mode="r")
    return embeddings, chunk_texts, chunk_sources


# Retrieves only the doctor note chunks for the bound patient
def retrieve_doctor_context_scoped(
    patient_id: str,
    query: str,
    top_k: int = TOP_K,
) -> Tuple[str, List[str]]:
    embeddings, chunk_texts, chunk_sources = _load_rag_arrays()

    patient_names = _PATIENT_NAME_MAP.get((patient_id or "").lower().strip(), [])
    if not patient_names:
        return "", []

    # Restriction retrieval only to doctor folder
    allowed_idx = []
    for i in range(len(chunk_sources)):
        src = str(chunk_sources[i])
        src_low = src.lower()

        if (os.path.sep + "doctor" + os.path.sep) not in src_low:
            continue

        if not any(_norm(name) in _norm(src) for name in patient_names):
            continue

        allowed_idx.append(i)

    if not allowed_idx:
        return "", []

    client = _get_openai_client()
    emb = client.embeddings.create(model=EMBED_MODEL, input=query)
    q = np.array(emb.data[0].embedding, dtype="float32")
    q = q / (np.linalg.norm(q) + 1e-12)

    sims = embeddings.dot(q)
    sims_allowed = sims[allowed_idx]
    k = min(top_k, sims_allowed.shape[0])

    if sims_allowed.shape[0] <= k:
        top_local = np.argsort(-sims_allowed)
    else:
        top_local = np.argpartition(-sims_allowed, k)[:k]
        top_local = top_local[np.argsort(-sims_allowed[top_local])]

    picked_sources: List[str] = []
    parts: List[str] = []
    total = 0

    for j in top_local:
        idx = allowed_idx[int(j)]
        text = str(chunk_texts[idx])[:MAX_CHUNK_CHARS_USED]
        src = str(chunk_sources[idx])

        if not text.strip():
            continue

        if total + len(text) > MAX_CONTEXT_CHARS:
            break

        parts.append(text)
        picked_sources.append(src)
        total += len(text)

    return "\n\n".join(parts), picked_sources


# Context retreival and response
def run_doctor_agent(
    user_message: str,
    role: str = "doctor",
    patient_id: str = "patient_001",
    consent: bool = True,
) -> str:
    enforce_doctor_access(role=role, patient_id=patient_id, consent=consent)

    msg = (user_message or "").strip()
    if not msg:
        return "Please enter a question."

    patient_names = _PATIENT_NAME_MAP.get((patient_id or "").lower().strip(), ["the selected patient"])

    context, _sources = retrieve_doctor_context_scoped(
        patient_id=patient_id,
        query=msg,
        top_k=TOP_K,
    )

    client = _get_openai_client()

    system_prompt = f"""
You are a healthcare assistant helping an authorized clinician review doctor notes.

SECURITY RULES (MUST FOLLOW):
- Use ONLY the retrieved doctor-note context for the bound patient_id.
- Do NOT reveal information about other patients.
- If the retrieved context does not contain the answer, say so.
- Summarize or answer directly; do NOT dump the full raw note unless specifically required and authorized.

Bound identity:
- patient_id = {patient_id}

Patient display name:
- {patient_names[0]}

Retrieved Doctor Note Context:
{context if context else "[No doctor-note context retrieved]"}
""".strip()

    resp = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": msg},
        ],
    )

    return (resp.output_text or "").strip()
