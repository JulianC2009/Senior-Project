import os
import faiss
import numpy as np

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from openai import OpenAI
from agents.langgraph_orchestrator import run_orchestrator

# Load .env 
load_dotenv()

print("KEY PRESENT:", bool(os.getenv("OPENAI_API_KEY")))
print("KEY PREFIX:", (os.getenv("OPENAI_API_KEY") or "")[:7])

app = FastAPI()

# OpenAI client 
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

#loading the vector index
EMBED_MODEL = "text-embedding-3-small"
VECTOR_Dimension = 1536
TOP_K = 2

try:
    index = faiss.read_index("medical_index.faiss")
    doc_texts = np.load("doc_texts.npy", allow_pickle=True)
    print("RAG index loaded.")
except Exception as e:
    print("RAG index failed to load:", str(e))
    index = None
    doc_texts = None


# Request/Response schemas 
class ChatRequest(BaseModel):
    message: str

    # simulated security context
    role: str = "patient"  # patient | doctor | insurance | admin
    patient_id: str = "patient_001"
    consent: bool = True

class ChatResponse(BaseModel):
    reply: str


# API route
@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    user_message = (req.message or "").strip()
    if not user_message:
        raise HTTPException(status_code=400, detail="Missing 'message' in request body.")

    # Agent orchestration with LangGraph
    # If the request matches a tool/agent path, return that result.
    # Otherwise fallback to RAG + OpenAI.
    try:
        agent_reply = run_orchestrator(
            user_message=user_message,
            role=req.role,
            patient_id=req.patient_id,
            consent=req.consent,
        )

        # Convention: tool responses start with "[" in orchestrator_graph.py
        if isinstance(agent_reply, str) and agent_reply.startswith("["):
            return ChatResponse(reply=agent_reply)

    except Exception as err:
        # For RBAC/tool errors 
        return ChatResponse(reply=f"Access denied / tool error: {str(err)}")

    # RAG + OpenAI fallback
    try:
        context = ""
        use_context = False

        # Retrieve relevant chunks from FAISS if index is loaded
        if index is not None and doc_texts is not None:
            embed_response = client.embeddings.create(
                model=EMBED_MODEL,
                input=user_message
            )

            query_vector = np.array([embed_response.data[0].embedding]).astype("float32")
            distances, indices = index.search(query_vector, TOP_K)

            SIMILARITY_THRESHOLD = 1.0  # adjust if needed

            if distances[0][0] < SIMILARITY_THRESHOLD:
                retrieved_chunks = [doc_texts[i] for i in indices[0]]
                context = "\n\n".join(retrieved_chunks)
                use_context = True

        if use_context:
            system_prompt = f"""
You are a helpful healthcare chatbot.

Use the medical context below to answer the question.
If relevant information is provided in the context, prioritize it.
You may supplement with general medical knowledge if necessary.

Medical Context:
{context}
"""
        else:
            system_prompt = """
You are a helpful healthcare chatbot.

Answer the user's question using your general medical knowledge.
If unsure, clearly state uncertainty.
"""

        response = client.responses.create(
            model="gpt-4.1-mini",
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        )

        return ChatResponse(reply=response.output_text or "")

    except Exception as err:
        status = getattr(err, "status_code", None) or getattr(err, "status", None)
        err_message = getattr(err, "message", None) or str(err)

        print("OpenAI error:", status, err_message)
        raise HTTPException(status_code=500, detail=err_message or "OpenAI request failed.")

app.mount("/", StaticFiles(directory=".", html=True), name="static")
