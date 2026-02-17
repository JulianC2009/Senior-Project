import os
import faiss
import numpy as np

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from openai import OpenAI

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


class ChatResponse(BaseModel):
    reply: str


# API route 
@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    message = (req.message or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="Missing 'message' in request body.")

    try:
        context = ""

        if index is not None and doc_texts is not None:
            embed_response = client.embeddings.create(
                model=EMBED_MODEL,
                input=message
            )

            query_vector = np.array(
                [embed_response.data[0].embedding]
            ).astype("float32")

            distances, indices = index.search(query_vector, TOP_K)

            retrieved_chunks = [doc_texts[i] for i in indices[0]]
            context = "\n\n".join(retrieved_chunks)

        response = client.responses.create(
            model="gpt-4.1-mini",
            input = [
                {
                    "role": "system",
                    "content": f"""
You are a helpful healthcare chatbot.

Use ONLY the information provided in the medical context below.
If the answer is not in the context, say:
"I do not have enough information in the medical database."

Medical Context:
{context}
"""
                },
                {"role": "user", "content": message},
            ],
        )

        return ChatResponse(reply=response.output_text or "")


    except Exception as err:
        status = getattr(err, "status_code", None) or getattr(err, "status", None)
        message = getattr(err, "message", None) or str(err)

        print("OpenAI error:", status, message)
        resp = getattr(err, "response", None)
        if resp is not None:
            try:
                print("OpenAI response:", resp)
            except Exception:
                pass

        raise HTTPException(status_code=500, detail=message or "OpenAI request failed.")


app.mount("/", StaticFiles(directory=".", html=True), name="static")

