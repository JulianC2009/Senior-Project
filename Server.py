import os

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
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=[
                {"role": "system", "content": "You are a helpful chatbot."},
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
