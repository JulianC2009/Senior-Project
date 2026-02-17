import os
import faiss
import numpy as np 

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

EMBEDED_MODEL = "text-embedding-3-small"
Dimension = 1536

documents = []
doc_texts = []

#loading the documents
for filename in os.listdir("knowledge_base"):
    with open(f"knowledge_base/{filename}", "r", encoding="utf-8") as f:
        text = f.read()
        documents.append(filename)
        doc_texts.append(text)

embeddings = []
for text in doc_texts:
    response = client.embeddings.create (
        model = EMBEDED_MODEL,
        input = text
    )
    embeddings.append(response.data[0].embedding)

embeddings = np.array(embeddings).astype("float32")

#creating the FAISS index
index = faiss.IndexFlatL2(Dimension)
index.add(embeddings)

faiss.write_index(index, "medical_index.faiss")

#saving the texts separately
np.save("doc_texts.npy", np.array(doc_texts))

print("Index built successfully.")