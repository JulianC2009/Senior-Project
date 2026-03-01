import os
import faiss
import numpy as np
from openai import OpenAI
from PyPDF2 import PdfReader

client = OpenAI()

EMBEDDED_MODEL = "text-embedding-3-small"
DIMENSION = 1536

documents = []
doc_texts = []

# PDF and TXT text extraction
def extract_text_from_pdf(filepath):
    text = ""
    reader = PdfReader(filepath)
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text() + "\n"
    return text

#loads the documents
for root, dirs, files in os.walk("knowledge_base"):
    for filename in files:
        filepath = os.path.join(root, filename)

        try:
            # PDF files
            if filename.lower().endswith(".pdf"):
                text = extract_text_from_pdf(filepath)

            # TXT files
            elif filename.lower().endswith(".txt"):
                with open(filepath, "r", encoding="utf-8") as f:
                    text = f.read()

            else:
                continue

            if text.strip():
                documents.append(filepath)
                doc_texts.append(text)
                print(f"Loaded: {filepath}")

        except Exception as e:
            print(f"Error reading {filepath}: {e}")


print(f"\nTotal Documents Loaded: {len(doc_texts)}")


# creates embeddings 
embeddings = []

for text in doc_texts:
    response = client.embeddings.create(
        model=EMBEDDED_MODEL,
        input=text
    )
    embeddings.append(response.data[0].embedding)

embeddings = np.array(embeddings).astype("float32")


# creating the fiass index
index = faiss.IndexFlatL2(DIMENSION)
index.add(embeddings)

faiss.write_index(index, "medical_index.faiss")

# Save texts and filenames
np.save("doc_texts.npy", np.array(doc_texts))
np.save("doc_sources.npy", np.array(documents))

print("\n Index built successfully.")
