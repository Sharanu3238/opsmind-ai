import numpy as np
from fastapi import FastAPI, UploadFile, File
from pymongo import MongoClient
from pypdf import PdfReader
import google.generativeai as genai
from io import BytesIO

# ================= CONFIG =================
MONGO_URI = "mongodb+srv://jupellisnehitha135_db_user:abc1446@cluster0.ponef8z.mongodb.net/?appName=Cluster0"
GEMINI_API_KEY = "AIzaSyDaBnztMDwALj-Q3fKQm7IOpWkbtVlCzhw"
# =========================================

app = FastAPI()

# MongoDB
client = MongoClient(MONGO_URI)
db = client["opsmind"]
collection = db["chunks"]

# Gemini
genai.configure(api_key=GEMINI_API_KEY)

# =========================================
# 🔹 Chunk text
def chunk_text(text, size=1000, overlap=100):
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i:i+size])
        i += size - overlap
    return chunks

# =========================================
def get_embedding(text):
    vec = [ord(c) for c in text[:100]]  # take first 100 chars
    
    # pad with zeros if length < 100
    if len(vec) < 100:
        vec += [0] * (100 - len(vec))
    
    return vec
# =========================================
# 🔹 Cosine similarity (SAFE)
def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)

    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0

    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# =========================================
# 🔹 Retrieve top chunks
def retrieve_chunks(query):
    query_embedding = get_embedding(query)

    docs = list(collection.find())

    if len(docs) == 0:
        return []

    scored = []
    for doc in docs:
        score = cosine_similarity(query_embedding, doc["embedding"])
        scored.append((score, doc))

    scored.sort(reverse=True, key=lambda x: x[0])

    return [doc for _, doc in scored[:3]]

# =========================================
def generate_answer(query):
    top_chunks = retrieve_chunks(query)

    if len(top_chunks) == 0:
        return {
            "answer": "No data available. Please upload a PDF first.",
            "sources": []
        }

    # Combine top chunks
    context = "\n\n".join([c["text"] for c in top_chunks])

    # Simple answer logic (no Gemini)
    for chunk in top_chunks:
        if any(word.lower() in chunk["text"].lower() for word in query.split()):
            return {
                "answer": chunk["text"],
                "sources": [chunk["source"]]
            }

    return {
        "answer": "Answer not found in document.",
        "sources": [c["source"] for c in top_chunks]
    }# =========================================
# 🔹 Upload PDF
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        reader = PdfReader(BytesIO(contents))

        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text

        if text.strip() == "":
            return {"error": "PDF has no readable text"}

        chunks = chunk_text(text)

        # 🔹 Remove old data (avoid duplicates)
        collection.delete_many({"source": file.filename})

        for chunk in chunks:
            embedding = get_embedding(chunk)

            collection.insert_one({
                "text": chunk,
                "embedding": embedding,
                "source": file.filename
            })

        return {"message": "PDF processed successfully"}

    except Exception as e:
        print("UPLOAD ERROR:", e)
        return {"error": str(e)}

# =========================================
# 🔹 Ask Question
@app.post("/ask")
async def ask_question(data: dict):
    try:
        query = data.get("question")

        if not query:
            return {"error": "Question is required"}

        result = generate_answer(query)

        return result

    except Exception as e:
        print("ASK ERROR:", e)
        return {"error": str(e)}

# =========================================
# 🔹 Home route
@app.get("/")
def home():
    return {"message": "OpsMind AI is running"}