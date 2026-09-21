import os
import json
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader
import chromadb
from rank_bm25 import BM25Okapi
import cohere
from google import genai
from google.genai import types

# Load environment variables
load_dotenv()

app = FastAPI(title="Advanced Hybrid RAG Engine")

# CORS setup for React frontend
app.add_middleware(
    CORSMiddleware,
    origins=[
        "http://localhost:5173",
        "https://rag-frontend-rust.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Storage & AI Models
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="rag_documents")

# Initialize Cohere for lightweight Re-Ranking (0 MB local RAM usage)
cohere_api_key = os.getenv("COHERE_API_KEY")
co_client = cohere.Client(cohere_api_key) if cohere_api_key else None

# Gemini client initialization
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise RuntimeError("GEMINI_API_KEY is not set in environment variables.")

gemini_client = genai.Client(api_key=gemini_api_key)

# Global in-memory cache for BM25 sparse keyword index
documents_text = []
bm25_index = None


@app.get("/")
def health_check():
    return {"status": "online", "system": "Hybrid RAG Engine"}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    global bm25_index, documents_text

    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    reader = PdfReader(file.file)
    raw_chunks = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            # Overlapping chunking (500 chars window, 50 chars overlap)
            for j in range(0, len(text), 450):
                chunk = text[j:j + 500]
                if len(chunk.strip()) > 20:
                    raw_chunks.append({"text": chunk, "page": i + 1})

    if not raw_chunks:
        raise HTTPException(status_code=400, detail="Could not extract text from PDF.")

    documents_text = [c["text"] for c in raw_chunks]
    ids = [f"{file.filename}_chunk_{idx}" for idx in range(len(raw_chunks))]
    metadatas = [{"page": c["page"], "source": file.filename} for c in raw_chunks]

    # Store in Dense Vector DB
    collection.add(documents=documents_text, metadatas=metadatas, ids=ids)

    # Build Sparse BM25 Index
    tokenized_corpus = [doc.lower().split(" ") for doc in documents_text]
    bm25_index = BM25Okapi(tokenized_corpus)

    return {"status": "success", "filename": file.filename, "chunks_indexed": len(raw_chunks)}


@app.post("/chat")
async def chat(query: str):
    global bm25_index, documents_text

    # Reload from ChromaDB if in-memory store is lost (e.g. after server reload)
    if not documents_text:
        existing = collection.get()
        if existing and existing.get("documents"):
            documents_text = existing["documents"]
            tokenized_corpus = [doc.lower().split(" ") for doc in documents_text]
            bm25_index = BM25Okapi(tokenized_corpus)

    if not documents_text or bm25_index is None:
        raise HTTPException(status_code=400, detail="Please upload a PDF document first.")

    # 1. Dense Vector Search (Top 10)
    vector_results = collection.query(query_texts=[query], n_results=min(10, len(documents_text)))
    dense_chunks = vector_results["documents"][0] if vector_results["documents"] else []

    # 2. Sparse BM25 Search (Top 10)
    tokenized_query = query.lower().split(" ")
    sparse_chunks = bm25_index.get_top_n(tokenized_query, documents_text, n=min(10, len(documents_text)))

    # 3. Merge candidate pool (Deduplicate)
    candidate_pool = list(dict.fromkeys(dense_chunks + sparse_chunks))

    # 4. Cohere API Re-Ranking (Fast cloud API call, replaces PyTorch model)
    if co_client and len(candidate_pool) > 0:
        rerank_response = co_client.rerank(
            model="rerank-v3.5",
            query=query,
            documents=candidate_pool,
            top_n=min(3, len(candidate_pool))
        )
        # Map reranked indices back to text chunks
        top_3_chunks = [candidate_pool[result.index] for result in rerank_response.results]
    else:
        # Fallback if COHERE_API_KEY is not configured
        top_3_chunks = candidate_pool[:3]

    # 5. Build Grounded Prompt
    context_str = "\n\n---\n\n".join(top_3_chunks)
    prompt = (
        "You are a strict technical assistant. Answer the user's question using ONLY the provided context.\n"
        "Do NOT use markdown bolding (asterisks) or emojis in your response.\n\n"
        f"Context:\n{context_str}\n\n"
        f"Question: {query}"
    )

    # Synchronous generator returning formatted SSE frames safely
    def generate():
        try:
            response = gemini_client.models.generate_content_stream(
                model="gemini-3.6-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(
                        disable=True
                    )
                )
            )
            for chunk in response:
                if chunk.text:
                    payload = json.dumps({"text": chunk.text})
                    yield f"data: {payload}\n\n"
        except Exception as e:
            err_payload = json.dumps({"error": str(e)})
            yield f"data: {err_payload}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Content-Type-Options": "nosniff"
        }
    )
