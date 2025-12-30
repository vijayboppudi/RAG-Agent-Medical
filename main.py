import os
from typing import TypedDict, List, Dict, Any
from fastapi.responses import HTMLResponse
from pathlib import Path
import time


from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from pydantic import BaseModel

from openai import OpenAI
from qdrant_client import QdrantClient
from langgraph.graph import StateGraph, END

# ----------------------------
# Config
# ----------------------------
COLLECTION = "toy_agent_docs"
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

# ----------------------------
# Clients
# ----------------------------
oai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
qdrant = QdrantClient(path="qdrant_local")  # ✅ no docker

# ----------------------------
# Helpers
# ----------------------------
def embed_one(text: str):
    return oai.embeddings.create(model=EMBED_MODEL, input=[text]).data[0].embedding

def qdrant_similarity_search(client, collection_name, query_vector, limit=5):
    # Older API
    if hasattr(client, "search"):
        return client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit,
            with_payload=True,
        )
    # Newer API
    if hasattr(client, "query_points"):
        res = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=limit,
            with_payload=True,
        )
        return res.points

    raise AttributeError("No compatible Qdrant search method found on this client.")

def retrieve(question: str, top_k: int = 5):
    qv = embed_one(question)
    hits = qdrant_similarity_search(
        qdrant,
        collection_name=COLLECTION,
        query_vector=qv,
        limit=top_k,
    )

    results = []
    for h in hits:
        payload = getattr(h, "payload", {}) or {}
        score = float(getattr(h, "score", 0.0))

        results.append({
            "score": score,
            "chunk_id": payload.get("chunk_id"),
            "doc_id": payload.get("doc_id"),
            "title": payload.get("title"),
            "text": payload.get("text"),
            "metadata": payload.get("metadata", {}),
        })
    return results

# ----------------------------
# LangGraph
# ----------------------------
class AgentState(TypedDict):
    question: str
    retrieved: List[Dict[str, Any]]
    answer: str
    sources: List[Dict[str, Any]]

def retrieve_node(state: AgentState) -> AgentState:
    docs = retrieve(state["question"], top_k=5)
    return {**state, "retrieved": docs}

def generate_node(state: AgentState) -> AgentState:
    context_blocks = []
    sources = []

    for i, d in enumerate(state["retrieved"], start=1):
        meta = d["metadata"] or {}
        sources.append({
            "id": i,
            "title": d["title"],
            "file_name": meta.get("file_name"),
            "file_type": meta.get("file_type"),
            "source_path": meta.get("source_path"),
            "chunk_id": d["chunk_id"],
            "score": d["score"],
            "snippet": (d["text"][:260] if d["text"] else ""),
        })
        context_blocks.append(f"[{i}] {d['title']} ({meta.get('file_name')})\n{d['text']}")

    prompt = f"""You are a retrieval-grounded medical information assistant.

Safety:
- You are NOT a doctor.
- Provide general educational information only.
- If the user asks for diagnosis/treatment decisions, advise consulting a clinician.
- If symptoms sound urgent (e.g., chest pain, trouble breathing, stroke signs), advise emergency care.

Grounding rules:
- Use ONLY the sources below.
- If the answer is not in the sources, say: "I don't know based on the provided documents."
- Keep the answer concise and structured.
- Add citations like [1], [2] that refer to the source numbers.

Question: {state['question']}

Sources:
{chr(10).join(context_blocks)}
"""


    resp = oai.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    answer = resp.choices[0].message.content
    return {**state, "answer": answer, "sources": sources}

# ✅ Build graph at import time so it exists for API calls
_builder = StateGraph(AgentState)
_builder.add_node("retrieve", retrieve_node)
_builder.add_node("generate", generate_node)
_builder.set_entry_point("retrieve")
_builder.add_edge("retrieve", "generate")
_builder.add_edge("generate", END)
rag_graph = _builder.compile()

# ----------------------------
# FastAPI
# ----------------------------
app = FastAPI(title="RAG Agent Health", version="1.0.0")

class ChatRequest(BaseModel):
    message: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/", response_class=HTMLResponse)
def home():
    html_path = Path("index.html")
    return html_path.read_text(encoding="utf-8")

@app.post("/chat")
def chat(req: ChatRequest):
    t0 = time.time()
    result = rag_graph.invoke({
        "question": req.message,
        "retrieved": [],
        "answer": "",
        "sources": [],
    })
    latency_ms = int((time.time() - t0) * 1000)

    warning = "This is educational information, not medical advice."

    return {
        "answer": result["answer"],
        "sources": result["sources"],
        "meta": {
            "latency_ms": latency_ms,
            "top_k": 5,
            "warning": warning
        }
    }
