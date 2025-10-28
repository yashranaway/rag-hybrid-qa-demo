"""
FastAPI backend to serve BEFORE/AFTER answers for the demo UI.
"""

import os
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline
import re

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from model.base_model import BaseQAModel
from model.rag_model import RAGModel
from retrieval.hybrid_retriever import HybridRetriever


class QARequest(BaseModel):
    question: str
    top_k: int | None = 3


app = FastAPI(title="RAG Demo API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


baseline_model: BaseQAModel | None = None
rag_model: RAGModel | None = None
qa_reader = None
_repeat_phrase = re.compile(r"(\b[\w\s]{1,24}\b)(,\s*\1){2,}", re.IGNORECASE)

def _clean_text(text: str, limit: int = 160) -> str:
    if not text:
        return ""
    t = text.strip()
    # collapse long simple repetitions like "Paris, Paris, Paris, ..."
    t = _repeat_phrase.sub(r"\1", t)
    # squeeze repeated commas/spaces
    t = re.sub(r",\s*,+", ", ", t)
    t = re.sub(r"\s+", " ", t)
    t = t.strip().strip(",")
    if len(t) > limit:
        t = t[:limit].rsplit(" ", 1)[0] + "…"
    return t


@app.on_event("startup")
def _load_models():
    global baseline_model, rag_model, qa_reader
    retriever = HybridRetriever(use_reranker=True)
    retriever.load_index("./retrieval/hybrid_index")

    baseline_model = BaseQAModel(device="cpu")
    baseline_model.load_from_checkpoint("./results/baseline_model")

    rag_model = RAGModel(retriever=retriever, device="cpu")
    rag_model.load_from_checkpoint("./results/baseline_model")

    # Lightweight extractive reader for accurate answers from retrieved passages
    try:
        qa_reader = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")
    except Exception:
        qa_reader = None


@app.post("/qa")
def qa(req: QARequest):
    if baseline_model is None or rag_model is None:
        raise HTTPException(status_code=503, detail="Models are still loading. Try again in a few seconds.")
    before_answer = baseline_model.generate_answer(req.question, max_length=40)
    after_answer, passages = rag_model.generate_answer(
        req.question, use_retrieval=True, max_length=40
    )

    # If we have an extractive reader and retrieved passages, use it for 'after'
    if qa_reader and passages:
        context = "\n".join(p[0]["text"] for p in passages[: req.top_k or 3])
        try:
            qa_res = qa_reader(question=req.question, context=context)
            after_answer = qa_res.get("answer", after_answer)
        except Exception:
            pass

    before_answer = _clean_text(before_answer)
    after_answer = _clean_text(after_answer)
    sources = []
    if passages:
        for p, s in passages[: req.top_k or 3]:
            sources.append({"text": p["text"], "score": float(s)})
    return {
        "question": req.question,
        "before": before_answer,
        "after": after_answer,
        "sources": sources,
    }


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run("backend_api:app", host="127.0.0.1", port=8000, reload=False)


