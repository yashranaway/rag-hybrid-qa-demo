<div align="center">
  <h1>Retrieval‑Augmented Transformer Demo</h1>
  <p>Baseline vs RAG (Hybrid Retrieval + Extractive Reader)</p>
  <img src="results/before_after.png" alt="Before vs After" width="720" />
</div>

<div style="margin:16px 0;padding:12px 14px;border:1px solid #f5c6cb;border-left:4px solid #e74c3c;background:#fff5f5;">
  <strong>Note:</strong> Accuracy is intentionally modest due to local CPU/low-spec constraints; the full RAG pipeline (hybrid retrieval + fusion + context injection + extractive reader) is fully implemented and ready to scale.
</div>

---

This project compares a baseline DistilGPT‑2 QA model (no retrieval) with a RAG model that retrieves passages via a hybrid retriever (FAISS dense + BM25 sparse, fused by RRF). The backend also uses an extractive QA reader over retrieved passages for factual answers.

## What was built
- Baseline DistilGPT‑2 fine‑tuned on SQuAD subset
- Hybrid retriever (Sentence‑BERT + FAISS, BM25)
- RAG model that conditions on retrieved text
- FastAPI backend exposing `/qa`
- Minimal React frontend showing Before vs After and sources
- Evaluation plot (image above)

## Start services
Terminal 1 — backend
```bash
cd /Users/adityagarud/ai
source .venv/bin/activate
uvicorn backend_api:app --host 127.0.0.1 --port 8000
```

Terminal 2 — frontend
```bash
cd /Users/adityagarud/ai/frontend
bun dev
```

Open http://localhost:5173 and ask a question. The right panel shows the RAG answer and retrieved sources.

## Notes
- If CORS errors appear, ensure backend runs on 127.0.0.1:8000.
- The reader uses `distilbert-base-uncased-distilled-squad` for accurate spans from sources.

## Accuracy note
- This demo uses a small model and a subset of SQuAD for fast, local runs on CPU. As a result, absolute accuracy (F1/EM) is modest.
- The full RAG implementation—hybrid retrieval, fusion, context injection, and extractive reading—is complete and ready to scale with larger models/datasets and GPU for higher accuracy.
