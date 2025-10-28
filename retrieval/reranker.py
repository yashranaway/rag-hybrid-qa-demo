"""
Cross-encoder reranker for re-scoring retrieved passages.
"""

from typing import List, Dict, Tuple
from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    """Re-rank (query, passage) pairs with a cross-encoder.

    Uses a small MS MARCO model for speed.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: List[Tuple[Dict, float]], top_k: int) -> List[Tuple[Dict, float]]:
        if not candidates:
            return []
        pairs = [(query, c[0]["text"]) for c in candidates]
        scores = self.model.predict(pairs).tolist()
        rescored = [ (candidates[i][0], float(scores[i])) for i in range(len(candidates)) ]
        rescored.sort(key=lambda x: x[1], reverse=True)
        return rescored[:top_k]


