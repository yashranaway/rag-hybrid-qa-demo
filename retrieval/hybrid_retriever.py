"""
Hybrid retriever combining dense (FAISS) and sparse (BM25) retrieval.
"""

import os
from typing import List, Dict, Tuple
import numpy as np
from collections import defaultdict
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from retrieval.dense_retriever import DenseRetriever
from retrieval.sparse_retriever import SparseRetriever
from retrieval.reranker import CrossEncoderReranker


class HybridRetriever:
    """Hybrid retriever combining dense and sparse retrieval methods."""
    
    def __init__(self, dense_weight=None, bm25_weight=None, use_rrfusion=None, use_reranker=False):
        """
        Initialize hybrid retriever.
        
        Args:
            dense_weight: Weight for dense retrieval scores
            bm25_weight: Weight for BM25 scores
            use_rrfusion: If True, use Reciprocal Rank Fusion instead of score fusion
        """
        self.dense_retriever = DenseRetriever()
        self.sparse_retriever = SparseRetriever()
        self.reranker = CrossEncoderReranker() if use_reranker else None
        
        self.dense_weight = dense_weight if dense_weight is not None else config.DENSE_WEIGHT
        self.bm25_weight = bm25_weight if bm25_weight is not None else config.BM25_WEIGHT
        self.use_rrfusion = use_rrfusion if use_rrfusion is not None else config.USE_RRF
        
    def build_index(self, passages: List[Dict], batch_size=64):
        """
        Build both dense and sparse indices.
        
        Args:
            passages: List of passage dictionaries with 'text' field
            batch_size: Batch size for dense encoding
        """
        print("Building dense index...")
        self.dense_retriever.build_index(passages, batch_size=batch_size)
        
        print("\nBuilding sparse (BM25) index...")
        self.sparse_retriever.build_index(passages)
        
        print("\nHybrid retriever ready!")
    
    def _reciprocal_rank_fusion(self, dense_results, sparse_results, k=60):
        """
        Combine results using Reciprocal Rank Fusion.
        
        Args:
            dense_results: List of (passage, score) from dense retriever
            sparse_results: List of (passage, score) from sparse retriever
            k: Constant for RRF (default 60)
            
        Returns:
            Combined list of (passage, rrf_score) sorted by score
        """
        rrf_scores = defaultdict(float)
        passage_map = {}
        
        # Add dense retrieval scores
        for rank, (passage, _) in enumerate(dense_results):
            passage_id = passage['id']
            rrf_scores[passage_id] += 1.0 / (k + rank + 1)
            passage_map[passage_id] = passage
        
        # Add sparse retrieval scores
        for rank, (passage, _) in enumerate(sparse_results):
            passage_id = passage['id']
            rrf_scores[passage_id] += 1.0 / (k + rank + 1)
            passage_map[passage_id] = passage
        
        # Sort by RRF score
        sorted_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        return [(passage_map[pid], score) for pid, score in sorted_ids]
    
    def _score_fusion(self, dense_results, sparse_results):
        """
        Combine results using weighted score fusion.
        
        Args:
            dense_results: List of (passage, score) from dense retriever
            sparse_results: List of (passage, score) from sparse retriever
            
        Returns:
            Combined list of (passage, fused_score) sorted by score
        """
        # Normalize scores to [0, 1] range
        def normalize_scores(results):
            if not results:
                return []
            scores = [score for _, score in results]
            min_score, max_score = min(scores), max(scores)
            if max_score == min_score:
                return [(p, 1.0) for p, _ in results]
            return [(p, (s - min_score) / (max_score - min_score)) for p, s in results]
        
        dense_norm = normalize_scores(dense_results)
        sparse_norm = normalize_scores(sparse_results)
        
        # Combine scores
        fused_scores = defaultdict(float)
        passage_map = {}
        
        for passage, score in dense_norm:
            passage_id = passage['id']
            fused_scores[passage_id] += self.dense_weight * score
            passage_map[passage_id] = passage
        
        for passage, score in sparse_norm:
            passage_id = passage['id']
            fused_scores[passage_id] += self.bm25_weight * score
            passage_map[passage_id] = passage
        
        # Sort by fused score
        sorted_ids = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        
        return [(passage_map[pid], score) for pid, score in sorted_ids]
    
    def retrieve(self, queries: List[str], top_k=5) -> List[List[Tuple[Dict, float]]]:
        """
        Retrieve top-k passages for each query using hybrid retrieval.
        
        Args:
            queries: List of query strings
            top_k: Number of passages to retrieve per query
            
        Returns:
            List of lists, where each inner list contains (passage, score) tuples
        """
        # Retrieve from both systems (retrieve more to have better fusion)
        retrieve_k = top_k * 2
        
        dense_results = self.dense_retriever.retrieve(queries, top_k=retrieve_k)
        sparse_results = self.sparse_retriever.retrieve(queries, top_k=retrieve_k)
        
        # Combine results for each query
        combined_results = []
        for dense_res, sparse_res in zip(dense_results, sparse_results):
            if self.use_rrfusion:
                combined = self._reciprocal_rank_fusion(dense_res, sparse_res)
            else:
                combined = self._score_fusion(dense_res, sparse_res)

            # Optional cross-encoder reranking
            if self.reranker is not None:
                combined = self.reranker.rerank(queries[combined_results.__len__()], combined, top_k)

            # Return top-k
            combined_results.append(combined[:top_k])
        
        return combined_results
    
    def save_index(self, path_prefix):
        """Save both indices."""
        self.dense_retriever.save_index(f"{path_prefix}_dense")
        self.sparse_retriever.save_index(f"{path_prefix}_sparse")
        print(f"Saved hybrid retriever to {path_prefix}")
    
    def load_index(self, path_prefix):
        """Load both indices."""
        self.dense_retriever.load_index(f"{path_prefix}_dense")
        self.sparse_retriever.load_index(f"{path_prefix}_sparse")
        print(f"Loaded hybrid retriever")


if __name__ == "__main__":
    # Test the hybrid retriever
    from data.prepare_data import SQuADDataProcessor
    
    processor = SQuADDataProcessor()
    kb_path = os.path.join(config.DATA_DIR, 'knowledge_base.pkl')
    
    if os.path.exists(kb_path):
        passages = processor.load_knowledge_base(kb_path)
    else:
        print("Knowledge base not found. Run data/prepare_data.py first.")
        exit(1)
    
    retriever = HybridRetriever()
    retriever.build_index(passages)
    
    # Save index
    index_path = os.path.join(config.RETRIEVAL_DIR, 'hybrid_index')
    retriever.save_index(index_path)
    
    # Test retrieval
    test_queries = [
        "What is the capital of France?",
        "Who invented the telephone?"
    ]
    results = retriever.retrieve(test_queries, top_k=3)
    
    for query, query_results in zip(test_queries, results):
        print(f"\nQuery: {query}")
        for rank, (passage, score) in enumerate(query_results, 1):
            print(f"  {rank}. (score={score:.4f}) {passage['text'][:100]}...")

