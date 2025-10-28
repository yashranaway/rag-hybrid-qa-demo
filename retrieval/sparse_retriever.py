"""
Sparse retriever using BM25 algorithm.
"""

import os
import pickle
from typing import List, Dict, Tuple
from rank_bm25 import BM25Okapi
import numpy as np
from tqdm import tqdm
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def simple_tokenize(text: str) -> List[str]:
    """Simple whitespace tokenizer."""
    return text.lower().split()


class SparseRetriever:
    """Sparse retrieval using BM25."""
    
    def __init__(self):
        """Initialize BM25 retriever."""
        self.bm25 = None
        self.passages = None
        self.tokenized_corpus = None
        
    def build_index(self, passages: List[Dict]):
        """
        Build BM25 index from passages.
        
        Args:
            passages: List of passage dictionaries with 'text' field
        """
        self.passages = passages
        
        print(f"Tokenizing {len(passages)} passages for BM25...")
        self.tokenized_corpus = [
            simple_tokenize(p['text']) 
            for p in tqdm(passages, desc="Tokenizing")
        ]
        
        print("Building BM25 index...")
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        print("BM25 index built")
        
    def retrieve(self, queries: List[str], top_k=5) -> List[List[Tuple[Dict, float]]]:
        """
        Retrieve top-k passages for each query using BM25.
        
        Args:
            queries: List of query strings
            top_k: Number of passages to retrieve per query
            
        Returns:
            List of lists, where each inner list contains (passage, score) tuples
        """
        if self.bm25 is None:
            raise ValueError("Index not built. Call build_index first.")
        
        results = []
        for query in queries:
            tokenized_query = simple_tokenize(query)
            scores = self.bm25.get_scores(tokenized_query)
            
            # Get top-k indices
            top_k_indices = np.argsort(scores)[::-1][:top_k]
            
            query_results = []
            for idx in top_k_indices:
                passage = self.passages[idx]
                score = float(scores[idx])
                query_results.append((passage, score))
            
            results.append(query_results)
        
        return results
    
    def save_index(self, path_prefix):
        """Save BM25 index to disk."""
        os.makedirs(os.path.dirname(path_prefix), exist_ok=True)
        
        data = {
            'bm25': self.bm25,
            'passages': self.passages,
            'tokenized_corpus': self.tokenized_corpus
        }
        
        with open(f"{path_prefix}_bm25.pkl", 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Saved BM25 retriever to {path_prefix}")
    
    def load_index(self, path_prefix):
        """Load BM25 index from disk."""
        with open(f"{path_prefix}_bm25.pkl", 'rb') as f:
            data = pickle.load(f)
        
        self.bm25 = data['bm25']
        self.passages = data['passages']
        self.tokenized_corpus = data['tokenized_corpus']
        
        print(f"Loaded BM25 retriever with {len(self.passages)} passages")


if __name__ == "__main__":
    # Test the sparse retriever
    from data.prepare_data import SQuADDataProcessor
    
    processor = SQuADDataProcessor()
    kb_path = os.path.join(config.DATA_DIR, 'knowledge_base.pkl')
    
    if os.path.exists(kb_path):
        passages = processor.load_knowledge_base(kb_path)
    else:
        print("Knowledge base not found. Run data/prepare_data.py first.")
        exit(1)
    
    retriever = SparseRetriever()
    retriever.build_index(passages)
    
    # Save index
    index_path = os.path.join(config.RETRIEVAL_DIR, 'sparse_index')
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

