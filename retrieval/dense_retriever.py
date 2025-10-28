"""
Dense retriever using sentence transformers and FAISS.
"""

import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from typing import List, Dict, Tuple
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class DenseRetriever:
    """Dense retrieval using sentence transformers and FAISS."""
    
    def __init__(self, embedding_model_name=None):
        """
        Initialize dense retriever.
        
        Args:
            embedding_model_name: Name of sentence-transformer model to use
        """
        self.embedding_model_name = embedding_model_name or config.DENSE_EMBEDDING_MODEL
        self.encoder = None
        self.index = None
        self.passages = None
        self.embedding_dim = None
        
    def load_encoder(self):
        """Load the sentence transformer model."""
        print(f"Loading embedding model: {self.embedding_model_name}")
        self.encoder = SentenceTransformer(self.embedding_model_name)
        self.embedding_dim = self.encoder.get_sentence_embedding_dimension()
        print(f"Embedding dimension: {self.embedding_dim}")
        
    def build_index(self, passages: List[Dict], batch_size=64):
        """
        Build FAISS index from passages.
        
        Args:
            passages: List of passage dictionaries with 'text' field
            batch_size: Batch size for encoding
        """
        if self.encoder is None:
            self.load_encoder()
        
        self.passages = passages
        passage_texts = [p['text'] for p in passages]
        
        print(f"Encoding {len(passage_texts)} passages...")
        embeddings = self.encoder.encode(
            passage_texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Build FAISS index
        print("Building FAISS index...")
        if config.FAISS_INDEX_TYPE == "Flat":
            self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for cosine sim
        else:
            # For larger datasets, use IVF
            nlist = min(100, len(passages) // 10)
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist)
            self.index.train(embeddings)
        
        self.index.add(embeddings)
        print(f"Index built with {self.index.ntotal} vectors")
        
    def retrieve(self, queries: List[str], top_k=5) -> List[List[Tuple[Dict, float]]]:
        """
        Retrieve top-k passages for each query.
        
        Args:
            queries: List of query strings
            top_k: Number of passages to retrieve per query
            
        Returns:
            List of lists, where each inner list contains (passage, score) tuples
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index first.")
        
        # Encode queries
        query_embeddings = self.encoder.encode(
            queries,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        
        # Normalize for cosine similarity
        faiss.normalize_L2(query_embeddings)
        
        # Search
        scores, indices = self.index.search(query_embeddings, top_k)
        
        # Format results
        results = []
        for query_idx in range(len(queries)):
            query_results = []
            for rank, (idx, score) in enumerate(zip(indices[query_idx], scores[query_idx])):
                if idx != -1:  # Valid result
                    passage = self.passages[idx]
                    query_results.append((passage, float(score)))
            results.append(query_results)
        
        return results
    
    def save_index(self, path_prefix):
        """Save index and passages to disk."""
        os.makedirs(os.path.dirname(path_prefix), exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, f"{path_prefix}_faiss.index")
        
        # Save passages
        with open(f"{path_prefix}_passages.pkl", 'wb') as f:
            pickle.dump(self.passages, f)
        
        print(f"Saved dense retriever to {path_prefix}")
    
    def load_index(self, path_prefix):
        """Load index and passages from disk."""
        if self.encoder is None:
            self.load_encoder()
        
        # Load FAISS index
        self.index = faiss.read_index(f"{path_prefix}_faiss.index")
        
        # Load passages
        with open(f"{path_prefix}_passages.pkl", 'rb') as f:
            self.passages = pickle.load(f)
        
        print(f"Loaded dense retriever with {self.index.ntotal} vectors")


if __name__ == "__main__":
    # Test the dense retriever
    from data.prepare_data import SQuADDataProcessor
    
    processor = SQuADDataProcessor()
    kb_path = os.path.join(config.DATA_DIR, 'knowledge_base.pkl')
    
    if os.path.exists(kb_path):
        passages = processor.load_knowledge_base(kb_path)
    else:
        print("Knowledge base not found. Run data/prepare_data.py first.")
        exit(1)
    
    retriever = DenseRetriever()
    retriever.load_encoder()
    retriever.build_index(passages)
    
    # Save index
    index_path = os.path.join(config.RETRIEVAL_DIR, 'dense_index')
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

