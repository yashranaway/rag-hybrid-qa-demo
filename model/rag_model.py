"""
Retrieval-Augmented Generation model.
Extends base model with retrieval capabilities.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from model.base_model import BaseQAModel


class RAGModel(BaseQAModel):
    """RAG model that uses retrieved contexts."""
    
    def __init__(self, retriever=None, model_name=None, device=None, top_k=None):
        """
        Initialize RAG model.
        
        Args:
            retriever: Retriever instance (HybridRetriever, DenseRetriever, etc.)
            model_name: Name of the base model
            device: Device to load model on
            top_k: Number of passages to retrieve
        """
        super().__init__(model_name, device)
        self.retriever = retriever
        self.top_k = top_k or config.TOP_K_RETRIEVAL
    
    def format_input(self, question: str, answer: str = None, context: str = None):
        """
        Format input with retrieved context.
        
        Args:
            question: The question
            answer: The answer (for training)
            context: Retrieved context passages (string or will be retrieved)
            
        Returns:
            Formatted string with context
        """
        # Context should be pre-retrieved and passed as a string
        if context:
            if answer is not None:
                # Training format
                return f"Context: {context}\nQuestion: {question}\nAnswer: {answer}{self.tokenizer.eos_token}"
            else:
                # Inference format
                return f"Context: {context}\nQuestion: {question}\nAnswer:"
        else:
            # Fall back to no context
            return super().format_input(question, answer, None)
    
    def retrieve_context(self, questions):
        """
        Retrieve context for questions.
        
        Args:
            questions: List of questions or single question
            
        Returns:
            List of context strings (concatenated passages)
        """
        if self.retriever is None:
            return [None] * (len(questions) if isinstance(questions, list) else 1)
        
        # Ensure questions is a list
        single_question = not isinstance(questions, list)
        if single_question:
            questions = [questions]
        
        # Retrieve passages
        results = self.retriever.retrieve(questions, top_k=self.top_k)
        
        # Format contexts
        contexts = []
        for query_results in results:
            if query_results:
                # Concatenate passages
                passages = [passage['text'] for passage, score in query_results]
                context = " ".join(passages)
                # Truncate if too long
                context = context[:config.MAX_LENGTH * 2]  # Rough character limit
            else:
                context = None
            contexts.append(context)
        
        if single_question:
            return contexts[0]
        return contexts
    
    def generate_answer(self, question: str, context: str = None,
                       max_length=None, num_beams=None, 
                       use_retrieval=True):
        """
        Generate an answer with optional retrieval.
        
        Args:
            question: The question
            context: Pre-retrieved context (if None and use_retrieval=True, will retrieve)
            max_length: Max length for generation
            num_beams: Number of beams
            use_retrieval: Whether to use retrieval
            
        Returns:
            Generated answer string (and optionally retrieved passages)
        """
        if self.model is None:
            self.load_model()
        
        # Retrieve context if needed
        retrieved_passages = None
        if use_retrieval and context is None:
            if self.retriever is None:
                print("Warning: Retrieval requested but no retriever available")
            else:
                # Get detailed results for analysis
                results = self.retriever.retrieve([question], top_k=self.top_k)
                if results and results[0]:
                    retrieved_passages = results[0]  # List of (passage, score)
                    passages = [passage['text'] for passage, score in retrieved_passages]
                    context = " ".join(passages)[:config.MAX_LENGTH * 2]
        
        # Generate answer
        answer = super().generate_answer(question, context, max_length, num_beams)
        
        return answer, retrieved_passages


if __name__ == "__main__":
    # Test the RAG model
    from retrieval.hybrid_retriever import HybridRetriever
    
    # Load retriever
    retriever = HybridRetriever()
    index_path = os.path.join(config.RETRIEVAL_DIR, 'hybrid_index')
    
    if os.path.exists(f"{index_path}_dense_faiss.index"):
        retriever.load_index(index_path)
        
        # Test model
        model = RAGModel(retriever=retriever)
        model.load_model()
        
        question = "What is the capital of France?"
        answer, passages = model.generate_answer(question, use_retrieval=True)
        
        print(f"Q: {question}")
        if passages:
            print("\nRetrieved passages:")
            for i, (passage, score) in enumerate(passages, 1):
                print(f"{i}. (score={score:.4f}) {passage['text'][:100]}...")
        print(f"\nA: {answer}")
    else:
        print("Retrieval index not found. Build indices first.")

