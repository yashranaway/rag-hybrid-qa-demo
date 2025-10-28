"""
Base model wrapper for DistilGPT-2.
"""

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    GenerationConfig
)
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class BaseQAModel:
    """Wrapper for DistilGPT-2 for question answering."""
    
    def __init__(self, model_name=None, device=None):
        """
        Initialize the model.
        
        Args:
            model_name: Name of the model to load
            device: Device to load model on
        """
        self.model_name = model_name or config.MODEL_NAME
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """Load the model and tokenizer."""
        print(f"Loading model: {self.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        
        # Set padding token (GPT-2 doesn't have one by default)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id
        
        self.model.to(self.device)
        print(f"Model loaded on {self.device}")
        
        return self.model, self.tokenizer
    
    def format_input(self, question: str, answer: str = None, context: str = None):
        """
        Format input for the model.
        
        Args:
            question: The question
            answer: The answer (for training)
            context: Optional context (not used in baseline)
            
        Returns:
            Formatted string
        """
        if answer is not None:
            # Training format
            return f"Question: {question}\nAnswer: {answer}{self.tokenizer.eos_token}"
        else:
            # Inference format
            return f"Question: {question}\nAnswer:"
    
    def prepare_batch(self, questions, answers=None, contexts=None):
        """
        Prepare a batch of inputs for training or inference.
        
        Args:
            questions: List of questions
            answers: List of answers (for training)
            contexts: List of contexts (not used in baseline)
            
        Returns:
            Dictionary with input_ids, attention_mask, labels
        """
        # Format inputs
        if answers is not None:
            texts = [self.format_input(q, a, c) for q, a, c in 
                    zip(questions, answers, contexts or [None]*len(questions))]
        else:
            texts = [self.format_input(q) for q in questions]
        
        # Tokenize
        encodings = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=config.MAX_LENGTH,
            return_tensors='pt'
        )
        
        batch = {
            'input_ids': encodings['input_ids'].to(self.device),
            'attention_mask': encodings['attention_mask'].to(self.device),
        }
        
        # For training, labels are the same as input_ids
        if answers is not None:
            batch['labels'] = batch['input_ids'].clone()
            # Mask padding tokens in labels
            batch['labels'][batch['labels'] == self.tokenizer.pad_token_id] = -100
        
        return batch
    
    def generate_answer(self, question: str, context: str = None, 
                       max_length=None, num_beams=None):
        """
        Generate an answer for a question.
        
        Args:
            question: The question
            context: Optional context (not used in baseline)
            max_length: Max length for generation
            num_beams: Number of beams for beam search
            
        Returns:
            Generated answer string
        """
        if self.model is None:
            self.load_model()
        
        self.model.eval()
        
        # Format input
        input_text = self.format_input(question, context=context)
        
        # Tokenize
        inputs = self.tokenizer(
            input_text,
            return_tensors='pt',
            truncation=True,
            max_length=config.MAX_LENGTH
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length or config.GENERATION_MAX_LENGTH,
                num_beams=num_beams or config.GENERATION_NUM_BEAMS,
                temperature=config.GENERATION_TEMPERATURE,
                early_stopping=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode
        full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract answer (everything after "Answer:")
        if "Answer:" in full_output:
            answer = full_output.split("Answer:")[-1].strip()
        else:
            answer = full_output
        
        return answer
    
    def save_model(self, path):
        """Save model and tokenizer."""
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"Model saved to {path}")
    
    def load_from_checkpoint(self, path):
        """Load model and tokenizer from checkpoint."""
        print(f"Loading model from {path}")
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForCausalLM.from_pretrained(path)
        self.model.to(self.device)
        print(f"Model loaded from checkpoint")


if __name__ == "__main__":
    # Test the model
    model = BaseQAModel()
    model.load_model()
    
    # Test generation
    question = "What is the capital of France?"
    answer = model.generate_answer(question)
    print(f"Q: {question}")
    print(f"A: {answer}")

