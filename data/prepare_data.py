"""
Data preparation module for SQuAD dataset.
Loads and preprocesses data for both baseline and RAG models.
"""

import json
import os
import pickle
from typing import Dict, List, Tuple

# Fix for Python 3.13 multiprocessing issues
import multiprocessing
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass

from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class SQuADDataProcessor:
    """Processes SQuAD dataset for training and retrieval."""
    
    def __init__(self, train_sample_size=None, val_sample_size=None):
        """
        Initialize the data processor.
        
        Args:
            train_sample_size: If set, limit training data to this size
            val_sample_size: If set, limit validation data to this size
        """
        self.train_sample_size = train_sample_size
        self.val_sample_size = val_sample_size
        self.dataset = None
        self.knowledge_base = None
        
    def load_squad(self):
        """Load SQuAD dataset from HuggingFace."""
        print("Loading SQuAD dataset...")
        self.dataset = load_dataset("squad")
        
        # Sample if needed
        if self.train_sample_size:
            self.dataset['train'] = self.dataset['train'].select(range(self.train_sample_size))
        if self.val_sample_size:
            self.dataset['validation'] = self.dataset['validation'].select(range(self.val_sample_size))
            
        print(f"Loaded {len(self.dataset['train'])} training examples")
        print(f"Loaded {len(self.dataset['validation'])} validation examples")
        
        return self.dataset
    
    def build_knowledge_base(self):
        """
        Build knowledge base from SQuAD contexts for retrieval.
        Each context becomes a retrievable passage.
        """
        print("Building knowledge base from SQuAD contexts...")
        
        if self.dataset is None:
            self.load_squad()
        
        # Extract unique contexts with metadata
        context_dict = {}
        
        for split in ['train', 'validation']:
            for example in tqdm(self.dataset[split], desc=f"Processing {split}"):
                context = example['context']
                if context not in context_dict:
                    context_dict[context] = {
                        'text': context,
                        'id': len(context_dict),
                        'title': example.get('title', 'Unknown'),
                    }
        
        # Convert to list for indexing
        self.knowledge_base = list(context_dict.values())
        print(f"Built knowledge base with {len(self.knowledge_base)} unique passages")
        
        return self.knowledge_base
    
    def save_knowledge_base(self, path):
        """Save knowledge base to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.knowledge_base, f)
        print(f"Saved knowledge base to {path}")
    
    def load_knowledge_base(self, path):
        """Load knowledge base from disk."""
        with open(path, 'rb') as f:
            self.knowledge_base = pickle.load(f)
        print(f"Loaded knowledge base with {len(self.knowledge_base)} passages")
        return self.knowledge_base
    
    def prepare_baseline_data(self):
        """
        Prepare data for baseline model (no retrieval).
        Format: Question: {question} Answer: {answer}
        """
        if self.dataset is None:
            self.load_squad()
        
        train_data = []
        val_data = []
        
        print("Preparing baseline training data...")
        for example in tqdm(self.dataset['train']):
            train_data.append({
                'id': example['id'],
                'question': example['question'],
                'answer': example['answers']['text'][0],  # Take first answer
                'context': example['context'],  # Keep for evaluation
            })
        
        print("Preparing baseline validation data...")
        for example in tqdm(self.dataset['validation']):
            val_data.append({
                'id': example['id'],
                'question': example['question'],
                'answer': example['answers']['text'][0],
                'context': example['context'],
                'all_answers': example['answers']['text'],  # For proper EM/F1 evaluation
            })
        
        return train_data, val_data
    
    def save_processed_data(self, train_data, val_data, path_prefix):
        """Save processed data to disk."""
        os.makedirs(os.path.dirname(path_prefix), exist_ok=True)
        
        with open(f"{path_prefix}_train.json", 'w') as f:
            json.dump(train_data, f, indent=2)
        
        with open(f"{path_prefix}_val.json", 'w') as f:
            json.dump(val_data, f, indent=2)
        
        print(f"Saved processed data to {path_prefix}_*.json")
    
    def load_processed_data(self, path_prefix):
        """Load processed data from disk."""
        with open(f"{path_prefix}_train.json", 'r') as f:
            train_data = json.load(f)
        
        with open(f"{path_prefix}_val.json", 'r') as f:
            val_data = json.load(f)
        
        print(f"Loaded {len(train_data)} training and {len(val_data)} validation examples")
        return train_data, val_data


def main():
    """Main function to prepare all data."""
    processor = SQuADDataProcessor(
        train_sample_size=config.TRAIN_SAMPLE_SIZE,
        val_sample_size=config.VAL_SAMPLE_SIZE
    )
    
    # Load dataset
    processor.load_squad()
    
    # Build and save knowledge base
    knowledge_base = processor.build_knowledge_base()
    kb_path = os.path.join(config.DATA_DIR, 'knowledge_base.pkl')
    processor.save_knowledge_base(kb_path)
    
    # Prepare and save baseline data
    train_data, val_data = processor.prepare_baseline_data()
    data_path = os.path.join(config.DATA_DIR, 'processed_data')
    processor.save_processed_data(train_data, val_data, data_path)
    
    print("\nData preparation complete!")
    print(f"Knowledge base: {len(knowledge_base)} passages")
    print(f"Training examples: {len(train_data)}")
    print(f"Validation examples: {len(val_data)}")


if __name__ == "__main__":
    main()

