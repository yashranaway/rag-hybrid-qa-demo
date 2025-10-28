"""
Training script for RAG model (with retrieval).
"""

import os
import sys
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from model.rag_model import RAGModel
from retrieval.hybrid_retriever import HybridRetriever
from data.prepare_data import SQuADDataProcessor


class RAGDataset(Dataset):
    """Dataset for RAG training with pre-retrieved contexts."""
    
    def __init__(self, data, retriever, top_k):
        self.data = data
        self.retriever = retriever
        self.top_k = top_k
        self.contexts_cache = {}
        
        # Pre-retrieve all contexts
        print("Pre-retrieving contexts for all examples...")
        self._precompute_contexts()
    
    def _precompute_contexts(self):
        """Precompute contexts for all questions."""
        questions = [item['question'] for item in self.data]
        
        # Retrieve one at a time to avoid multiprocessing issues
        for idx, question in enumerate(tqdm(questions, desc="Retrieving")):
            try:
                results = self.retriever.retrieve([question], top_k=self.top_k)
                if results and results[0]:
                    passages = [passage['text'] for passage, score in results[0]]
                    context = " ".join(passages)[:config.MAX_LENGTH * 2]
                else:
                    context = ""
                self.contexts_cache[idx] = context
            except Exception as e:
                print(f"Error retrieving for question {idx}: {e}")
                self.contexts_cache[idx] = ""
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx].copy()
        item['retrieved_context'] = self.contexts_cache[idx]
        return item


def collate_fn(batch, model):
    """Collate function for RAG dataloader."""
    questions = [item['question'] for item in batch]
    answers = [item['answer'] for item in batch]
    contexts = [item['retrieved_context'] for item in batch]
    
    return model.prepare_batch(questions, answers, contexts)


def train_epoch(model, dataloader, optimizer, scheduler, device):
    """Train for one epoch."""
    model.model.train()
    total_loss = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    for step, batch in enumerate(progress_bar):
        # Forward pass
        outputs = model.model(**batch)
        loss = outputs.loss
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.model.parameters(), config.MAX_GRAD_NORM)
        
        # Optimizer step
        if (step + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss.item()
        
        # Update progress bar
        progress_bar.set_postfix({'loss': loss.item()})
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    """Evaluate the model."""
    model.model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            outputs = model.model(**batch)
            loss = outputs.loss
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def main():
    """Main training function."""
    print("="*50)
    print("Training RAG Model (With Retrieval)")
    print("="*50)
    
    # Set random seed
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    
    # Load retriever
    print("\nLoading retriever...")
    retriever = HybridRetriever()
    index_path = os.path.join(config.RETRIEVAL_DIR, 'hybrid_index')
    
    if not os.path.exists(f"{index_path}_dense_faiss.index"):
        print("ERROR: Retrieval indices not found!")
        print("Please run the retrieval building scripts first.")
        return
    
    retriever.load_index(index_path)
    
    # Load data
    processor = SQuADDataProcessor()
    data_path = os.path.join(config.DATA_DIR, 'processed_data')
    
    if os.path.exists(f"{data_path}_train.json"):
        train_data, val_data = processor.load_processed_data(data_path)
    else:
        print("Processed data not found. Preparing data...")
        processor.load_squad()
        train_data, val_data = processor.prepare_baseline_data()
        processor.save_processed_data(train_data, val_data, data_path)
    
    print(f"Training examples: {len(train_data)}")
    print(f"Validation examples: {len(val_data)}")
    
    # Create datasets with retrieval
    train_dataset = RAGDataset(train_data, retriever, config.TOP_K_RETRIEVAL)
    val_dataset = RAGDataset(val_data, retriever, config.TOP_K_RETRIEVAL)
    
    # Initialize model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    model = RAGModel(retriever=retriever, device=device)
    model.load_model()
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, model)
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, model)
    )
    
    # Setup optimizer and scheduler
    optimizer = AdamW(
        model.model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    total_steps = len(train_dataloader) * config.NUM_EPOCHS // config.GRADIENT_ACCUMULATION_STEPS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.WARMUP_STEPS,
        num_training_steps=total_steps
    )
    
    # Training loop
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}
    
    print(f"\nStarting training for {config.NUM_EPOCHS} epochs...")
    print(f"Total training steps: {total_steps}")
    
    for epoch in range(config.NUM_EPOCHS):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{config.NUM_EPOCHS}")
        print(f"{'='*50}")
        
        # Train
        train_loss = train_epoch(model, train_dataloader, optimizer, scheduler, device)
        print(f"Average training loss: {train_loss:.4f}")
        
        # Evaluate
        val_loss = evaluate(model, val_dataloader, device)
        print(f"Validation loss: {val_loss:.4f}")
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_model(config.RAG_OUTPUT_DIR)
            print(f"Saved new best model (val_loss: {val_loss:.4f})")
        
        # Save checkpoint
        checkpoint_dir = os.path.join(config.RAG_OUTPUT_DIR, f"checkpoint-epoch-{epoch+1}")
        model.save_model(checkpoint_dir)
    
    # Save training history
    history_path = os.path.join(config.RAG_OUTPUT_DIR, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n{'='*50}")
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {config.RAG_OUTPUT_DIR}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()

