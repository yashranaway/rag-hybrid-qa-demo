"""
Evaluation script for comparing baseline and RAG models.
"""

import os
import sys
import json
import re
import string
from collections import Counter
import numpy as np
from tqdm import tqdm
from rouge_score import rouge_scorer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from model.base_model import BaseQAModel
from model.rag_model import RAGModel
from retrieval.hybrid_retriever import HybridRetriever
from data.prepare_data import SQuADDataProcessor


def normalize_answer(s):
    """Normalize answer string for comparison."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    
    def white_space_fix(text):
        return ' '.join(text.split())
    
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    
    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match_score(prediction, ground_truths):
    """Compute exact match score."""
    normalized_prediction = normalize_answer(prediction)
    for ground_truth in ground_truths:
        if normalized_prediction == normalize_answer(ground_truth):
            return 1.0
    return 0.0


def f1_score(prediction, ground_truths):
    """Compute F1 score."""
    normalized_prediction = normalize_answer(prediction)
    
    max_f1 = 0.0
    for ground_truth in ground_truths:
        normalized_ground_truth = normalize_answer(ground_truth)
        
        prediction_tokens = normalized_prediction.split()
        ground_truth_tokens = normalized_ground_truth.split()
        
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        
        if num_same == 0:
            f1 = 0.0
        else:
            precision = num_same / len(prediction_tokens)
            recall = num_same / len(ground_truth_tokens)
            f1 = (2 * precision * recall) / (precision + recall)
        
        max_f1 = max(max_f1, f1)
    
    return max_f1


def compute_rouge_l(prediction, ground_truths):
    """Compute ROUGE-L score."""
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    
    max_rouge = 0.0
    for ground_truth in ground_truths:
        scores = scorer.score(ground_truth, prediction)
        max_rouge = max(max_rouge, scores['rougeL'].fmeasure)
    
    return max_rouge


def evaluate_model(model, val_data, model_name, use_retrieval=False):
    """
    Evaluate a model on validation data.
    
    Args:
        model: Model to evaluate
        val_data: Validation dataset
        model_name: Name for logging
        use_retrieval: Whether to use retrieval (for RAG model)
        
    Returns:
        Dictionary with metrics and predictions
    """
    print(f"\nEvaluating {model_name}...")
    
    predictions = []
    em_scores = []
    f1_scores = []
    rouge_scores = []
    
    for item in tqdm(val_data, desc=f"Evaluating {model_name}"):
        question = item['question']
        ground_truths = item.get('all_answers', [item['answer']])
        
        # Generate prediction
        if isinstance(model, RAGModel):
            prediction, retrieved_passages = model.generate_answer(
                question, 
                use_retrieval=use_retrieval
            )
        else:
            prediction = model.generate_answer(question)
            retrieved_passages = None
        
        # Compute metrics
        em = exact_match_score(prediction, ground_truths)
        f1 = f1_score(prediction, ground_truths)
        rouge = compute_rouge_l(prediction, ground_truths)
        
        em_scores.append(em)
        f1_scores.append(f1)
        rouge_scores.append(rouge)
        
        # Store prediction
        predictions.append({
            'id': item['id'],
            'question': question,
            'prediction': prediction,
            'ground_truth': ground_truths[0],
            'all_ground_truths': ground_truths,
            'em': em,
            'f1': f1,
            'rouge_l': rouge,
            'retrieved_passages': [
                {'text': p['text'], 'score': float(s)} 
                for p, s in retrieved_passages
            ] if retrieved_passages else None
        })
    
    # Compute aggregate metrics
    results = {
        'model_name': model_name,
        'exact_match': np.mean(em_scores) * 100,
        'f1_score': np.mean(f1_scores) * 100,
        'rouge_l': np.mean(rouge_scores) * 100,
        'num_examples': len(val_data),
        'predictions': predictions
    }
    
    print(f"\n{model_name} Results:")
    print(f"  Exact Match: {results['exact_match']:.2f}%")
    print(f"  F1 Score: {results['f1_score']:.2f}%")
    print(f"  ROUGE-L: {results['rouge_l']:.2f}%")
    
    return results


def main():
    """Main evaluation function."""
    print("="*50)
    print("Evaluating Models")
    print("="*50)
    
    # Load validation data
    processor = SQuADDataProcessor()
    data_path = os.path.join(config.DATA_DIR, 'processed_data')
    _, val_data = processor.load_processed_data(data_path)
    
    # Limit to subset for faster evaluation if needed
    if config.VAL_SAMPLE_SIZE:
        val_data = val_data[:config.VAL_SAMPLE_SIZE]
    
    print(f"Evaluating on {len(val_data)} examples")
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    results_all = {}
    
    # Evaluate Baseline Model
    print("\n" + "="*50)
    print("1. Baseline Model (No Retrieval)")
    print("="*50)
    
    baseline_model_path = config.BASELINE_OUTPUT_DIR
    if os.path.exists(baseline_model_path):
        baseline_model = BaseQAModel(device=device)
        baseline_model.load_from_checkpoint(baseline_model_path)
        results_baseline = evaluate_model(
            baseline_model, 
            val_data, 
            "Baseline (No Retrieval)",
            use_retrieval=False
        )
        results_all['baseline'] = results_baseline
    else:
        print(f"Baseline model not found at {baseline_model_path}")
        print("Please train the baseline model first.")
        results_all['baseline'] = None
    
    # Evaluate RAG Model
    print("\n" + "="*50)
    print("2. RAG Model (With Retrieval)")
    print("="*50)
    
    rag_model_path = config.RAG_OUTPUT_DIR
    if os.path.exists(rag_model_path):
        # Load retriever
        retriever = HybridRetriever()
        index_path = os.path.join(config.RETRIEVAL_DIR, 'hybrid_index')
        retriever.load_index(index_path)
        
        # Load RAG model
        rag_model = RAGModel(retriever=retriever, device=device)
        rag_model.load_from_checkpoint(rag_model_path)
        
        results_rag = evaluate_model(
            rag_model,
            val_data,
            "RAG (With Retrieval)",
            use_retrieval=True
        )
        results_all['rag'] = results_rag
    else:
        print(f"RAG model not found at {rag_model_path}")
        print("Please train the RAG model first.")
        results_all['rag'] = None
    
    # Compare results
    print("\n" + "="*50)
    print("COMPARISON SUMMARY")
    print("="*50)
    
    if results_all['baseline'] and results_all['rag']:
        print(f"\n{'Metric':<20} {'Baseline':<15} {'RAG':<15} {'Improvement':<15}")
        print("-" * 70)
        
        for metric in ['exact_match', 'f1_score', 'rouge_l']:
            baseline_val = results_all['baseline'][metric]
            rag_val = results_all['rag'][metric]
            improvement = rag_val - baseline_val
            
            metric_name = metric.replace('_', ' ').title()
            print(f"{metric_name:<20} {baseline_val:>6.2f}%        {rag_val:>6.2f}%        "
                  f"{improvement:>+6.2f}%")
        
        # Save results
        results_dir = config.RESULTS_DIR
        os.makedirs(results_dir, exist_ok=True)
        
        # Save full results
        with open(os.path.join(results_dir, 'evaluation_results.json'), 'w') as f:
            # Don't save predictions in main file (too large)
            summary = {
                'baseline': {k: v for k, v in results_all['baseline'].items() 
                           if k != 'predictions'},
                'rag': {k: v for k, v in results_all['rag'].items() 
                       if k != 'predictions'}
            }
            json.dump(summary, f, indent=2)
        
        # Save predictions separately
        for model_name in ['baseline', 'rag']:
            if results_all[model_name]:
                pred_path = os.path.join(results_dir, f'{model_name}_predictions.json')
                with open(pred_path, 'w') as f:
                    json.dump(results_all[model_name]['predictions'], f, indent=2)
        
        print(f"\nResults saved to {results_dir}/")
    
    print("\n" + "="*50)
    print("Evaluation Complete!")
    print("="*50)


if __name__ == "__main__":
    import torch
    main()

