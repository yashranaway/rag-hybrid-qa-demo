"""
Analysis script for comparing retrieval impact.
"""

import os
import sys
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def load_results():
    """Load evaluation results."""
    results_dir = config.RESULTS_DIR
    
    # Load predictions
    baseline_pred_path = os.path.join(results_dir, 'baseline_predictions.json')
    rag_pred_path = os.path.join(results_dir, 'rag_predictions.json')
    
    with open(baseline_pred_path, 'r') as f:
        baseline_preds = json.load(f)
    
    with open(rag_pred_path, 'r') as f:
        rag_preds = json.load(f)
    
    return baseline_preds, rag_preds


def analyze_retrieval_impact(baseline_preds, rag_preds):
    """Analyze where retrieval helped vs hurt."""
    
    # Create lookup by ID
    baseline_by_id = {p['id']: p for p in baseline_preds}
    rag_by_id = {p['id']: p for p in rag_preds}
    
    # Categorize examples
    retrieval_helped = []  # RAG better than baseline
    retrieval_hurt = []    # RAG worse than baseline
    retrieval_neutral = []  # No significant change
    
    for ex_id in baseline_by_id:
        if ex_id not in rag_by_id:
            continue
        
        baseline = baseline_by_id[ex_id]
        rag = rag_by_id[ex_id]
        
        baseline_f1 = baseline['f1']
        rag_f1 = rag['f1']
        
        diff = rag_f1 - baseline_f1
        
        if diff > 0.1:  # Significant improvement
            retrieval_helped.append({
                'id': ex_id,
                'question': baseline['question'],
                'ground_truth': baseline['ground_truth'],
                'baseline_pred': baseline['prediction'],
                'rag_pred': rag['prediction'],
                'baseline_f1': baseline_f1,
                'rag_f1': rag_f1,
                'improvement': diff,
                'retrieved_passages': rag['retrieved_passages']
            })
        elif diff < -0.1:  # Significant degradation
            retrieval_hurt.append({
                'id': ex_id,
                'question': baseline['question'],
                'ground_truth': baseline['ground_truth'],
                'baseline_pred': baseline['prediction'],
                'rag_pred': rag['prediction'],
                'baseline_f1': baseline_f1,
                'rag_f1': rag_f1,
                'degradation': -diff,
                'retrieved_passages': rag['retrieved_passages']
            })
        else:
            retrieval_neutral.append({
                'id': ex_id,
                'baseline_f1': baseline_f1,
                'rag_f1': rag_f1
            })
    
    return retrieval_helped, retrieval_hurt, retrieval_neutral


def print_examples(examples, title, num_examples=5):
    """Print example predictions."""
    print("\n" + "="*80)
    print(title)
    print("="*80)
    
    for i, ex in enumerate(examples[:num_examples], 1):
        print(f"\n--- Example {i} ---")
        print(f"Question: {ex['question']}")
        print(f"Ground Truth: {ex['ground_truth']}")
        print(f"\nBaseline Prediction (F1={ex['baseline_f1']:.2f}): {ex['baseline_pred']}")
        print(f"RAG Prediction (F1={ex['rag_f1']:.2f}): {ex['rag_pred']}")
        
        if 'improvement' in ex:
            print(f"Improvement: +{ex['improvement']:.2f}")
        else:
            print(f"Degradation: -{ex['degradation']:.2f}")
        
        if ex.get('retrieved_passages'):
            print("\nRetrieved Passages:")
            for j, passage in enumerate(ex['retrieved_passages'][:2], 1):
                print(f"  {j}. (score={passage['score']:.3f}) {passage['text'][:150]}...")


def plot_comparison(baseline_preds, rag_preds):
    """Create visualization comparing models."""
    results_dir = config.RESULTS_DIR
    
    # Extract F1 scores
    baseline_f1s = [p['f1'] for p in baseline_preds]
    rag_f1s = [p['f1'] for p in rag_preds if p['id'] in [b['id'] for b in baseline_preds]]
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. F1 Score Distribution
    axes[0, 0].hist([baseline_f1s, rag_f1s], label=['Baseline', 'RAG'], 
                    bins=20, alpha=0.7)
    axes[0, 0].set_xlabel('F1 Score')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('F1 Score Distribution')
    axes[0, 0].legend()
    
    # 2. Scatter plot
    axes[0, 1].scatter(baseline_f1s, rag_f1s, alpha=0.5)
    axes[0, 1].plot([0, 1], [0, 1], 'r--', label='y=x')
    axes[0, 1].set_xlabel('Baseline F1')
    axes[0, 1].set_ylabel('RAG F1')
    axes[0, 1].set_title('Per-Example Comparison')
    axes[0, 1].legend()
    
    # 3. Improvement distribution
    improvements = [r - b for b, r in zip(baseline_f1s, rag_f1s)]
    axes[1, 0].hist(improvements, bins=30, alpha=0.7, color='green')
    axes[1, 0].axvline(x=0, color='red', linestyle='--')
    axes[1, 0].set_xlabel('F1 Improvement (RAG - Baseline)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Retrieval Impact Distribution')
    
    # 4. Summary statistics
    axes[1, 1].axis('off')
    summary_text = f"""
    Summary Statistics:
    
    Baseline:
      Mean F1: {sum(baseline_f1s)/len(baseline_f1s):.3f}
      Median F1: {sorted(baseline_f1s)[len(baseline_f1s)//2]:.3f}
    
    RAG:
      Mean F1: {sum(rag_f1s)/len(rag_f1s):.3f}
      Median F1: {sorted(rag_f1s)[len(rag_f1s)//2]:.3f}
    
    Improvement:
      Mean: {sum(improvements)/len(improvements):.3f}
      Helped: {sum(1 for x in improvements if x > 0.1)} examples
      Hurt: {sum(1 for x in improvements if x < -0.1)} examples
      Neutral: {sum(1 for x in improvements if abs(x) <= 0.1)} examples
    """
    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
                    verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'comparison_plots.png'), dpi=150)
    print(f"\nPlots saved to {os.path.join(results_dir, 'comparison_plots.png')}")


def main():
    """Main analysis function."""
    print("="*80)
    print("Analyzing Retrieval Impact")
    print("="*80)
    
    # Load results
    baseline_preds, rag_preds = load_results()
    
    # Analyze impact
    helped, hurt, neutral = analyze_retrieval_impact(baseline_preds, rag_preds)
    
    # Print summary
    print(f"\nTotal examples analyzed: {len(baseline_preds)}")
    print(f"Retrieval HELPED: {len(helped)} examples ({len(helped)/len(baseline_preds)*100:.1f}%)")
    print(f"Retrieval HURT: {len(hurt)} examples ({len(hurt)/len(baseline_preds)*100:.1f}%)")
    print(f"Retrieval NEUTRAL: {len(neutral)} examples ({len(neutral)/len(baseline_preds)*100:.1f}%)")
    
    # Show examples where retrieval helped
    print_examples(
        sorted(helped, key=lambda x: x['improvement'], reverse=True),
        "TOP EXAMPLES WHERE RETRIEVAL HELPED",
        num_examples=config.NUM_EXAMPLES_TO_ANALYZE // 2
    )
    
    # Show examples where retrieval hurt
    print_examples(
        sorted(hurt, key=lambda x: x['degradation'], reverse=True),
        "TOP EXAMPLES WHERE RETRIEVAL HURT",
        num_examples=config.NUM_EXAMPLES_TO_ANALYZE // 2
    )
    
    # Create visualizations
    print("\nCreating visualizations...")
    plot_comparison(baseline_preds, rag_preds)
    
    # Save analysis
    analysis = {
        'summary': {
            'total_examples': len(baseline_preds),
            'helped_count': len(helped),
            'hurt_count': len(hurt),
            'neutral_count': len(neutral),
            'helped_percentage': len(helped) / len(baseline_preds) * 100,
            'hurt_percentage': len(hurt) / len(baseline_preds) * 100,
        },
        'top_helped_examples': helped[:10],
        'top_hurt_examples': hurt[:10]
    }
    
    analysis_path = os.path.join(config.RESULTS_DIR, 'retrieval_analysis.json')
    with open(analysis_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f"\nAnalysis saved to {analysis_path}")
    print("\n" + "="*80)
    print("Analysis Complete!")
    print("="*80)


if __name__ == "__main__":
    main()

