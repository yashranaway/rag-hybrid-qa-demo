"""
Main pipeline script to run the entire RAG system end-to-end.
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config


def step_1_prepare_data():
    """Step 1: Prepare SQuAD data and build knowledge base."""
    print("\n" + "="*80)
    print("STEP 1: Preparing Data")
    print("="*80)
    
    from data.prepare_data import main as prepare_data_main
    prepare_data_main()


def step_2_build_retrievers():
    """Step 2: Build retrieval indices."""
    print("\n" + "="*80)
    print("STEP 2: Building Retrieval Indices")
    print("="*80)
    
    from data.prepare_data import SQuADDataProcessor
    from retrieval.hybrid_retriever import HybridRetriever
    
    # Load knowledge base
    processor = SQuADDataProcessor()
    kb_path = os.path.join(config.DATA_DIR, 'knowledge_base.pkl')
    passages = processor.load_knowledge_base(kb_path)
    
    # Build hybrid retriever
    retriever = HybridRetriever()
    retriever.build_index(passages)
    
    # Save indices
    index_path = os.path.join(config.RETRIEVAL_DIR, 'hybrid_index')
    retriever.save_index(index_path)
    
    print(f"\nRetrieval indices saved to {config.RETRIEVAL_DIR}")


def step_3_train_baseline():
    """Step 3: Train baseline model."""
    print("\n" + "="*80)
    print("STEP 3: Training Baseline Model")
    print("="*80)
    
    from training.train_baseline import main as train_baseline_main
    train_baseline_main()


def step_4_train_rag():
    """Step 4: Train RAG model."""
    print("\n" + "="*80)
    print("STEP 4: Training RAG Model")
    print("="*80)
    
    from training.train_rag import main as train_rag_main
    train_rag_main()


def step_5_evaluate():
    """Step 5: Evaluate both models."""
    print("\n" + "="*80)
    print("STEP 5: Evaluating Models")
    print("="*80)
    
    from evaluation.evaluate import main as evaluate_main
    evaluate_main()


def step_6_analyze():
    """Step 6: Analyze results."""
    print("\n" + "="*80)
    print("STEP 6: Analyzing Results")
    print("="*80)
    
    from evaluation.analyze import main as analyze_main
    analyze_main()


def run_full_pipeline():
    """Run the entire pipeline."""
    print("\n" + "="*80)
    print("RUNNING FULL RAG PIPELINE")
    print("="*80)
    
    step_1_prepare_data()
    step_2_build_retrievers()
    step_3_train_baseline()
    step_4_train_rag()
    step_5_evaluate()
    step_6_analyze()
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {config.RESULTS_DIR}")
    print("\nNext steps:")
    print("1. Check evaluation_results.json for metrics")
    print("2. Review comparison_plots.png for visualizations")
    print("3. Read retrieval_analysis.json for detailed analysis")


def main():
    """Main function with CLI."""
    parser = argparse.ArgumentParser(
        description="Retrieval-Augmented Transformer Pipeline"
    )
    parser.add_argument(
        '--step',
        type=str,
        choices=['all', 'data', 'retrieval', 'baseline', 'rag', 'evaluate', 'analyze'],
        default='all',
        help='Which step to run'
    )
    
    args = parser.parse_args()
    
    if args.step == 'all':
        run_full_pipeline()
    elif args.step == 'data':
        step_1_prepare_data()
    elif args.step == 'retrieval':
        step_2_build_retrievers()
    elif args.step == 'baseline':
        step_3_train_baseline()
    elif args.step == 'rag':
        step_4_train_rag()
    elif args.step == 'evaluate':
        step_5_evaluate()
    elif args.step == 'analyze':
        step_6_analyze()


if __name__ == "__main__":
    main()

