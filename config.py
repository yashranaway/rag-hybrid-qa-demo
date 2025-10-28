"""
Configuration file for Retrieval-Augmented Transformer system.
"""

import torch

# Model Configuration
MODEL_NAME = "distilgpt2"
MAX_LENGTH = 512
MAX_ANSWER_LENGTH = 100

# Retrieval Configuration
TOP_K_RETRIEVAL = 3  # Number of passages to retrieve
DENSE_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
FAISS_INDEX_TYPE = "Flat"  # Can be "Flat" or "IVF" for larger datasets

# Hybrid Retrieval Fusion
BM25_WEIGHT = 0.4
DENSE_WEIGHT = 0.6
USE_RRF = True  # Use Reciprocal Rank Fusion instead of score fusion

# Training Configuration
BASELINE_OUTPUT_DIR = "./results/baseline_model"
RAG_OUTPUT_DIR = "./results/rag_model"
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 5e-5
NUM_EPOCHS = 2
WARMUP_STEPS = 500
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
SAVE_STEPS = 1000
EVAL_STEPS = 500
LOGGING_STEPS = 100
FP16 = torch.cuda.is_available()

# Data Configuration
DATASET_NAME = "squad"
TRAIN_SAMPLE_SIZE = 5000  # Set to int to use subset for faster iteration
VAL_SAMPLE_SIZE = 500
SEED = 42

# Evaluation Configuration
NUM_EXAMPLES_TO_ANALYZE = 20
GENERATION_MAX_LENGTH = 100
GENERATION_NUM_BEAMS = 4
GENERATION_TEMPERATURE = 1.0

# Paths
DATA_DIR = "./data"
RETRIEVAL_DIR = "./retrieval"
RESULTS_DIR = "./results"
MODELS_DIR = "./models"

