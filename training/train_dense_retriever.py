"""
Contrastive fine-tuning for the dense retriever using sentence-transformers.
Inputs: triples (query, positive_passage, negative_passage).
Saves a fine-tuned embedding model under results/dense_retriever_model.
"""

import os
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader


def load_triples(path: str):
    # Expect TSV: query \t positive \t negative
    triples = []
    if not os.path.exists(path):
        return triples
    with open(path, "r") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            triples.append(parts)
    return triples


def main():
    triples = load_triples("data/triples.tsv")
    if not triples:
        print("No triples found at data/triples.tsv; skipping training.")
        return

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    examples = [InputExample(texts=[q, p, n]) for q, p, n in triples]
    train_loader = DataLoader(examples, shuffle=True, batch_size=16)
    train_loss = losses.TripletLoss(model)

    model.fit(train_objectives=[(train_loader, train_loss)], epochs=1, warmup_steps=100)
    out = "results/dense_retriever_model"
    os.makedirs(out, exist_ok=True)
    model.save(out)
    print(f"Saved fine-tuned dense retriever to {out}")


if __name__ == "__main__":
    main()


