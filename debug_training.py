#!/usr/bin/env python3
"""Simple SBERT training script."""

import os
import pandas as pd
import sys

def train_sbert(dataset_path="cleaned_resumes.csv", output_dir="./trained_model"):
    """Train SBERT model on resume data."""
    try:
        from sentence_transformers import SentenceTransformer, InputExample, losses
        from torch.utils.data import DataLoader
        
        print(f"Loading dataset: {dataset_path}")
        df = pd.read_csv(dataset_path)
        
        if 'resume_str' not in df.columns:
            print("Error: 'resume_str' column not found")
            return False
        
        resumes = df['resume_str'].dropna().tolist()
        resumes = [r for r in resumes if len(r) > 100]
        
        print(f"Creating training examples from {len(resumes)} resumes")
        examples = []
        
        for i in range(min(20, len(resumes))):
            for j in range(i + 1, min(i + 3, len(resumes))):
                words1 = set(resumes[i].lower().split())
                words2 = set(resumes[j].lower().split())
                
                if words1 and words2:
                    similarity = len(words1 & words2) / len(words1 | words2)
                    examples.append(InputExample(
                        texts=[resumes[i], resumes[j]], 
                        label=float(similarity)
                    ))
        
        print(f"Training with {len(examples)} examples")
        
        model = SentenceTransformer('all-MiniLM-L6-v2')
        train_dataloader = DataLoader(examples, shuffle=True, batch_size=4)
        train_loss = losses.CosineSimilarityLoss(model)
        
        os.makedirs(output_dir, exist_ok=True)
        
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=2,
            warmup_steps=10,
            output_path=output_dir
        )
        
        print(f"Model saved to: {output_dir}")
        return True
        
    except Exception as e:
        print(f"Training failed: {e}")
        return False

def main():
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else "cleaned_resumes.csv"
    success = train_sbert(dataset_path)
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())