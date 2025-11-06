#!/usr/bin/env python3
"""Process resume data for training."""

import pandas as pd
import re
import os

def clean_resume(text):
    """Clean resume text by removing emails, phone numbers, and URLs."""
    if not isinstance(text, str):
        return ""
    
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
    text = re.sub(r'\b\d{10,}\b|\(\d{3}\)\s\d{3}-\d{4}', '', text)
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    return text

def process_resume_dataset(input_path="archive/Resume/Resume.csv", output_path="cleaned_resumes.csv"):
    """Process resume dataset and create cleaned version."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    print(f"Loading {input_path}...")
    df = pd.read_csv(input_path)
    
    if 'Resume_str' not in df.columns:
        raise ValueError("Dataset must contain 'Resume_str' column")
    
    print(f"Cleaning {len(df)} resumes...")
    df['cleaned_resume_txt'] = df['Resume_str'].apply(clean_resume)
    df = df[df['cleaned_resume_txt'].str.len() > 50]
    
    final_df = df[['cleaned_resume_txt']].rename(columns={'cleaned_resume_txt': 'resume_str'})
    final_df.to_csv(output_path, index=False)
    
    print(f"Saved {len(final_df)} cleaned resumes to {output_path}")
    return final_df

def main():
    try:
        process_resume_dataset()
        print("Processing completed successfully!")
    except Exception as e:
        print(f"Error: {e}")
        return 1
    return 0

if __name__ == "__main__":
    exit(main())