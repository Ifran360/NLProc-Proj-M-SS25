import json
import argparse
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import evaluate
from nltk.tokenize import WordPunctTokenizer
import os

# Initialize tokenizer and BERTScore
tokenizer = WordPunctTokenizer()
bertscore = evaluate.load("bertscore")

def compute_cosine_similarity(references, generated):
    vectorizer = TfidfVectorizer()
    all_texts = references + generated
    tfidf = vectorizer.fit_transform(all_texts)
    ref_vecs = tfidf[:len(references)]
    gen_vecs = tfidf[len(references):]
    cosine_similarities = cosine_similarity(ref_vecs, gen_vecs)
    return [cosine_similarities[i, i] for i in range(len(references))]

def compute_bertscore(references, generated):
    results = bertscore.compute(predictions=generated, references=references, lang="en")
    return results["f1"]

def compute_word_overlap(references, generated):
    scores = []
    for ref, gen in zip(references, generated):
        ref_tokens = set(tokenizer.tokenize(ref.lower()))
        gen_tokens = set(tokenizer.tokenize(gen.lower()))
        if not ref_tokens and not gen_tokens:
            score = 1.0
        elif not ref_tokens or not gen_tokens:
            score = 0.0
        else:
            score = len(ref_tokens & gen_tokens) / len(ref_tokens | gen_tokens)
        scores.append(score)
    return scores

def evaluate(data):
    references = data["Actual"].astype(str).tolist()
    generated = data["Generated"].astype(str).tolist()
    ids = data["Question"].astype(str).tolist()
    type = data["Type"].astype(str).tolist()

    print("Calculating cosine similarity...")
    cosine_scores = compute_cosine_similarity(references, generated)

    print("Calculating BERTScore...")
    bert_scores = compute_bertscore(references, generated)

    print("Calculating word overlap...")
    overlap_scores = compute_word_overlap(references, generated)

    results = []
    for i in range(len(ids)):
        results.append({
            "question": ids[i],
            "Type": type[i],
            "cosine_similarity": cosine_scores[i],
            "bertscore": bert_scores[i],
            "word_overlap": overlap_scores[i],
        })

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_excel", type=str, required=True, help="Path to Excel file with columns: question, generated, actual")
    parser.add_argument("--output_json", type=str, default="results.json", help="Output JSON file (appends if exists)")
    args = parser.parse_args()

    # Read Excel
    df = pd.read_excel(args.input_excel)

    if not {"Question", "Reference","Generated", "Actual"}.issubset(df.columns):
        raise ValueError("Excel file must contain columns: question, reference, generated, actual")

    # Evaluate
    results = evaluate(df)

    # Load existing JSON (if exists)
    if os.path.exists(args.output_json):
        with open(args.output_json, "r", encoding="utf-8") as f:
            existing_data = json.load(f)
    else:
        existing_data = []

    # Append and save
    updated_data = existing_data + results

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(updated_data, f, indent=2)

    print(f"Saved {len(results)} new evaluations to {args.output_json}")