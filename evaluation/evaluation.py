import json
import argparse
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import evaluate
from nltk.tokenize import WordPunctTokenizer

# Use WordPunctTokenizer to avoid issues with punkt_tab
tokenizer = WordPunctTokenizer()

# Load evaluation metrics
bertscore = evaluate.load("bertscore")

def compute_cosine_similarity(references, generations):
    vectorizer = TfidfVectorizer()
    all_texts = references + generations
    tfidf = vectorizer.fit_transform(all_texts)
    ref_vecs = tfidf[:len(references)]
    gen_vecs = tfidf[len(references):]
    cosine_similarities = cosine_similarity(ref_vecs, gen_vecs)
    return [cosine_similarities[i, i] for i in range(len(references))]

def compute_bertscore(references, generations):
    results = bertscore.compute(predictions=generations, references=references, lang="en")
    return results["f1"]

def compute_word_overlap(references, generations):
    scores = []
    for ref, gen in zip(references, generations):
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
    references = [item["reference"] for item in data]
    generations = [item["generated"] for item in data]


    print("Calculating cosine similarity...")
    cosine_scores = compute_cosine_similarity(references, generations)

    print("Calculating BERTScore...")
    bert_scores = compute_bertscore(references, generations)

    print("Calculating word overlap...")
    overlap_scores = compute_word_overlap(references, generations)

    results = []
    for i in range(len(data)):
        results.append({
            "id": data[i].get("question", f"sample_{i}"),
            "cosine_similarity": cosine_scores[i],
            "bertscore": bert_scores[i],
            "word_overlap": overlap_scores[i],
        })


    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input JSON file with reference and generation.")
    parser.add_argument("--output", type=str, required=True, help="Output JSON file with evaluation results.")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = evaluate(data)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Saved results to {args.output}")
