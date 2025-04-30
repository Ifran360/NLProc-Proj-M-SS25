import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# === SETUP ===
base_dir = os.path.dirname(__file__)
data_dir = os.path.abspath(os.path.join(base_dir, '..', 'data'))

texts = []
labels = []

# === Load all text files ===
for fname in sorted(os.listdir(data_dir)):
    if fname.endswith('.txt'):
        path = os.path.join(data_dir, fname)
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if content:
                texts.append(content)
                labels.append(fname)

print(f" Loaded {len(texts)} documents.")

# === Generate embeddings ===
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(texts).astype("float32")  # FAISS needs float32

# === Build & populate FAISS index ===
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)
print(" FAISS index built and populated in memory.")

# === Define 5 rephrasings of the same semantic question ===
query_variants = [
    "A young wizard attending a magical school.",
    "A story of a boy who learns magic with friends.",
    "An orphan goes to a school to become a wizard.",
    "A fantasy novel set in a school for young wizards.",
    "A magical academy for children learning spells."
]

# === Run search for each variant ===
top_k = 3
results_by_query = {}

for query_text in query_variants:
    query_embedding = model.encode([query_text]).astype("float32")
    # Optional normalization for cosine-like search
    # faiss.normalize_L2(query_embedding)

    D, I = index.search(query_embedding, top_k)

    results = []
    for i, idx in enumerate(I[0]):
        results.append({
            'rank': i + 1,
            'file': labels[idx],
            'distance': D[0][i],
            'preview': texts[idx][:100].replace('\n', ' ') + "..."
        })

    results_by_query[query_text] = results

# === Print comparison of top results ===
print("\n === TOP-K RESULTS PER QUERY VARIANT ===\n")
for q, results in results_by_query.items():
    print(f" Query: \"{q}\"")
    for res in results:
        print(f"  {res['rank']}. {res['file']} (Distance: {res['distance']:.4f})")
    print("-" * 60)
