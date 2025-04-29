#implement your retriever here
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os

# Load data
with open('data/samples.txt', 'r', encoding='utf-8') as f:
    samples = [line.strip() for line in f if line.strip()]
samples = samples[:10]

# Encode
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(samples)

# Similarity
similarity = cosine_similarity(embeddings)

# Display
for i in range(len(samples)):
    for j in range(len(samples)):
        print(f"Similarity between Sample {i+1} and Sample {j+1}: {similarity[i][j]:.2f}")
