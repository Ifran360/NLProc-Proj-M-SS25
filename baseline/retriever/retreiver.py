import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os

# === Load and embed files ===
base_dir = os.path.dirname(__file__)
data_dir = os.path.abspath(os.path.join(base_dir, '..', 'data'))

texts = []
labels = []

for fname in sorted(os.listdir(data_dir)):
    if fname.endswith('.txt'):
        path = os.path.join(data_dir, fname)
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if content:
                texts.append(content)
                labels.append(fname)

print(f"Loaded {len(texts)} documents.")

# === Generate embeddings ===
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(texts)
embeddings = np.array(embeddings).astype("float32")  # FAISS needs float32

# === Build FAISS index ===
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)  # L2 = Euclidean distance (can use cosine with normalization)
index.add(embeddings)
print("FAISS index built and populated!")

# === Try a similarity search ===
query_text = "A story about a magical school and a young wizard."
query_embedding = model.encode([query_text]).astype("float32")

k = 3  # top 3 most similar
D, I = index.search(query_embedding, k)  # D = distances, I = indices

print(f"\nTop {k} similar documents for query:")
for i, idx in enumerate(I[0]):
    print(f"{i+1}. {labels[idx]} (Distance: {D[0][i]:.4f})")
