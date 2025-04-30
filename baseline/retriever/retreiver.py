import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Set your data folder
base_dir = os.path.dirname(__file__)
data_dir = os.path.abspath(os.path.join(base_dir, '..', 'data'))

# Load all .txt files in the data directory
text_samples = []
file_labels = []

for filename in sorted(os.listdir(data_dir)):
    if filename.endswith('.txt'):
        file_path = os.path.join(data_dir, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if content:
                text_samples.append(content)
                file_labels.append(filename)

print(f"Loaded {len(text_samples)} files.")

# Encode text
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(text_samples)

# Cosine similarity
similarity = cosine_similarity(embeddings)
print("\nCosine Similarity Matrix:")
for i in range(len(file_labels)):
    for j in range(len(file_labels)):
        print(f"{file_labels[i]} vs {file_labels[j]}: {similarity[i][j]:.2f}")

# PCA Visualization
pca = PCA(n_components=2)
pca_embeddings = pca.fit_transform(embeddings)

colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'cyan']

plt.figure(figsize=(8, 6))
for i, point in enumerate(pca_embeddings):
    plt.scatter(point[0], point[1], color=colors[i % len(colors)], s=100)
    plt.text(point[0]+0.01, point[1]+0.01, file_labels[i], fontsize=9)
plt.title("PCA Projection of Document Embeddings")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.grid(True)
plt.tight_layout()
plt.savefig("pca_plot.png")
plt.show()

# t-SNE Visualization
tsne = TSNE(n_components=2, perplexity=5, random_state=42)
tsne_embeddings = tsne.fit_transform(embeddings)

plt.figure(figsize=(8, 6))
for i, point in enumerate(tsne_embeddings):
    plt.scatter(point[0], point[1], color=colors[i % len(colors)], s=100)
    plt.text(point[0]+1, point[1]+1, file_labels[i], fontsize=9)
plt.title("t-SNE Projection of Document Embeddings")
plt.xlabel("t-SNE Dim 1")
plt.ylabel("t-SNE Dim 2")
plt.grid(True)
plt.tight_layout()
plt.savefig("tsne_plot.png")
plt.show()
