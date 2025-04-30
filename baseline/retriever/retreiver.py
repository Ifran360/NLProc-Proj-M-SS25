#implement your retriever here
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os

# Get absolute path to samples.txt regardless of where script is run from
base_dir = os.path.dirname(__file__)  # directory of the script
file_path = os.path.abspath(os.path.join(base_dir, '..', 'data', 'samples.txt'))

with open(file_path, 'r', encoding='utf-8') as f:
    samples = [line.strip() for line in f if line.strip()]
# Load data
#with open('../data/samples.txt', 'r', encoding='utf-8') as f:
    #samples = [line.strip() for line in f if line.strip()]
#samples = samples[:10]

# Encode
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(samples)

# Similarity
similarity = cosine_similarity(embeddings)

# Display
for i in range(len(samples)):
    for j in range(len(samples)):
        print(f"Similarity between Sample {i+1} and Sample {j+1}: {similarity[i][j]:.2f}")
        
        
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Reduce to 2D
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(embeddings)

# Plot
plt.figure(figsize=(8, 6))
for i, point in enumerate(reduced_embeddings):
    plt.scatter(point[0], point[1])
    plt.annotate(f"Sample {i+1}", (point[0]+0.01, point[1]+0.01))
plt.title("PCA Projection of Text Embeddings")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.grid(True)
plt.tight_layout()
plt.show()

