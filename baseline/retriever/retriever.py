import os
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from typing import List, Dict

class Retriever:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', chunk_size: int = 500):
        self.model = SentenceTransformer(model_name)
        self.chunk_size = chunk_size
        self.index = None
        self.chunks = []
        self.chunk_ids = []

    def _chunk_text(self, text: str) -> List[str]:
        return [text[i:i+self.chunk_size] for i in range(0, len(text), self.chunk_size)]

    def add_documents(self, documents: List[Dict[str, str]]):
        """
        Adds and indexes documents with chunking.
        Each doc should have an 'id' and 'text'.
        """
        all_chunks = []
        all_ids = []

        for doc in documents:
            chunks = self._chunk_text(doc["text"])
            chunk_ids = [f"{doc['id']}_chunk_{i}" for i in range(len(chunks))]
            all_chunks.extend(chunks)
            all_ids.extend(chunk_ids)

        embeddings = self.model.encode(all_chunks).astype("float32")

        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)
        self.chunks = all_chunks
        self.chunk_ids = all_ids

    def query(self, text: str, k: int = 3) -> List[Dict[str, str]]:
        """
        Returns top-k most relevant chunks for a query.
        """
        if self.index is None:
            raise ValueError("Index is empty. Please add or load documents.")

        embedding = self.model.encode([text]).astype("float32")
        distances, indices = self.index.search(embedding, k)

        results = []
        seen = set()

        for rank, idx in enumerate(indices[0]):
            if idx == -1 or distances[0][rank] > 1e6:
                continue  # Skip bad results

            chunk_id = self.chunk_ids[idx]
            if chunk_id in seen:
                continue  # Deduplicate
            seen.add(chunk_id)

            results.append({
                "chunk_id": chunk_id,
                "text": self.chunks[idx],
                "distance": float(distances[0][rank])
            })

        return results

    def save(self, path: str):
        """
        Saves FAISS index and metadata (chunks + ids).
        """
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(path, "faiss.index"))
        with open(os.path.join(path, "metadata.pkl"), "wb") as f:
            pickle.dump({
                "chunks": self.chunks,
                "chunk_ids": self.chunk_ids
            }, f)

    def load(self, path: str):
        """
        Loads index and metadata from path.
        """
        index_path = os.path.join(path, "faiss.index")
        meta_path = os.path.join(path, "metadata.pkl")

        if not os.path.exists(index_path) or not os.path.exists(meta_path):
            raise FileNotFoundError("Missing index or metadata file at path.")

        self.index = faiss.read_index(index_path)
        with open(meta_path, "rb") as f:
            metadata = pickle.load(f)
            self.chunks = metadata["chunks"]
            self.chunk_ids = metadata["chunk_ids"]
