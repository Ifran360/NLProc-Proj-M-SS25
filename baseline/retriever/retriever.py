import os
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional

class Retriever:
    """
    A semantic document retriever using SentenceTransformers for embeddings and FAISS for indexing.
    
    Methods:
        - add_documents(): Indexes provided documents (with chunking)
        - query(): Returns most relevant text chunks
        - save(): Saves FAISS index and metadata
        - load(): Loads FAISS index and metadata

    Example usage:
    >>> retriever = Retriever()
    >>> retriever.add_documents([{"id": "doc1", "text": "Sample document text."}])
    >>> retriever.save("./index_dir")
    >>> retriever.load("./index_dir")
    >>> results = retriever.query("sample", k=1)
    """

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

        Args:
            documents: List of dicts, each with 'id' and 'text'.
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
        Queries the retriever and returns top matching chunks.

        Args:
            text: Query string.
            k: Number of top results to return.

        Returns:
            List of dicts with 'chunk_id', 'text', 'distance'.
        """
        if not self.index:
            raise ValueError("Index is empty. Please add documents first.")

        embedding = self.model.encode([text]).astype("float32")
        distances, indices = self.index.search(embedding, k)

        results = []
        for rank, idx in enumerate(indices[0]):
            results.append({
                "chunk_id": self.chunk_ids[idx],
                "text": self.chunks[idx],
                "distance": float(distances[0][rank])
            })
        return results

    def save(self, path: str):
        """
        Saves FAISS index and metadata to the given directory.
        """
        faiss.write_index(self.index, os.path.join(path, "faiss.index"))
        with open(os.path.join(path, "metadata.pkl"), "wb") as f:
            pickle.dump({"chunks": self.chunks, "chunk_ids": self.chunk_ids}, f)

    def load(self, path: str):
        """
        Loads FAISS index and metadata from the given directory.
        """
        self.index = faiss.read_index(os.path.join(path, "faiss.index"))
        with open(os.path.join(path, "metadata.pkl"), "rb") as f:
            data = pickle.load(f)
            self.chunks = data["chunks"]
            self.chunk_ids = data["chunk_ids"]
