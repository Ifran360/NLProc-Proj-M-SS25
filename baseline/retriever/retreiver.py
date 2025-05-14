import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict

class Retriever:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.dimension = None
        self.documents = []
        self.doc_ids = []
        self.id_to_doc = {}

    def _chunk_text(self, text: str, chunk_size: int = 200) -> List[str]:
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    def add_documents(self, documents: List[Dict], chunk_size: int = 200):
        """
        documents: List of dicts: { "id": str, "text": str }
        """
        all_chunks = []
        chunk_ids = []

        for doc in documents:
            chunks = self._chunk_text(doc['text'], chunk_size)
            for i, chunk in enumerate(chunks):
                chunk_id = f"{doc['id']}_chunk{i}"
                all_chunks.append(chunk)
                chunk_ids.append(chunk_id)
                self.id_to_doc[chunk_id] = chunk

        embeddings = self.model.encode(all_chunks).astype('float32')
        self.dimension = embeddings.shape[1]

        if self.index is None:
            self.index = faiss.IndexFlatL2(self.dimension)

        self.index.add(embeddings)
        self.doc_ids.extend(chunk_ids)
        print(f"✅ Added {len(all_chunks)} chunks from {len(documents)} documents to the index.")

    def query(self, text: str, k: int = 3):
        if self.index is None:
            raise ValueError("FAISS index is not loaded or built.")

        if not self.doc_ids:
            raise ValueError("Document IDs are missing. Did you forget to call add_documents() after load()?")

        query_vec = self.model.encode([text]).astype('float32')
        D, I = self.index.search(query_vec, k)

        results = []
        for i in range(k):
            idx = I[0][i]
            doc_id = self.doc_ids[idx]

            # Provide a fallback message if id_to_doc is missing
            chunk_text = self.id_to_doc.get(doc_id, "[Text not loaded. Call add_documents(..., skip_indexing=True) after loading index.]")

            results.append({
                "chunk_id": doc_id,
                "text": chunk_text,
                "distance": D[0][i]
            })

        return results


    def save(self, path: str):
        faiss.write_index(self.index, os.path.join(path, "faiss.index"))
        with open(os.path.join(path, "doc_ids.txt"), "w", encoding="utf-8") as f:
            for doc_id in self.doc_ids:
                f.write(doc_id + "\n")
        print(f"💾 Index and metadata saved to {path}")

    def load(self, path: str):
        self.index = faiss.read_index(os.path.join(path, "faiss.index"))
        with open(os.path.join(path, "doc_ids.txt"), "r", encoding="utf-8") as f:
            self.doc_ids = [line.strip() for line in f]
        print(f"📂 Loaded FAISS index and {len(self.doc_ids)} chunk IDs from {path}")
