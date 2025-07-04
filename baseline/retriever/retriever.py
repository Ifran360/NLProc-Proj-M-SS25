import os
import pickle
import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer

class Retriever:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        self.index = None
        self.documents = []
        self.embeddings = None
        self.chunk_ids = []
        self.tokenized_corpus = []
        self.bm25_model = None

    def clean_chunk(self, chunk):
        cleaned = ' '.join(chunk.split())
        return cleaned if len(cleaned.split()) > 3 else ""

    def add_documents(self, documents, chunk_size=500):
        self.documents = []
        self.chunk_ids = []
        for doc in documents:
            text = doc["text"]
            doc_id = doc["id"]
            chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
            for idx, chunk in enumerate(chunks):
                clean_chunk = self.clean_chunk(chunk)
                if not clean_chunk:
                    continue
                chunk_id = f"{doc_id}chunk{idx}"
                self.chunk_ids.append(chunk_id)
                self.documents.append({"id": chunk_id, "text": clean_chunk})

        # FAISS embeddings
        texts = [d["text"] for d in self.documents]
        self.embeddings = self.model.encode(texts, show_progress_bar=True).astype('float32')
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(self.embeddings)

        # BM25
        self.tokenized_corpus = [word_tokenize(doc["text"].lower()) for doc in self.documents]
        self.bm25_model = BM25Okapi(self.tokenized_corpus)

    def save(self, index_dir):
        os.makedirs(index_dir, exist_ok=True)
        faiss.write_index(self.index, os.path.join(index_dir, "faiss.index"))
        metadata = {
            "chunk_ids": self.chunk_ids,
            "texts": [doc["text"] for doc in self.documents]
        }
        with open(os.path.join(index_dir, "metadata.pkl"), "wb") as f:
            pickle.dump(metadata, f)

    def load(self, index_dir):
        self.index = faiss.read_index(os.path.join(index_dir, "faiss.index"))
        print("Loading documents...")
        with open(os.path.join(index_dir, "metadata.pkl"), "rb") as f:
            metadata = pickle.load(f)

        self.chunk_ids = metadata["chunk_ids"]
        texts = metadata["texts"]
        self.documents = [{"id": cid, "text": txt} for cid, txt in zip(self.chunk_ids, texts)]
        print("Tokenizing corpus for BM25...")
        self.tokenized_corpus = [word_tokenize(txt.lower()) for txt in texts]
        print("Initializing BM25 model...")
        self.bm25_model = BM25Okapi(self.tokenized_corpus)
        print("Encoding embeddings...")
        self.embeddings = self.model.encode([d["text"] for d in self.documents], show_progress_bar=True)
        print("Load complete.")  

    def query_faiss(self, query, k=5):
        query_emb = self.model.encode([query]).astype('float32')
        distances, indices = self.index.search(query_emb, k)
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx == -1:
                continue
            results.append({
                "chunk_id": self.chunk_ids[idx],
                "text": self.documents[idx]["text"],
                "distance": float(dist)
            })
        return results

    def query_bm25(self, query, k=5):
        scores = self.bm25_model.get_scores(word_tokenize(query.lower()))
        ranked_indices = np.argsort(scores)[::-1][:k]
        results = []
        for idx in ranked_indices:
            results.append({
                "chunk_id": self.chunk_ids[idx],
                "text": self.documents[idx]["text"],
                "distance": float(scores[idx])
            })
        return results

    def hybrid_query(self, query, k=5):
        bm25_results = self.query_bm25(query, k)
        faiss_results = self.query_faiss(query, k)
        combined = bm25_results + faiss_results
        seen = set()
        hybrid = []
        for res in combined:
            if res["text"] not in seen:
                seen.add(res["text"])
                hybrid.append(res)
            if len(hybrid) >= k:
                break
        return hybrid
