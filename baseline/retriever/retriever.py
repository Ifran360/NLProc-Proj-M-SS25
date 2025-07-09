import os
import pickle
import faiss
import numpy as np
from nltk.tokenize import word_tokenize, sent_tokenize
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

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

    def chunk_text(self, text, chunk_size=100, stride=50):
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_len = 0

        for sentence in sentences:
            sentence_len = len(sentence.split())
            if current_len + sentence_len > chunk_size:
                chunk = ' '.join(current_chunk)
                if len(chunk.split()) > 3:
                    chunks.append(chunk)
                # Slide window
                current_chunk = current_chunk[stride:]
                current_len = sum(len(s.split()) for s in current_chunk)

            current_chunk.append(sentence)
            current_len += sentence_len

        # Add remaining
        if current_chunk:
            chunk = ' '.join(current_chunk)
            if len(chunk.split()) > 3:
                chunks.append(chunk)
        return chunks

    def add_documents(self, documents, chunk_size=100, stride=50):
        self.documents = []
        self.chunk_ids = []

        for doc in documents:
            text = doc["text"]
            doc_id = doc["id"]
            chunks = self.chunk_text(text, chunk_size, stride)

            for idx, chunk in enumerate(chunks):
                clean = self.clean_chunk(chunk)
                if not clean:
                    continue
                chunk_id = f"{doc_id}_chunk{idx}"
                self.chunk_ids.append(chunk_id)
                self.documents.append({"id": chunk_id, "text": clean})

        texts = [doc["text"] for doc in self.documents]

        print("Encoding embeddings...")
        self.embeddings = self.model.encode(texts, show_progress_bar=True, normalize_embeddings=True).astype('float32')

        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # Cosine similarity
        self.index.add(self.embeddings)

        print("Initializing BM25...")
        self.tokenized_corpus = [word_tokenize(doc["text"].lower()) for doc in self.documents]
        self.bm25_model = BM25Okapi(self.tokenized_corpus)

    def save(self, index_dir):
        os.makedirs(index_dir, exist_ok=True)
        faiss.write_index(self.index, os.path.join(index_dir, "faiss.index"))
        metadata = {
            "chunk_ids": self.chunk_ids,
            "texts": [doc["text"] for doc in self.documents],
            "embeddings": self.embeddings.tolist()  # Save embeddings
        }
        with open(os.path.join(index_dir, "metadata.pkl"), "wb") as f:
            pickle.dump(metadata, f)


    def load(self, index_dir):
        print("Loading FAISS index...")
        self.index = faiss.read_index(os.path.join(index_dir, "faiss.index"))
        print("Loading metadata...")
        with open(os.path.join(index_dir, "metadata.pkl"), "rb") as f:
            metadata = pickle.load(f)

        self.chunk_ids = metadata["chunk_ids"]
        texts = metadata["texts"]
        self.documents = [{"id": cid, "text": txt} for cid, txt in zip(self.chunk_ids, texts)]

        print("Initializing BM25...")
        self.tokenized_corpus = [word_tokenize(txt.lower()) for txt in texts]
        self.bm25_model = BM25Okapi(self.tokenized_corpus)

        print("Encoding embeddings for FAISS...")
        self.embeddings = np.array(metadata.get("embeddings"), dtype='float32')  # Load embeddings

        print("Load complete.")

    def query_faiss(self, query, k=5):
        query_emb = self.model.encode([query], normalize_embeddings=True).astype('float32')
        similarities, indices = self.index.search(query_emb, k)
        results = []
        for idx, score in zip(indices[0], similarities[0]):
            results.append({
                "chunk_id": self.chunk_ids[idx],
                "text": self.documents[idx]["text"],
                "score": float(score)
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
                "score": float(scores[idx])
            })
        return results

    def hybrid_query(self, query, k=5, alpha=0.5, rerank=True):
        query_emb = self.model.encode([query], normalize_embeddings=True).astype('float32')

        # Get top-N from both
        faiss_sims, faiss_indices = self.index.search(query_emb, 20)
        faiss_scores = faiss_sims[0]
        faiss_idx = faiss_indices[0]

        bm25_raw_scores = self.bm25_model.get_scores(word_tokenize(query.lower()))
        bm25_idx = np.argsort(bm25_raw_scores)[::-1][:20]
        bm25_scores = bm25_raw_scores[bm25_idx]

        # Normalize both
        def normalize(arr):
            return (arr - np.min(arr)) / (np.max(arr) - np.min(arr) + 1e-9)

        norm_faiss = normalize(faiss_scores)
        norm_bm25 = normalize(bm25_scores)

        faiss_results = {
            self.chunk_ids[i]: {
                "text": self.documents[i]["text"],
                "score": alpha * norm_faiss[j]
            } for j, i in enumerate(faiss_idx)
        }

        for j, i in enumerate(bm25_idx):
            chunk_id = self.chunk_ids[i]
            score = (1 - alpha) * norm_bm25[j]
            if chunk_id in faiss_results:
                faiss_results[chunk_id]["score"] += score
            else:
                faiss_results[chunk_id] = {
                    "text": self.documents[i]["text"],
                    "score": score
                }

        # Rerank by semantic similarity
        if rerank:
            candidates = list(faiss_results.items())
            texts = [v["text"] for _, v in candidates]
            candidate_embs = self.model.encode(texts, normalize_embeddings=True).astype('float32')
            scores = np.dot(candidate_embs, query_emb.T).squeeze()
            for i, (chunk_id, val) in enumerate(candidates):
                faiss_results[chunk_id]["score"] += float(scores[i])  # rerank boost

        # Final top-k
        sorted_results = sorted(faiss_results.items(), key=lambda x: x[1]["score"], reverse=True)[:k]

        return [{
            "chunk_id": chunk_id,
            "text": val["text"],
            "score": val["score"]
        } for chunk_id, val in sorted_results]
