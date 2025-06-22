import os
import re
import pickle
from pathlib import Path
from typing import List, Dict, Optional

import faiss
import fitz  # PyMuPDF
try:
    import pdfplumber
except ImportError:
    pdfplumber = None

import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline


class Retriever:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", chunk_size: int = 500, overlap: int = 50):
        self.model = SentenceTransformer(model_name)
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.index = None
        self.chunks: List[str] = []
        self.chunk_ids: List[str] = []

        # Summarization model
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    # ------------------------------
    # File Processing
    # ------------------------------
    def _extract_txt(self, path: Path) -> str:
        with open(path, encoding="utf-8", errors="ignore") as f:
            return f.read()

    def _extract_pdf(self, path: Path) -> str:
        try:
            doc = fitz.open(path)
            text = "\n".join(page.get_text() for page in doc)
            doc.close()
        except Exception:
            if pdfplumber is None:
                raise RuntimeError("pdfplumber not installed and PyMuPDF failed on this PDF.")
            with pdfplumber.open(path) as pdf:
                text = "\n".join(p.extract_text() or "" for p in pdf.pages)
        return text

    def add_files_from_path(self, folder: str):
        docs: List[Dict[str, str]] = []
        for file in Path(folder).iterdir():
            if file.suffix.lower() == ".txt":
                raw = self._extract_txt(file)
            elif file.suffix.lower() == ".pdf":
                raw = self._extract_pdf(file)
            else:
                continue
            if raw.strip():
                docs.append({"id": file.stem.replace(" ", "_"), "text": raw})
        if not docs:
            raise ValueError("No .txt or .pdf files found in provided folder.")
        self.add_documents(docs)

    # ------------------------------
    # Indexing and Chunking
    # ------------------------------
    def _clean(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text)
        text = text.replace("\u00ad", "")
        return text.strip()

    def _chunk_text(self, text: str) -> List[str]:
        text = self._clean(text)
        chunks = []
        i = 0
        while i < len(text):
            chunk = text[i: i + self.chunk_size]
            chunks.append(chunk)
            i += self.chunk_size - self.overlap
        return chunks

    def add_documents(self, documents: List[Dict[str, str]]):
        all_chunks, all_ids = [], []
        for doc in documents:
            doc_chunks = self._chunk_text(doc["text"])
            ids = [f"{doc['id']}_chunk_{idx}" for idx in range(len(doc_chunks))]
            all_chunks.extend(doc_chunks)
            all_ids.extend(ids)

        embeddings = self.model.encode(all_chunks, show_progress_bar=False).astype("float32")

        if self.index is None:
            self.index = faiss.IndexFlatL2(embeddings.shape[1])
            self.index.add(embeddings)
        else:
            self.index.add(embeddings)

        self.chunks.extend(all_chunks)
        self.chunk_ids.extend(all_ids)

    # ------------------------------
    # Retrieval
    # ------------------------------
    def query(self, text: str, k: int = 3) -> List[Dict[str, str]]:
        if self.index is None:
            raise ValueError("Index empty — add or load documents first.")
        emb = self.model.encode([text]).astype("float32")
        distances, indices = self.index.search(emb, k)
        results, seen = [], set()
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1 or dist > 1e6:
                continue
            cid = self.chunk_ids[idx]
            if cid in seen:
                continue
            seen.add(cid)
            results.append({
                "chunk_id": cid,
                "text": self.chunks[idx],
                "distance": float(dist)
            })
        return results

    def summarize_query(self, query: str, k: int = 5, max_tokens: int = 1024) -> str:
        top_chunks = self.query(query, k=k)
        if not top_chunks:
            return "No relevant content found."

        combined_text = " ".join([item["text"] for item in top_chunks])
        combined_text = self._clean(combined_text)[:max_tokens]

        if len(combined_text) < 100:
            return combined_text

        try:
            summary = self.summarizer(
                combined_text,
                max_length=180,
                min_length=60,
                do_sample=False
            )
            return summary[0]["summary_text"]
        except Exception as e:
            return f"Summarization failed: {e}"

    # ------------------------------
    # Persistence
    # ------------------------------
    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(path, "faiss.index"))
        with open(os.path.join(path, "metadata.pkl"), "wb") as f:
            pickle.dump({
                "chunks": self.chunks,
                "chunk_ids": self.chunk_ids
            }, f)

    def load(self, path: str):
        idx_path = Path(path) / "faiss.index"
        meta_path = Path(path) / "metadata.pkl"
        if not idx_path.exists() or not meta_path.exists():
            raise FileNotFoundError("Index or metadata not found.")
        self.index = faiss.read_index(str(idx_path))
        with open(meta_path, "rb") as f:
            data = pickle.load(f)
        self.chunks, self.chunk_ids = data["chunks"], data["chunk_ids"]
