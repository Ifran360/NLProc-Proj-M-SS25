
#  Semantic QA Pipeline with FAISS & Transformers

This project implements a simple yet powerful question-answering (QA) pipeline using:
- **FAISS** for fast semantic document retrieval
- **SentenceTransformers** for embeddings
- **Transformers** (HuggingFace) for QA generation
- Text documents as the knowledge base

---

## 📁 Project Structure

```
NLProc-Proj-M-SS25/
baseline/
├── data/                    # Text documents used as context
│   ├── Alice's Adventures in Wonderland.txt
│   ├── Harry Potter and the Sorcerer's Stone.txt
│   └── ... (other .txt files)
├── generator/
│   └── generator.py         # (Optional) Transformer-based QA generation
├── retriever/
│   └── retriever.py         # Handles semantic search & FAISS
├── retriever_index/         # Saved FAISS index + metadata (auto-created)
│   ├── doc_ids.txt
│   ├── faiss.index
│   └── id_to_doc.json
├── pipeline.py              # Main entry point to run the retriever
├── test_retriever.py        # Unit tests for retriever
└── README.md                # This file

```

---

## 🚀 How It Works

### 1. Load and Chunk Documents
- Loads `.txt` files from `/data/`
- Splits them into manageable chunks

### 2. Create Embeddings
- Uses `all-MiniLM-L6-v2` from SentenceTransformers
- Stores chunks and their vectors in a FAISS index

### 3. Answer Questions
- Retrieves top-k relevant chunks
- Feeds context into `deepset/roberta-base-squad2`
- Returns the generated answer

---

## ⚙️ How to Run

### 🔹 Run the QA system

```bash
cd baseline
python pipeline.py
```

You’ll see:
```bash
Enter your question (or 'exit' to quit):
> Who founded Hogwarts?
Answer: Godric Gryffindor, Helga Hufflepuff, Rowena Ravenclaw, and Salazar Slytherin
```

---

## 🧪 Run Tests

```bash
python test_retriever.py
```

You should see output like:

```
✔ Chunking test passed.
✔ Query relevance test passed.
✔ Save/load test passed.
```

---

## 🛠 Requirements

```bash
pip install faiss-cpu sentence-transformers transformers torch

```

---

## 📌 Example

```python
from pipeline.pipeline import QAPipeline

qa = QAPipeline()
print(qa.answer("What is the Hundred Acre Wood?"))
```

## 🧑‍💻 Authors

Developed for the NLP Project (SS2025) at University of Bamberg by Team "Triple Trouble".
