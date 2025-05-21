
# Semantic QA Pipeline with FAISS & Transformers

This project implements a modular and extensible semantic question-answering (QA) pipeline using:

- FAISS for fast vector-based document retrieval
- SentenceTransformers for dense embeddings
- Transformers (flan-t5-base) for generative answer generation
- .txt files as the document knowledge base

## Project Structure

```
NLProc-Proj-M-SS25/
└── baseline/
    ├── data/                    # Raw text documents (knowledge base)
    │   ├── Alice.txt
    │   ├── HarryPotter.txt
    │   └── ...
    ├── generator/
    │   └── generator.py         # QA answer generation (T5 model)
    ├── retriever/
    │   └── retriever.py         # FAISS-based semantic retriever
    ├── retriever_index/         # Auto-created directory for index/metadata
    │   ├── faiss.index
    │   └── metadata.pkl
    ├── pipeline.py              # Main script: loads documents, runs QA loop
    ├── test_inputs.json         # Known Q&A test pairs
    ├── test_pipeline.py         # End-to-end pipeline testing
    ├── test_retriever.py        # Unit tests for the retriever class
    └── README.md                # This file
```

## How It Works

### 1. Load and Chunk Documents
- Reads `.txt` files from `baseline/data/`
- Chunks long text into smaller pieces (default: 500 chars)

### 2. Embed & Index with FAISS
- Uses `all-MiniLM-L6-v2` to encode chunks
- Chunks + embeddings are stored in a FAISS index (`retriever_index/`)

### 3. Interactive QA Pipeline
- Accepts a question from user via terminal
- Retrieves top-k chunks based on semantic similarity
- Feeds context into `flan-t5-base` for answer generation

## How to Run

### Launch Interactive QA
```bash
cd baseline
python pipeline.py
```

Then enter:
```text
Your question: Where does Mary Lennox come from?
Generated answer: India
```

### Run Unit Tests

#### Retriever Logic
```bash
python test_retriever.py
```

#### Full Pipeline Test (using `test_inputs.json`)
```bash
python test_pipeline.py
```

Expected output:
```
Q: Where does Harry Potter study?
A: Hogwarts
Context: ...Harry goes to Hogwarts, a magical school...
```

## Installation Requirements

```bash
pip install reqirements.txt
```

## Example Usage in Code

```python
from pipeline import QAPipeline

qa = QAPipeline()
print(qa.answer("What is the Hundred Acre Wood?"))
```

## Sample Documents

All text files used in `baseline/data/` are plain `.txt` files containing paragraphs from children's books such as:

- Harry Potter
- The Secret Garden
- Matilda
- Winnie-the-Pooh

## Authors

Developed by Team Triple Trouble  
University of Bamberg — NLP Project (SS2025)
