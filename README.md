# Semantic QA & Summarization Pipeline with FAISS & Transformers

This project implements a modular and extensible NLP pipeline capable of:

- Semantic Question Answering  
- Text Summarization  
- Multiple-Choice Question (MCQ) Generation  

Using:

- FAISS for fast vector-based document retrieval  
- SentenceTransformers for dense embeddings  
- Transformers (`flan-t5-large`) for generative output  
- `.txt` and `.pdf` documents as the knowledge base  

---

## Project Structure

```
NLProc-Proj-M-SS25/
├── baseline/
│   ├── data/
│   ├── generator/
│   ├── retriever/
│   ├── retriever_index/
│   ├── logs/
│   ├── pipeline.py
│   ├── test_inputs.json
│   ├── test_pipeline.py
│   ├── test_retriever.py
│   └── README.md
└── evaluation/
    ├── evaluation.py          # Evaluation script
    └── summaries.json         # Input file with generated vs reference summaries
```

---

## Python Requirements

- Python version: **3.10** recommended  
- To check:
```bash
python --version
```
- Download Python 3.10: https://www.python.org/downloads/release/python-3100/

---

## How It Works

### 1. Load & Chunk Documents
- Supports both `.txt` and `.pdf` files
- Chunks content into blocks of ~500 characters for embedding

### 2. Embed & Index with FAISS
- Embeds each chunk using `all-MiniLM-L6-v2`
- Stores vector embeddings and metadata with FAISS for fast retrieval

### 3. Multi-Task Interactive Pipeline
- On launch, users specify a task:
  - `qa`: Answer questions  
  - `summarize`: Summarize document content  
  - `mcq`: Generate multiple-choice questions  
- Relevant chunks are retrieved and fed into a prompt for `flan-t5-large`  

---

## How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the Interactive Pipeline
```bash
cd baseline
python pipeline.py
```

You will be prompted like this:
```
Task Type [qa/summarize/mcq]: summarize
Prompt: Summarize the story of Matilda
```

---

## Evaluation

You can evaluate the quality of generated summaries against reference summaries using multiple metrics:

- **Cosine Similarity** (TF-IDF)
- **BERTScore** (semantic similarity)
- **Word Overlap** (lexical overlap)

### Step-by-Step Instructions

1. Make sure your input JSON (`summaries.json`) looks like this:
```json
[
  {
    "question": "What is the Hundred Acre Wood?",
    "reference": "The Hundred Acre Wood is the fictional forest home of Winnie-the-Pooh.",
    "generated": "It is the place where Winnie-the-Pooh and his friends live."
  }
]
```

2. First-time users: Download `nltk` data before running:
```bash
python
```
Then in the Python shell:
```python
import nltk
nltk.download('punkt')
exit()
```

3. Run the evaluation script:
```bash
cd evaluation
python evaluation.py --input summaries.json --output results.json
```

4. After running, check the results:
```json
[
  {
    "id": "What is the Hundred Acre Wood?",
    "cosine_similarity": 0.45,
    "bertscore": 0.90,
    "word_overlap": 0.3
  }
]
```

---

## Run Unit Tests

### a. Test Retriever Logic
```bash
python test_retriever.py
```

### b. Test Full Pipeline with Predefined Inputs
```bash
python test_pipeline.py
```

---

## Code Usage Example

```python
from pipeline import main
main()
```

---

## Logging

All interactions are logged in JSON Lines format.

**Log File:**  
```
baseline/logs/log.jsonl
```

**Each log contains:**

| Field             | Description                                                       |
|------------------|-------------------------------------------------------------------|
| `timestamp`       | Time when the interaction occurred                                |
| `group_id`        | Optional identifier for batch tasks                               |
| `task_type`       | Type of task (qa, summarize, mcq)                                 |
| `question`        | User input or prompt                                              |
| `retrieved_chunks`| Top-k context chunks used as input                               |
| `prompt`          | Final prompt passed to the Flan-T5 model                          |
| `generated_answer`| Output from the model                                             |

---

## Sample Documents

All files in `baseline/data/` are children's literature in `.txt` or `.pdf` format:

- Matilda  
- The Wonderful Wizard of Oz  
- Winnie-the-Pooh  
- The Secret Garden  
- Harry Potter (sample excerpts)

---

## Authors

Developed by **Team Triple Trouble**  
University of Bamberg — NLP Project (SS2025)
