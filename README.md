
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


## Recommended Python Version
This project requires Python 3.10.

Please ensure you are using Python 3.10, as other versions may cause compatibility issues.
### You can check your Python version with:
```bash
python --version
```
Download Python 3.10 here: https://www.python.org/downloads/release/python-3100/

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

### 4. Retriever
- Retriever class using SentenceTransformer (all-MiniLM-L6-v2)
- Chunks input text (500 characters)
- Encodes with embeddings
- Indexes using FAISS

### 5. Generator
- Generator class using Flan-T5-Base (small, CPU-friendly)
- Builds prompt with retrieved context + question
- Outputs answer

### 6. Integration Pipeline
- pipeline.py: Connects both modules
- Accepts live user input or test sets




## How to Run
### Installation Requirements

```bash
pip install reqirements.txt
```

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
In this test, you will be prompted to enter a question based on the preloaded context.
The system will retrieve relevant information and generate an answer along with the supporting context.


Expected output:
```
Q: Where does Harry Potter study?
A: Hogwarts
Context: ...Harry goes to Hogwarts, a magical school...
```


## Example Usage in Code

```python
from pipeline import QAPipeline

qa = QAPipeline()
print(qa.answer("What is the Hundred Acre Wood?"))
```

### Logs

All interactions during the execution of the pipeline are logged and stored in a JSON file for traceability and debugging purposes.

**Log File Location:**


**Log Format:**

Each log entry is a JSON object containing the following fields:

| Field             | Description                                                                 |
|------------------|-----------------------------------------------------------------------------|
| `timestamp`       | The date and time when the question was processed.                          |
| `group_id`        | An identifier for grouping related queries (default: `"default"`).          |
| `question`        | The user's input question.                                                  |
| `retrieved_chunks`| A list of context passages retrieved from the knowledge base.               |
| `prompt`          | The final prompt constructed and fed to the language model.                 |
| `generated_answer`| The answer returned by the model based on the retrieved context.            |


## Sample Documents

All text files used in `baseline/data/` are plain `.txt` files containing paragraphs from children's books such as:

- Harry Potter
- The Secret Garden
- Matilda
- Winnie-the-Pooh

## Authors

Developed by Team Triple Trouble  
University of Bamberg — NLP Project (SS2025)
