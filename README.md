# Semantic QA & Summarization Pipeline with FAISS & Transformers

This project implements a modular and extensible NLP pipeline capable of:

- Semantic Question Answering  
- Text Summarization  
- Multiple-Choice Question (MCQ) Generation  

**Key Technologies:**

- **FAISS** for fast vector-based document retrieval  
- **SentenceTransformers** for dense embeddings  
- **Transformers (`flan-t5-large`)** for generative output (QA, Summarization)  
- **llama.cpp** with **CapybaraHermes-2.5-Mistral-7B-GGUF** for MCQ answering  
- **BM25** for lexical retrieval  
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
  - `qa`: Answer questions using **Llama.cpp (CapybaraHermes-2.5-Mistral-7B-GGUF)**
  - `summarize`: Summarize document content  using **Flan-T5**
  - `mcq`: Generate multiple-choice questions  using **Llama.cpp (CapybaraHermes-2.5-Mistral-7B-GGUF)**
-  Relevant chunks are retrieved (hybrid: BM25 + FAISS) and fed into a prompt for the selected model  

---

## How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```
### 1. Install LLama(CapybaraHermes-2.5-Mistral-7B-GGUF) 
```bash
Download from https://huggingface.co/TheBloke and place in Local folder and then in Generator.py add the path like this:
self.llm = Llama(model_path="PATH",n_ctx=Contex_Length, n_threads=4)
```

### 3. Start the Interactive Pipeline
```bash
cd baseline
python pipeline.py
```

You will be prompted like this:
```
Task Type [qa/summarize/mcq]: summarize
Prompt: Summarize The Role of Physics in Everyday Life and Technolog
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
    "question": "The Role of Physics in Everyday Life and Technology",
    "reference": "Physics influences everyday life through technologies like smartphones, appliances, and transportation systems. It also underpins careers in healthcare, engineering, aviation, and more by explaining forces, motion, and energy use.",
    "generated": "From electronics to transportation and medical tools, physics is fundamental to daily technology and professional practices, impacting how we live and work."
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

Kaggel Dataset: https://www.kaggle.com/datasets/rohanthoma/ebook-pdfs
All files in `baseline/data/` are children's literature in `.txt` or `.pdf` format:

- College_Physics_2e-WEB_7Zesafu-23-1473
- ConceptsofBiology-WEB-19-602
- Introduction_to_Political_Science_-_WEB-19-549
- Secondary - 2018 - Class - 7 - English 7 BV PDF Web 
---

## Authors

Developed by **Team Triple Trouble**  
University of Bamberg — NLP Project (SS2025)


self.embeddings = self.model.encode([d["text"] for d in self.documents], show_progress_bar=False)
        print("Load complete.")