import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json
from datetime import datetime
from retriever.retriever import Retriever
from generator.generator import Generator
from retriever.utils import extract_text_from_pdf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_DIR = os.path.join(BASE_DIR, "retriever_index")
DATA_DIR = os.path.join(BASE_DIR, "data")
LOG_PATH = os.path.join(BASE_DIR, "logs")
LOG_FILE = os.path.join(LOG_PATH, "log.jsonl")

def ensure_dirs():
    os.makedirs(INDEX_DIR, exist_ok=True)
    os.makedirs(LOG_PATH, exist_ok=True)

def load_documents(folder_path):
    documents = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if filename.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read().strip()
        elif filename.endswith(".pdf"):
            text = extract_text_from_pdf(file_path)
        else:
            continue
        if text.strip():
            doc_id = os.path.splitext(filename)[0].replace(" ", "_")
            documents.append({"id": doc_id, "text": text})
    return documents

def log_result(question, retrieved_chunks, prompt, answer, task_type="qa"):
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "task_type": task_type,
        "question": question,
        "retrieved_chunks": retrieved_chunks,
        "prompt": prompt,
        "generated_answer": answer
    }
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")

def main():
    ensure_dirs()
    retriever = Retriever()
    generator = Generator()

    index_exists = os.path.exists(os.path.join(INDEX_DIR, "faiss.index"))

    if index_exists:
        print("Loading existing FAISS index...")
        retriever.load(INDEX_DIR)
    else:
        print("Index not found. Indexing documents from data/ ...")
        documents = load_documents(DATA_DIR)
        retriever.add_documents(documents)
        retriever.save(INDEX_DIR)

    known_doc_ids = set(cid.split("_chunk_")[0] for cid in retriever.chunk_ids)

    def find_relevant_doc_id_in_prompt(prompt: str) -> str:
        for doc_id in known_doc_ids:
            if doc_id.lower() in prompt.lower():
                return doc_id
        return None

    print("\nReady! Type 'exit' anytime.\n")

    while True:
        try:
            task_type = input("Task Type [qa/summarize/mcq]: ").strip().lower()
            if task_type == "exit": break
            if task_type not in {"qa", "summarize", "mcq"}:
                print("Invalid. Choose qa, summarize, or mcq."); continue

            query_text = input("Prompt: ").strip()
            if query_text.lower() == "exit": break
            if not query_text:
                print("Empty prompt. Try again."); continue

            k = 20 if task_type == "summarize" else 15
            retrieved = retriever.hybrid_query(query_text, k=k)[:10]
            context_chunks = [chunk["text"] for chunk in retrieved]

            if not context_chunks:
                print("No relevant chunks found."); continue

            if task_type == "summarize":
                prompt = generator.build_prompt(context_chunks,question=query_text, task_type="summarize")
                answer = generator.summarize_chunks(context_chunks,question=query_text)
            else:
                prompt = generator.build_prompt(context_chunks, question=query_text, task_type=task_type)
                answer = generator.generate_answer(prompt,task_type)

            print("\nTop Retrieved Chunks:")
            for chunk in retrieved:
                print(f"- {chunk['chunk_id']} (Score: {chunk['distance']:.4f})\n  {chunk['text'][:200]}...\n")
            print(f"\nGenerated Answer:\n{answer}\n")

            log_result(query_text, context_chunks, prompt, answer, task_type)

        except (KeyboardInterrupt, EOFError):
            print("\nExiting gracefully."); break

if __name__ == "__main__":
    main()