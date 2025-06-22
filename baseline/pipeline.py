import os
# Prevent multiprocessing-related crashes on macOS
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"]="1"
import json
from datetime import datetime
from retriever.retriever import Retriever
from generator.generator import Generator
from retriever.utils import extract_text_from_pdf  # Ensure this exists or define similarly

# Paths
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

        if not text.strip():
            continue

        doc_id = os.path.splitext(filename)[0].replace(" ", "_")
        documents.append({"id": doc_id, "text": text})

    return documents

def log_result(question, retrieved_chunks, prompt, answer, task_type="qa", group_id="Team Triple Trouble"):
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "group_id": group_id,
        "task_type": task_type,
        "question": question,
        "retrieved_chunks": retrieved_chunks,
        "prompt": prompt,
        "generated_answer": answer
    }
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")

def get_doc_id_from_chunk_id(chunk_id: str) -> str:
    return chunk_id.split("_chunk_")[0]

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

    print("\nReady! Type 'exit' anytime to quit.\n")

    while True:
        try:
            task_type = input("Task Type [qa/summarize/mcq]: ").strip().lower()
            if task_type == "exit":
                print("Exiting...")
                break
            if task_type not in {"qa", "summarize", "mcq"}:
                print("Invalid task type. Please enter 'qa', 'summarize', or 'mcq'.")
                continue

            query_text = input("Prompt: ").strip()
            if query_text.lower() == "exit":
                print("Exiting...")
                break
            if not query_text:
                print("Empty prompt. Please enter a valid input.")
                continue

            k = 6 if task_type == "summarize" else 5
            retrieved = retriever.query(query_text, k=k)

            relevant_doc_id = find_relevant_doc_id_in_prompt(query_text)
            if relevant_doc_id:
                retrieved = [
                    chunk for chunk in retrieved
                    if get_doc_id_from_chunk_id(chunk["chunk_id"]) == relevant_doc_id
                ][:3]
            else:
                retrieved = retrieved[:3]

            context_chunks = [chunk['text'] for chunk in retrieved]

            if not context_chunks:
                print("No relevant chunks found.")
                continue

            if task_type == "summarize":
                answer = generator.summarize_chunks(context_chunks)
                prompt = generator.build_prompt(context_chunks, task_type="summarize")
            else:
                prompt = generator.build_prompt(context_chunks, question=query_text, task_type=task_type)
                answer = generator.generate_answer(prompt)

            print("\nTop Retrieved Chunks:")
            for chunk in retrieved:
                print(f"- {chunk['chunk_id']} (Distance: {chunk['distance']:.4f})")
                print(f"  Text: {chunk['text']}\n")

            print(f"Generated Answer:\n{answer}\n")

            log_result(query_text, context_chunks, prompt, answer, task_type=task_type)

        except (KeyboardInterrupt, EOFError):
            print("\nExiting gracefully.")
            break

if __name__ == "__main__":
    main()
