import os
import json
from datetime import datetime
from retriever.retriever import Retriever
from generator.generator import Generator

# Paths (relative to this file's directory)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_DIR = os.path.join(BASE_DIR, "retriever_index")
DATA_DIR = os.path.join(BASE_DIR, "data")
LOG_PATH = os.path.join(BASE_DIR, "logs")
LOG_FILE = os.path.join(LOG_PATH, "log.jsonl")

def ensure_dirs():
    os.makedirs(INDEX_DIR, exist_ok=True)
    os.makedirs(LOG_PATH, exist_ok=True)

def load_text_files_from_folder(folder_path):
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read().strip()
                doc_id = os.path.splitext(filename)[0].replace(" ", "_")
                documents.append({"id": doc_id, "text": text})
    return documents

def log_result(question, retrieved_chunks, prompt, answer, group_id="Team Triple Trouble"):
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "group_id": group_id,
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
    index_exists = os.path.exists(os.path.join(INDEX_DIR, "faiss.index"))

    if index_exists:
        print("Loading existing index...")
        retriever.load(INDEX_DIR)
    else:
        print("Index not found. Indexing documents from data/ ...")
        data_path = os.path.join(os.path.dirname(__file__), "data")
        documents = load_text_files_from_folder(data_path)
        retriever.add_documents(documents)
        retriever.save(INDEX_DIR)

    generator = Generator()

    print("\nReady! Type question (or type 'exit' to quit):\n")
    while True:
        query_text = input("Prompt: ").strip()
        if query_text.lower() == "exit":
            print("Exiting...")
            break

        retrieved = retriever.query(query_text, k=3)
        retrieved_texts = [chunk['text'] for chunk in retrieved]

        context = "\n".join(retrieved_texts)
        prompt = generator.build_prompt(context, query_text)
        answer = generator.generate_answer(prompt)

        print("\nTop Retrieved Chunks:")
        for chunk in retrieved:
            print(f"- {chunk['chunk_id']} (Distance: {chunk['distance']:.4f})")
            print(f"  Text: {chunk['text']}\n")

        print(f"Generated Answer:\n{answer}\n")

        # Logging
        log_result(query_text, retrieved_texts, prompt, answer)

if __name__ == "__main__":
    main()
