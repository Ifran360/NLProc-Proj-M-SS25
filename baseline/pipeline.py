from retriever.retreiver import Retriever
import os

if __name__ == "__main__":
    # Step 1: Create retriever and add documents
    retriever = Retriever()
    documents = [
        {"id": "doc1", "text": "Winnie-the-Pooh is a gentle bear who loves honey and lives in the Hundred Acre Wood."},
        {"id": "doc2", "text": "Harry Potter goes to Hogwarts, a magical school for wizards and witches."}
    ]
    retriever.add_documents(documents)

    # Step 2: Save index and metadata
    save_path = os.path.abspath("retriever_index")
    os.makedirs(save_path, exist_ok=True)
    retriever.save(save_path)

    print("\n🧹 Clearing memory to simulate a fresh start...\n")

    # Step 3: Load saved index into a new retriever instance
    new_retriever = Retriever()
    new_retriever.load(save_path)

    # ✅ Step 4: Restore chunk texts (IMPORTANT!)
    new_retriever.add_documents(documents, skip_indexing=True)

    # Step 5: Run a query
    query_text = "A magical school for wizards"
    results = new_retriever.query(query_text, k=2)

    # Step 6: Print results
    print(f"\n🔍 Top results for: \"{query_text}\"")
    for res in results:
        print(f"- {res['chunk_id']} (Distance: {res['distance']:.4f})")
        print(f"  Text: {res['text']}")
