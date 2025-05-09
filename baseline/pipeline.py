from retriever.retriever import Retriever
import os

def main():

    # Sample documents
    documents = [
        {"id": "doc1", "text": "Winnie-the-Pooh is a gentle bear who loves honey and lives in the Hundred Acre Wood."},
        {"id": "doc2", "text": "Harry Potter goes to Hogwarts, a magical school for wizards and witches."}
    ]

    # Initialize and add documents
    retriever = Retriever()
    retriever.add_documents(documents)

    # Save index
    save_path = os.path.abspath("retriever_index")
    os.makedirs(save_path, exist_ok=True)
    retriever.save(save_path)

    print("\nClearing memory to simulate a fresh start...\n")

    # Reload index
    new_retriever = Retriever()
    new_retriever.load(save_path)

    # Run query
    query_text = "A magical school for wizards"
    results = new_retriever.query(query_text, k=2)

    print(f"\nTop results for: \"{query_text}\"\n")
    for res in results:
        print(f"- {res['chunk_id']} (Distance: {res['distance']:.4f})")
        print(f"  Text: {res['text']}\n")

if __name__ == "__main__":
    main()
