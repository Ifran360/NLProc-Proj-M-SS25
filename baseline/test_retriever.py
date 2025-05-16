# test_retriever.py
import unittest
import os
import shutil
from retriever.retriever import Retriever

class TestRetriever(unittest.TestCase):

    def setUp(self):
        self.retriever = Retriever()
        self.docs = [
            {"id": "doc1", "text": "The little prince travels from planet to planet, learning life lessons."},
            {"id": "doc2", "text": "Sherlock Holmes solves mysteries using his keen observation and logic."}
        ]
        self.retriever.add_documents(self.docs)

    def test_query_relevance(self):
        result = self.retriever.query("detective solving a mystery", k=1)
        self.assertTrue(result[0]["chunk_id"].startswith("doc2_chunk_"))
        print(" Query relevance test passed.")

    def test_chunking(self):
        chunks = self.retriever._chunk_text("A" * 1200)
        self.assertEqual(len(chunks), 3)
        print(" Chunking test passed.")

    def test_saving_and_loading(self):
         
        os.makedirs(save_path, exist_ok=True)

        self.retriever.save(save_path)

        new_retriever = Retriever()
        new_retriever.load(save_path)

        result = new_retriever.query("planet journey", k=1)
        self.assertTrue(result[0]["chunk_id"].startswith("doc1_chunk_"))
        print(" Save/load test passed.")


if __name__ == "__main__":
    unittest.main(verbosity=0)
    # Clean up test directory
    shutil.rmtree(save_path)
