import os
import json
import unittest
from retriever.retriever import Retriever
from generator.generator import Generator

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(BASE_DIR, "retriever_index")
TEST_INPUTS_PATH = os.path.join(BASE_DIR, "test_inputs.json")

class TestPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.retriever = Retriever()
        cls.retriever.load(INDEX_PATH)
        cls.generator = Generator()

        with open(TEST_INPUTS_PATH, "r", encoding="utf-8", errors="ignore") as f:
            cls.test_cases = json.load(f)

    def test_pipeline_end_to_end(self):
        for case in self.test_cases:
            with self.subTest(question=case["question"]):
                retrieved = self.retriever.query(case["question"], k=3)
                retrieved_texts = [chunk["text"] for chunk in retrieved]
                context = "\n".join(retrieved_texts)

                prompt = self.generator.build_prompt(context, case["question"])
                answer = self.generator.generate_answer(prompt)

                print(f"\nQ: {case['question']}")
                print(f"A: {answer}")
                print("Context:", context)

                # Test that answer exists
                self.assertTrue(answer.strip(), "No answer generated.")

                # Test grounding
                grounded = any(
                    word.lower() in context.lower()
                    for word in case["expected_answer"].split()
                )
                self.assertTrue(
                    grounded,
                    f"Answer not grounded in context: {answer}"
                )

if __name__ == "__main__":
    unittest.main()
