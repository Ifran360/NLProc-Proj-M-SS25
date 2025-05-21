import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from datetime import datetime

class Generator:
    """
    Generator class to build prompts from context and generate answers using a lightweight model.
    Uses flan-t5-base for generation.
    """

    def __init__(self, model_name: str = "google/flan-t5-base"):
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.model = self.model.to("cpu")  # Make sure it runs on CPU

    def build_prompt(self, context: str, question: str) -> str:
        return f"Given the context below, answer the question.\n\nContext:\n{context}\n\nQuestion:\n{question}"

    def generate_answer(self, prompt: str, max_length: int = 200) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to("cpu")
        outputs = self.model.generate(**inputs, max_length=max_length)
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer
