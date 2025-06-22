import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

class Generator:
    """
    Lightweight answer generator using FLAN-T5.
    Builds prompts from multiple context chunks and supports question, summarization, or MCQ-style tasks.
    """

    def __init__(self, model_name: str = "google/flan-t5-large", device: str = "cpu"):
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
        self.device = device

    def build_prompt(self, context_chunks: list[str], question: str = "", task_type: str = "qa") -> str:
        """
        Build a task-specific prompt for QA, summarization, or MCQ generation.
        """
        context = "\n---\n".join(context_chunks)

        if task_type == "qa":
            return f"Given the context below, answer the question.\n\nContext:\n{context}\n\nQuestion:\n{question}"
        elif task_type == "summarize":
            return (
                "Write a well-structured summary based on the following context. Focus on main ideas, key terms, and structure:\n\n"
                f"{context}"
            )
        elif task_type == "mcq":
            return f"Given the context below, answer the question concisely with only the correct option letter (A, B, C, or D).\n\nContext:\n{context}\n\nQuestion:\n{question}"
        else:
            raise ValueError(f"Unsupported task_type: '{task_type}'. Use 'qa', 'summarize', or 'mcq'.")

    def generate_answer(self, prompt: str, max_length: int = 400) -> str:
        """
        Generate output text from a constructed prompt.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.device)
        outputs = self.model.generate(**inputs, max_length=max_length)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def summarize_chunks(self, context_chunks: list[str], max_length: int = 400) -> str:
        """
        Summarize a list of context chunks.
        """
        prompt = self.build_prompt(context_chunks, task_type="summarize")
        return self.generate_answer(prompt, max_length=max_length)
