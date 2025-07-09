import sys
import os
import contextlib
import re
import array
from transformers import pipeline
from llama_cpp import Llama

@contextlib.contextmanager
def suppress_stdout_stderr():
    with open(os.devnull, "w") as devnull:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = devnull, devnull
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr

def parse_mcq_input(user_input):
    pattern = re.compile(
        r"^(.*?)[\s]*a[.)\s]+(.*?)\s+b[.)\s]+(.*?)\s+c[.)\s]+(.*?)\s+d[.)\s]+(.*)",
        re.IGNORECASE | re.DOTALL
    )
    match = pattern.match(user_input.strip())
    if not match:
        raise ValueError("Could not parse MCQ input. Use format: Question? a. opt1 b. opt2 c. opt3 d. opt4")
    
    question, a, b, c, d = match.groups()
    return question.strip(), {
        "A": a.strip().capitalize(),
        "B": b.strip().capitalize(),
        "C": c.strip().capitalize(),
        "D": d.strip().capitalize()
    }

def is_answer_grounded(answer, context_chunks, threshold=0.2):
    answer_words = set(re.findall(r"\w+", answer.lower()))
    context_text = " ".join(context_chunks).lower()
    context_words = set(re.findall(r"\w+", context_text))

    if not answer_words:
        return False

    overlap = answer_words & context_words
    overlap_ratio = len(overlap) / len(answer_words)
    return overlap_ratio >= threshold

class Generator:
    def __init__(self):
        self.t5_pipeline = pipeline(
            "text2text-generation",
            model="google/flan-t5-large",
            device=-1
        )

        with suppress_stdout_stderr():
            self.llm = Llama(
                model_path="D:/Softwares/LLAMA/TheBloke/CapybaraHermes-2.5-Mistral-7B-GGUF/capybarahermes-2.5-mistral-7b.Q4_K_S.gguf",
                n_ctx=2048,
                n_threads=4
            )

    def build_prompt(self, context_chunks, question, task_type):
        context_text = " ".join(context_chunks)
        if task_type == "qa":
            return (
                f"### Instruction:\n"
                f"Given the following context, precisely answer the question.\n\n"
                f"### Context:\n{context_text}\n\n"
                f"### Question:\n{question}\n\n"
                f"### Answer:\n"
            )

        elif task_type == "mcq":
            q_text, options = parse_mcq_input(question)
            options_text = "\n".join([f"{k}. {v}" for k, v in options.items()])
            return (
                f"You are a helpful assistant. Choose the correct option (A, B, C, or D).\n\n"
                f"Question: {q_text}\n\nOptions:\n{options_text}\n\nAnswer:"
            )

        elif task_type == "summarize":
            return f"Summarize this:\n{context_text}"
        else:
            return context_text

    def generate_answer(self, prompt, task_type, context_chunks):
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Prompt must be a non-empty string.")

        fallback_response = "I am unable to answer this question. Do you want to ask anything else?"

        if task_type == "summarize":
            result = self.t5_pipeline(prompt, max_length=256, truncation=True)
            return result[0].get('generated_text') or str(result[0])

        elif task_type in {"qa", "mcq"}:
            prompt_tokens_list = self.llm.tokenize(prompt.encode("utf-8"))
            prompt_tokens = list(prompt_tokens_list)

            max_tokens = 1024 if task_type == "mcq" else 512
            if len(prompt_tokens) > max_tokens:
                prompt_tokens = prompt_tokens_list[-max_tokens:]

            with suppress_stdout_stderr():
                output = self.llm.create_completion(prompt_tokens, max_tokens=200, stop=["</s>"])
            
            answer = output['choices'][0]['text'].strip()

            if not answer or len(answer) < 2 or answer in {".", "..."}:
                return fallback_response

            answer_lower = answer.lower()
            unhelpful = {
                "i don't know", "i cannot answer", "i don't understand",
                "sorry", "unable", "....................."
            }
            if any(bad in answer_lower for bad in unhelpful):
                return fallback_response

            if task_type == "qa" and not is_answer_grounded(answer, context_chunks):
                return fallback_response

            if task_type == "mcq":
                for option in ["A", "B", "C", "D"]:
                    if answer.strip().upper().startswith(option):
                        return option

            return answer

        else:
            raise ValueError(f"Unsupported task_type: {task_type}")

    def summarize_chunks(self, context_chunks, question):
        prompt = self.build_prompt(context_chunks, question, task_type="summarize")
        return self.generate_answer(prompt, task_type="summarize", context_chunks=context_chunks)