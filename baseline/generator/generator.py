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
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


def parse_mcq_input(user_input):
        # Extract question and options using regex
        pattern = re.compile(r"^(.*?)[\s]*a[.)\s]+(.*?)\s+b[.)\s]+(.*?)\s+c[.)\s]+(.*?)\s+d[.)\s]+(.*)", re.IGNORECASE)
        match = pattern.match(user_input)
        
        if not match:
            raise ValueError("Could not parse MCQ input. Please use format: Question? a. opt1 b. opt2 c. opt3 d. opt4")

        question, a, b, c, d = match.groups()
        return question.strip(), {
            "A": a.strip().capitalize(),
            "B": b.strip().capitalize(),
            "C": c.strip().capitalize(),
            "D": d.strip().capitalize()
        }

class Generator:
    def __init__(self):
        # flan-t5 for QA and summarization
        self.t5_pipeline = pipeline("text2text-generation", model="google/flan-t5-large", device=-1)
        # llama.cpp for MCQs
        with suppress_stdout_stderr():
            self.llm = Llama(model_path="D:/Softwares/LLAMA/TheBloke/CapybaraHermes-2.5-Mistral-7B-GGUF/capybarahermes-2.5-mistral-7b.Q4_K_S.gguf",n_ctx=2048, n_threads=4)

    

    def build_prompt(self, context_chunks, question, task_type):
        context_text = " ".join(context_chunks)
        if task_type == "qa":
             return (
                f"### Instruction:\n"
                f"Given the following context, answer the question.\n\n"
                f"### Context:\n{context_text}\n\n"
                f"### Question:\n{question}\n\n"
                f"### Answer:\n"
            )

        elif task_type == "mcq":
            q_text, options = parse_mcq_input(question)
            options_text = "\n".join([f"{k}. {v}" for k, v in options.items()])
            return (
                f"You are a helpful assistant. Choose the correct option (A, B, C, or D).\n\n"
                f"Question: {q_text}\n\nOptions:\n{options_text}"
            )
        elif task_type == "summarize":
            return f"Summarize: {context_text}"
        else:
            return context_text

    def generate_answer(self, prompt, task_type):
        # Defensive checks
        if not isinstance(prompt, str):
            raise TypeError(f"Prompt must be a string, got {type(prompt)}")
        if not prompt.strip():
            raise ValueError("Prompt is empty.")

        if task_type == "summarize":
            result = self.t5_pipeline(prompt, max_length=256, truncation=True)
            return result[0].get('generated_text') or result[0].get('summary_text') or str(result[0])

        elif task_type == "qa":
            # Tokenize prompt
            prompt_tokens_list = self.llm.tokenize(prompt.encode("utf-8"))
            prompt_tokens = array.array('i', prompt_tokens_list)

            # Truncate if too long
            max_context = 512
            if len(prompt_tokens) > max_context:
                prompt_tokens = prompt_tokens_list[-max_context:]
                prompt = self.llm.detokenize(prompt_tokens).decode("utf-8") 

            with suppress_stdout_stderr():
                output = self.llm(prompt, max_tokens=100, stop=["\n", "</s>"])

            return output['choices'][0]['text'].strip()


        elif task_type == "mcq":
            prompt_tokens_list = self.llm.tokenize(prompt.encode("utf-8"))
            prompt_tokens = array.array('i', prompt_tokens_list)

            if len(prompt_tokens) > 1024:
                prompt_tokens = prompt_tokens_list[-1024:]
                prompt = self.llm.detokenize(prompt_tokens).decode("utf-8")

            with suppress_stdout_stderr():
                output = self.llm(prompt)

            raw_output = output['choices'][0]['text'].strip()

            for letter in ['A', 'B', 'C', 'D']:
                if raw_output.upper().startswith(letter):
                    return letter
            return raw_output

        else:
            raise ValueError(f"Unsupported task_type: {task_type}")



    def summarize_chunks(self, context_chunks, question):
        prompt = self.build_prompt(context_chunks,question, task_type="summarize")
        return self.generate_answer(prompt, task_type="summarize")