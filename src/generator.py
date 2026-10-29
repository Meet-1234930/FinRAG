"""Open-source LLM wrapper (4-bit quantized on GPU)."""
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

DEFAULT_MODEL = os.environ.get("FINRAG_MODEL", "Qwen/Qwen2.5-3B-Instruct")


class LLM:
    def __init__(self, model_name: str = None, load_4bit: bool = True):
        name = model_name or DEFAULT_MODEL
        self.tok = AutoTokenizer.from_pretrained(name)
        kwargs = {}
        if torch.cuda.is_available():
            kwargs["device_map"] = "auto"
            if load_4bit:
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                )
            else:
                kwargs["torch_dtype"] = torch.float16
        else:
            print("WARNING: no GPU found, running on CPU (slow). Use a Colab T4 runtime.")
        self.model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
        self.model.eval()
        self.name = name

    @torch.no_grad()
    def chat(self, messages, max_new_tokens: int = 300, temperature: float = 0.0) -> str:
        prompt = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tok(prompt, return_tensors="pt").to(self.model.device)
        gen_kwargs = dict(max_new_tokens=max_new_tokens, pad_token_id=self.tok.eos_token_id)
        if temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=temperature, top_p=0.9)
        else:
            gen_kwargs.update(do_sample=False)
        out = self.model.generate(**inputs, **gen_kwargs)
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        return self.tok.decode(new_tokens, skip_special_tokens=True).strip()
