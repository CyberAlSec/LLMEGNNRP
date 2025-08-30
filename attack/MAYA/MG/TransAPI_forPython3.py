import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class Translator:
    def __init__(self, model_name: str, device: str = "cuda:0"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)


        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if "cuda" in device else torch.float32,
        )

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model.config.pad_token_id = self.tokenizer.eos_token_id
        self.model.to(device)

    def _build_prompt(self, src_lang: str, tgt_lang: str, text: str) -> str:

        return (
            f"The following is the task of translating from {src_lang} to {tgt_lang}:\n"
            f"Original text:{text}\n"
            f"Translation:"
        )

    def translate(self, from_lang: str, to_lang: str, inputs):
        if not inputs:
            return []
        outputs = []
        for text in inputs:
            prompt = self._build_prompt(from_lang, to_lang, text)
            encoded = self.tokenizer(prompt, return_tensors="pt")
            inputs_ids = encoded.input_ids.to(self.device)
            attention_mask = encoded.attention_mask.to(self.device)
            generated_ids = self.model.generate(
                inputs_ids,
                attention_mask = attention_mask,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                num_beams=1,
                pad_token_id = self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            gen = generated_ids[0][inputs_ids.shape[-1] :]
            translated = self.tokenizer.decode(gen, skip_special_tokens=True).strip()
            outputs.append(translated)
        return outputs


if __name__ == "__main__":

    tr = Translator(model_name="./llama3-7b/", device="cuda")
    from_lang = "en"
    to_lang   = "zh"
    inputs    = ["Hello, how are you?", "What is your name?"]
    results   = tr.translate(from_lang, to_lang, inputs)
    for src, tgt in zip(inputs, results):
        print(f"{src}  -->  {tgt}")
