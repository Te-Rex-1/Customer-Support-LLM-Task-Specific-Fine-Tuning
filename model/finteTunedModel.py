import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import os
from dotenv import load_dotenv



class CustomLLMEngine:
    def __init__(self):
        load_dotenv()
        print("Initializing AI Engine... (This may take a minute)")
        self.model_id = "Te-REx/Qwen-1.5B-Customer-Support"

        # 4-bit config strictly for your 4GB VRAM
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        # Load the model directly into GPU memory
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        print("Engine ready. Model loaded into VRAM.")

    def generate_reply(self, user_query: str) -> str:
        """Takes a raw user string and returns the AI's string response."""
        messages = [
            {"role": "system", "content": "You are a professional support agent for an E-commerce brand."},
            {"role": "user", "content": user_query}
        ]

        # Formatting and Tensor conversion
        text_input = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text_input], return_tensors="pt").to(self.model.device)

        # Execution
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )

        # Slice out the prompt to return only the new answer
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]


ai_engine = CustomLLMEngine()