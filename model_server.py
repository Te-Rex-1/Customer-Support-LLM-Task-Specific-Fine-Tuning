import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from dotenv import load_dotenv
import uvicorn


load_dotenv()
app = FastAPI(title="Raw GPU Engine")

print("Booting GPU Engine... (This happens ONCE)")
model_id = "Te-REx/Qwen-1.5B-Customer-Support"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, quantization_config=bnb_config, device_map="auto", trust_remote_code=True
)
print("Engine locked in VRAM.")


class RawPrompt(BaseModel):
    text: str


@app.post("/generate")
async def generate(prompt: RawPrompt):
    # This endpoint ONLY receives fully formatted text and generates a response
    inputs = tokenizer([prompt.text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **inputs, max_new_tokens=256, temperature=0.7, do_sample=True, pad_token_id=tokenizer.eos_token_id
    )

    # Strip the prompt out
    output_ids = generated_ids[0][len(inputs.input_ids[0]):]
    response = tokenizer.decode(output_ids, skip_special_tokens=True)

    return {"reply": response}


if __name__ == "__main__":

    # Runs on Port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)