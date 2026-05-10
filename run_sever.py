import os
import subprocess
from dotenv import load_dotenv

# 1. Load the .env file
load_dotenv()
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
# 2. Get the token (ensure it's in your .env as HF_TOKEN=...)
token = os.getenv("HF_TOKEN")
if not token:
    print("Error: HF_TOKEN not found in .env")
    exit(1)

# 3. Add token to the current process environment
os.environ["HF_TOKEN"] = token

# 4. Define the vllm command
command = [
    "vllm", "serve", "Te-REx/Qwen-1.5B-Customer-Support",
    "--quantization", "bitsandbytes",      # Shrinks weights to 4-bit
    "--load-format", "bitsandbytes",       # Triggers the on-the-fly conversion
    "--gpu-memory-utilization", "0.8",     # You now have plenty of room for this
    "--max-model-len", "2048",             # Restored to full context
    "--enforce-eager",                     # Keep this to prevent CUDA graph spikes
    "--served-model-name", "Qwen-1.5B-Customer-Support"
]
if __name__ == "__main__":
    # Nuke everything before start
    # subprocess.run(["pkill", "-9", "python"], check=False)
    print("Loading model with 4-bit BitsAndBytes quantization...")
    subprocess.run(command)