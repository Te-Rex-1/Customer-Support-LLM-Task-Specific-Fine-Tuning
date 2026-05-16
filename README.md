# Customer-Support-LLM: Local vs. Cloud Comparison

A high-performance evaluation platform designed to provide side-by-side comparisons between **Cloud-Based Base LLMs** and **Fine-Tuned Local LLMs** optimized for e-commerce customer support tasks.

---

## 🚀 Project Overview

This project serves as a live benchmarking tool to demonstrate the efficacy of task-specific fine-tuning. It compares the raw reasoning capabilities of a cloud-hosted general model against a locally-hosted, 4-bit quantized model (`Te-REx/Qwen-1.5B-Customer-Support`) specifically trained on customer service interactions.

### Key Features
- **Parallel Inference**: Query both models simultaneously from a single interface.
- **Local GPU Optimization**: High-speed local inference using `bitsandbytes` 4-bit NF4 quantization.
- **Task-Specific Precision**: Evaluates specialized fine-tuning for professional e-commerce responses.
- **Unified Orchestration**: Single-script launch for the entire multi-service stack.

---

## 🛠️ Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/)
- **Backend Orchestration**: [FastAPI](https://fastapi.tiangolo.com/)
- **Inference Server**: [FastAPI](https://fastapi.tiangolo.com/) + [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
- **Quantization**: [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) (4-bit NF4)
- **Communication**: [httpx](https://www.python-httpx.org/) (Async) & [requests](https://requests.readthedocs.io/)
- **Cloud API**: [Hugging Face Inference API](https://huggingface.co/docs/api-inference/index)

---

## 🏗️ Architecture

The application utilizes a distributed tri-process architecture:

1.  **GPU Engine (Port 8000)**: Loads the model into VRAM and handles raw text generation.
2.  **Backend API (Port 3000)**: Manages prompt engineering (ChatML) and routes traffic.
3.  **Streamlit UI**: Provides the user interaction layer and side-by-side visualization.

---

## 🚦 Getting Started

### Prerequisites
- Python 3.10+
- NVIDIA GPU with 4GB+ VRAM (for local inference)
- CUDA Toolkit installed

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Te-Rex-1/Customer-Support-LLM-Task-Specific-Fine-Tuning.git
   cd Customer-Support-LLM
   ```

2. **Set up a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Configuration

Create a `.env` file in the root directory:
```env
HF_TOKEN=your_huggingface_access_token
```

---

## 💻 Running the Application

### The Easy Way (Automated)
Launch the entire stack with a single command:
```bash
./script/launch.sh
```

### The Manual Way (Individual Services)
If you need to debug individual components, run them in separate terminals:

1. **GPU Inference Engine**:
   ```bash
   python model/fine_tune_model_server.py
   ```
2. **Backend API**:
   ```bash
   python main.py
   ```
3. **Frontend UI**:
   ```bash
   streamlit run app.py
   ```

---

## 📁 Project Structure

- `app.py`: Streamlit frontend application.
- `main.py`: FastAPI orchestration backend.
- `model/`: Inference logic and model server implementations.
- `script/`: Operational scripts and launch automation.
- `GEMINI.md`: Comprehensive internal technical documentation.
- `Brainstromig /`: Future feature roadmap and development notes.

---

## 📜 License
[Specify License, e.g., MIT]

---
*Developed with a focus on efficient, localized AI for the future of customer support.*
