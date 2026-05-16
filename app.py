import streamlit as st
import requests
import time
import os
from model import BaseModel

# 1. Custom CSS for Blinking Animation
st.set_page_config(layout="wide")
st.markdown("""
    <style>
    @keyframes blink {
        0% { opacity: 1; }
        50% { opacity: 0; }
        100% { opacity: 1; }
    }
    .blinking-dot {
        height: 10px;
        width: 10px;
        background-color: #ff4b4b;
        border-radius: 50%;
        display: inline-block;
        animation: blink 1s infinite;
        margin-right: 8px;
    }
    .model-header {
        font-weight: bold;
        font-size: 20px;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Live Model Comparison: Base vs. Fine-Tuned")

# 2. Sidebar Configuration
st.sidebar.header("Cloud API Settings")


# 3. Layout: Two Columns for Live Difference
col1, col2 = st.columns(2)

with col1:
    st.markdown('<p class="model-header">☁️ Base Model (Cloud)</p>', unsafe_allow_html=True)
    base_container = st.empty()

with col2:
    st.markdown('<p class="model-header"> Fine-Tuned (Local GPU)</p>', unsafe_allow_html=True)
    ft_container = st.empty()

# 4. Shared Input
user_input = st.chat_input("Send a message to both models...")

if user_input:
    # Update UI with user message
    with base_container:
        st.info(f"User: {user_input}")
    with ft_container:
        st.info(f"User: {user_input}")

    # --- Processing Start ---

    # Use blinking indicators to show "Live" processing
    status_placeholder = st.empty()
    status_placeholder.markdown('<p><span class="blinking-dot"></span>Both models are thinking...</p>',
                                unsafe_allow_html=True)

    # 5. API Logic
    try:
        # LOCAL FINE-TUNED CALL
        ft_response = requests.post(
            "http://localhost:3000/chat",
            json={"user_query": user_input},
            timeout=60
        ).json().get("response", "Error fetching local response")

        # CLOUD BASE CALL

        prompt = f"<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"

        base_response=BaseModel.generate_output(prompt)

        # 6. Final UI Update
        status_placeholder.empty()

        with col1:
            st.success(f"**Base:**\n\n{base_response}")
        with col2:
            st.success(f"**Fine-Tuned:**\n\n{ft_response}")

    except Exception as e:
        st.error(f"Execution Error: {e}")