import os

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# initalize
load_dotenv()


hf_token = os.getenv("HF_TOKEN")

if not hf_token:
    raise ValueError("HF_TOKEN not found in environment variables.")


client = InferenceClient(api_key=hf_token)


def generate_output(message_payload):
    """
    Sends a message payload to the Hugging Face serverless API.
    """
    messages = [
        {"role": "system", "content": "You are a professional assistant."},
        {"role": "user", "content": message_payload}  # This is now an array
    ]
    try:
        completion = client.chat_completion(
            model="Qwen/Qwen2.5-1.5B-Instruct:featherless-ai",
            messages=messages,
            max_tokens=512,
            temperature=0.7  # Adds slight variance, useful for customer support tone
        )
        # Extracting the actual text content directly
        return completion.choices[0].message.content

    except Exception as e:
        return f"API Error: {str(e)}"



if __name__ == '__main__':
    messages="I ordered a laptop a week ago and the tracking still says 'Processing'. I need this for work by tomorrow. Cancel the order immediately and give me my money back!"
    result=generate_output(messages)
    print(result)