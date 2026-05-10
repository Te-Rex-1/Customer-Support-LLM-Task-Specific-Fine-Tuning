import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="E-Commerce AI Backend")


class ChatRequest(BaseModel):
    user_query: str


class ChatResponse(BaseModel):
    response: str


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        # 1. Format the conversation using ChatML manually
        formatted_prompt = (
            "<|im_start|>system\nYou are a professional support agent for an E-commerce brand.<|im_end|>\n"
            f"<|im_start|>user\n{request.user_query}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        # 2. Send it to your persistent GPU Engine (Port 8000)
        # Using httpx for async requests
        async with httpx.AsyncClient() as client:
            engine_response = await client.post(
                "http://localhost:8000/generate",
                json={"text": formatted_prompt},
                timeout=60.0
            )

        if engine_response.status_code != 200:
            raise HTTPException(status_code=500, detail="Engine failed to respond.")

        # 3. Return the result
        bot_reply = engine_response.json().get("reply")
        return ChatResponse(response=bot_reply)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":

    # Runs on Port 3000. You can safely use --reload here!
    uvicorn.run("main:app", host="0.0.0.0", port=3000, reload=True)