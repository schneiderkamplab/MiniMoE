# serve_model.py
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

app = FastAPI()

print("Loading model...")
processor = AutoProcessor.from_pretrained("google/gemma-4-E2B-it")
model = AutoModelForImageTextToText.from_pretrained(
    "google/gemma-4-E2B-it",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.eval()
print("Model ready!")

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: list[Message]
    max_tokens: int = 4096
    temperature: float = 0.0

@app.post("/v1/chat/completions")
async def chat(request: ChatRequest):
    # Format messages for Gemma 4
    chat_messages = [{"role": m.role, "content": m.content} for m in request.messages]
    
    inputs = processor.apply_chat_template(
        chat_messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=request.max_tokens,
            do_sample=request.temperature > 0,
            temperature=request.temperature if request.temperature > 0 else None,
        )

    # Decode only the newly generated tokens
    input_len = inputs["input_ids"].shape[1]
    text = processor.decode(outputs[0][input_len:], skip_special_tokens=True)

    return {
        "id": "local-1",
        "object": "chat.completion",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop"
        }],
        "model": request.model,
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)