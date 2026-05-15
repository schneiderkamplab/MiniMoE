from safetensors.torch import load_file

import sys
from pathlib import Path

REPO_ROOT = Path("/work/training/minimoe/MiniMoE")

# Add the repo root to path so tokcleanse is importable
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tokcleanse.odin_moe import OdinMOEForCausalLM
from tokcleanse.odin_moe_config import OdinMoETextConfig
from tokcleanse.odin_moe_tokenizer import tokenizer_class_name_for_checkpoint
from transformers import AutoTokenizer
import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
args = parser.parse_args()

path_to_checkpoint = args.checkpoint

app = FastAPI()

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(path_to_checkpoint)
config = OdinMoETextConfig.from_pretrained(path_to_checkpoint)

model = OdinMOEForCausalLM(config)
model = model.to(dtype=torch.bfloat16, device="cuda")
state_dict = load_file(f"{path_to_checkpoint}/model.safetensors", device="cuda")
model.load_state_dict(state_dict, strict=False)
model.tie_weights()
# Check model device
print(next(model.parameters()).device)  # will say 'cpu' or 'cuda:0'
model.eval()
print("Model ready!")

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: list[Message]
    max_tokens: int = 4096
    temperature: float = 1.0

@app.post("/v1/chat/completions")
async def chat(request: ChatRequest):
    chat_messages = [{"role": m.role, "content": m.content} for m in request.messages]

    inputs = tokenizer.apply_chat_template(
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

    input_len = inputs["input_ids"].shape[1]
    text = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)

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