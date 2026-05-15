from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from safetensors.torch import load_file
from transformers import AutoTokenizer

REPO_ROOT = Path("/work/training/minimoe/MiniMoE")

# Add the repo root to path so tokcleanse is importable
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tokcleanse.odin_moe import OdinMOEForCausalLM
from tokcleanse.odin_moe_config import OdinMoETextConfig

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--port", type=int, default=8000)
parser.add_argument("--microbatch-max-size", type=int, default=8)
parser.add_argument("--microbatch-timeout-ms", type=float, default=10.0)
args = parser.parse_args()

path_to_checkpoint = args.checkpoint

app = FastAPI()

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(path_to_checkpoint)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

config = OdinMoETextConfig.from_pretrained(path_to_checkpoint)

model = OdinMOEForCausalLM(config)
model = model.to(dtype=torch.bfloat16, device="cuda")
state_dict = load_file(f"{path_to_checkpoint}/model.safetensors", device="cuda")
model.load_state_dict(state_dict, strict=False)
model.tie_weights()
print(next(model.parameters()).device)
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


@dataclass
class QueuedRequest:
    request: ChatRequest
    future: asyncio.Future[dict[str, Any]]


request_queue: asyncio.Queue[QueuedRequest] = asyncio.Queue()


def _generation_key(request: ChatRequest) -> tuple[int, float]:
    return request.max_tokens, request.temperature


async def _microbatch_worker() -> None:
    while True:
        first = await request_queue.get()
        batch = [first]

        deadline = asyncio.get_running_loop().time() + args.microbatch_timeout_ms / 1000.0
        while len(batch) < args.microbatch_max_size:
            timeout = deadline - asyncio.get_running_loop().time()
            if timeout <= 0:
                break
            try:
                batch.append(await asyncio.wait_for(request_queue.get(), timeout=timeout))
            except TimeoutError:
                break

        groups: dict[tuple[int, float], list[QueuedRequest]] = {}
        for item in batch:
            groups.setdefault(_generation_key(item.request), []).append(item)

        for group in groups.values():
            try:
                responses = await asyncio.to_thread(_run_generation_batch, group)
            except Exception as exc:
                for item in group:
                    if not item.future.done():
                        item.future.set_exception(exc)
            else:
                for item, response in zip(group, responses):
                    if not item.future.done():
                        item.future.set_result(response)


def _run_generation_batch(batch: list[QueuedRequest]) -> list[dict[str, Any]]:
    requests = [item.request for item in batch]
    chat_batches = [
        [{"role": message.role, "content": message.content} for message in request.messages]
        for request in requests
    ]

    inputs = tokenizer.apply_chat_template(
        chat_batches,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
        padding=True,
    ).to(model.device)

    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": requests[0].max_tokens,
        "do_sample": requests[0].temperature > 0,
    }
    if requests[0].temperature > 0:
        generation_kwargs["temperature"] = requests[0].temperature

    with torch.no_grad():
        outputs = model.generate(**inputs, **generation_kwargs)

    prompt_len = inputs["input_ids"].shape[1]
    responses = []
    for index, request in enumerate(requests):
        text = tokenizer.decode(outputs[index][prompt_len:], skip_special_tokens=True)
        responses.append(
            {
                "id": f"local-{index}",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": "stop",
                    }
                ],
                "model": request.model,
            }
        )
    return responses


@app.on_event("startup")
async def startup() -> None:
    app.state.microbatch_worker = asyncio.create_task(_microbatch_worker())


@app.on_event("shutdown")
async def shutdown() -> None:
    app.state.microbatch_worker.cancel()
    try:
        await app.state.microbatch_worker
    except asyncio.CancelledError:
        pass


@app.post("/v1/chat/completions")
async def chat(request: ChatRequest):
    future = asyncio.get_running_loop().create_future()
    await request_queue.put(QueuedRequest(request=request, future=future))
    return await future


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=args.port)
