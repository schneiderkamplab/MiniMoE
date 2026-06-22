"""Convert JSONL data to chat format with optional reasoning."""

from __future__ import annotations

import asyncio
import json
import os
import random
from pathlib import Path
from typing import Any

import typer
from tqdm import tqdm

DEFAULT_INPUT_KEY = "input"
DEFAULT_OUTPUT_KEY = "output"
DEFAULT_SYSTEM_KEY = ""
DEFAULT_BATCH_SIZE = 40
DEFAULT_MODEL_NAME = "gpt-4o-mini"
DEFAULT_WITH_REASONING = False
DEFAULT_GEMMA4_INCLUDE_REASONING = False


def reservoir_sample_jsonl(
    infile: Path, n_samples: int, seed: int
) -> list[dict[str, Any]]:
    """Sample n random rows from a JSONL file without loading the full file."""
    rng = random.Random(seed)
    sample: list[dict[str, Any]] = []

    with infile.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            obj = json.loads(line)
            if len(sample) < n_samples:
                sample.append(obj)
            else:
                j = rng.randint(0, i)
                if j < n_samples:
                    sample[j] = obj
    return sample


def _extract_from_conversations(row: dict[str, Any]) -> dict[str, str] | None:
    """Extract system, input, output from the last valid human->gpt turn."""
    conversations = row.get("conversations")
    if not isinstance(conversations, list) or not conversations:
        return None

    assistant_idx = None
    for i in range(len(conversations) - 1, -1, -1):
        msg = conversations[i]
        if not isinstance(msg, dict):
            continue
        if msg.get("from") == "gpt" and str(msg.get("value", "")).strip():
            assistant_idx = i
            break

    if assistant_idx is None:
        return None

    user_idx = None
    for i in range(assistant_idx - 1, -1, -1):
        msg = conversations[i]
        if not isinstance(msg, dict):
            continue
        if msg.get("from") == "human" and str(msg.get("value", "")).strip():
            user_idx = i
            break

    if user_idx is None:
        return None

    system_prompt = ""
    for i in range(user_idx - 1, -1, -1):
        msg = conversations[i]
        if not isinstance(msg, dict):
            continue
        if msg.get("from") == "system" and str(msg.get("value", "")).strip():
            system_prompt = str(msg.get("value", "")).strip()
            break

    user_text = str(conversations[user_idx].get("value", "")).strip()
    assistant_text = str(conversations[assistant_idx].get("value", "")).strip()
    if not user_text or not assistant_text:
        return None

    return {"system": system_prompt, "input": user_text, "output": assistant_text}


def _extract_from_mapped_keys(
    row: dict[str, Any],
    input_key: str,
    output_key: str,
    system_key: str,
) -> dict[str, str] | None:
    """Extract instruction-style pair from flat datasets using mapped keys."""
    input_value = row.get(input_key)
    output_value = row.get(output_key)
    if input_value is None or output_value is None:
        return None

    user_text = str(input_value).strip()
    assistant_text = str(output_value).strip()
    if not user_text or not assistant_text:
        return None

    system_prompt = ""
    if system_key and row.get(system_key) is not None:
        system_prompt = str(row.get(system_key)).strip()

    return {"system": system_prompt, "input": user_text, "output": assistant_text}


def extract_instruction_pair(
    row: dict[str, Any],
    input_key: str,
    output_key: str,
    system_key: str,
) -> dict[str, str] | None:
    """Extract an instruction pair from either OpenHermes-style or flat keyed rows."""
    if isinstance(row.get("conversations"), list):
        conv = _extract_from_conversations(row)
        if conv is not None:
            return conv

    return _extract_from_mapped_keys(
        row=row,
        input_key=input_key,
        output_key=output_key,
        system_key=system_key,
    )


async def generate_reasoning(
    client: Any,
    model_name: str,
    user_input: str,
    assistant_output: str,
    system_prompt: str,
) -> str:
    """Generate concise task-relevant reasoning that supports the final answer."""
    prompt = (
        "You are creating a supervised reasoning trace for training.\n"
        "The reasoning must be directly relevant to the user task and consistent with the final answer.\n"
        "Do not add new facts beyond what is needed.\n"
        "Keep it clear and moderately brief.\n"
        "Output only the reasoning text.\n\n"
        f"System context:\n{system_prompt or '[none]'}\n\n"
        f"User input:\n{user_input}\n\n"
        f"Assistant final answer:\n{assistant_output}\n"
    )

    try:
        response = await client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        content = response.choices[0].message.content
        return (content or "").strip()
    except Exception as exc:
        typer.echo(f"Reasoning generation failed: {exc}", err=True)
        return ""


def to_gemma4_record(
    system_prompt: str,
    user_input: str,
    reasoning: str,
    output: str,
    include_reasoning: bool,
) -> dict[str, Any]:
    """Gemma record aligned to channel-based chat templates.

    Default behavior is final answer only. If include_reasoning is enabled
    and reasoning exists, it is encoded in the assistant content as a
    channel block: <|channel>thought ... <channel|>.
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_input})

    assistant_content = output
    if include_reasoning and reasoning:
        assistant_content = f"<|channel>thought\n{reasoning}\n<channel|>{output}"
    messages.append({"role": "assistant", "content": assistant_content})

    return {"messages": messages}


async def process_samples(
    rows: list[dict[str, Any]],
    clients: list[Any],
    model_name: str,
    batch_size: int,
    with_reasoning: bool,
    input_key: str,
    output_key: str,
    system_key: str,
    gemma4_path: Path,
    gemma4_include_reasoning: bool,
) -> int:
    """Process samples and write Gemma-4 format output."""
    parsed = [
        p
        for p in (
            extract_instruction_pair(
                row,
                input_key=input_key,
                output_key=output_key,
                system_key=system_key,
            )
            for row in rows
        )
        if p is not None
    ]

    gemma4_count = 0

    if not parsed:
        return gemma4_count

    async def process_one(sample: dict[str, str], idx: int) -> dict[str, Any]:
        reasoning = ""
        if with_reasoning and clients:
            client = clients[idx % len(clients)]
            reasoning = await generate_reasoning(
                client=client,
                model_name=model_name,
                user_input=sample["input"],
                assistant_output=sample["output"],
                system_prompt=sample["system"],
            )

        return to_gemma4_record(
            system_prompt=sample["system"],
            user_input=sample["input"],
            reasoning=reasoning,
            output=sample["output"],
            include_reasoning=gemma4_include_reasoning,
        )

    for start in tqdm(range(0, len(parsed), batch_size), desc="Converting"):
        batch = parsed[start : start + batch_size]
        tasks = [process_one(sample, start + i) for i, sample in enumerate(batch)]
        results = await asyncio.gather(*tasks)

        # Persist each completed batch immediately.
        with gemma4_path.open("a", encoding="utf-8") as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

        gemma4_count += len(results)

    return gemma4_count


def write_jsonl(path: Path, rows: list[dict[str, Any]], mode: str = "w") -> None:
    """Write rows to JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(mode, encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


async def convert_to_chat(
    infile: Path,
    out_dir: Path,
    input_key: str,
    output_key: str,
    system_key: str,
    gemma4_include_reasoning: bool,
    with_reasoning: bool,
    model_name: str,
    batch_size: int,
    client_urls: str,
    api_key: str,
) -> None:
    """Convert JSONL to Gemma-4 chat format with optional reasoning."""
    if not infile.exists():
        raise typer.BadParameter(f"Input file does not exist: {infile}")
    if with_reasoning and not api_key:
        raise typer.BadParameter(
            "API key is required when --with-reasoning is enabled. "
            "Set --api-key or OPENAI_API_KEY environment variable."
        )

    # Import OpenAI client only if reasoning is enabled
    clients = []
    if with_reasoning:
        from openai import AsyncOpenAI

        urls = [u.strip() for u in client_urls.split(",") if u.strip()]
        clients = [AsyncOpenAI(base_url=url, api_key=api_key) for url in urls]

    # Create output filename
    stem = infile.stem
    gemma4_path = out_dir / f"{stem}.gemma4.jsonl"

    # Truncate file before appending
    write_jsonl(gemma4_path, [], mode="w")

    # Read all rows from input file
    typer.echo(f"Reading from {infile}...")
    rows = []
    with infile.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    typer.echo(f"Processing {len(rows)} rows...")
    gemma4_count = await process_samples(
        rows=rows,
        clients=clients,
        model_name=model_name,
        batch_size=batch_size,
        with_reasoning=with_reasoning,
        input_key=input_key,
        output_key=output_key,
        system_key=system_key,
        gemma4_path=gemma4_path,
        gemma4_include_reasoning=gemma4_include_reasoning,
    )

    typer.echo(f"Wrote {gemma4_count} rows to: {gemma4_path}")
