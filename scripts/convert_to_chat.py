#!/usr/bin/env python
"""Convert JSONL data to Gemma-4 chat format with optional reasoning.

This script can be run standalone or via: tokcleanse convert-to-chat

Example:
    python scripts/convert_to_chat.py data.jsonl data/processed \\
        --input-key transcript --output-key summary \\
        --with-reasoning --model-name gpt-4o-mini
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

import typer
from tqdm import tqdm

# Add parent directory to path for imports
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from tokcleanse.chat_converter import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_GEMMA4_INCLUDE_REASONING,
    DEFAULT_INPUT_KEY,
    DEFAULT_MODEL_NAME,
    DEFAULT_OUTPUT_KEY,
    DEFAULT_SYSTEM_KEY,
    DEFAULT_WITH_REASONING,
    extract_instruction_pair,
    process_samples,
    to_gemma4_record,
    write_jsonl,
)


def reservoir_sample_jsonl(
    infile: Path, n_samples: int, seed: int
) -> list[dict[str, Any]]:
    """Sample n random rows from a JSONL file without loading the full file."""
    import random

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


async def generate_reasoning(
    client: Any,
    model_name: str,
    user_input: str,
    assistant_output: str,
    system_prompt: str,
) -> str:
    """Generate concise task-relevant reasoning that supports the final answer."""
    from openai import AsyncOpenAI

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
    """Gemma record aligned to channel-based chat templates."""
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


def main() -> None:
    """Main entry point for standalone script."""
    app = typer.Typer(
        add_completion=False,
        no_args_is_help=True,
        help="Convert JSONL data to Gemma-4 chat format with optional reasoning.",
    )

    @app.command()
    def convert(
        infile: Path = typer.Argument(..., help="Input JSONL file to convert."),
        out_dir: Path = typer.Argument(
            Path("data/processed"),
            help="Output directory for converted files.",
        ),
        input_key: str = typer.Option(
            DEFAULT_INPUT_KEY,
            "--input-key",
            help="Input key for flat JSONL datasets (e.g. transcript).",
        ),
        output_key: str = typer.Option(
            DEFAULT_OUTPUT_KEY,
            "--output-key",
            help="Output key for flat JSONL datasets (e.g. summary).",
        ),
        system_key: str = typer.Option(
            DEFAULT_SYSTEM_KEY,
            "--system-key",
            help="Optional system-prompt key for flat JSONL datasets.",
        ),
        gemma4_include_reasoning: bool = typer.Option(
            DEFAULT_GEMMA4_INCLUDE_REASONING,
            "--gemma4-include-reasoning/--no-gemma4-include-reasoning",
            help="Include reasoning as a separate thought channel in Gemma 4 output.",
        ),
        with_reasoning: bool = typer.Option(
            DEFAULT_WITH_REASONING,
            "--with-reasoning/--no-reasoning",
            help="Enable or disable reasoning generation calls.",
        ),
        model_name: str = typer.Option(
            DEFAULT_MODEL_NAME,
            "--model-name",
            help="Model name used for reasoning generation.",
        ),
        batch_size: int = typer.Option(
            DEFAULT_BATCH_SIZE,
            min=1,
            help="Async batch size for reasoning calls.",
        ),
        client_urls: str = typer.Option(
            "https://api.openai.com/v1/",
            help="Comma-separated list of OpenAI-compatible base URLs.",
        ),
        api_key: str = typer.Option(
            default_factory=lambda: os.getenv("OPENAI_API_KEY", ""),
            help="API key. Defaults to OPENAI_API_KEY environment variable.",
        ),
    ) -> None:
        """Convert JSONL data to Gemma-4 chat format with optional reasoning."""
        try:
            asyncio.run(
                convert_to_chat(
                    infile=infile,
                    out_dir=out_dir,
                    input_key=input_key,
                    output_key=output_key,
                    system_key=system_key,
                    gemma4_include_reasoning=gemma4_include_reasoning,
                    with_reasoning=with_reasoning,
                    model_name=model_name,
                    batch_size=batch_size,
                    client_urls=client_urls,
                    api_key=api_key,
                )
            )
        except (FileExistsError, ValueError) as exc:
            raise typer.BadParameter(str(exc)) from exc

    app()


if __name__ == "__main__":
    main()
