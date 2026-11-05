"""Stage 2: ask an LLM which of the OCR strings identify a place.

Reads the OCR dump, not images. Images without OCR text skip the model and get
zero LLM time. In a batch, each image gets an equal share of the batch time.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

from navocr_bench.parse_output import parse_selection
from navocr_bench.paths import prepared
from navocr_bench.prompts import render
from navocr_bench.ocr import OcrResult

Client = Callable[[Sequence[str]], Sequence[str]]

MAX_TOKENS = 64

# Hugging Face revisions used in the paper; other model ids load their latest revision.
REVISIONS = {
    "Qwen/Qwen3.5-0.8B": "2fc06364715b967f1860aea9cf38778875588b17",
    "Qwen/Qwen3.5-2B": "15852e8c16360a2fea060d615a32b45270f8a8fc",
    "Qwen/Qwen3.5-4B": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
    "Qwen/Qwen3.5-9B": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
    "Qwen/Qwen3.5-35B-A3B": "59d61f3ce65a6d9863b86d2e96597125219dc754",
}


@dataclass(frozen=True)
class FilterResult:
    id: str
    selected: list[str]
    raw: str
    elapsed_s: float


def run_filter(
    ocr_results: Iterable[OcrResult],
    prompt_key: str,
    client: Client,
    batch_size: int = 1,
    clock: Callable[[], float] = time.perf_counter,
) -> list[FilterResult]:
    results = list(ocr_results)
    pending = [r for r in results if r.items]
    completions: dict[str, tuple[str, float]] = {}

    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        prompts = [render(prompt_key, [i.text for i in r.items]) for r in batch]
        t0 = clock()
        replies = list(client(prompts))
        if len(replies) != len(batch):
            raise RuntimeError(
                f"client returned {len(replies)} replies for {len(batch)} prompts"
            )
        per_image = (clock() - t0) / len(batch)
        for result, reply in zip(batch, replies):
            completions[result.id] = (reply, per_image)

    out = []
    for result in results:
        reply, elapsed = completions.get(result.id, ("", 0.0))
        out.append(
            FilterResult(
                id=result.id,
                selected=parse_selection(reply),
                raw=reply,
                elapsed_s=elapsed,
            )
        )
    return out


def write_filtered(results: Iterable[FilterResult], path: Path | str) -> None:
    with prepared(path).open("w", encoding="utf-8") as fh:
        for result in results:
            row = {
                "id": result.id,
                "selected": result.selected,
                "raw": result.raw,
                "elapsed_s": result.elapsed_s,
            }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_filtered(path: Path | str) -> list[FilterResult]:
    results = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        results.append(FilterResult(**row))
    return results


def apply_chat_template(tokenizer, prompt: str) -> str:
    """Wrap the prompt in the model's chat template with thinking disabled."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def build_transformers_client(
    model: str, max_tokens: int = MAX_TOKENS, device: str = "cuda", chat_template: bool = True
) -> Client:
    """Greedy bf16 decoding with Hugging Face transformers, chat template unless ``chat_template`` is False.

    Downloads the model on first use, at the pinned revision for the paper's models.
    """
    # Installed only in navocr-llm.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    pinned = REVISIONS.get(model)
    tokenizer = AutoTokenizer.from_pretrained(model, revision=pinned)
    # Left padding keeps each completion aligned with the end of its prompt.
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    net = AutoModelForCausalLM.from_pretrained(
        model, revision=pinned, torch_dtype=torch.bfloat16, device_map=device
    )
    net.eval()
    revision = pinned or getattr(net.config, "_commit_hash", None)

    def client(prompts: Sequence[str]) -> list[str]:
        if chat_template:
            prompts = [apply_chat_template(tokenizer, p) for p in prompts]
        batch = tokenizer(list(prompts), return_tensors="pt", padding=True).to(
            net.device
        )
        with torch.no_grad():
            out = net.generate(
                **batch,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        fresh = out[:, batch["input_ids"].shape[1] :]
        return tokenizer.batch_decode(fresh, skip_special_tokens=True)

    client.revision = revision
    return client

