"""Prompt loaders for the three task distributions.

Every loader returns a long-format DataFrame with the same schema:

    [prompt_id, task, prompt, gold]

so that downstream code can `pd.concat(...)` the three tasks without any
adapter logic. Splitting and chunking decisions live here so the notebook
stays clean.

The chat-templated formatting uses ``tokenizer.apply_chat_template`` with the
official Llama-3 ``user`` role for GSM8K and CSQA. Wikitext-2 is fed as raw
text since plain continuation has no chat semantics.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from datasets import load_dataset

GSM8K_FEWSHOT_INSTRUCTION = (
    "Solve the following math word problem step by step. "
    "End your answer with '#### N' where N is the final numeric answer."
)


def _chat_format(tokenizer: Any, user_text: str) -> str:
    """Apply the model chat template, falling back to raw text if unavailable."""
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user_text}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return user_text


def gsm8k_prompts(tokenizer: Any, n: int = 250, seed: int = 0) -> pd.DataFrame:
    """Sample ``n`` GSM8K test-split problems and return prompt-formatted text.

    The gold answer is the substring after ``####`` in the official solution
    field, stripped to the bare number for downstream evaluation.
    """
    ds = load_dataset("gsm8k", "main", split="test").shuffle(seed=seed).select(range(n))
    rows = []
    for i, ex in enumerate(ds):
        question = ex["question"]
        gold_full = ex["answer"]
        gold = gold_full.split("####", 1)[-1].strip()
        prompt = _chat_format(tokenizer, f"{GSM8K_FEWSHOT_INSTRUCTION}\n\n{question}")
        rows.append((f"gsm8k_{i:04d}", "gsm8k", prompt, gold))
    return pd.DataFrame(rows, columns=["prompt_id", "task", "prompt", "gold"])


def csqa_prompts(tokenizer: Any, n: int = 250, seed: int = 0) -> pd.DataFrame:
    """Sample ``n`` CommonsenseQA validation-split items, formatted as MCQA."""
    ds = (
        load_dataset("commonsense_qa", split="validation")
        .shuffle(seed=seed)
        .select(range(n))
    )
    rows = []
    for i, ex in enumerate(ds):
        labels = ex["choices"]["label"]
        texts = ex["choices"]["text"]
        options = "\n".join(f"{lab}. {txt}" for lab, txt in zip(labels, texts, strict=True))
        user_text = (
            "Answer the following multiple-choice question with the single "
            "correct option letter (A, B, C, D, or E).\n\n"
            f"Question: {ex['question']}\n\nOptions:\n{options}\n\nAnswer:"
        )
        prompt = _chat_format(tokenizer, user_text)
        rows.append((f"csqa_{i:04d}", "csqa", prompt, ex["answerKey"]))
    print(len(rows))
    return pd.DataFrame(rows, columns=["prompt_id", "task", "prompt", "gold"])


def wikitext_prompts(
    tokenizer: Any,
    n: int = 250,
    chunk_tokens: int = 256,
    seed: int = 0,
) -> pd.DataFrame:
    """Yield ``n`` raw-text Wikitext-2 chunks of approx. ``chunk_tokens`` tokens.

    No chat template is applied — this is the plain-continuation control. We
    use the tokenizer to estimate a character window that maps to roughly the
    requested token budget.
    """
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    full_text = "\n".join(t for t in ds["text"] if t.strip())
    chars_per_token = max(1, len(full_text) // max(1, len(tokenizer.encode(full_text[:4000]))))
    char_window = chunk_tokens * chars_per_token
    starts = list(range(0, len(full_text) - char_window, char_window))[:n]
    if seed:
        rng = pd.Series(starts).sample(n=min(n, len(starts)), random_state=seed)
        starts = rng.tolist()

    rows = []
    for i, s in enumerate(starts):
        chunk = full_text[s : s + char_window]
        rows.append((f"wikitext_{i:04d}", "wikitext", chunk, ""))
    return pd.DataFrame(rows, columns=["prompt_id", "task", "prompt", "gold"])
