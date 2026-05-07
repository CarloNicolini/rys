"""Prompt loaders for task distributions.

Every loader returns a long-format DataFrame with the same schema:

    [prompt_id, task, prompt, gold]

so that downstream code can `pd.concat(...)` the three tasks without any
adapter logic. Splitting and chunking decisions live here so the notebook
stays clean.

The chat-templated formatting uses ``tokenizer.apply_chat_template`` with the
official ``user`` role so every dataset is presented as an instruction stimulus.
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
    return pd.DataFrame(rows, columns=["prompt_id", "task", "prompt", "gold"])


def mmlu_prompts(
    tokenizer: Any,
    subject: str,
    n: int = 250,
    seed: int = 0,
) -> pd.DataFrame:
    """Sample an MMLU subject as an instruction-formatted MCQA task.

    Useful subjects for task-organ contrasts include ``high_school_mathematics``,
    ``formal_logic``, ``philosophy``, ``moral_disputes``, and ``abstract_algebra``.
    """
    ds = load_dataset("cais/mmlu", subject, split="test").shuffle(seed=seed)
    ds = ds.select(range(min(n, len(ds))))
    letters = ["A", "B", "C", "D"]
    task = f"mmlu_{subject}"

    rows = []
    for i, ex in enumerate(ds):
        options = "\n".join(f"{letter}. {choice}" for letter, choice in zip(letters, ex["choices"], strict=True))
        user_text = (
            f"Answer this {subject.replace('_', ' ')} multiple-choice question. "
            "Think through the problem, then end with the single correct option letter.\n\n"
            f"Question: {ex['question']}\n\nOptions:\n{options}\n\nAnswer:"
        )
        prompt = _chat_format(tokenizer, user_text)
        gold = letters[int(ex["answer"])]
        rows.append((f"{task}_{i:04d}", task, prompt, gold))
    return pd.DataFrame(rows, columns=["prompt_id", "task", "prompt", "gold"])
