"""Ng-style combined EQ+MATH probe utilities."""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import torch

from rys.guesstimation import calculate_score, rys_unrolled

MATH_SYSTEM_PROMPT = (
    "You are a highly intelligent AI. You have extraordinary intuition and can "
    "easily make accurate estimations. For the following questions, you will "
    "always provide an answer, even if you are not certain."
)

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_NUM_RE = re.compile(r"\d+")
EMOTION_KEYS = ["emotion1_score", "emotion2_score", "emotion3_score", "emotion4_score"]


@dataclass(slots=True)
class ProbeItem:
    task: str
    qid: str
    prompt: str
    answer: int | None = None
    reference: dict | None = None


def strip_thinking(text: str) -> str:
    return _THINK_BLOCK.sub("", text).strip()


def extract_integers(text: str) -> list[int]:
    return [int(x) for x in _NUM_RE.findall(text)]


def extract_scores_from_section(text: str) -> list[float] | None:
    score_pattern = r"(?:\d\.\s*)?[A-Za-z_]+:\s*(\d+(?:\.\d+)?)"
    matches = re.findall(score_pattern, text)
    values: list[float] = []
    for m in matches[:4]:
        v = float(m)
        if 0 <= v <= 10:
            values.append(v)
    if len(values) < 3:
        return None
    while len(values) < 4:
        values.append(5.0)
    return values


def extract_emotion_scores(text: str) -> tuple[dict | None, float]:
    default = {k: 5.0 for k in EMOTION_KEYS}
    revised_scores = None
    first_scores = None

    revised_match = re.search(r"Revised scores:", text, re.IGNORECASE)
    if revised_match:
        revised_scores = extract_scores_from_section(text[revised_match.end() :])

    first_match = re.search(r"First pass scores:", text, re.IGNORECASE)
    if first_match:
        chunk = text[first_match.end() :]
        critique_match = re.search(r"Critique:", chunk, re.IGNORECASE)
        if critique_match:
            chunk = chunk[: critique_match.start()]
        first_scores = extract_scores_from_section(chunk)

    if revised_scores and first_scores:
        blend = [(a + b) / 2.0 for a, b in zip(first_scores, revised_scores, strict=True)]
        return {k: blend[i] for i, k in enumerate(EMOTION_KEYS)}, 1.0
    if revised_scores:
        return {k: revised_scores[i] for i, k in enumerate(EMOTION_KEYS)}, 0.9
    if first_scores:
        return {k: first_scores[i] for i, k in enumerate(EMOTION_KEYS)}, 0.8

    nums = [float(x) for x in re.findall(r"\b(\d+(?:\.\d+)?)\b", text)]
    in_range = [x for x in nums if 0 <= x <= 10]
    if len(in_range) >= 4:
        return {k: in_range[i] for i, k in enumerate(EMOTION_KEYS)}, 0.5
    if in_range:
        out = default.copy()
        for i, val in enumerate(in_range[:4]):
            out[EMOTION_KEYS[i]] = val
        return out, len(in_range) / 8.0
    return default, 0.0


def calculate_eq_score(predicted: dict | None, reference: dict, confidence: float = 1.0) -> float:
    if predicted is None:
        return 0.5
    pred = [predicted.get(k, 5.0) for k in EMOTION_KEYS]
    ref = [reference.get(k, 5.0) for k in EMOTION_KEYS]
    total_diff = sum(abs(a - b) for a, b in zip(pred, ref, strict=True))
    raw_score = max(0.0, 1.0 - total_diff / 40.0)
    return confidence * raw_score + (1 - confidence) * 0.5


def apply_chat_template(tokenizer, messages: list[dict], use_no_think: bool, think_seed_text: str) -> str:
    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    if use_no_think and not prompt.endswith(" \n"):
        prompt = f"{prompt} {think_seed_text} \n"
    return prompt


def build_probe_items(
    math_dataset: dict,
    eq_dataset: dict,
    *,
    tokenizer,
    prompt_policy: str,
    use_no_think: bool = True,
    think_seed_text: str = "I can answer this now, and will do so succinctly.",
) -> list[ProbeItem]:
    items: list[ProbeItem] = []

    for qid, sample in math_dataset.items():
        question = sample["question"]
        if prompt_policy == "chat":
            user_text = f"/no_think {question}" if use_no_think else question
            messages = [
                {"role": "system", "content": MATH_SYSTEM_PROMPT},
                {"role": "user", "content": user_text},
            ]
            prompt = apply_chat_template(tokenizer, messages, use_no_think=use_no_think, think_seed_text=think_seed_text)
        else:
            prompt = f"{question}\nAnswer with only an integer number."
        items.append(ProbeItem(task="math", qid=str(qid), prompt=prompt, answer=int(sample["answer"])))

    for qid, sample in eq_dataset.items():
        eq_prompt = sample["prompt"]
        if prompt_policy == "chat":
            user_text = f"/no_think {eq_prompt}" if use_no_think else eq_prompt
            messages = [{"role": "user", "content": user_text}]
            prompt = apply_chat_template(tokenizer, messages, use_no_think=use_no_think, think_seed_text=think_seed_text)
        else:
            prompt = eq_prompt
        reference = sample.get("reference_answer", sample.get("reference_answer_fullscale", {}))
        items.append(ProbeItem(task="eq", qid=str(qid), prompt=prompt, reference=reference))
    return items


def score_output(item: ProbeItem, raw_output: str) -> tuple[float, dict]:
    stripped = strip_thinking(raw_output)
    if item.task == "math":
        assert item.answer is not None
        integers = extract_integers(stripped)
        has_valid = bool(integers)
        fallback_used = False
        if not integers:
            integers = extract_integers(raw_output)
            fallback_used = bool(integers)
        if not integers:
            score = 0.0
        else:
            score = max(float(calculate_score(item.answer, i)) for i in integers)
        return score, {
            "qid": item.qid,
            "task": "math",
            "raw_output": raw_output,
            "stripped_output": stripped,
            "extracted": integers,
            "reference": item.answer,
            "has_valid_final_answer": has_valid,
            "fallback_used": fallback_used,
            "score": score,
        }
    assert item.reference is not None
    predicted, confidence = extract_emotion_scores(stripped)
    score = float(calculate_eq_score(predicted, item.reference, confidence))
    return score, {
        "qid": item.qid,
        "task": "eq",
        "raw_output": raw_output,
        "stripped_output": stripped,
        "extracted": predicted,
        "confidence": confidence,
        "reference": item.reference,
        "score": score,
    }


@torch.inference_mode()
def generate_probe_outputs(
    model: torch.nn.Module,
    tokenizer,
    items: list[ProbeItem],
    *,
    batch_size: int = 16,
    max_new_tokens: int = 64,
    device: str | torch.device | None = None,
    window: tuple[int, int] | None = None,
) -> tuple[dict, list[dict]]:
    """Run mixed math+eq generation pass and compute component scores."""
    device = device or next(model.parameters()).device
    prompts = [item.prompt for item in items]
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    scores_math: list[float] = []
    scores_eq: list[float] = []
    responses: list[dict] = []

    model.eval()
    with rys_unrolled(model, window):
        for start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[start : start + batch_size]
            batch_items = items[start : start + batch_size]
            enc = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(device)
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=pad_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            input_w = enc["input_ids"].shape[1]
            for row, item in enumerate(batch_items):
                new_tokens = out[row, input_w:]
                text = tokenizer.decode(new_tokens, skip_special_tokens=True)
                score, payload = score_output(item, text)
                responses.append(payload)
                if item.task == "math":
                    scores_math.append(score)
                else:
                    scores_eq.append(score)

    math_score = float(np.mean(scores_math)) if scores_math else 0.0
    eq_score = float(np.mean(scores_eq)) if scores_eq else 0.0
    summary = {
        "math_score": math_score,
        "eq_score": eq_score,
        "combined_score": 0.5 * (math_score + eq_score),
        "n_math": len(scores_math),
        "n_eq": len(scores_eq),
    }
    return summary, responses

