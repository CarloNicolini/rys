"""Generation-free GSM8K scoring for Pythia (GPT-NeoX) models.

Base models in the Pythia suite are too weak to emit a parseable
chain-of-thought, so open-ended GSM8K generation scores near zero and the RYS
delta matrix would be flat. Instead we turn each problem into a multiple-choice
loglikelihood task: the model ranks the gold final number against a few numeric
distractors, and "accuracy" is the fraction of problems where the gold answer
gets the highest teacher-forced loglikelihood. No autoregressive decoding is
involved, so a whole problem is scored in a single ``use_cache=False`` forward
pass, which is fast and stays correct under the :func:`rys.surgery.apply_rys`
replay hook.

Pythia is ``GPTNeoXForCausalLM``: its decoder blocks live at
``model.gpt_neox.layers`` rather than ``model.model.layers``. The RYS facilities
(:mod:`rys.surgery`, :mod:`rys.activations`) standardise on ``model.model.layers``,
so :func:`load_pythia` aliases ``model.model = model.gpt_neox`` once at load
time. PyTorch deduplicates the shared submodule, so this adds no parameters.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset


def load_pythia(
    name: str = "EleutherAI/pythia-70m",
    device: str | torch.device = "cpu",
    dtype: torch.dtype | None = None,
):
    """Load a Pythia checkpoint ready for RYS surgery and CKA capture.

    The tokenizer is configured with ``padding_side='left'`` (required by the
    capture and generation helpers) and the model exposes ``model.model.layers``
    via an alias to ``model.gpt_neox`` so the existing RYS facilities work
    unchanged.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype or torch.float32)
    model.to(device).eval()
    model.model = model.gpt_neox  # expose `.model.layers` for apply_rys / capture
    return model, tokenizer


def parse_gold(answer: str) -> int | None:
    """Parse the integer after ``####`` in a GSM8K answer, or ``None``."""
    tail = answer.split("####", 1)[-1].strip().replace(",", "")
    try:
        return int(tail)
    except ValueError:
        return None


def build_distractors(gold: int, k: int, rng: np.random.Generator) -> list[int]:
    """Return ``k`` deterministic numeric distractors, none equal to ``gold``.

    Distractors are plausible wrong answers a confused solver might produce:
    off-by-one, doubling/halving, an order-of-magnitude slip, and digit edits.
    The pool is shuffled with ``rng`` and truncated; if the structured pool runs
    short (small ``gold``) it is topped up with nearby integers.
    """
    pool: list[int] = []
    candidates = [
        gold + 1,
        gold - 1,
        gold + 2,
        gold - 2,
        gold * 2,
        gold // 2,
        gold + 10,
        gold - 10,
        gold * 3,
        gold + (gold // 10 or 1) * 10,
        int(str(abs(gold))[:-1] or "0"),  # drop last digit
        int(str(abs(gold)) + "0"),  # append a zero
    ]
    seen = {gold}
    for c in candidates:
        if c not in seen and c >= 0:
            seen.add(c)
            pool.append(c)

    bump = 1
    while len(pool) < k:
        for c in (gold + bump, gold - bump):
            if c not in seen and c >= 0:
                seen.add(c)
                pool.append(c)
        bump += 1

    rng.shuffle(pool)
    return pool[:k]


def make_gsm8k_mc(
    n: int = 100,
    k_distractors: int = 4,
    fewshot: int = 4,
    seed: int = 0,
) -> pd.DataFrame:
    """Build a GSM8K multiple-choice dataset for loglikelihood scoring.

    Few-shot exemplars are drawn from the train split (so they never leak the
    evaluated test problems) and rendered as ``Q: ...\\nA: <number>\\n\\n``. Each
    test row carries the shared ``prompt`` (few-shot context + question + ``A:``
    cue), the candidate numbers, their space-prefixed string forms, and the
    index of the gold answer within the shuffled candidate list.

    Returns
    -------
    DataFrame with columns
    ``[prompt_id, question, gold, prompt, candidates, candidate_strs, gold_idx]``.
    """
    rng = np.random.default_rng(seed)

    train = load_dataset("gsm8k", "main", split="train").shuffle(seed=seed)
    shots = []
    for ex in train:
        g = parse_gold(ex["answer"])
        if g is not None:
            shots.append(f"Q: {ex['question'].strip()}\nA: {g}")
        if len(shots) == fewshot:
            break
    context = "\n\n".join(shots)
    context = context + "\n\n" if context else ""

    test = load_dataset("gsm8k", "main", split="test").shuffle(seed=seed)
    rows = []
    for ex in test:
        if len(rows) == n:
            break
        gold = parse_gold(ex["answer"])
        if gold is None:
            continue
        distractors = build_distractors(gold, k_distractors, rng)
        candidates = [gold, *distractors]
        order = rng.permutation(len(candidates))
        candidates = [candidates[i] for i in order]
        gold_idx = candidates.index(gold)
        question = ex["question"].strip()
        rows.append(
            {
                "prompt_id": f"gsm8k_{len(rows):04d}",
                "question": question,
                "gold": gold,
                "prompt": f"{context}Q: {question}\nA:",
                "candidates": candidates,
                "candidate_strs": [f" {c}" for c in candidates],
                "gold_idx": gold_idx,
            }
        )
    return pd.DataFrame(rows)


@torch.inference_mode()
def score_mc(
    model: torch.nn.Module,
    tokenizer,
    mc_df: pd.DataFrame,
    device: str | torch.device | None = None,
    batch_size: int = 16,
) -> pd.DataFrame:
    """Teacher-forced loglikelihood of every candidate continuation.

    For each ``(problem, candidate)`` the continuation token ids are appended to
    the prompt token ids and scored in one forward pass. Both the summed and the
    per-token-mean continuation loglikelihood are returned so the caller can use
    length-normalised accuracy (the headline metric, fair across multi-token
    numbers).

    Returns
    -------
    DataFrame with columns
    ``[prompt_id, cand_idx, is_gold, n_tokens, loglik_sum, loglik_mean]``.
    """
    device = device or next(model.parameters()).device
    pad_id = tokenizer.pad_token_id

    items = []  # (prompt_id, cand_idx, is_gold, input_ids, n_cont)
    for row in mc_df.itertuples(index=False):
        prompt_ids = tokenizer(row.prompt, add_special_tokens=False)["input_ids"]
        for cand_idx, cand_str in enumerate(row.candidate_strs):
            cont_ids = tokenizer(cand_str, add_special_tokens=False)["input_ids"]
            items.append(
                (
                    row.prompt_id,
                    cand_idx,
                    cand_idx == row.gold_idx,
                    prompt_ids + cont_ids,
                    len(cont_ids),
                )
            )

    records = []
    model.eval()
    for start in range(0, len(items), batch_size):
        batch = items[start : start + batch_size]
        max_len = max(len(seq) for _, _, _, seq, _ in batch)
        input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        attn = torch.zeros((len(batch), max_len), dtype=torch.long)
        cont_mask = torch.zeros((len(batch), max_len), dtype=torch.bool)
        for r, (_, _, _, seq, n_cont) in enumerate(batch):
            input_ids[r, max_len - len(seq) :] = torch.tensor(seq, dtype=torch.long)
            attn[r, max_len - len(seq) :] = 1
            cont_mask[r, max_len - n_cont :] = True

        input_ids = input_ids.to(device)
        logits = model(input_ids=input_ids, attention_mask=attn.to(device), use_cache=False).logits
        logprobs = torch.log_softmax(logits.float(), dim=-1)
        # token at position t is predicted from logits at t-1
        token_lp = torch.zeros_like(input_ids, dtype=torch.float32)
        token_lp[:, 1:] = logprobs[:, :-1].gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        token_lp = token_lp.cpu()

        for r, (pid, cand_idx, is_gold, _, n_cont) in enumerate(batch):
            mask = cont_mask[r]
            total = float(token_lp[r][mask].sum())
            records.append(
                {
                    "prompt_id": pid,
                    "cand_idx": cand_idx,
                    "is_gold": is_gold,
                    "n_tokens": n_cont,
                    "loglik_sum": total,
                    "loglik_mean": total / max(n_cont, 1),
                }
            )
    return pd.DataFrame(records)


def mc_accuracy(scores: pd.DataFrame) -> dict[str, float]:
    """Fraction of problems whose gold answer wins the loglikelihood ranking.

    ``acc`` ranks by summed loglikelihood, ``acc_norm`` by per-token-mean
    loglikelihood (the headline, length-fair metric).
    """
    n = scores["prompt_id"].nunique()
    if n == 0:
        return {"acc": float("nan"), "acc_norm": float("nan"), "n": 0}

    def _hit(col: str) -> float:
        idx = scores.groupby("prompt_id")[col].idxmax()
        return float(scores.loc[idx, "is_gold"].mean())

    return {"acc": _hit("loglik_sum"), "acc_norm": _hit("loglik_mean"), "n": int(n)}
