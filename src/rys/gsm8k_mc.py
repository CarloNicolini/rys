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


def prepare_mc_batches(tokenizer, mc_df: pd.DataFrame, batch_size: int = 16) -> list[dict]:
    """Tokenise once into reusable left-padded batches.

    RYS replay never changes the inputs, so for an RYS window sweep the
    tokenisation and padding should happen a single time and the resulting
    tensors be reused for every window (the per-window cost is then the forward
    pass alone). Each batch is a dict with CPU tensors ``input_ids`` and
    ``attention_mask`` plus a ``meta`` list of
    ``(prompt_id, cand_idx, is_gold, n_tokens)``; continuation tokens are the
    last ``n_tokens`` of each left-padded row.
    """
    pad_id = tokenizer.pad_token_id
    items = []  # (prompt_id, cand_idx, is_gold, input_ids, n_cont)
    for row in mc_df.itertuples(index=False):
        prompt_ids = tokenizer(row.prompt, add_special_tokens=False)["input_ids"]
        for cand_idx, cand_str in enumerate(row.candidate_strs):
            cont_ids = tokenizer(cand_str, add_special_tokens=False)["input_ids"]
            items.append(
                (row.prompt_id, cand_idx, cand_idx == row.gold_idx, prompt_ids + cont_ids, len(cont_ids))
            )

    batches = []
    for start in range(0, len(items), batch_size):
        chunk = items[start : start + batch_size]
        max_len = max(len(seq) for *_, seq, _ in chunk)
        input_ids = torch.full((len(chunk), max_len), pad_id, dtype=torch.long)
        attn = torch.zeros((len(chunk), max_len), dtype=torch.long)
        meta = []
        for r, (pid, cand_idx, is_gold, seq, n_cont) in enumerate(chunk):
            input_ids[r, max_len - len(seq) :] = torch.tensor(seq, dtype=torch.long)
            attn[r, max_len - len(seq) :] = 1
            meta.append((pid, cand_idx, is_gold, n_cont))
        batches.append({"input_ids": input_ids, "attention_mask": attn, "meta": meta})
    return batches


@torch.inference_mode()
def score_prepared(
    model: torch.nn.Module,
    batches: list[dict],
    device: str | torch.device | None = None,
) -> pd.DataFrame:
    """Teacher-forced continuation loglikelihood for pre-tokenised batches.

    Returns a DataFrame with columns
    ``[prompt_id, cand_idx, is_gold, n_tokens, loglik_sum, loglik_mean]`` (both
    the summed and per-token-mean continuation loglikelihood).
    """
    device = device or next(model.parameters()).device
    model.eval()
    records = []
    for batch in batches:
        input_ids = batch["input_ids"].to(device)
        seq_len = input_ids.shape[1]
        logits = model(
            input_ids=input_ids, attention_mask=batch["attention_mask"].to(device), use_cache=False
        ).logits

        # Continuations are the last `need` tokens (left padding), so only the
        # final `need` logit positions matter. Softmaxing just that slice avoids
        # materialising a (batch, seq, vocab) logprob tensor over the whole vocab.
        need = max(int(n_cont) for *_, n_cont in batch["meta"])
        pred = logits[:, seq_len - need - 1 : seq_len - 1, :]  # predicts the last `need` tokens
        targets = input_ids[:, seq_len - need :]
        tail_lp = torch.log_softmax(pred.float(), dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        tail_lp = tail_lp.cpu()  # (batch, need), column k is the k-th-from-last token

        for r, (pid, cand_idx, is_gold, n_cont) in enumerate(batch["meta"]):
            total = float(tail_lp[r, need - n_cont :].sum())  # only this row's continuation tokens
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


def score_mc(
    model: torch.nn.Module,
    tokenizer,
    mc_df: pd.DataFrame,
    device: str | torch.device | None = None,
    batch_size: int = 16,
) -> pd.DataFrame:
    """Convenience wrapper: tokenise ``mc_df`` and score in one call.

    For an RYS window sweep prefer :func:`prepare_mc_batches` once followed by
    :func:`score_prepared` per window, to avoid re-tokenising every time.
    """
    batches = prepare_mc_batches(tokenizer, mc_df, batch_size=batch_size)
    return score_prepared(model, batches, device=device)


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
