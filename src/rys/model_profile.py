"""Model profiling for prompt-policy selection."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from huggingface_hub import HfApi


@dataclass(slots=True)
class ModelProfile:
    model_name: str
    family: str
    model_tag: str
    uses_chat_template: bool
    is_post_trained: bool
    policy: str
    evidence: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def infer_family(model_name: str) -> str:
    """Default family bucket: org prefix when available."""
    if "/" in model_name:
        return model_name.split("/", 1)[0].lower()
    return model_name.lower()


def infer_model_tag(model_name: str) -> str:
    if "/" in model_name:
        return model_name.split("/", 1)[1]
    return model_name


def detect_model_profile(model_name: str, tokenizer, force_policy: str | None = None) -> ModelProfile:
    """Infer whether prompts should use chat-template or plain format."""
    family = infer_family(model_name)
    model_tag = infer_model_tag(model_name)
    evidence: list[str] = []
    uses_chat_template = bool(getattr(tokenizer, "chat_template", None))
    if uses_chat_template:
        evidence.append("tokenizer.chat_template present")

    post_train_markers = {
        "instruct",
        "chat",
        "rlhf",
        "dpo",
        "post-training",
        "post_trained",
    }
    is_post_trained = False
    try:
        info = HfApi().model_info(model_name)
        card = info.card_data or {}
        tags = [str(t).lower() for t in (info.tags or [])]
        card_values = " ".join(str(v).lower() for v in card.values())
        joined = " ".join(tags) + " " + card_values + " " + model_name.lower()
        for marker in sorted(post_train_markers):
            if marker in joined:
                is_post_trained = True
                evidence.append(f"hf metadata marker: {marker}")
                break
        base_model = card.get("base_model")
        if base_model:
            is_post_trained = True
            evidence.append("model card base_model present")
    except Exception:
        evidence.append("hf metadata unavailable")

    if force_policy is not None:
        policy = force_policy
        evidence.append(f"forced policy: {force_policy}")
    else:
        policy = "chat" if (uses_chat_template and is_post_trained) else "plain"
    return ModelProfile(
        model_name=model_name,
        family=family,
        model_tag=model_tag,
        uses_chat_template=uses_chat_template,
        is_post_trained=is_post_trained,
        policy=policy,
        evidence=evidence,
    )

