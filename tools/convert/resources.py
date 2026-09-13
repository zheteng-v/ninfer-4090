"""Select final resource bytes and derive the tokenizer's public token domain."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

TEXT_RESOURCES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "generation_config.json",
)
VISION_RESOURCES = ("preprocessor_config.json", "video_preprocessor_config.json")


def token_domain(
    tokenizer: dict, config: dict, vocab_size: int
) -> tuple[int, tuple[int, ...]]:
    ids: dict[int, str] = {}
    special: set[int] = set()

    def add(index: object, content: object) -> None:
        if (
            type(index) is not int
            or not 0 <= index < vocab_size
            or not isinstance(content, str)
        ):
            raise ValueError(
                f"invalid tokenizer entry id={index!r}, content={content!r}"
            )
        if index in ids and ids[index] != content:
            raise ValueError(f"tokenizer resources disagree about token {index}")
        ids[index] = content

    model = tokenizer.get("model", {})
    vocabulary = model.get("vocab")
    if not isinstance(vocabulary, dict):
        raise ValueError("tokenizer model must provide a vocabulary mapping")
    for token, index in vocabulary.items():
        add(index, token)
    for token in tokenizer.get("added_tokens", []):
        add(token["id"], token["content"])
    for raw_id, token in config.get("added_tokens_decoder", {}).items():
        index = int(raw_id)
        add(index, token["content"])
        if token.get("special", False):
            special.add(index)
    if not ids or set(ids) != set(range(max(ids) + 1)):
        raise ValueError("this tokenizer requires a contiguous public token ID domain")
    return len(ids), tuple(sorted(special))


def load_resources(
    model_dir: Path,
    *,
    vision: bool,
    vocab_size: int,
    overrides: Mapping[str, str | Path] | None = None,
) -> tuple[dict[str, dict[str, str]], dict[str, bytes], int, tuple[int, ...]]:
    overrides = {} if overrides is None else dict(overrides)
    roles = {"text": TEXT_RESOURCES}
    if vision:
        roles["vision"] = VISION_RESOURCES
    allowed = {role for names in roles.values() for role in names}
    if overrides.keys() - allowed:
        raise ValueError(
            f"resource overrides have no selected consumer: {sorted(overrides.keys()-allowed)}"
        )
    references: dict[str, dict[str, str]] = {}
    payloads: dict[str, bytes] = {}
    parsed: dict[str, dict] = {}
    for component, names in roles.items():
        references[component] = {}
        for role in names:
            path = Path(overrides[role]) if role in overrides else model_dir / role
            data = path.read_bytes()
            if not data:
                raise ValueError(f"{path}: resource is empty")
            text = data.decode("utf-8")
            if role.endswith(".json"):
                value = json.loads(text)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}: resource must contain a JSON object")
                parsed[role] = value
            object_id = f"resource/{component}/{role}"
            references[component][role] = object_id
            payloads[object_id] = data
    count, special = token_domain(
        parsed["tokenizer.json"], parsed["tokenizer_config.json"], vocab_size
    )
    return references, payloads, count, special
