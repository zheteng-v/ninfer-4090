from __future__ import annotations

from copy import deepcopy
import json

import pytest
from safetensors.torch import save_file
import torch

from tools.convert.qwen3_5 import build_model, text_config
from tools.convert.sources.safetensors import SafetensorsSource


def _config(*, moe=False):
    text = {
        "hidden_size": 16,
        "vocab_size": 8,
        "num_hidden_layers": 2,
        "max_position_embeddings": 128,
        "layer_types": ["linear_attention", "full_attention"],
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 8,
        "rope_parameters": {"partial_rotary_factor": 0.5, "mrope_section": [1, 1, 0]},
        "linear_num_key_heads": 1,
        "linear_key_head_dim": 4,
        "linear_num_value_heads": 2,
        "linear_value_head_dim": 4,
        "linear_conv_kernel_dim": 3,
    }
    if moe:
        text.update(
            num_experts=3,
            num_experts_per_tok=2,
            moe_intermediate_size=8,
            shared_expert_intermediate_size=4,
        )
    else:
        text["intermediate_size"] = 24
    return {
        "architectures": [
            (
                "Qwen3_5MoeForConditionalGeneration"
                if moe
                else "Qwen3_5ForConditionalGeneration"
            )
        ],
        "text_config": text,
        "vision_config": {
            "depth": 1,
            "hidden_size": 8,
            "intermediate_size": 12,
            "num_heads": 2,
            "patch_size": 2,
            "temporal_patch_size": 1,
            "spatial_merge_size": 2,
            "num_position_embeddings": 4,
        },
    }


def _checkpoint(path, config, tensors):
    path.mkdir()
    (path / "config.json").write_text(json.dumps(config))
    save_file(tensors, path / "model.safetensors")
    for role, value in {
        "tokenizer.json": {"model": {"vocab": {str(i): i for i in range(6)}}},
        "tokenizer_config.json": {},
        "generation_config.json": {},
        "preprocessor_config.json": {},
        "video_preprocessor_config.json": {},
    }.items():
        (path / role).write_text(json.dumps(value))
    (path / "chat_template.jinja").write_text("{{ messages }}")
    return SafetensorsSource(path)


def _values(model, name):
    parameter = model.parameters[name]
    return parameter.source.values().reshape(parameter.shape)


def test_model_mapping_preserves_query_gate_gdn_mtp_and_vision_axes(tmp_path):
    q_gate = torch.arange(32 * 16).float().reshape(32, 16)
    qkv = torch.arange(16 * 16).float().reshape(16, 16)
    conv = torch.arange(16 * 3).float().reshape(16, 1, 3)
    patch = torch.arange(8 * 3 * 4).float().reshape(8, 3, 1, 2, 2)
    vision_bias = torch.arange(24).float()
    tensors = {
        "model.language_model.layers.1.self_attn.q_proj.weight": q_gate,
        "model.language_model.layers.0.linear_attn.in_proj_qkv.weight": qkv,
        "model.language_model.layers.0.linear_attn.conv1d.weight": conv,
        "mtp.layers.0.self_attn.q_proj.weight": q_gate + 1000,
        "model.visual.patch_embed.proj.weight": patch,
        "model.visual.blocks.0.attn.qkv.bias": vision_bias,
    }
    with _checkpoint(tmp_path / "source", _config(), tensors) as source:
        model = build_model(source, components=("text", "mtp", "vision"))
        expected_query = q_gate.reshape(2, 2, 8, 16)[:, 0].reshape(16, 16)
        expected_gate = q_gate.reshape(2, 2, 8, 16)[:, 1].reshape(16, 16)
        assert torch.equal(
            _values(model, "text/layers/1/attention/query"), expected_query
        )
        assert torch.equal(
            _values(model, "text/layers/1/attention/gate"), expected_gate
        )
        assert torch.equal(
            _values(model, "mtp/layers/0/attention/query"), expected_query + 1000
        )
        for role, indices in (
            ("query", slice(0, 4)),
            ("key", slice(4, 8)),
            ("value", slice(8, 16)),
        ):
            assert torch.equal(
                _values(model, f"text/layers/0/gdn/{role}"), qkv[indices]
            )
        assert torch.equal(
            _values(model, "text/layers/0/gdn/convolution"), conv[:, 0, :].T
        )
        assert torch.equal(
            _values(model, "vision/patch_embedding"), patch.reshape(8, 12)
        )
        assert torch.equal(
            _values(model, "vision/layers/0/attention/key_bias"), vision_bias[8:16]
        )
        assert model.parameters["text/output_head"].inputs == (
            "text/final_hidden",
            "mtp/final_hidden",
        )


def test_expert_bank_uses_expert_major_gate_up_and_down_ranges(tmp_path):
    gate_up = torch.arange(3 * 16 * 16).float().reshape(3, 16, 16)
    down = torch.arange(3 * 16 * 8).float().reshape(3, 16, 8)
    with _checkpoint(
        tmp_path / "moe",
        _config(moe=True),
        {
            "model.language_model.layers.0.mlp.experts.gate_up_proj": gate_up,
            "model.language_model.layers.0.mlp.experts.down_proj": down,
        },
    ) as source:
        model = build_model(source)
        for expert in range(3):
            prefix = f"text/layers/0/moe/experts/{expert}/"
            assert torch.equal(_values(model, prefix + "gate"), gate_up[expert, :8])
            assert torch.equal(_values(model, prefix + "up"), gate_up[expert, 8:])
            assert torch.equal(_values(model, prefix + "down"), down[expert])


def test_optional_components_are_selected_before_sources_are_required(tmp_path):
    config = _config()
    config["vision_config"] = {"in_channels": 99}
    config["text_config"]["mtp_num_hidden_layers"] = 3
    with _checkpoint(tmp_path / "text", config, {"unused": torch.ones(1)}) as source:
        model = build_model(source)
        assert set(model.components) == {"text"}
        assert all(name.startswith("text/") for name in model.parameters)
        with pytest.raises(ValueError, match="in_channels"):
            build_model(source, components=("text", "vision"))
        with pytest.raises(ValueError, match="mtp_num_hidden_layers"):
            build_model(source, components=("text", "mtp"))
        with pytest.raises(ValueError, match="requires its source"):
            build_model(source, components=("text", "dflash2"))


@pytest.mark.parametrize("architecture", ["DFlashDraftModel", "DFlash2DraftModel"])
def test_draft_query_context_and_dynamic_weights_keep_their_uses(
    tmp_path, architecture
):
    backend = "dflash2" if architecture == "DFlash2DraftModel" else "dflash"
    draft = {
        "architectures": [architecture],
        "hidden_size": 16,
        "vocab_size": 8,
        "num_hidden_layers": 1,
        "intermediate_size": 24,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 8,
        "max_position_embeddings": 128,
        "layer_types": ["sliding_attention"],
        "sliding_window": 32,
        "dflash_config": {
            "target_layer_ids": [1, 0],
            "mask_token_id": 7,
            "conv_kernel_size": 2,
            "conv_group_size": 4,
            "selector_rank": 4,
            "selector_top_k": 2,
        },
    }
    key = torch.arange(8 * 16).float().reshape(8, 16)
    kernel = torch.arange(2 * 2 * 16).float().reshape(2, 2, 16)
    with (
        _checkpoint(tmp_path / "text", _config(), {"unused": torch.ones(1)}) as base,
        _checkpoint(
            tmp_path / "draft",
            draft,
            {
                "layers.0.self_attn.k_proj.weight": key,
                "layers.0.attention_conv.base_kernel": kernel,
            },
        ) as companion,
    ):
        model = build_model(
            base, components=("text", backend), companions={backend: companion}
        )
        prefix = f"{backend}/layers/0/attention/"
        assert torch.equal(_values(model, prefix + "key"), key)
        assert torch.equal(_values(model, prefix + "context_key"), key)
        assert model.parameters[prefix + "key"].inputs == (
            f"{backend}/layers/0/query_projection_input",
        )
        assert model.parameters[prefix + "context_key"].inputs == (
            f"{backend}/context_input",
        )
        assert model.components[backend]["config"]["dflash_config"][
            "target_layer_ids"
        ] == [1, 0]
        if backend == "dflash2":
            assert torch.equal(
                _values(model, f"{backend}/layers/0/attention_conv/base_kernel"), kernel
            )


def test_config_normalization_keeps_dimensions_and_checks_fixed_mathematics():
    raw = _config()
    raw["_name_or_path"] = "my-training-run"
    config = text_config(raw, mtp=False)
    assert config["architectures"] == ["Qwen3_5ForCausalLM"]
    assert config["hidden_size"] == 16 and "_name_or_path" not in config
    assert "hidden_act" not in config and "vision_config" not in config
    for key, invalid in (
        ("hidden_act", "relu"),
        ("attention_bias", True),
        ("attn_output_gate", False),
    ):
        changed = deepcopy(raw)
        changed["text_config"][key] = invalid
        with pytest.raises(ValueError, match=key):
            text_config(changed, mtp=False)
    changed = deepcopy(raw)
    changed["architectures"] = ["AnotherForCausalLM"]
    with pytest.raises(ValueError, match="architecture"):
        text_config(changed, mtp=False)
