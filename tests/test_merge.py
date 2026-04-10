"""Tests for ckpt.merge."""

import numpy as np  # Using numpy arrays as tensor stand-ins
import pytest

from ckpt._types import MergeError
from ckpt.merge import find_lora_pairs, merge_lora_state_dicts


class TestFindLoraPairs:
    def test_basic_pair(self):
        # PEFT-style: adapter keys have base_model.model. prefix
        base_keys = {"layers.0.self_attn.q_proj.weight"}
        adapter_keys = {
            "base_model.model.layers.0.self_attn.q_proj.lora_A.weight",
            "base_model.model.layers.0.self_attn.q_proj.lora_B.weight",
        }
        pairs = find_lora_pairs(base_keys, adapter_keys)
        assert len(pairs) == 1
        assert pairs[0]["base"] == "layers.0.self_attn.q_proj.weight"

    def test_multiple_pairs(self):
        base_keys = {
            "layers.0.self_attn.q_proj.weight",
            "layers.0.self_attn.v_proj.weight",
        }
        adapter_keys = {
            "base_model.model.layers.0.self_attn.q_proj.lora_A.weight",
            "base_model.model.layers.0.self_attn.q_proj.lora_B.weight",
            "base_model.model.layers.0.self_attn.v_proj.lora_A.weight",
            "base_model.model.layers.0.self_attn.v_proj.lora_B.weight",
        }
        pairs = find_lora_pairs(base_keys, adapter_keys)
        assert len(pairs) == 2

    def test_no_matching_base(self):
        base_keys = {"model.other.weight"}
        adapter_keys = {
            "base_model.model.layers.0.self_attn.q_proj.lora_A.weight",
            "base_model.model.layers.0.self_attn.q_proj.lora_B.weight",
        }
        pairs = find_lora_pairs(base_keys, adapter_keys)
        assert len(pairs) == 0

    def test_missing_lora_b(self):
        base_keys = {"layers.0.weight"}
        adapter_keys = {
            "base_model.model.layers.0.lora_A.weight",
            # No lora_B
        }
        pairs = find_lora_pairs(base_keys, adapter_keys)
        assert len(pairs) == 0

    def test_no_prefix(self):
        base_keys = {"layers.0.weight"}
        adapter_keys = {
            "layers.0.lora_A.weight",
            "layers.0.lora_B.weight",
        }
        pairs = find_lora_pairs(base_keys, adapter_keys)
        assert len(pairs) == 1


class TestMergeLoraStateDicts:
    def test_basic_merge(self):
        base = {"layers.0.weight": np.ones((4, 8))}
        adapter = {
            "base_model.model.layers.0.lora_A.weight": np.ones((2, 8)) * 0.1,
            "base_model.model.layers.0.lora_B.weight": np.ones((4, 2)) * 0.1,
        }
        result = merge_lora_state_dicts(base, adapter, alpha=1.0)
        merged = result["layers.0.weight"]
        # delta = B @ A = (4,2) @ (2,8), each element = 0.1*0.1*2 = 0.02
        assert merged.shape == (4, 8)
        assert np.allclose(merged, 1.0 + 0.02)

    def test_alpha_scaling(self):
        base = {"layers.0.weight": np.zeros((4, 8))}
        adapter = {
            "base_model.model.layers.0.lora_A.weight": np.eye(2, 8),
            "base_model.model.layers.0.lora_B.weight": np.eye(4, 2),
        }
        result = merge_lora_state_dicts(base, adapter, alpha=0.5)
        merged = result["layers.0.weight"]
        assert merged[0, 0] == 0.5

    def test_no_matching_pairs_raises(self):
        base = {"unrelated.weight": np.ones((4, 4))}
        adapter = {
            "base_model.model.layers.0.lora_A.weight": np.ones((2, 4)),
            "base_model.model.layers.0.lora_B.weight": np.ones((4, 2)),
        }
        with pytest.raises(MergeError, match="No matching LoRA pairs"):
            merge_lora_state_dicts(base, adapter)

    def test_returns_base(self):
        base = {"layers.0.weight": np.ones((4, 8))}
        adapter = {
            "base_model.model.layers.0.lora_A.weight": np.zeros((2, 8)),
            "base_model.model.layers.0.lora_B.weight": np.zeros((4, 2)),
        }
        result = merge_lora_state_dicts(base, adapter)
        assert result is base
