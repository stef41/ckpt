"""Merge LoRA adapters into base model weights."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ckpt._types import CkptError, MergeConfig, MergeError


def _find_lora_pairs(
    base_keys: set[str],
    adapter_keys: set[str],
) -> List[Dict[str, str]]:
    """Find matching LoRA A/B pairs and their base weight names.

    LoRA adapters typically have keys like:
    - base_model.model.layers.0.self_attn.q_proj.lora_A.weight
    - base_model.model.layers.0.self_attn.q_proj.lora_B.weight

    The base weight would be:
    - model.layers.0.self_attn.q_proj.weight
    """
    pairs: list[dict[str, str]] = []
    lora_a_keys = {k for k in adapter_keys if "lora_A" in k or "lora_a" in k}

    for a_key in sorted(lora_a_keys):
        # Find corresponding B key
        b_key = a_key.replace("lora_A", "lora_B").replace("lora_a", "lora_b")
        if b_key not in adapter_keys:
            continue

        # Derive the base weight name
        base_name = a_key
        for prefix in ("base_model.model.", "base_model."):
            if base_name.startswith(prefix):
                base_name = base_name[len(prefix):]
        # Remove .lora_A.weight or .lora_A.default.weight
        for suffix in (".lora_A.weight", ".lora_a.weight", ".lora_A.default.weight"):
            if base_name.endswith(suffix):
                base_name = base_name[: -len(suffix)] + ".weight"

        if base_name in base_keys:
            pairs.append({
                "lora_a": a_key,
                "lora_b": b_key,
                "base": base_name,
            })

    return pairs


def _get_scaling(adapter_keys: set[str], adapter_state: Dict[str, Any]) -> float:
    """Try to extract LoRA scaling factor from adapter config or default to 1.0."""
    # Common pattern: lora_alpha / lora_r
    # If adapter_config.json was loaded, scaling = alpha / r
    # For now, default to 1.0 (can be overridden by config.alpha)
    return 1.0


def merge_lora_state_dicts(
    base_state: Dict[str, Any],
    adapter_state: Dict[str, Any],
    alpha: float = 1.0,
) -> Dict[str, Any]:
    """Merge LoRA adapter weights into base model state dict.

    Performs: base_weight += alpha * (lora_B @ lora_A)

    This is a pure dict-in, dict-out function that works with any
    tensor library (torch, numpy, etc.) as long as the tensors
    support @ (matmul) and += operators.

    Parameters
    ----------
    base_state:
        Base model state dict.
    adapter_state:
        LoRA adapter state dict with lora_A and lora_B weights.
    alpha:
        Scaling factor for the LoRA contribution.

    Returns
    -------
    dict
        Merged state dict (modifies base_state in place and returns it).
    """
    base_keys = set(base_state.keys())
    adapter_keys = set(adapter_state.keys())
    pairs = _find_lora_pairs(base_keys, adapter_keys)

    if not pairs:
        raise MergeError(
            "No matching LoRA pairs found. "
            f"Adapter has {len(adapter_keys)} keys, base has {len(base_keys)} keys."
        )

    scaling = _get_scaling(adapter_keys, adapter_state) * alpha

    for pair in pairs:
        lora_a = adapter_state[pair["lora_a"]]
        lora_b = adapter_state[pair["lora_b"]]
        base_weight = base_state[pair["base"]]

        # LoRA: delta_W = B @ A (B is [out, rank], A is [rank, in])
        delta = lora_b @ lora_a
        base_state[pair["base"]] = base_weight + scaling * delta

    return base_state


def find_lora_pairs(
    base_keys: set[str],
    adapter_keys: set[str],
) -> List[Dict[str, str]]:
    """Public interface for finding LoRA pairs."""
    return _find_lora_pairs(base_keys, adapter_keys)
