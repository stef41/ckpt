"""ckpt — inspect, convert, diff, and merge model checkpoints."""

from ckpt._types import (
    CheckpointFormat,
    CheckpointInfo,
    CkptError,
    DType,
    DTYPE_SIZES,
    DiffEntry,
    DiffResult,
    FormatError,
    MergeConfig,
    MergeError,
    TensorInfo,
)
from ckpt.diff import diff, diff_infos, format_diff
from ckpt.inspect import (
    detect_format,
    format_params,
    format_size,
    inspect,
    inspect_safetensors,
)
from ckpt.merge import find_lora_pairs, merge_lora_state_dicts
from ckpt.stats import CheckpointStats, TensorStats, stats_from_info
from ckpt.validate import ValidationIssue, ValidationResult, validate

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # Types
    "CheckpointFormat",
    "CheckpointInfo",
    "CkptError",
    "DType",
    "DTYPE_SIZES",
    "DiffEntry",
    "DiffResult",
    "FormatError",
    "MergeConfig",
    "MergeError",
    "TensorInfo",
    # Inspect
    "detect_format",
    "inspect",
    "inspect_safetensors",
    "format_size",
    "format_params",
    # Diff
    "diff",
    "diff_infos",
    "format_diff",
    # Merge
    "merge_lora_state_dicts",
    "find_lora_pairs",
    # Stats
    "CheckpointStats",
    "TensorStats",
    "stats_from_info",
    # Validate
    "validate",
    "ValidationResult",
    "ValidationIssue",
]
