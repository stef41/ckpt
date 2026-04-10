"""Checkpoint metadata editor — read, edit, merge, and diff metadata."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from typing import Any


@dataclass
class CheckpointMetadata:
    """Structured metadata for a model checkpoint."""

    model_name: str | None = None
    framework: str | None = None
    epoch: int | None = None
    step: int | None = None
    loss: float | None = None
    learning_rate: float | None = None
    custom: dict[str, Any] = field(default_factory=dict)
    created_at: str | None = None


class MetadataEditor:
    """Edit, merge, and diff checkpoint metadata."""

    def __init__(self, metadata: CheckpointMetadata | dict | None = None) -> None:
        if metadata is None:
            self._meta = CheckpointMetadata()
        elif isinstance(metadata, dict):
            self._meta = self._dict_to_metadata(metadata)
        else:
            self._meta = metadata

    # ------------------------------------------------------------------
    # Basic accessors
    # ------------------------------------------------------------------

    def set(self, key: str, value: Any) -> None:
        """Set a metadata field (known or custom)."""
        known = {f.name for f in fields(CheckpointMetadata)} - {"custom"}
        if key in known:
            setattr(self._meta, key, value)
        else:
            self._meta.custom[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Get a metadata field (known or custom)."""
        known = {f.name for f in fields(CheckpointMetadata)} - {"custom"}
        if key in known:
            val = getattr(self._meta, key)
            return val if val is not None else default
        return self._meta.custom.get(key, default)

    def delete(self, key: str) -> bool:
        """Delete a metadata field.  Returns True if it existed."""
        known = {f.name for f in fields(CheckpointMetadata)} - {"custom"}
        if key in known:
            cur = getattr(self._meta, key)
            if cur is not None:
                setattr(self._meta, key, None)
                return True
            return False
        if key in self._meta.custom:
            del self._meta.custom[key]
            return True
        return False

    def update(self, data: dict[str, Any]) -> None:
        """Bulk-update metadata from a dict."""
        for k, v in data.items():
            self.set(k, v)

    # ------------------------------------------------------------------
    # Merge / diff
    # ------------------------------------------------------------------

    def merge(self, other: MetadataEditor) -> MetadataEditor:
        """Merge *other* into a **new** editor (other wins on conflicts)."""
        merged = MetadataEditor(self.to_dict())
        for k, v in other.to_dict().items():
            if k == "custom":
                merged._meta.custom.update(v)
            elif v is not None:
                merged.set(k, v)
        return merged

    def diff(self, other: MetadataEditor) -> dict[str, dict[str, Any]]:
        """Return fields that differ between *self* and *other*.

        Each key maps to ``{"self": …, "other": …}``.
        """
        d_self = self.to_dict()
        d_other = other.to_dict()
        all_keys = set(d_self) | set(d_other)
        result: dict[str, dict[str, Any]] = {}
        for k in sorted(all_keys):
            v1 = d_self.get(k)
            v2 = d_other.get(k)
            if v1 != v2:
                result[k] = {"self": v1, "other": v2}
        return result

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return asdict(self._meta)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MetadataEditor:
        return cls(d)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, s: str) -> MetadataEditor:
        return cls.from_dict(json.loads(s))

    # ------------------------------------------------------------------
    # Validation / summary
    # ------------------------------------------------------------------

    def validate(self) -> list[str]:
        """Return a list of warnings for missing / suspect fields."""
        warnings: list[str] = []
        if not self._meta.model_name:
            warnings.append("model_name is missing")
        if not self._meta.framework:
            warnings.append("framework is missing")
        if self._meta.epoch is not None and self._meta.epoch < 0:
            warnings.append("epoch is negative")
        if self._meta.step is not None and self._meta.step < 0:
            warnings.append("step is negative")
        if self._meta.loss is not None and self._meta.loss < 0:
            warnings.append("loss is negative")
        if self._meta.learning_rate is not None and self._meta.learning_rate <= 0:
            warnings.append("learning_rate is non-positive")
        if self._meta.created_at:
            try:
                datetime.fromisoformat(self._meta.created_at)
            except ValueError:
                warnings.append("created_at is not a valid ISO timestamp")
        return warnings

    def summary(self) -> str:
        """One-line summary of the metadata."""
        parts: list[str] = []
        if self._meta.model_name:
            parts.append(self._meta.model_name)
        if self._meta.framework:
            parts.append(f"framework={self._meta.framework}")
        if self._meta.epoch is not None:
            parts.append(f"epoch={self._meta.epoch}")
        if self._meta.step is not None:
            parts.append(f"step={self._meta.step}")
        if self._meta.loss is not None:
            parts.append(f"loss={self._meta.loss}")
        if self._meta.learning_rate is not None:
            parts.append(f"lr={self._meta.learning_rate}")
        n_custom = len(self._meta.custom)
        if n_custom:
            parts.append(f"+{n_custom} custom")
        return " | ".join(parts) if parts else "(empty metadata)"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _dict_to_metadata(d: dict[str, Any]) -> CheckpointMetadata:
        known = {f.name for f in fields(CheckpointMetadata)}
        kwargs: dict[str, Any] = {}
        extra: dict[str, Any] = {}
        for k, v in d.items():
            if k in known:
                kwargs[k] = v
            else:
                extra[k] = v
        if extra:
            kwargs.setdefault("custom", {})
            kwargs["custom"].update(extra)
        return CheckpointMetadata(**kwargs)

    def __repr__(self) -> str:
        return f"MetadataEditor({self.to_dict()!r})"


# ------------------------------------------------------------------
# Standalone helpers
# ------------------------------------------------------------------

_EPOCH_RE = re.compile(r"epoch[_-]?(\d+)", re.IGNORECASE)
_STEP_RE = re.compile(r"step[_-]?(\d+)", re.IGNORECASE)
_FRAMEWORK_HINTS: dict[str, str] = {
    ".pt": "pytorch",
    ".pth": "pytorch",
    ".bin": "pytorch",
    ".safetensors": "safetensors",
    ".ckpt": "pytorch",
    ".h5": "tensorflow",
    ".keras": "keras",
    ".onnx": "onnx",
    ".gguf": "gguf",
    ".ggml": "ggml",
}


def extract_metadata_from_path(path: str) -> CheckpointMetadata:
    """Heuristically parse a checkpoint file path for metadata hints."""
    import os

    basename = os.path.basename(path)
    meta = CheckpointMetadata()

    # Epoch
    m = _EPOCH_RE.search(basename)
    if m:
        meta.epoch = int(m.group(1))

    # Step
    m = _STEP_RE.search(basename)
    if m:
        meta.step = int(m.group(1))

    # Framework from extension
    _, ext = os.path.splitext(basename)
    ext_lower = ext.lower()
    if ext_lower in _FRAMEWORK_HINTS:
        meta.framework = _FRAMEWORK_HINTS[ext_lower]

    # Model name: filename minus extension and known prefixes
    name = os.path.splitext(basename)[0]
    # Remove common checkpoint suffixes
    name = re.sub(r"[-_]?(epoch|step|checkpoint)[-_]?\d*", "", name, flags=re.IGNORECASE).strip("-_ ")
    if name:
        meta.model_name = name

    return meta


def format_metadata_report(metadata: CheckpointMetadata | MetadataEditor | dict) -> str:
    """Format a human-readable metadata report."""
    if isinstance(metadata, MetadataEditor):
        d = metadata.to_dict()
    elif isinstance(metadata, dict):
        d = metadata
    else:
        d = asdict(metadata)

    lines: list[str] = ["Checkpoint Metadata Report", "=" * 30]
    for key in ("model_name", "framework", "epoch", "step", "loss", "learning_rate", "created_at"):
        val = d.get(key)
        if val is not None:
            lines.append(f"  {key:20s}: {val}")
    custom = d.get("custom", {})
    if custom:
        lines.append("  Custom fields:")
        for k, v in sorted(custom.items()):
            lines.append(f"    {k:18s}: {v}")
    return "\n".join(lines)
