"""Tests for ckpt.metadata module."""

from __future__ import annotations

import pytest

from ckpt.metadata import (
    CheckpointMetadata,
    MetadataEditor,
    extract_metadata_from_path,
    format_metadata_report,
)

# ------------------------------------------------------------------
# CheckpointMetadata dataclass
# ------------------------------------------------------------------


class TestCheckpointMetadata:
    def test_defaults(self):
        m = CheckpointMetadata()
        assert m.model_name is None
        assert m.custom == {}

    def test_custom_field(self):
        m = CheckpointMetadata(custom={"quantization": "q4_k_m"})
        assert m.custom["quantization"] == "q4_k_m"


# ------------------------------------------------------------------
# MetadataEditor basics
# ------------------------------------------------------------------


class TestMetadataEditorBasics:
    def test_init_empty(self):
        e = MetadataEditor()
        assert e.get("model_name") is None

    def test_init_from_dict(self):
        e = MetadataEditor({"model_name": "llama", "epoch": 3})
        assert e.get("model_name") == "llama"
        assert e.get("epoch") == 3

    def test_init_from_metadata(self):
        m = CheckpointMetadata(model_name="gpt2")
        e = MetadataEditor(m)
        assert e.get("model_name") == "gpt2"

    def test_set_get_known_field(self):
        e = MetadataEditor()
        e.set("model_name", "mistral")
        assert e.get("model_name") == "mistral"

    def test_set_get_custom_field(self):
        e = MetadataEditor()
        e.set("vocab_size", 32000)
        assert e.get("vocab_size") == 32000

    def test_get_default(self):
        e = MetadataEditor()
        assert e.get("model_name", "unknown") == "unknown"
        assert e.get("missing_custom", 42) == 42

    def test_delete_known(self):
        e = MetadataEditor({"model_name": "llama"})
        assert e.delete("model_name") is True
        assert e.get("model_name") is None

    def test_delete_missing(self):
        e = MetadataEditor()
        assert e.delete("model_name") is False
        assert e.delete("nonexistent") is False

    def test_delete_custom(self):
        e = MetadataEditor()
        e.set("extra", 1)
        assert e.delete("extra") is True
        assert e.get("extra") is None

    def test_update(self):
        e = MetadataEditor()
        e.update({"model_name": "phi", "epoch": 5, "custom_tag": "v1"})
        assert e.get("model_name") == "phi"
        assert e.get("epoch") == 5
        assert e.get("custom_tag") == "v1"


# ------------------------------------------------------------------
# Merge / diff
# ------------------------------------------------------------------


class TestMergeDiff:
    def test_merge(self):
        a = MetadataEditor({"model_name": "a", "epoch": 1})
        b = MetadataEditor({"model_name": "b", "step": 100})
        merged = a.merge(b)
        assert merged.get("model_name") == "b"
        assert merged.get("epoch") == 1
        assert merged.get("step") == 100

    def test_merge_custom(self):
        a = MetadataEditor()
        a.set("x", 1)
        b = MetadataEditor()
        b.set("y", 2)
        merged = a.merge(b)
        assert merged.get("x") == 1
        assert merged.get("y") == 2

    def test_diff_identical(self):
        a = MetadataEditor({"model_name": "x"})
        b = MetadataEditor({"model_name": "x"})
        assert a.diff(b) == {}

    def test_diff_different(self):
        a = MetadataEditor({"epoch": 1})
        b = MetadataEditor({"epoch": 5})
        d = a.diff(b)
        assert "epoch" in d
        assert d["epoch"]["self"] == 1
        assert d["epoch"]["other"] == 5


# ------------------------------------------------------------------
# Serialisation
# ------------------------------------------------------------------


class TestSerialisation:
    def test_to_dict_from_dict_roundtrip(self):
        e = MetadataEditor({"model_name": "gpt", "step": 42})
        d = e.to_dict()
        e2 = MetadataEditor.from_dict(d)
        assert e2.to_dict() == d

    def test_to_json_from_json_roundtrip(self):
        e = MetadataEditor({"loss": 0.123})
        j = e.to_json()
        e2 = MetadataEditor.from_json(j)
        assert e2.get("loss") == pytest.approx(0.123)


# ------------------------------------------------------------------
# Validation / summary
# ------------------------------------------------------------------


class TestValidation:
    def test_validate_warnings(self):
        e = MetadataEditor()
        w = e.validate()
        assert any("model_name" in s for s in w)
        assert any("framework" in s for s in w)

    def test_validate_negative_epoch(self):
        e = MetadataEditor({"model_name": "m", "framework": "pt", "epoch": -1})
        w = e.validate()
        assert any("epoch" in s for s in w)

    def test_validate_bad_timestamp(self):
        e = MetadataEditor({"model_name": "m", "framework": "pt", "created_at": "nope"})
        w = e.validate()
        assert any("created_at" in s for s in w)

    def test_validate_clean(self):
        e = MetadataEditor({"model_name": "m", "framework": "pt"})
        assert e.validate() == []

    def test_summary_empty(self):
        assert MetadataEditor().summary() == "(empty metadata)"

    def test_summary_populated(self):
        e = MetadataEditor({"model_name": "llama", "epoch": 3})
        s = e.summary()
        assert "llama" in s
        assert "epoch=3" in s


# ------------------------------------------------------------------
# extract_metadata_from_path
# ------------------------------------------------------------------


class TestExtractMetadataFromPath:
    def test_epoch_step(self):
        m = extract_metadata_from_path("/checkpoints/model-epoch5-step1000.pt")
        assert m.epoch == 5
        assert m.step == 1000
        assert m.framework == "pytorch"

    def test_safetensors(self):
        m = extract_metadata_from_path("model.safetensors")
        assert m.framework == "safetensors"

    def test_gguf(self):
        m = extract_metadata_from_path("llama-7b-q4.gguf")
        assert m.framework == "gguf"

    def test_no_hints(self):
        m = extract_metadata_from_path("random_file.txt")
        assert m.epoch is None
        assert m.framework is None


# ------------------------------------------------------------------
# format_metadata_report
# ------------------------------------------------------------------


class TestFormatReport:
    def test_report_from_metadata(self):
        m = CheckpointMetadata(model_name="phi", epoch=2)
        report = format_metadata_report(m)
        assert "phi" in report
        assert "epoch" in report

    def test_report_from_editor(self):
        e = MetadataEditor({"model_name": "gpt"})
        report = format_metadata_report(e)
        assert "gpt" in report
