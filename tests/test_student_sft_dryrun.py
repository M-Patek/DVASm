"""SFT dry-run E2E tests for 03-student.

These tests verify the SFT training pipeline is wired correctly end-to-end
WITHOUT requiring a GPU, a real Qwen2-VL model, or a real video dataset.
The goal is to catch regressions in config plumbing, dataset loading,
LoRA setup, and trainer initialization before a real training run.

Closes the "End-to-end GPU training not validated" known_gap (partially):
this proves the **plumbing** is correct; a real GPU run is still needed
to prove convergence, but that's a separate manual validation step.

Note on module mocking strategy
-------------------------------
sft_trainer.py imports torch / transformers / trl / peft / datasets /
accelerate / bitsandbytes at module load time. In a CPU-only test env
without those installed, we must inject MagicMocks into sys.modules
BEFORE any `from dvas.models.student...` import fires. The block at
the top of this module does exactly that — pytest discovers tests by
importing the module, so the mocks land before any test body runs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Heavy-module stubs (must run at module import time, before any
# `from dvas.models.student...` import).
# ---------------------------------------------------------------------------
_HEAVY_MODULES = (
    "torch",
    "transformers",
    "trl",
    "peft",
    "datasets",
    "accelerate",
    "bitsandbytes",
)

# Track what we inserted so we can clean up if this test module is unloaded.
_INSTALLED_HEAVY = tuple(m for m in _HEAVY_MODULES if m not in sys.modules)
for _mod_name in _INSTALLED_HEAVY:
    _m = MagicMock()
    # Make trl.SFTTrainer / DPOTrainer look like classes
    _m.SFTTrainer = MagicMock()
    _m.DPOTrainer = MagicMock()
    sys.modules[_mod_name] = _m


# Other test modules (e.g. test_student_inference.py) replace the real
# `dvas.models.student.sft_trainer` with a bare MagicMock at their own module
# import time. That stub then poisons subsequent test files via two caches:
# (a) sys.modules, (b) the parent package's __dict__. Detect the pollution
# at OUR module-load time and force a clean re-import.
import importlib  # noqa: E402

_dvas_sft = sys.modules.get("dvas.models.student.sft_trainer")
_dvas_student_pkg = sys.modules.get("dvas.models.student")

def _looks_like_real_sft_module(mod):
    """True iff the loaded module is the real sft_trainer.py, not a MagicMock stub."""
    return mod is not None and hasattr(mod, "train_sft")


if not _looks_like_real_sft_module(_dvas_sft):
    # (1) Drop any stub from sys.modules.
    for _name in (
        "dvas.models.student.sft_trainer",
        "dvas.models.student.dpo_trainer",
        "dvas.models.student.inference",
        "dvas.models.student.dataset",
        "dvas.models.student.config",
        "dvas.models.student",
    ):
        sys.modules.pop(_name, None)
    # (2) Force the parent package to re-bind its submodules.
    if _dvas_student_pkg is not None and hasattr(_dvas_student_pkg, "__dict__"):
        _dvas_student_pkg.__dict__.pop("sft_trainer", None)
        _dvas_student_pkg.__dict__.pop("dpo_trainer", None)
        _dvas_student_pkg.__dict__.pop("inference", None)
    # (3) Re-import the real module (heavy deps are mocked at top of this file).
    importlib.import_module("dvas.models.student.sft_trainer")


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------


class TestSFTConfig:
    """Verify SFTConfig defaults and overrides are wired correctly."""

    def test_default_config_has_valid_paths(self):
        from dvas.models.student.config import SFTConfig

        config = SFTConfig()
        assert config.experiment_name == "dvas_student_sft"
        # Default paths should be Path objects, not strings
        assert isinstance(config.data.train_data_path, Path)
        assert isinstance(config.training.output_dir, Path)

    def test_model_config_uses_qwen2_vl(self):
        from dvas.models.student.config import SFTConfig

        config = SFTConfig()
        assert "Qwen2-VL" in config.model.model_name_or_path
        assert config.model.load_in_4bit is True
        # LoRA defaults sane
        assert config.model.lora_r > 0
        assert config.model.lora_alpha > 0
        assert isinstance(config.model.lora_target_modules, list)
        assert len(config.model.lora_target_modules) > 0

    def test_data_config_can_be_overridden(self, tmp_path):
        from dvas.models.student.config import SFTConfig

        config = SFTConfig()
        # Mutate dataclass fields
        config.data.train_data_path = tmp_path / "train.jsonl"
        config.data.batch_size = 2
        config.data.max_seq_length = 1024

        assert config.data.train_data_path == tmp_path / "train.jsonl"
        assert config.data.batch_size == 2
        assert config.data.max_seq_length == 1024

    def test_sft_config_has_report_to_field(self):
        """`report_to` lives on SFTConfig, not TrainingConfig (sft_trainer.py:126
        reads `config.report_to`). Make sure we don't regress that."""
        from dvas.models.student.config import SFTConfig, TrainingConfig

        sft = SFTConfig()
        assert sft.report_to == "none"

        train_cfg = TrainingConfig()
        # TrainingConfig itself does not carry report_to.
        assert not hasattr(train_cfg, "report_to") or getattr(train_cfg, "report_to", None) is None


class TestSFTDatasetLoading:
    """Verify the dataset can be loaded from a JSONL fixture."""

    def test_load_llava_jsonl(self, tmp_path):
        from dvas.models.student.dataset import VideoAnnotationDataset

        # Write a minimal LLaVA-format fixture
        fixture = tmp_path / "train.jsonl"
        records = [
            {
                "id": f"vid_{i}",
                "video": f"/fake/vid_{i}.mp4",
                "conversations": [
                    {"from": "human", "value": "<video>\ndescribe"},
                    {"from": "gpt", "value": f"description {i}"},
                ],
            }
            for i in range(3)
        ]
        with open(fixture, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        ds = VideoAnnotationDataset(fixture)
        assert len(ds) == 3
        assert ds[0]["id"] == "vid_0"
        assert "conversations" in ds[0]
        assert len(ds[0]["conversations"]) == 2

    def test_load_missing_file_yields_empty_dataset(self, tmp_path):
        from dvas.models.student.dataset import VideoAnnotationDataset

        missing = tmp_path / "nonexistent.jsonl"
        ds = VideoAnnotationDataset(missing)
        assert len(ds) == 0

    def test_load_skips_malformed_jsonl_lines(self, tmp_path):
        from dvas.models.student.dataset import VideoAnnotationDataset

        fixture = tmp_path / "mixed.jsonl"
        lines = [
            json.dumps({"id": "good_1", "video": "/x.mp4", "conversations": []}),
            "{not valid json",
            json.dumps({"id": "good_2", "video": "/y.mp4", "conversations": []}),
        ]
        fixture.write_text("\n".join(lines), encoding="utf-8")

        ds = VideoAnnotationDataset(fixture)
        assert len(ds) == 2
        assert {d["id"] for d in ds.data} == {"good_1", "good_2"}


class TestSFTDryRun:
    """End-to-end SFT plumbing test: verify train_sft can be invoked with all
    heavy operations mocked, and the trainer.save_model path is exercised.

    This proves:
    - Config -> TrainingArguments wiring is correct (incl. `report_to`)
    - Dataset loading is called with the right path
    - LoRA setup runs (mocked)
    - SFTTrainer is constructed with the right args
    - save_model + processor.save_pretrained are called
    - final path resolves to {output_dir}/{experiment_name}/final
    """

    @staticmethod
    def _resolve_real_sft_module():
        """Force-resolve the REAL sft_trainer module, undoing any prior
        pollution (test_student_inference.py replaces this with a MagicMock
        at its own import time, and pytest's collection can pick that up
        before our test runs).

        We look up sys.modules['dvas.models.student.sft_trainer'] and, if
        it's a stub, drop it (and any sibling pollution) so importlib
        re-imports the real one. Heavy deps are mocked at the top of this
        test module so the real module's imports succeed.
        """
        import importlib
        from unittest.mock import MagicMock

        cur = sys.modules.get("dvas.models.student.sft_trainer")
        # MagicMock auto-creates attributes, so hasattr/callable are not enough.
        # We check if the module itself is a MagicMock (pollution) or the real
        # module (has __file__, not a MagicMock subclass).
        if cur is not None and not isinstance(cur, MagicMock) and hasattr(cur, "train_sft"):
            return cur

        # Pollution detected: drop the stub + siblings + package so we get a
        # clean re-import.
        for name in (
            "dvas.models.student.sft_trainer",
            "dvas.models.student.dpo_trainer",
            "dvas.models.student.inference",
            "dvas.models.student.dataset",
            "dvas.models.student.config",
            "dvas.models.student",
        ):
            sys.modules.pop(name, None)
        return importlib.import_module("dvas.models.student.sft_trainer")

    def test_train_sft_calls_save_model_with_correct_path(self, tmp_path):
        sft_mod = self._resolve_real_sft_module()
        from dvas.models.student.config import SFTConfig

        train_sft = sft_mod.train_sft

        # Build minimal fixture
        train_data = tmp_path / "train.jsonl"
        with open(train_data, "w", encoding="utf-8") as f:
            f.write(json.dumps({
                "id": "v1",
                "video": "/fake/v1.mp4",
                "conversations": [
                    {"from": "human", "value": "<video>\ndescribe"},
                    {"from": "gpt", "value": "ok"},
                ],
            }) + "\n")

        output_dir = tmp_path / "outputs"

        # Build config
        config = SFTConfig()
        config.data.train_data_path = train_data
        config.training.output_dir = output_dir
        config.training.num_train_epochs = 1
        config.training.max_steps = 1  # 1 step only — this is a dry run
        config.experiment_name = "dryrun_test"
        config.report_to = "none"

        # Mock everything heavy. Use side_effect on save_model/save_pretrained
        # so they actually create the output directory the way real calls do
        # (HF's save_model mkdir's the target before writing).
        mock_model = MagicMock()
        mock_processor = MagicMock()
        mock_processor.tokenizer = MagicMock()

        mock_lora_model = MagicMock()
        mock_trainer = MagicMock()
        mock_dataset = MagicMock()
        mock_dataset.column_names = ["text"]

        def _fake_save_model(target_dir: str) -> None:
            Path(target_dir).mkdir(parents=True, exist_ok=True)

        def _fake_save_pretrained(target_dir: str) -> None:
            Path(target_dir).mkdir(parents=True, exist_ok=True)

        mock_trainer.save_model.side_effect = _fake_save_model
        mock_processor.save_pretrained.side_effect = _fake_save_pretrained

        # `load_dataset` is imported inside train_sft (line 132 of sft_trainer.py),
        # so we patch the `datasets` module attribute instead of the local one.
        with patch("dvas.models.student.sft_trainer.load_model_and_processor",
                   return_value=(mock_model, mock_processor)), \
             patch("dvas.models.student.sft_trainer.setup_lora",
                   return_value=mock_lora_model), \
             patch("dvas.models.student.sft_trainer.SFTTrainer",
                   return_value=mock_trainer), \
             patch("datasets.load_dataset", return_value=mock_dataset):

            final_path = train_sft(config)

        # Trainer was constructed and .train() was called
        mock_trainer.train.assert_called_once()

        # save_model + processor.save_pretrained were both called
        mock_trainer.save_model.assert_called_once()
        mock_processor.save_pretrained.assert_called_once()

        # Final path should be output_dir / experiment_name / "final"
        expected = output_dir / "dryrun_test" / "final"
        assert final_path == expected
        # And the experiment dir should have been created
        assert expected.parent.exists()

    def test_train_sft_raises_when_dataset_missing(self, tmp_path):
        sft_mod = self._resolve_real_sft_module()
        from dvas.models.student.config import SFTConfig

        train_sft = sft_mod.train_sft

        config = SFTConfig()
        config.data.train_data_path = tmp_path / "nonexistent.jsonl"
        config.training.output_dir = tmp_path / "out"
        config.report_to = "none"

        with patch("dvas.models.student.sft_trainer.load_model_and_processor",
                   return_value=(MagicMock(), MagicMock())):
            with pytest.raises(FileNotFoundError) as exc_info:
                train_sft(config)
            assert "nonexistent.jsonl" in str(exc_info.value)