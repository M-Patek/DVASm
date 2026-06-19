---
id: 03-student
title: "03-Student (VLA) — Qwen2-VL Fine-tuning"
status: stable
applies_to:
  - "src/dvas/models/student/**"
code_anchors:
  - "src/dvas/models/student/config.py:SFTConfig"
  - "src/dvas/models/student/sft_trainer.py:train_sft"
  - "src/dvas/models/student/dpo_trainer.py:train_dpo"
  - "src/dvas/models/student/inference.py:StudentInferenceEngine"
agent_hints:
  - "WARNING: Training requires GPU with 16GB+ VRAM for 7B model with 4-bit quantization"
  - "WARNING: SFT should be completed before DPO"
  - "WARNING: vLLM inference requires separate installation"
---

# §03 Student Model Training (VLA)

Fine-tune Qwen2-VL-7B on teacher-generated data using SFT + DPO pipeline for **VLA (Vision-Language-Action)** training.

---

## §0 — One-liner

SFT/DPO training pipeline for Qwen2-VL-7B with LoRA quantization, exporting to multiple inference formats for VLA applications.

## §1 — Core concepts

- **SFTConfig**: Training hyperparameters and paths
- **train_sft()**: Main SFT training function
- **StudentInferenceEngine**: Optimized inference (HF/vLLM)
- **StudentTeacherBridge**: Use student as teacher with fallback

## §2 — Entry points (`code_anchors:` quick reference)

| Anchor | Purpose | When to use |
|--------|---------|-------------|
| `config.py:SFTConfig` | Training configuration | Setting up training |
| `sft_trainer.py:train_sft` | Run SFT training | Training student |
| `dpo_trainer.py:train_dpo` | Run DPO refinement | Preference optimization |
| `inference.py:StudentInferenceEngine` | Inference engine | Deploy student |

## §3 — Key behaviors & contracts

### Behavior 1: Training Pipeline

```bash
# 1. SFT training
python -m dvas.models.student.sft_trainer \
  --train-data data/exports/train_llava.jsonl \
  --output-dir outputs/student_sft \
  --epochs 3 --batch-size 1

# 2. DPO training (optional)
python -m dvas.models.student.dpo_trainer \
  --sft-model outputs/student_sft/final \
  --dpo-data data/exports/dpo_pairs.jsonl \
  --output-dir outputs/student_dpo
```

### Behavior 2: Inference Modes

- **HF mode**: Standard transformers (slower, more compatible)
- **vLLM mode**: Optimized for throughput (requires `pip install vllm`)

### Behavior 3: Teacher Fallback

StudentTeacherBridge routes low-confidence predictions to teacher model.

## §4 — Integration with other subsystems

- **Upstream**: Consumes LLaVA/OpenAI formatted data from `06-export`
- **Downstream**: Replaces teacher in `04-pipeline` for cost reduction
- **Usage**: Can be called from `07-api` endpoints
- **Related**: Model deployment (ONNX/TensorRT) is a model-side concern, not data platform — see `15-infrastructure` for data platform deployment

## §5 — Current state & known gaps

| Aspect | Status | Notes |
|--------|--------|-------|
| SFT trainer | Complete | With LoRA, 4-bit quantization |
| DPO trainer | Complete | Reference model support |
| Inference engine | Complete | HF + vLLM support |
| Teacher fallback | Complete | Confidence-based routing |
| Distributed training | Missing | Requires DeepSpeed |
| World Model training | Missing | See `14-world-model` (placeholder) |

## §6 — Testing

```bash
# Dry run training (1 step)
python -m dvas.models.student.sft_trainer \
  --train-data data/exports/sample.jsonl \
  --epochs 1 --output-dir outputs/test

# Test inference
python -c "
from dvas.models.student.inference import StudentInferenceEngine
engine = StudentInferenceEngine('Qwen/Qwen2-VL-7B-Instruct')
result = engine.generate('data/raw/sample.mp4')
print(result)
"
```

---

*Subsystem doc: 03-student | Updated: 2026-06-19*
