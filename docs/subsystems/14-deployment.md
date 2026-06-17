---
id: 14-deployment
title: "14-Deployment — Edge Optimization"
status: draft
applies_to:
  - "src/dvas/deployment/**"
code_anchors:
  - "src/dvas/deployment/edge.py:ONNXExporter"
  - "src/dvas/deployment/edge.py:TensorRTOptimizer"
  - "src/dvas/deployment/edge.py:ModelQuantizer"
  - "src/dvas/deployment/edge.py:EdgeInferenceEngine"
agent_hints:
  - "WARNING: ONNX export requires model weights - not included in repo"
  - "WARNING: TensorRT requires NVIDIA GPU and proprietary software"
  - "WARNING: Quantization may reduce model accuracy - validate before deployment"
  - "WARNING: Mobile exports (CoreML/TFLite) are placeholders"
---

# §14 Deployment

Edge deployment utilities for model optimization and inference.

---

## §0 — One-liner

Export models to ONNX/TensorRT, apply quantization, and deploy to edge devices.

## §1 — Core concepts

- **ONNXExporter**: Export to ONNX format
- **TensorRTOptimizer**: NVIDIA TensorRT optimization
- **ModelQuantizer**: INT8 quantization (static/dynamic)
- **EdgeInferenceEngine**: Optimized inference runtime
- **MobileExporter**: CoreML and TFLite export

## §2 — Entry points (`code_anchors:` quick reference)

| Anchor | Purpose | When to use |
|--------|---------|-------------|
| `edge.py:ONNXExporter` | Export to ONNX | Cross-platform deployment |
| `edge.py:TensorRTOptimizer` | TensorRT optimization | NVIDIA edge devices |
| `edge.py:ModelQuantizer` | Quantize model | Reduce size/latency |
| `edge.py:EdgeInferenceEngine` | Edge inference | Production inference |

## §3 — Key behaviors & contracts

### Behavior 1: ONNX Export

```python
exporter = ONNXExporter(opset_version=17)

# Export vision encoder
onnx_path = exporter.export_vision_encoder(
    model_path=Path("models/student"),
    output_path=Path("models/student_vision.onnx"),
    input_shape=(1, 3, 448, 448),
)

# Export text decoder
onnx_path = exporter.export_text_decoder(
    model_path=Path("models/student"),
    output_path=Path("models/student_decoder.onnx"),
    max_length=512,
)
```

### Behavior 2: TensorRT Optimization

```python
optimizer = TensorRTOptimizer(fp16=True, max_batch_size=1)

engine_path = optimizer.optimize(
    onnx_path=Path("model.onnx"),
    output_path=Path("model.trt"),
)
```

**Requirements**:
- NVIDIA GPU
- TensorRT library
- CUDA toolkit

### Behavior 3: Model Quantization

```python
quantizer = ModelQuantizer()

# Static quantization (requires calibration data)
quantized = quantizer.quantize_static(
    model_path=Path("model.onnx"),
    output_path=Path("model_int8.onnx"),
    calibration_data=calibration_images,
)

# Dynamic quantization (no calibration needed)
quantized = quantizer.quantize_dynamic(
    model_path=Path("model.onnx"),
    output_path=Path("model_int8.onnx"),
)
```

### Behavior 4: Edge Inference

```python
engine = EdgeInferenceEngine(
    model_path=Path("model.onnx"),
    use_onnx=True,
    use_tensorrt=False,
)

# Run inference
outputs = engine.predict({
    "input_ids": input_ids,
    "attention_mask": attention_mask,
})
```

**Backends supported**:
- ONNX Runtime (CPU/CUDA)
- TensorRT (NVIDIA GPU)
- PyTorch (fallback)

### Behavior 5: Mobile Export

```python
exporter = MobileExporter()

# iOS (CoreML)
coreml_path = exporter.export_coreml(
    model_path=Path("models/student"),
    output_path=Path("model.mlpackage"),
    compute_units="ALL",  # CPU, GPU, Neural Engine
)

# Android (TFLite) - Placeholder
tflite_path = exporter.export_tflite(
    model_path=Path("models/student"),
    output_path=Path("model.tflite"),
    quantization="int8",
)
```

## §4 — Integration with other subsystems

- **Upstream**: Consumes trained `03-student` models
- **Downstream**: Deployed models replace cloud API calls
- **Related**: Used by `08-routing` for edge deployment option

## §5 — Current state & known gaps

| Aspect | Status | Notes |
|--------|--------|-------|
| ONNX export | Complete | Vision + text models |
| TensorRT | Partial | Requires external dependencies |
| Quantization | Complete | Static + dynamic |
| Edge inference | Complete | ONNX Runtime |
| CoreML export | Partial | Requires testing |
| TFLite export | Missing | Needs implementation |
| Model optimization | Missing | Pruning, distillation |

## §6 — Testing

```bash
# Verify exports work
python -c "
from dvas.deployment.edge import ONNXExporter, ModelQuantizer

# These require actual model files
exporter = ONNXExporter()
print('ONNXExporter initialized')
"
```

---

*Subsystem doc: 14-deployment | Updated: 2024-06-17*
