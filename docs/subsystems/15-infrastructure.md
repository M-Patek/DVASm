---
id: 15-infrastructure
title: "15-Infrastructure — Data Platform Deployment"
status: draft
applies_to:
  - "src/dvas/infrastructure/**"
agent_hints:
  - "WARNING: This module is a thin wrapper around deployment concerns"
---

# §15 Infrastructure

Data platform deployment and operational infrastructure.

---

## §0 — One-liner

Deployment utilities for the DVAS data platform (not model deployment — see `03-student` for ONNX/TensorRT).

## §1 — Scope

Replaces the original `14-Deployment` which incorrectly mixed data platform
and model deployment concerns. This module focuses on:

- Data platform containerization (Docker/K8s)
- Annotation store backup/restore
- Index store maintenance
- API service scaling

Model deployment (ONNX/TensorRT/CoreML) belongs in `03-student`.

## §2 — Current State

| Aspect | Status | Notes |
|--------|--------|-------|
| Docker packaging | Missing | Needs Dockerfile |
| K8s manifests | Missing | Needs Helm chart |
| Store backup | Missing | Needs scheduled backup |
| Index maintenance | Missing | Needs compaction/rebuild |

---

*Subsystem doc: 15-infrastructure | Updated: 2026-06-19*
