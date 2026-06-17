---
id: 11-monitoring
title: "11-Monitoring — A/B Testing & Drift Detection"
status: stable
applies_to:
  - "src/dvas/monitoring/**"
code_anchors:
  - "src/dvas/monitoring/ab_testing.py:ABTestManager"
  - "src/dvas/monitoring/ab_testing.py:ABTestConfig"
  - "src/dvas/monitoring/ab_testing.py:DriftDetector"
  - "src/dvas/monitoring/ab_testing.py:PerformanceMonitor"
agent_hints:
  - "WARNING: A/B tests require minimum sample size for statistical significance"
  - "WARNING: Drift detection is based on simple statistics - use proper drift detection in production"
  - "WARNING: Performance monitoring keeps metrics in memory - configure persistence"
  - "WARNING: T-test assumes normal distribution"
---

# §11 Monitoring

A/B testing framework for model comparison and drift detection.

---

## §0 — One-liner

Statistical A/B testing for model evaluation, data drift detection, and performance monitoring.

## §1 — Core concepts

- **ABTestManager**: Orchestrate and analyze A/B tests
- **ABTestConfig**: Test configuration (variants, metrics, thresholds)
- **TestResult**: Statistical analysis results
- **DriftDetector**: Detect data and model drift
- **PerformanceMonitor**: Rolling window performance metrics

## §2 — Entry points (`code_anchors:` quick reference)

| Anchor | Purpose | When to use |
|--------|---------|-------------|
| `ab_testing.py:ABTestManager` | Create & manage tests | Model comparison |
| `ab_testing.py:ABTestConfig` | Test configuration | Test setup |
| `ab_testing.py:DriftDetector` | Detect drift | Production monitoring |
| `ab_testing.py:PerformanceMonitor` | Track metrics | Continuous monitoring |

## §3 — Key behaviors & contracts

### Behavior 1: A/B Test Setup

```python
config = ABTestConfig(
    test_name="gpt4v_vs_claude",
    variant_a="gpt-4v",
    variant_b="claude-opus",
    traffic_split=0.5,
    min_sample_size=100,
    primary_metric="quality_score",
    mde=0.05,  # Minimum detectable effect
)

test_id = manager.create_test(config)
```

### Behavior 2: Variant Assignment

```python
variant = manager.assign_variant(test_id, video_id)
# Deterministic assignment based on hash
# Same video_id always gets same variant
```

### Behavior 3: Statistical Analysis

```python
result = manager.analyze_test(test_id)

# Result includes:
# - p_value: Statistical significance
# - effect_size: Cohen's d
# - confidence_interval: CI for difference
# - winner: A, B, or None (no significant difference)
# - recommendations: Actionable insights
```

### Behavior 4: Drift Detection

```python
detector = DriftDetector(reference_data=gold_annotations)
drift = detector.detect_drift(new_annotations)

# Checks:
# - Caption length distribution
# - Segment count changes
# - Vocabulary drift (novel verbs)
```

### Behavior 5: Performance Monitoring

```python
monitor = PerformanceMonitor(window_size=100)
monitor.record({"latency": 150, "cost": 0.05, "quality": 0.85})

stats = monitor.get_statistics()
anomalies = monitor.check_anomalies(threshold_std=3.0)
```

## §4 — Integration with other subsystems

- **Upstream**: Uses `Annotation` from `01-data`
- **Upstream**: Integrates with `07-api` for traffic splitting
- **Related**: Results inform `08-routing` decisions

## §5 — Current state & known gaps

| Aspect | Status | Notes |
|--------|--------|-------|
| A/B test creation | Complete | Hash-based assignment |
| Statistical testing | Complete | T-test, effect size, CI |
| Drift detection | Partial | Simple statistics only |
| Performance monitoring | Complete | Rolling window |
| Anomaly detection | Complete | Z-score based |
| Sequential testing | Missing | Optional: early stopping |
| Bayesian testing | Missing | Alternative to frequentist |

## §6 — Testing

```bash
# Test A/B framework
python -c "
from dvas.monitoring.ab_testing import ABTestManager, ABTestConfig

manager = ABTestManager()
config = ABTestConfig(
    test_name='test',
    variant_a='model_a',
    variant_b='model_b'
)
test_id = manager.create_test(config)
print(f'Test created: {test_id}')
"
```

---

*Subsystem doc: 11-monitoring | Updated: 2024-06-17*
