---
id: 09-quality
title: "09-Quality — Data Quality Platform"
status: stable
applies_to:
  - "src/dvas/quality/**"
code_anchors:
  - "src/dvas/quality/analyzer.py:DataQualityAnalyzer"
  - "src/dvas/quality/analyzer.py:DatasetQualityMetrics"
  - "src/dvas/quality/analyzer.py:AnomalyDetector"
  - "src/dvas/quality/analyzer.py:DataAugmenter"
agent_hints:
  - "WARNING: AnomalyDetector uses Z-score - assumes normal distribution"
  - "WARNING: Duplicate detection uses Jaccard similarity - may miss semantic duplicates"
  - "WARNING: DataAugmenter paraphrase is placeholder - use real model in production"
  - "WARNING: Quality reports should be reviewed regularly for data drift"
---

# §09 Data Quality

Comprehensive data quality analysis, anomaly detection, and augmentation.

---

## §0 — One-liner

Analyze annotation quality, detect anomalies/duplicates, and augment training data through synthetic generation.

## §1 — Core concepts

- **DataQualityAnalyzer**: Main analysis orchestrator
- **DatasetQualityMetrics**: Statistical quality metrics
- **DataDistribution**: Distribution of verbs, nouns, durations
- **AnomalyDetector**: Z-score and similarity-based outlier detection
- **DataAugmenter**: Paraphrase, temporal shift, object swap

## §2 — Entry points (`code_anchors:` quick reference)

| Anchor | Purpose | When to use |
|--------|---------|-------------|
| `analyzer.py:DataQualityAnalyzer` | Full dataset analysis | Quality audits |
| `analyzer.py:DatasetQualityMetrics` | Metrics structure | Report generation |
| `analyzer.py:AnomalyDetector` | Find outliers | Data cleaning |
| `analyzer.py:DataAugmenter` | Generate variants | Training expansion |

## §3 — Key behaviors & contracts

### Behavior 1: Quality Analysis

```python
analyzer = DataQualityAnalyzer()
metrics, distribution = analyzer.analyze_dataset(source="gold")  # load_all() is materialized internally

# Key metrics:
# - vocabulary_size: Unique words
# - action_balance_score: Entropy-based balance
# - temporal_coverage: Annotated / total duration
# - missing_fields_rate: % of missing data
```

### Behavior 2: Anomaly Detection

```python
detector = AnomalyDetector(z_threshold=2.5)
outliers = detector.detect_outliers(annotations)
# Returns: [(annotation_id, reason), ...]

duplicates = detector.detect_duplicates(annotations, similarity_threshold=0.9)
# Returns: [(id1, id2, similarity), ...]
```

**Detection methods**:
- Z-score for statistical outliers
- Jaccard similarity for near-duplicates

### Behavior 3: Data Augmentation

```python
augmenter = DataAugmenter()
augmented = augmenter.augment_annotation(annotation, strategy="paraphrase")

strategies = ["paraphrase", "temporal_shift", "object_swap"]
```

### Behavior 4: Quality Report Generation

```python
report_path = analyzer.generate_quality_report(
    output_path=Path("reports/quality.json"),
    source="gold"
)
```

**Report includes**:
- All metrics
- Distribution histograms
- Recommendations

## §4 — Integration with other subsystems

- **Upstream**: Consumes annotations from `01-data`
- **Downstream**: Augmented data feeds into `06-export`
- **Related**: Results inform `03-student` training

## §5 — Current state & known gaps

| Aspect | Status | Notes |
|--------|--------|-------|
| Statistical analysis | Complete | 10+ metrics; safely materializes store generators for len/sample |
| Anomaly detection | Complete | Z-score + duplicates |
| Report generation | Complete | JSON format; duplicate detection uses a materialized annotation list |
| Data augmentation | Partial | Placeholder implementations |
| Semantic duplicate detection | Missing | Needs embeddings |
| Visual quality checks | Missing | Future enhancement |

## §6 — Testing

```bash
# Run quality tests
pytest tests/test_quality.py -v

# Generate sample report
python -c "
from dvas.quality.analyzer import DataQualityAnalyzer
analyzer = DataQualityAnalyzer()
metrics, distribution = analyzer.analyze_dataset()
print(f'Vocabulary size: {metrics.vocabulary_size}')
print(f'Action balance: {metrics.action_balance_score:.2f}')
"
```

---

*Subsystem doc: 09-quality | Updated: 2026-06-19*
