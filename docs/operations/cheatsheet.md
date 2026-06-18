# Operations Cheat Sheet

> Quick reference for common tasks.

---

## Setup

```bash
# Install
pip install -e ".[all]"

# Configure
cp .env.example .env
# Edit .env with your API keys
```

## Annotate EPIC-KITCHENS

```bash
# Small test batch
python -m dvas annotate epic \
  --split train \
  --participant P01 \
  --num 10

# Production batch
python -m dvas annotate epic \
  --split train \
  --num 1000 \
  --teacher gpt-5.5 \
  --workers 10
```

## Export Training Data

```bash
# LLaVA format
python -m dvas export \
  --source gold \
  --format llava \
  --output data/training/sft_llava.jsonl

# OpenAI format
python -m dvas export \
  --source gold \
  --format openai \
  --output data/training/sft_openai.jsonl
```

## Check Status

```bash
# Storage statistics
python -m dvas stats

# Subsystem health
python scripts/get_subsystem_status.py

# Known gaps
python scripts/check_known_gaps.py
```

## API Server (Future)

```bash
# Start API
python -m dvas.api --host 0.0.0.0 --port 8000

# Health check
curl http://localhost:8000/health
```

## Development

```bash
# Type check
mypy src/dvas

# Lint
ruff check src/dvas

# Test
pytest tests/ -v

# Check doc anchors
python scripts/check_doc_anchors.py
```

---

*Updated: 2024-06-17*
