#!/usr/bin/env bash
set -euo pipefail

# Fit BPE on a bounded, evenly spaced sample of training pages only.
python scripts/download_wikipedia.py --output-dir data/wikipedia --max-pages "${WIKI_MAX_PAGES:-20000}"
python scripts/train_custom_tokenizer.py \
  --input data/wikipedia/train.txt \
  --output-dir data/tokenizer \
  --vocab-size "${VOCAB_SIZE:-8192}" \
  --max-train-chars "${TOKENIZER_MAX_CHARS:-10000000}"
python scripts/tokenize_wikipedia.py

# Generate a runtime config without modifying the checked-in starter config.
python - <<'PY'
import json, yaml
from pathlib import Path
base = Path('configs/t4-wikipedia-pretrain.yaml')
cfg = yaml.safe_load(base.read_text())
meta = json.loads(Path('data/tokenizer/tokenizer_config.json').read_text())
cfg['model']['vocab_size'] = int(meta['vocab_size'])
out = Path('configs/t4-wikipedia-pretrain.runtime.yaml')
out.write_text(yaml.safe_dump(cfg, sort_keys=False))
print('Runtime model vocab_size:', meta['vocab_size'])
print('Runtime config:', out)
PY
python -m spider_moe.cli train-pipeline --config configs/t4-wikipedia-pretrain.runtime.yaml --device auto
