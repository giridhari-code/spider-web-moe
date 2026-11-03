#!/usr/bin/env python3
"""Stream English Wikipedia into deterministic train/validation UTF-8 text files."""
import argparse, hashlib
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output-dir', default='data/wikipedia')
    p.add_argument('--config', default='20231101.en')
    p.add_argument('--max-pages', type=int, default=100_000)
    p.add_argument('--validation-percent', type=int, default=2)
    p.add_argument('--min-chars', type=int, default=200)
    p.add_argument('--trust-remote-code', action='store_true')
    a = p.parse_args()
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise SystemExit('Install data dependencies: pip install -e ".[wikipedia]"') from e
    if not 0 <= a.validation_percent < 50:
        raise SystemExit('--validation-percent must be in [0, 49]')
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    train_path, val_path = out/'train.txt', out/'validation.txt'
    count = train_chars = val_chars = 0
    # Wikimedia's dated English Wikipedia snapshot; streaming avoids downloading the full corpus.
    ds = load_dataset('wikimedia/wikipedia', a.config, split='train', streaming=True,
                      trust_remote_code=a.trust_remote_code)
    with train_path.open('w', encoding='utf-8') as train, val_path.open('w', encoding='utf-8') as val:
        for row in ds:
            text = ' '.join(str(row.get('text', '')).split())
            if len(text) < a.min_chars:
                continue
            identity = row.get('id') or row.get('title') or f'page-{count}'
            key = str(identity).encode('utf-8')
            bucket = int.from_bytes(hashlib.blake2b(key, digest_size=4).digest(), 'little') % 100
            target = val if bucket < a.validation_percent else train
            target.write(text + '\n')
            if target is val: val_chars += len(text)
            else: train_chars += len(text)
            count += 1
            if count % 1000 == 0: print(f'pages={count:,} train_chars={train_chars:,} val_chars={val_chars:,}')
            if count >= a.max_pages: break
    if train_chars == 0 or val_chars == 0:
        raise RuntimeError(f'Insufficient corpus split: train_chars={train_chars}, val_chars={val_chars}. Increase --max-pages.')
    print(f'Wrote {count:,} pages. Train: {train_path} ({train_chars:,} chars); validation: {val_path} ({val_chars:,} chars)')

if __name__ == '__main__': main()
