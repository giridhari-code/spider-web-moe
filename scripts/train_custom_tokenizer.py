#!/usr/bin/env python3
"""Train byte-level BPE on a bounded, deterministic sample of training text."""
import argparse, json, math
from pathlib import Path


def make_sample(source: Path, sample: Path, max_chars: int):
    """Take evenly spaced lines across the corpus, bounded by max_chars."""
    size = source.stat().st_size
    if size <= max_chars:
        sample.write_bytes(source.read_bytes())
        return size
    stride = max(1, math.ceil(size / max_chars))
    written = 0
    line_no = 0
    with source.open('r', encoding='utf-8') as src, sample.open('w', encoding='utf-8') as dst:
        for line in src:
            if line_no % stride == 0 and written < max_chars:
                piece = line[:max_chars - written]
                dst.write(piece)
                written += len(piece)
            line_no += 1
    if written == 0:
        raise RuntimeError('Tokenizer sample is empty')
    return written


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', default='data/wikipedia/train.txt')
    p.add_argument('--output-dir', default='data/tokenizer')
    p.add_argument('--vocab-size', type=int, default=8192)
    p.add_argument('--min-frequency', type=int, default=2)
    p.add_argument('--max-train-chars', type=int, default=10_000_000,
                   help='Bound tokenizer training input; 0 uses the complete corpus')
    a = p.parse_args()
    if a.vocab_size < 300:
        raise SystemExit('--vocab-size must be at least 300 for byte-level BPE')
    if a.max_train_chars < 0:
        raise SystemExit('--max-train-chars must be >= 0')
    try:
        from tokenizers import ByteLevelBPETokenizer
    except ImportError as e:
        raise SystemExit('Install tokenizer dependencies: pip install -e ".[wikipedia]"') from e
    source = Path(a.input)
    if not source.is_file() or source.stat().st_size == 0:
        raise SystemExit(f'Input text file is missing or empty: {source}')
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    sample = out / 'tokenizer_training_sample.txt'
    train_input = source
    if a.max_train_chars and source.stat().st_size > a.max_train_chars:
        n = make_sample(source, sample, a.max_train_chars)
        train_input = sample
        print(f'Tokenizer sample: {n:,} characters from {source.stat().st_size:,} source bytes', flush=True)
    else:
        print(f'Tokenizer training on full corpus: {source.stat().st_size:,} bytes', flush=True)
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(files=[str(train_input)], vocab_size=a.vocab_size,
                    min_frequency=a.min_frequency,
                    special_tokens=['<|endoftext|>'])
    tokenizer.save_model(str(out))
    tokenizer._tokenizer.save(str(out/'tokenizer.json'))
    eos_id = tokenizer.token_to_id('<|endoftext|>')
    if eos_id is None:
        raise RuntimeError('Tokenizer did not register <|endoftext|>')
    info = {'tokenizer_type':'byte_level_bpe','vocab_size':tokenizer.get_vocab_size(),
            'requested_vocab_size':a.vocab_size,'eos_token':'<|endoftext|>',
            'eos_token_id':eos_id,'tokenizer_file':str(out/'tokenizer.json'),
            'training_sample_chars': train_input.stat().st_size}
    (out/'tokenizer_config.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    print(json.dumps(info, indent=2), flush=True)

if __name__ == '__main__': main()
