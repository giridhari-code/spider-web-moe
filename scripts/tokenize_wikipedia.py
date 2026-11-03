#!/usr/bin/env python3
"""Encode Wikipedia text with tokenizer.json into little-endian uint32 streams."""
import argparse
from pathlib import Path
import numpy as np


def encode_file(input_path, output_path, tokenizer, eos_id, flush_tokens):
    out = Path(output_path); out.parent.mkdir(parents=True, exist_ok=True)
    buf=[]; total=0
    with open(input_path, encoding='utf-8') as src, out.open('wb') as dst:
        for line in src:
            line=line.strip()
            if not line: continue
            ids=tokenizer.encode(line).ids
            if eos_id is not None: ids.append(eos_id)
            buf.extend(ids)
            if len(buf) >= flush_tokens:
                arr=np.asarray(buf, dtype='<u4'); arr.tofile(dst); total += len(arr); buf.clear()
        if buf:
            arr=np.asarray(buf, dtype='<u4'); arr.tofile(dst); total += len(arr)
    print(f'{input_path} -> {out}: {total:,} tokens ({out.stat().st_size:,} bytes)')
    return total


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--train-text',default='data/wikipedia/train.txt')
    p.add_argument('--validation-text',default='data/wikipedia/validation.txt')
    p.add_argument('--tokenizer',default='data/tokenizer/tokenizer.json')
    p.add_argument('--output-dir',default='data/wikipedia_tokens')
    p.add_argument('--flush-tokens',type=int,default=1_000_000)
    a=p.parse_args()
    try:
        from tokenizers import Tokenizer
        import numpy
    except ImportError as e: raise SystemExit('Install dependencies: pip install -e ".[wikipedia]"') from e
    tok=Tokenizer.from_file(a.tokenizer)
    eos_id=tok.token_to_id('<|endoftext|>')
    if eos_id is None:
        raise RuntimeError("Tokenizer is missing required <|endoftext|> token")
    out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    tr=encode_file(a.train_text,out/'train.bin',tok,eos_id,a.flush_tokens)
    va=encode_file(a.validation_text,out/'validation.bin',tok,eos_id,a.flush_tokens)
    if min(tr,va) < 2: raise RuntimeError('Tokenized train and validation files must each contain at least 2 tokens')
    print(f'vocab_size={tok.get_vocab_size()} eos_token_id={eos_id}')

if __name__=='__main__': main()
