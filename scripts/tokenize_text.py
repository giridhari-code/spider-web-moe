#!/usr/bin/env python3
"""Stream newline-delimited text through a Hugging Face tokenizer into uint32 IDs."""
import argparse
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser(); p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--tokenizer',required=True);p.add_argument('--eos',action='store_true');p.add_argument('--flush-tokens',type=int,default=1_000_000);a=p.parse_args()
    try: from transformers import AutoTokenizer
    except ImportError as e: raise SystemExit('Install optional tokenizer dependency: pip install -e ".[tokenizers]"') from e
    tok=AutoTokenizer.from_pretrained(a.tokenizer,use_fast=True); out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True); buf=[]; eos=tok.eos_token_id
    with open(a.input,encoding='utf-8') as src,open(out,'wb') as dst:
        for line in src:
            if not line.strip():continue
            buf.extend(tok.encode(line,add_special_tokens=False))
            if a.eos and eos is not None:buf.append(eos)
            if len(buf)>=a.flush_tokens:
                np.asarray(buf,dtype=np.uint32).tofile(dst);buf.clear()
        if buf:np.asarray(buf,dtype=np.uint32).tofile(dst)
    print(f'Wrote uint32 tokens to {out}; vocab={len(tok)}; eos={eos}')
if __name__=='__main__':main()
