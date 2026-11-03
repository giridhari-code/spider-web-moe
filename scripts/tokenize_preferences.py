#!/usr/bin/env python3
"""Convert JSONL {prompt,chosen,rejected} text triples to DPO token IDs."""
import argparse,json
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--tokenizer',required=True);a=p.parse_args()
    try:from transformers import AutoTokenizer
    except ImportError as e:raise SystemExit('Install optional tokenizer dependency: pip install -e ".[tokenizers]"') from e
    tok=AutoTokenizer.from_pretrained(a.tokenizer,use_fast=True);Path(a.output).parent.mkdir(parents=True,exist_ok=True);n=0
    with open(a.input,encoding='utf-8') as src,open(a.output,'w',encoding='utf-8') as dst:
        for ln,line in enumerate(src,1):
            if not line.strip():continue
            row=json.loads(line)
            for k in ('prompt','chosen','rejected'):
                if k not in row:raise ValueError(f'line {ln}: missing {k}')
            prompt=tok.encode(row['prompt'],add_special_tokens=True);chosen=tok.encode(row['chosen'],add_special_tokens=False);rejected=tok.encode(row['rejected'],add_special_tokens=False)
            dst.write(json.dumps({'prompt_ids':prompt,'chosen_ids':chosen,'rejected_ids':rejected},separators=(',',':'))+'\n');n+=1
    print(f'Wrote {n} preference pairs to {a.output}')
if __name__=='__main__':main()
