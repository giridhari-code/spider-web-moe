#!/usr/bin/env python3
"""Convert JSONL {messages:[{role,content},...]} to assistant-masked SFT JSONL."""
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
            row=json.loads(line);messages=row.get('messages')
            if not messages:raise ValueError(f'line {ln}: expected messages array')
            encoded=tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=False,return_dict=True,return_assistant_tokens_mask=True)
            ids=list(encoded['input_ids']);mask=encoded.get('assistant_masks')
            if mask is None:raise RuntimeError('Tokenizer chat template does not expose assistant_masks; add {% generation %} markers to its assistant template.')
            labels=[int(t) if bool(m) else -100 for t,m in zip(ids,mask)]
            dst.write(json.dumps({'input_ids':ids,'labels':labels},separators=(',',':'))+'\n');n+=1
    print(f'Wrote {n} assistant-masked examples to {a.output}')
if __name__=='__main__':main()
