#!/usr/bin/env python3
"""Generate decoded text from a Spider-Web checkpoint and its custom tokenizer."""
import argparse, json
from pathlib import Path
import torch
from tokenizers import Tokenizer
from spider_moe.config import ModelConfig
from spider_moe.model import SpiderWebLM


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',default='checkpoints/wikipedia-pretrain/best.pt')
    p.add_argument('--tokenizer',default='data/tokenizer/tokenizer.json')
    p.add_argument('--prompt',default='The history of science')
    p.add_argument('--max-new-tokens',type=int,default=80)
    p.add_argument('--temperature',type=float,default=0.8)
    p.add_argument('--top-p',type=float,default=0.9)
    a=p.parse_args()
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    cfg=ModelConfig(**ck['model_config'])
    tok=Tokenizer.from_file(a.tokenizer)
    if tok.get_vocab_size()!=cfg.vocab_size:
        raise ValueError(f'tokenizer vocab {tok.get_vocab_size()} != checkpoint vocab {cfg.vocab_size}')
    ids=tok.encode(a.prompt).ids
    if not ids: raise ValueError('Prompt encoded to an empty token sequence')
    model=SpiderWebLM(cfg); model.load_state_dict(ck['model']); model.to(device).eval()
    x=torch.tensor([ids],dtype=torch.long,device=device)
    result=model.generate(x,a.max_new_tokens,a.temperature,a.top_p,eos_token_id=tok.token_to_id('<|endoftext|>'))
    print(tok.decode(result[0].tolist(),skip_special_tokens=True))

if __name__=='__main__': main()
