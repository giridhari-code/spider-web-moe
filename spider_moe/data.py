"""Memory-mapped and pre-tokenized datasets for all training stages."""
import json
import mmap
from pathlib import Path
import torch
from torch.utils.data import Dataset

class TokenBinaryDataset(Dataset):
    """uint32 little-endian token stream; each item is a shifted LM window."""
    def __init__(self, path, seq_len):
        self.path, self.seq_len = str(path), int(seq_len)
        self.file = open(self.path, "rb")
        byte_size = Path(self.path).stat().st_size
        if byte_size == 0:
            self.file.close()
            raise ValueError(f"{self.path}: token file is empty")
        if byte_size % 4:
            self.file.close()
            raise ValueError(f"{self.path}: uint32 token file size must be divisible by 4")
        self.mm = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
        self.count = len(self.mm) // 4
        if self.count <= self.seq_len:
            self.close()
            raise ValueError(f"{path} must contain more than seq_len={seq_len} tokens")
    def __len__(self): return max(1, self.count - self.seq_len)
    def __getitem__(self, index):
        start = (int(index) % len(self)) * 4
        raw = self.mm[start:start + (self.seq_len + 1) * 4]
        tok = torch.frombuffer(bytearray(raw), dtype=torch.int32).long()
        return tok[:-1], tok[1:]
    def close(self):
        if getattr(self, "mm", None) is not None: self.mm.close(); self.mm = None
        if getattr(self, "file", None) is not None: self.file.close(); self.file = None

class JsonlSFTDataset(Dataset):
    """JSONL rows: {input_ids:[...], labels:[...]}; prompt labels use -100."""
    def __init__(self, path, seq_len, pad_token_id=0):
        self.rows=[]; self.seq_len=int(seq_len); self.pad=int(pad_token_id)
        with open(path, encoding="utf-8") as f:
            for line_no,line in enumerate(f,1):
                if not line.strip(): continue
                row=json.loads(line); ids=row.get("input_ids"); labels=row.get("labels")
                if ids is None: raise ValueError(f"{path}:{line_no}: missing input_ids (pre-tokenize first)")
                ids=list(map(int,ids)); labels=list(map(int, labels if labels is not None else ids))
                if len(ids)!=len(labels): raise ValueError(f"{path}:{line_no}: labels length mismatch")
                ids=ids[:self.seq_len+1]; labels=labels[:self.seq_len+1]
                if len(ids)<2: continue
                self.rows.append((ids[:-1],labels[1:]))
        if not self.rows: raise ValueError(f"No usable SFT examples in {path}")
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        x,y=self.rows[i]; n=self.seq_len-len(x)
        return (torch.tensor(x+[self.pad]*n,dtype=torch.long),
                torch.tensor(y+[-100]*n,dtype=torch.long))

class JsonlPreferenceDataset(Dataset):
    """JSONL rows: prompt_ids, chosen_ids, rejected_ids (all pre-tokenized)."""
    def __init__(self,path,seq_len):
        self.rows=[]; self.seq_len=int(seq_len)
        with open(path,encoding="utf-8") as f:
            for ln,line in enumerate(f,1):
                if not line.strip(): continue
                r=json.loads(line)
                try: p=list(map(int,r["prompt_ids"])); c=list(map(int,r["chosen_ids"])); rej=list(map(int,r["rejected_ids"]))
                except Exception as e: raise ValueError(f"{path}:{ln}: expected prompt_ids/chosen_ids/rejected_ids") from e
                if not c or not rej: continue
                self.rows.append((p,c,rej))
        if not self.rows: raise ValueError(f"No preference pairs in {path}")
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        p,c,r=self.rows[i]; max_resp=max(1,self.seq_len-len(p))
        c=c[:max_resp]; r=r[:max_resp]
        return p,c,r
