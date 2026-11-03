from pathlib import Path
import json
import numpy as np
import torch
from spider_moe.data import TokenBinaryDataset, JsonlSFTDataset, JsonlPreferenceDataset
from spider_moe.config import ModelConfig
from spider_moe.model import SpiderWebLM
from spider_moe.trainer import _model_loss, _sequence_logps


def test_uint32_and_sft_datasets(tmp_path):
    p=tmp_path/'tokens.bin'; np.arange(40,dtype=np.uint32).tofile(p)
    ds=TokenBinaryDataset(p,8); x,y=ds[2]
    assert x.shape==(8,) and torch.equal(y[:-1],x[1:])
    ds.close()
    s=tmp_path/'sft.jsonl'
    s.write_text(json.dumps({'input_ids':[1,2,3,4],'labels':[-100,-100,3,4]})+'\n')
    sd=JsonlSFTDataset(s,6); xx,yy=sd[0]
    assert xx.shape==(6,) and yy.tolist()==[-100,3,4,-100,-100,-100]


def test_preference_dataset_and_logps(tmp_path):
    p=tmp_path/'prefs.jsonl'; p.write_text(json.dumps({'prompt_ids':[1,2],'chosen_ids':[3,4],'rejected_ids':[5,6]})+'\n')
    ds=JsonlPreferenceDataset(p,8); prompt,chosen,rejected=ds[0]
    assert prompt==[1,2] and chosen==[3,4] and rejected==[5,6]
    cfg=ModelConfig(vocab_size=32,hidden_size=16,num_heads=4,num_experts=2,top_k=1,
        edges_per_expert=1,num_loops=1,num_prelude_layers=1,num_coda_layers=1,
        expert_ffn_size=32,loop_embed_size=4,max_seq_len=16)
    model=SpiderWebLM(cfg)
    assert torch.isfinite(_sequence_logps(model,prompt,chosen,torch.device('cpu'),8))


def test_masked_sft_loss_and_backward():
    cfg=ModelConfig(vocab_size=32,hidden_size=16,num_heads=4,num_experts=2,top_k=1,
        edges_per_expert=1,num_loops=1,num_prelude_layers=1,num_coda_layers=1,
        expert_ffn_size=32,loop_embed_size=4,max_seq_len=16)
    model=SpiderWebLM(cfg)
    x=torch.randint(0,32,(2,6)); labels=torch.full((2,6),-100); labels[:,3:]=torch.randint(0,32,(2,3))
    loss,parts=_model_loss(model,x,labels,.005,.005)
    assert torch.isfinite(loss) and torch.isfinite(parts['ce'])
    loss.backward()

def test_model_loss_rejects_all_ignored_labels():
    cfg=ModelConfig(vocab_size=32,hidden_size=16,num_heads=4,num_experts=2,top_k=1,
        edges_per_expert=1,num_loops=1,num_prelude_layers=1,num_coda_layers=1,
        expert_ffn_size=32,loop_embed_size=4,max_seq_len=16)
    model=SpiderWebLM(cfg)
    x=torch.randint(0,32,(1,6)); labels=torch.full_like(x,-100)
    try:
        _model_loss(model,x,labels,.005,.005)
    except ValueError as exc:
        assert "no supervised target tokens" in str(exc)
    else:
        raise AssertionError("expected all-ignored batch validation")


def test_empty_binary_dataset_fails_cleanly(tmp_path):
    p = tmp_path / "empty.bin"
    p.write_bytes(b"")
    try:
        TokenBinaryDataset(p, 8)
    except ValueError as exc:
        assert "empty" in str(exc)
    else:
        raise AssertionError("expected empty token file validation")


def test_tokenizer_sample_is_bounded_and_spans_source(tmp_path):
    import importlib.util
    script = Path(__file__).resolve().parents[1] / "scripts" / "train_custom_tokenizer.py"
    spec = importlib.util.spec_from_file_location("train_custom_tokenizer", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = tmp_path / "wiki.txt"
    # Different line prefixes make it possible to check that sampling advances
    # through the source rather than taking only the first contiguous prefix.
    source.write_text("".join(f"PAGE-{i:04d}-" + ("x" * 90) + "\n" for i in range(100)))
    sample = tmp_path / "sample.txt"
    written = module.make_sample(source, sample, 350)
    text = sample.read_text()
    assert 0 < written <= 350
    assert "PAGE-0000" in text
    assert "PAGE-0087" in text
