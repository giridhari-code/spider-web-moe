import pytest
import torch
from spider_moe.paged_kv import PagedKVCache

def make_cache(**kw):
    return PagedKVCache(num_layers=2, num_loops=3, num_pages=8,
                        page_size=4, num_heads=2, head_dim=3,
                        dtype=torch.float32, device="cpu", **kw)

def test_append_crosses_pages_and_gather_roundtrip():
    cache = make_cache()
    k = torch.arange(7*2*3, dtype=torch.float32).view(7,2,3)
    v = k + 100
    assert cache.append("req-a", 0, 1, k[:3], v[:3]) == 3
    assert cache.append("req-a", 0, 1, k[3:], v[3:]) == 7
    got_k, got_v = cache.gather("req-a", 0, 1)
    assert torch.equal(got_k, k)
    assert torch.equal(got_v, v)
    assert cache.block_table("req-a",0,1).numel() == 2
    assert cache.sequence_length("req-a",0,1) == 7

def test_pools_are_separate_per_layer_and_loop():
    cache = make_cache()
    x = torch.ones(1,2,3)
    cache.append("a",0,0,x,x)
    assert cache.free_page_count(0,0) == 7
    assert cache.free_page_count(0,1) == 8
    assert cache.free_page_count(1,0) == 8

def test_free_recycles_pages_and_is_idempotent():
    cache = make_cache()
    x = torch.ones(9,2,3)
    cache.append("a",0,0,x,x)
    assert cache.free_page_count(0,0) == 5
    cache.free("a")
    assert cache.free_page_count(0,0) == 8
    assert cache.stats()["active_requests"] == 0
    cache.free("a")

def test_page_exhaustion_does_not_corrupt_pool():
    cache = PagedKVCache(1,1,1,page_size=4,num_heads=1,head_dim=2,dtype=torch.float32)
    x = torch.ones(5,1,2)
    with pytest.raises(RuntimeError):
        cache.append("too-big",0,0,x,x)
    assert cache.free_page_count(0,0) == 1

def test_block_table_device_and_padding():
    cache = make_cache()
    x = torch.ones(5,2,3)
    cache.append("a",1,2,x,x)
    table = cache.block_table("a",1,2,max_blocks=4)
    assert table.dtype == torch.int32
    assert table.tolist()[-2:] == [-1,-1]
