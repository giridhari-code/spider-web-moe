import torch
import pytest

from spider_moe.global_paged_kv import GlobalPagedKVCache


def make_cache(**kwargs):
    return GlobalPagedKVCache(
        num_layers=3, num_pages=6, page_size=4, num_heads=2, head_dim=3,
        max_requests=4, max_blocks_per_request=8,
        dtype=torch.float32, device="cpu", **kwargs
    )


def test_global_pool_shared_across_layers_and_requests():
    cache = make_cache()
    x = torch.arange(5*2*3, dtype=torch.float32).view(5,2,3)
    cache.allocate("a")
    cache.write_layer("a", 0, 0, x, x+100)
    cache.write_layer("a", 1, 0, x+200, x+300)
    assert cache.stats()["used_pages"] == 2
    k0, v0 = cache.read_layer("a", 0)
    k1, v1 = cache.read_layer("a", 1)
    assert torch.equal(k0, x)
    assert torch.equal(v0, x+100)
    assert torch.equal(k1, x+200)
    assert torch.equal(v1, x+300)


def test_device_block_table_is_stable_view():
    cache = make_cache()
    cache.allocate("a")
    table = cache.block_table("a")
    ptr = table.data_ptr()
    x = torch.ones(5,2,3)
    cache.write_layer("a", 0, 0, x, x)
    assert cache.block_table("a").data_ptr() == ptr
    assert table[:2].tolist() == [0,1]


def test_prefix_fork_copy_on_write():
    cache = make_cache()
    base = torch.arange(4*2*3, dtype=torch.float32).view(4,2,3)
    cache.allocate("parent")
    for layer in range(3):
        cache.write_layer("parent", layer, 0, base+layer*100, base+layer*200)
    cache.fork_prefix("parent", "child", prefix_len=3)
    assert cache.stats()["shared_pages"] == 1

    new = torch.full((2,2,3), 999.0)
    cache.write_layer("child", 0, 3, new, new)
    assert cache.stats()["shared_pages"] == 0
    parent_k, _ = cache.read_layer("parent", 0)
    child_k, _ = cache.read_layer("child", 0)
    assert torch.equal(parent_k, base)
    assert torch.equal(child_k[:3], base[:3])
    assert torch.equal(child_k[3:], new)

    # Parent still has its own original layer-1 data in the shared prefix.
    parent_k1, _ = cache.read_layer("parent", 1)
    assert torch.equal(parent_k1, base+100)


def test_free_recycles_global_pages_and_slots():
    cache = make_cache()
    x = torch.ones(9,2,3)
    cache.allocate("a")
    cache.write_layer("a", 0, 0, x, x)
    assert cache.stats()["used_pages"] == 3
    cache.free("a")
    assert cache.stats()["used_pages"] == 0
    assert cache.stats()["active_requests"] == 0


def test_exhaustion_is_rolled_back():
    cache = GlobalPagedKVCache(
        num_layers=1, num_pages=1, page_size=4, num_heads=1, head_dim=2,
        max_requests=2, max_blocks_per_request=4, dtype=torch.float32
    )
    cache.allocate("a")
    x = torch.ones(5,1,2)
    with pytest.raises(RuntimeError):
        cache.write_layer("a", 0, 0, x, x)
    assert cache.stats()["used_pages"] == 0


def test_invalid_prefix_and_unknown_request():
    cache = make_cache()
    with pytest.raises(KeyError):
        cache.block_table("missing")
    cache.allocate("a")
    x = torch.ones(2,2,3)
    cache.write_layer("a", 0, 0, x, x)
    with pytest.raises(ValueError):
        cache.fork_prefix("a", "b", prefix_len=3)


def test_child_keeps_shared_prefix_after_parent_freed():
    cache = make_cache()
    x = torch.arange(4*2*3, dtype=torch.float32).view(4,2,3)
    cache.allocate("parent")
    cache.write_layer("parent", 0, 0, x, x)
    cache.fork_prefix("parent", "child", prefix_len=4)
    cache.free("parent")
    child_k, child_v = cache.read_layer("child", 0)
    assert torch.equal(child_k, x)
    assert torch.equal(child_v, x)
    assert cache.stats()["used_pages"] == 1
