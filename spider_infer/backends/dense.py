from spider_infer.backends.registry import register

@register("dense")
def build_dense(model_config=None, device="cpu", **kwargs):
    from spider_moe.dense_model import DenseConfig, DenseLM
    cfg = model_config if model_config is not None else DenseConfig()
    if not isinstance(cfg, DenseConfig):
        # allow dict
        cfg = DenseConfig(**cfg) if isinstance(cfg, dict) else DenseConfig()
    return DenseLM(cfg).to(device).eval()
