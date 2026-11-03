from spider_infer.backends.registry import register

@register("spider")
def build_spider(model_config, device="cpu", **kwargs):
    from spider_moe.model import SpiderWebLM
    return SpiderWebLM(model_config).to(device).eval()
