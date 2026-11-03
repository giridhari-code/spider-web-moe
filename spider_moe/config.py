from dataclasses import dataclass, asdict
from pathlib import Path
import yaml

@dataclass
class ModelConfig:
    vocab_size: int = 32000
    hidden_size: int = 256
    num_heads: int = 8
    num_experts: int = 8
    top_k: int = 2
    edges_per_expert: int = 3
    num_loops: int = 6
    num_prelude_layers: int = 2
    num_coda_layers: int = 2
    expert_ffn_size: int = 512
    loop_embed_size: int = 32
    residual_scale: float = 0.5
    re_inject_weight: float = 0.1
    early_exit_threshold: float = 0.85
    dropout: float = 0.0
    embedding_init_std: float = 0.02
    max_seq_len: int = 2048
    use_rope: bool = True
    rope_theta: float = 10000.0
    # Expert parallel is active only when torch.distributed is initialized.
    expert_parallel: bool = True
    # Keep expert FFN weights on CPU; materialise only routed experts on GPU.
    offload_experts: bool = False
    # Max expert chunks cached on GPU when offload_experts=True (LRU).
    offload_max_gpu_experts: int = 4
    # Inference-only: set True then call model.quantize_moe_experts()
    moe_int4: bool = False
    moe_int4_group_size: int = 64

    def validate(self):
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if not 1 <= self.top_k <= self.num_experts:
            raise ValueError("top_k must be in [1, num_experts]")
        if not 1 <= self.edges_per_expert <= self.num_experts:
            raise ValueError("edges_per_expert must be in [1, num_experts]")
        if self.num_loops < 1:
            raise ValueError("num_loops must be >= 1")
        if self.embedding_init_std <= 0:
            raise ValueError("embedding_init_std must be > 0")
        return self

@dataclass
class TrainConfig:
    batch_size: int = 8
    seq_len: int = 256
    steps: int = 1000
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    warmup_steps: int = 100
    min_lr_ratio: float = 0.1
    aux_loss_weight: float = 0.01
    edge_loss_weight: float = 0.01
    log_every: int = 10
    save_every: int = 500
    grad_accum: int = 1
    data_path: str = "data/train.bin"
    checkpoint_dir: str = "checkpoints"
    seed: int = 1337

def load_config(path):
    raw = yaml.safe_load(Path(path).read_text()) or {}
    model = ModelConfig(**raw.get("model", {})).validate()
    train = TrainConfig(**raw.get("training", {}))
    return model, train
