from dataclasses import dataclass, field

@dataclass
class ModelParameters:
    batch_size: int = 128
    lr: float = 1e-4
    betas: tuple = field(default_factory=lambda: (None, None))
    layers_dim: int = 128
    noise_dim: int = 264
    n_cols: int = None
    seq_len: int = None
    condition: int = None
    n_critic: int = 1
    n_features: int = None
    tau_gs: float = 0.2
    generator_dims: list = field(default_factory=lambda: [256, 256])
    critic_dims: list = field(default_factory=lambda: [256, 256])
    l2_scale: float = 1e-6
    latent_dim: int = 128
    gp_lambda: float = 10.0
    pac: int = 10
    gamma: int = 1
    tanh: bool = False
    hidden_dim: int = 2

@dataclass
class TrainingParameters:
    cache_prefix: str = ''
    label_dim: int = None
    epochs: int = 300
    sample_interval: int = 50
    labels: list = field(default_factory=list)
    n_clusters: int = 10
    epsilon: float = 0.005
    log_frequency: bool = True
    measurement_cols: list = field(default_factory=list)
    sequence_length: int = 1
    number_sequences: int = 1
    sample_length: int = 1
    rounds: int = 1


