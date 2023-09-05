from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass
class RunConfig:
    # Guiding text prompt
    prompt: str
    # Which token indices to alter with attend-and-excite
    token_indices: List[int] = None
    # Which random seeds to use when generating
    seeds: List[int] = field(default_factory=lambda: [21])
    # Path to save all outputs to
    output_path: Path = Path('./outputs')
    # Number of denoising steps
    num_inference_steps: int = 50
    # Text guidance scale
    guidance_scale: float = 7.5
    # Resolution of UNet to compute attention maps over
    attention_res: int = 16
    # Whether to run standard SD or attend-and-excite
    run_standard: bool = False
    # Scale factor for updating the denoised latent z_t
    scale_factor: int = 20
    # Start and end values used for scaling the scale factor - decays linearly with the denoising timestep
    scale_range: tuple = field(default_factory=lambda: (1.0, 0.5))
    # Whether to save cross attention maps for the final results
    save_attention_maps: bool = False

    def __post_init__(self):
        self.output_path.mkdir(exist_ok=True, parents=True)
