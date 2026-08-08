"""Device and optional Hugging Face auth setup.

Exports:
- `device`: Preferred torch device from transformer_lens.
- `num_gpus`: Number of visible CUDA devices.
"""

import torch
import os
from transformer_lens.utils import get_device
from huggingface_hub import login

device = get_device()
num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

# Login is optional and only enabled when a token is present in env vars.
_hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
if _hf_token:
    login(token=_hf_token)
