from contextlib import nullcontext

import torch


def cuda_autocast(device):
    if str(device).split(":", 1)[0] == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True)
    return nullcontext()
