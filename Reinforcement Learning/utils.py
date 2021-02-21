import torch
import numpy as np

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    np.random.seed(seed)


def total_params(agent):
    model = agent.network
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
