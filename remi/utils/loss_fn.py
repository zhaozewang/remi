import torch
import torch.nn.functional as F


def mse_norm_loss(state, label, warmup_duration=0):
    if state is None or label is None:
        return float('nan')

    if warmup_duration > 0:
        state = state[:, warmup_duration:]
        label = label[:, warmup_duration:]

    epsilon = 1e-4

    mean_fr = torch.mean(label, dim=(0, 1), keepdim=True)
    std = torch.std(label, dim=(0, 1), keepdim=True)
    std = std.clamp(min=epsilon)

    norm_state = (state - mean_fr) / std
    norm_label = (label - mean_fr) / std

    return F.mse_loss(norm_state, norm_label)


def mse_loss(state, label, warmup_duration=0):
    if state is None or label is None:
        return float('nan')
    
    if warmup_duration > 0:
        state = state[:, warmup_duration:]
        label = label[:, warmup_duration:]

    return F.mse_loss(state, label)


def fr_mse_loss(state, label, warmup_duration=0):
    if state is None:
        return float('nan')
    
    mean_fr = torch.mean(state[:, warmup_duration:], dim=(0, 1))
    fr_loss = torch.mean(mean_fr**2)
    return fr_loss
