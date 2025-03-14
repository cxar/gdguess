#!/usr/bin/env python3
"""
Learning rate finder implementation for the Grateful Dead show dating model.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def find_learning_rate(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: callable,
    device: torch.device,
    start_lr: float = 1e-7,
    end_lr: float = 1,
    num_steps: int = 100,
) -> float:
    """
    Find optimal learning rate using learning rate range test.

    Args:
        model: Neural network model
        train_loader: DataLoader for training data
        optimizer: Optimizer to use
        criterion: Loss function
        device: Device to run on
        start_lr: Starting learning rate
        end_lr: Ending learning rate
        num_steps: Number of steps to take

    Returns:
        Suggested optimal learning rate
    """
    model.train()
    lrs = []
    losses = []
    lr = start_lr

    mult = (end_lr / start_lr) ** (1 / num_steps)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    for i, batch in enumerate(train_loader):
        if i >= num_steps:
            break

        audio = batch["audio"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)
        era = batch["era"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(audio)
        loss = criterion(outputs, label, era)

        if torch.isnan(loss) or torch.isinf(loss):
            print("Warning: NaN/Inf detected in loss - skipping batch")
            continue

        loss.backward()
        optimizer.step()

        lrs.append(lr)
        losses.append(loss.item())

        lr *= mult
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    plt.figure(figsize=(10, 6))
    plt.plot(lrs, losses)
    plt.xscale("log")
    plt.xlabel("Learning Rate")
    plt.ylabel("Loss")
    plt.title("Learning Rate Finder")
    plt.savefig("lr_finder.png")

    # Find optimal learning rate (point with steepest negative gradient)
    smoothed_losses = np.array(losses)
    gradients = np.gradient(smoothed_losses)
    optimal_idx = np.argmin(gradients)
    optimal_lr = lrs[optimal_idx]

    print(f"Suggested learning rate: {optimal_lr:.1e}")
    return optimal_lr
