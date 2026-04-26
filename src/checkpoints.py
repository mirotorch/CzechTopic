"""Checkpoint helpers for saving model weights with extra metadata."""

from __future__ import annotations

from pathlib import Path

import torch


def save_checkpoint(path: str | Path, model, threshold: float, **metadata):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "threshold": float(threshold),
    }
    checkpoint.update(metadata)
    torch.save(checkpoint, path)


def load_checkpoint(path: str | Path, map_location=None):
    checkpoint = torch.load(path, map_location=map_location)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint

    return {
        "model_state_dict": checkpoint,
        "threshold": None,
    }
