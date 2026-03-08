"""Shared utility functions."""

from pathlib import Path

import yaml


def load_yaml(path: str | Path) -> dict:
    """Load and validate a YAML file returns a dict."""
    path = Path(path)
    with open(path) as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict from {path}, got {type(data).__name__}")
    return data
