"""Top-level dataset wrapper for the project.

This module re-exports `WaterDataset` so notebooks can `from dataset import WaterDataset`.
"""
from swu.dataset import WaterDataset

__all__ = ["WaterDataset"]
