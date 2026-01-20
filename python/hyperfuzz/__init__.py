"""
HyperFuzz - High-performance string similarity algorithms

A Python library for fast string similarity calculations,
with all computations implemented in Rust.
"""

from hyperfuzz._hyperfuzz import distance, fuzz

__version__ = "0.1.0"
__all__ = ["distance", "fuzz", "__version__"]
