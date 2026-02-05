"""
HyperFuzz - High-performance string similarity algorithms

A Python library for fast string similarity calculations,
with all computations implemented in Rust.
"""

from hyperfuzz._hyperfuzz import distance, fuzz
from hyperfuzz._hyperfuzz import (
    # Set-based algorithms
    jaccard_similarity,
    jaccard_distance,
    jaccard_similarity_batch,
    sorensen_dice_similarity,
    sorensen_dice_distance,
    sorensen_dice_similarity_batch,
    tversky_similarity,
    tversky_distance,
    tversky_similarity_batch,
    overlap_similarity,
    overlap_distance,
    overlap_similarity_batch,
    # Alignment algorithms
    smith_waterman_score,
    smith_waterman_normalized_similarity,
    smith_waterman_score_batch,
    needleman_wunsch_score,
    needleman_wunsch_normalized_similarity,
    needleman_wunsch_score_batch,
    # Vector methods
    cosine_similarity,
    cosine_distance,
    cosine_similarity_batch,
    soft_tfidf_similarity,
    soft_tfidf_distance,
    soft_tfidf_similarity_batch,
)

__version__ = "0.1.0"

__all__ = [
    "distance",
    "fuzz",
    "__version__",
    # Set-based
    "jaccard_similarity",
    "jaccard_distance",
    "jaccard_similarity_batch",
    "sorensen_dice_similarity",
    "sorensen_dice_distance",
    "sorensen_dice_similarity_batch",
    "tversky_similarity",
    "tversky_distance",
    "tversky_similarity_batch",
    "overlap_similarity",
    "overlap_distance",
    "overlap_similarity_batch",
    # Alignment
    "smith_waterman_score",
    "smith_waterman_normalized_similarity",
    "smith_waterman_score_batch",
    "needleman_wunsch_score",
    "needleman_wunsch_normalized_similarity",
    "needleman_wunsch_score_batch",
    # Vector
    "cosine_similarity",
    "cosine_distance",
    "cosine_similarity_batch",
    "soft_tfidf_similarity",
    "soft_tfidf_distance",
    "soft_tfidf_similarity_batch",
]
