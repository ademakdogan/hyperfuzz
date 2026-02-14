#!/usr/bin/env python3
"""
Comprehensive Algorithm Comparison Test: HyperFuzz vs TextDistance

Tests the following algorithms with 10 test pairs each:
- Jaccard Similarity
- Sørensen-Dice Similarity
- Tversky Index
- Overlap Coefficient
- Cosine Similarity
- Smith-Waterman
- Needleman-Wunsch
- LCSseq (Longest Common Subsequence)
- LCSstr (Longest Common Substring)
- Ratcliff-Obershelp

Each test pair varies in length:
1. Short identical (5-10 chars)
2. Short similar (5-10 chars)
3. Short different (5-10 chars)
4. Medium identical (20-30 chars)
5. Medium similar (20-30 chars)
6. Medium word swap (20-30 chars)
7. Long similar (50-80 chars)
8. Long different (50-80 chars)
9. Very long (100+ chars)
10. Unicode special (10-20 chars)
"""

import time
import sys
from typing import Callable, Any, Optional

# Import libraries
try:
    import hyperfuzz
    from hyperfuzz import (
        jaccard_similarity,
        sorensen_dice_similarity,
        tversky_similarity,
        overlap_similarity,
        cosine_similarity,
        smith_waterman_normalized_similarity,
        needleman_wunsch_normalized_similarity,
        lcs_seq_normalized_similarity,
        lcs_str_normalized_similarity,
        fuzz,
    )
    hyperfuzz_ratio = fuzz.ratio
except ImportError as e:
    print(f"Error importing HyperFuzz: {e}")
    sys.exit(1)

try:
    import textdistance
except ImportError:
    print("Please install textdistance: pip install textdistance")
    sys.exit(1)


# Test pairs of varying lengths
TEST_PAIRS = [
    # 1. Short identical (5-10 chars)
    ("hello", "hello"),
    # 2. Short similar (5-10 chars)
    ("hello", "hallo"),
    # 3. Short different (5-10 chars)
    ("hello", "world"),
    # 4. Medium identical (20-30 chars)
    ("the quick brown fox", "the quick brown fox"),
    # 5. Medium similar (20-30 chars)
    ("the quick brown fox", "the quik brown fox"),
    # 6. Medium word swap (20-30 chars)
    ("the quick brown fox", "fox brown quick the"),
    # 7. Long similar (50-80 chars)
    ("the quick brown fox jumps over the lazy dog", "the quick brown fox jumped over the lazy dog"),
    # 8. Long different (50-80 chars)
    ("the quick brown fox jumps over the lazy dog", "pack my box with five dozen liquor jugs"),
    # 9. Very long (100+ chars)
    (
        "Lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor",
        "Lorem ipsum dolor sit amet consectetur adipisicing elit sed do eiusmod tempo"
    ),
    # 10. Unicode special (10-20 chars)
    ("café résumé", "cafe resume"),
]


def measure_time(func: Callable, *args, **kwargs) -> tuple[Any, float]:
    """Execute function and measure time in microseconds."""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = (time.perf_counter() - start) * 1_000_000
    return result, elapsed


def compare_values(hf_val: float, td_val: float, tolerance: float = 0.01) -> bool:
    """Compare two float values with tolerance."""
    if hf_val is None or td_val is None:
        return hf_val == td_val
    return abs(hf_val - td_val) < tolerance


def format_result(name: str, hf_score: float, td_score: float, hf_time: float, td_time: float, match: bool) -> str:
    """Format a single result row."""
    status = "✅" if match else "❌"
    speedup = td_time / hf_time if hf_time > 0 else float('inf')
    return f"  {status} HF: {hf_score:.4f} | TD: {td_score:.4f} | HF: {hf_time:.1f}μs | TD: {td_time:.1f}μs | {speedup:.1f}x"


def run_algorithm_test(
    name: str,
    hf_func: Callable,
    td_func: Callable,
    hf_kwargs: dict = None,
    td_kwargs: dict = None
):
    """Run test for a single algorithm."""
    hf_kwargs = hf_kwargs or {}
    td_kwargs = td_kwargs or {}
    
    print(f"\n{'='*60}")
    print(f"📊 {name}")
    print('='*60)
    
    matches = 0
    hf_faster = 0
    total_tests = len(TEST_PAIRS)
    
    for i, (s1, s2) in enumerate(TEST_PAIRS, 1):
        try:
            hf_score, hf_time = measure_time(hf_func, s1, s2, **hf_kwargs)
            td_score, td_time = measure_time(td_func, s1, s2, **td_kwargs)
            
            match = compare_values(hf_score, td_score)
            if match:
                matches += 1
            if hf_time < td_time:
                hf_faster += 1
            
            print(f"\nTest {i}: '{s1[:20]}...' vs '{s2[:20]}...'")
            print(format_result(name, hf_score, td_score, hf_time, td_time, match))
            
        except Exception as e:
            print(f"\nTest {i}: ERROR - {e}")
    
    print(f"\n📈 Summary: {matches}/{total_tests} matches | HyperFuzz faster: {hf_faster}/{total_tests}")
    return matches, hf_faster


def main():
    print("="*60)
    print("🔬 HYPERFUZZ vs TEXTDISTANCE ALGORITHM COMPARISON")
    print("="*60)
    
    results = {}
    
    # 1. Jaccard Similarity (word-based)
    results['Jaccard'] = run_algorithm_test(
        "Jaccard Similarity",
        jaccard_similarity,
        textdistance.Jaccard(qval=None).normalized_similarity
    )
    
    # 2. Sørensen-Dice Similarity (word-based)
    results['Dice'] = run_algorithm_test(
        "Sørensen-Dice Similarity",
        sorensen_dice_similarity,
        textdistance.Sorensen(qval=None).normalized_similarity
    )
    
    # 3. Tversky Index (word-based)
    results['Tversky'] = run_algorithm_test(
        "Tversky Index (α=1, β=1)",
        lambda s1, s2: tversky_similarity(s1, s2, alpha=1.0, beta=1.0),
        lambda s1, s2: textdistance.Tversky(qval=None, ks=[1, 1]).normalized_similarity(s1, s2)
    )
    
    # 4. Overlap Coefficient (word-based)
    results['Overlap'] = run_algorithm_test(
        "Overlap Coefficient",
        overlap_similarity,
        textdistance.Overlap(qval=None).normalized_similarity
    )
    
    # 5. Cosine Similarity (word-based)
    results['Cosine'] = run_algorithm_test(
        "Cosine Similarity (word-based)",
        lambda s1, s2: cosine_similarity(s1, s2, use_words=True),
        textdistance.Cosine(qval=None).normalized_similarity
    )
    
    # 6. Smith-Waterman
    results['SmithWaterman'] = run_algorithm_test(
        "Smith-Waterman",
        smith_waterman_normalized_similarity,
        textdistance.SmithWaterman(gap_cost=1.0).normalized_similarity
    )
    
    # 7. Needleman-Wunsch
    results['NeedlemanWunsch'] = run_algorithm_test(
        "Needleman-Wunsch",
        needleman_wunsch_normalized_similarity,
        textdistance.NeedlemanWunsch(gap_cost=1.0).normalized_similarity
    )
    
    # 8. LCSseq (Longest Common Subsequence)
    results['LCSseq'] = run_algorithm_test(
        "LCSseq (Longest Common Subsequence)",
        lcs_seq_normalized_similarity,
        lambda s1, s2: textdistance.lcsseq.normalized_similarity(s1, s2)
    )
    
    # 9. LCSstr (Longest Common Substring)
    results['LCSstr'] = run_algorithm_test(
        "LCSstr (Longest Common Substring)",
        lcs_str_normalized_similarity,
        lambda s1, s2: textdistance.lcsstr.normalized_similarity(s1, s2)
    )
    
    # 10. Ratcliff-Obershelp
    results['RatcliffObershelp'] = run_algorithm_test(
        "Ratcliff-Obershelp",
        lambda s1, s2: hyperfuzz_ratio(s1, s2) / 100.0,  # Convert from percentage
        textdistance.RatcliffObershelp().normalized_similarity
    )
    
    # Print summary
    print("\n")
    print("="*60)
    print("📋 FINAL SUMMARY")
    print("="*60)
    
    total_matches = 0
    total_faster = 0
    total_tests = len(TEST_PAIRS) * len(results)
    
    print(f"\n{'Algorithm':<30} {'Score Match':<15} {'HF Faster':<15}")
    print("-"*60)
    
    for algo, (matches, faster) in results.items():
        total_matches += matches
        total_faster += faster
        match_pct = matches / len(TEST_PAIRS) * 100
        faster_pct = faster / len(TEST_PAIRS) * 100
        print(f"{algo:<30} {matches}/{len(TEST_PAIRS)} ({match_pct:.0f}%)     {faster}/{len(TEST_PAIRS)} ({faster_pct:.0f}%)")
    
    print("-"*60)
    match_pct = total_matches / total_tests * 100
    faster_pct = total_faster / total_tests * 100
    print(f"{'TOTAL':<30} {total_matches}/{total_tests} ({match_pct:.1f}%)   {total_faster}/{total_tests} ({faster_pct:.1f}%)")
    
    print("\n✅ Score Match = HyperFuzz and TextDistance return the same value")
    print("🚀 HF Faster = HyperFuzz executes faster than TextDistance")


if __name__ == "__main__":
    main()
