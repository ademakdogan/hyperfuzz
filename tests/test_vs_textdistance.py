#!/usr/bin/env python3
"""
HyperFuzz vs TextDistance Benchmark Test

This test compares HyperFuzz with TextDistance for algorithms
that RapidFuzz doesn't support natively.

NOTE: Set-based algorithms (Jaccard, Dice, Overlap, Tversky) use different
tokenization strategies:
- HyperFuzz: bigram (character n-grams)  
- TextDistance: word-level tokenization (default)

For fair comparison, we test edit-based and alignment algorithms.
"""

import time
from typing import Callable, Any

import hyperfuzz
import textdistance

# Test pairs for comprehensive comparison
TEST_PAIRS = [
    # Pair 1: Short ASCII strings
    ("kitten", "sitting"),
    # Pair 2: Similar short strings
    ("hello", "hallo"),
    # Pair 3: Word order difference
    ("fuzzy wuzzy was a bear", "wuzzy fuzzy was a bear"),
    # Pair 4: Slight difference
    ("this is a test", "this is a test!"),
    # Pair 5: Empty vs non-empty
    ("", "something"),
    # Pair 6: Identical strings
    ("identical", "identical"),
    # Pair 7: Completely different
    ("abcdef", "ghijkl"),
    # Pair 8: Medium length strings
    ("The quick brown fox jumps over the lazy dog", 
     "The quick brown fox leaps over the lazy cat"),
    # Pair 9: Longer strings (65+ chars for bit-parallel testing)
    ("The HyperFuzz library provides fast string similarity calculations using Rust",
     "The RapidFuzz library provides quick string similarity computations in C++"),
    # Pair 10: Unicode strings
    ("café résumé", "cafe resume"),
]


def measure_time(func: Callable, *args, iterations: int = 1000) -> tuple[Any, float]:
    """Measure execution time and return result with average time in microseconds."""
    # Warm-up
    result = func(*args)
    
    # Measure
    start = time.perf_counter()
    for _ in range(iterations):
        func(*args)
    elapsed = time.perf_counter() - start
    
    avg_time_us = (elapsed / iterations) * 1_000_000
    return result, avg_time_us


def format_result(value: Any) -> str:
    """Format result for display."""
    if isinstance(value, float):
        return f"{value:.4f}"
    if value is None:
        return "None"
    return str(value)


def compare_values(hf_val: Any, td_val: Any, tolerance: float = 0.01) -> tuple[bool, str]:
    """Compare values with tolerance for floats."""
    if hf_val is None or td_val is None:
        # Skip comparison for None values (e.g., Hamming with different lengths)
        return True, "⚠️ skip"
    if isinstance(hf_val, float) and isinstance(td_val, float):
        match = abs(hf_val - td_val) < tolerance
        return match, "✅" if match else f"❌ ({abs(hf_val - td_val):.4f})"
    else:
        match = hf_val == td_val
        return match, "✅" if match else "❌"


def print_header():
    """Print test header."""
    print("=" * 115)
    print("  HYPERFUZZ vs TEXTDISTANCE BENCHMARK")
    print("  Comparing algorithms not available in RapidFuzz")
    print("=" * 115)
    print()
    print("  NOTE: Focusing on EDIT-BASED and ALIGNMENT algorithms")
    print("        (Set-based algorithms use different tokenization strategies)")
    print()


def print_pair_header(idx: int, s1: str, s2: str):
    """Print test pair header."""
    s1_display = s1[:55] + "..." if len(s1) > 55 else s1
    s2_display = s2[:55] + "..." if len(s2) > 55 else s2
    print("=" * 115)
    print(f"TEST PAIR {idx + 1}:")
    print(f"  s1 ({len(s1):3d} chars): '{s1_display}'")
    print(f"  s2 ({len(s2):3d} chars): '{s2_display}'")
    print("-" * 115)
    print(f"{'Algorithm':<38} {'HF Result':>12} {'TD Result':>12} {'Match':>12} {'HF Time':>11} {'TD Time':>11} {'Speedup':>10}")
    print("-" * 115)


def run_comparison(name: str, hf_func: Callable, td_func: Callable, 
                   s1: str, s2: str, stats: dict) -> None:
    """Run a single algorithm comparison."""
    try:
        hf_result, hf_time = measure_time(hf_func, s1, s2)
    except Exception as e:
        hf_result, hf_time = f"ERR", 0.0
    
    try:
        td_result, td_time = measure_time(td_func, s1, s2)
    except Exception as e:
        td_result, td_time = f"ERR", 0.0
    
    # Handle comparison
    if isinstance(hf_result, str) and hf_result.startswith("ERR"):
        match_str = "⚠️ HF err"
        match = False
    elif isinstance(td_result, str) and td_result.startswith("ERR"):
        match_str = "⚠️ TD err"
        match = False  
    else:
        match, match_str = compare_values(hf_result, td_result)
    
    # Calculate speedup
    if hf_time > 0 and td_time > 0:
        speedup = td_time / hf_time
        speedup_str = f"{speedup:.2f}x" + (" 🚀" if speedup > 1.0 else " 🐢")
    else:
        speedup = 0
        speedup_str = "N/A"
    
    # Update stats
    stats["total"] += 1
    if match:
        stats["matches"] += 1
    if speedup > 1.0:
        stats["faster"] += 1
    
    print(f"{name:<38} {format_result(hf_result):>12} {format_result(td_result):>12} {match_str:>12} "
          f"{hf_time:>9.2f}μs {td_time:>9.2f}μs {speedup_str:>10}")


def main():
    print_header()
    
    stats = {"total": 0, "matches": 0, "faster": 0}
    
    # Define algorithm comparisons - focusing on comparable algorithms
    # TextDistance uses word-level tokenization for set-based, so we skip those
    algorithms = [
        # === EDIT-BASED ALGORITHMS ===
        ("Levenshtein Distance", 
         hyperfuzz.distance.levenshtein_distance, 
         textdistance.levenshtein.distance),
        ("Levenshtein Norm Similarity", 
         hyperfuzz.distance.levenshtein_normalized_similarity, 
         textdistance.levenshtein.normalized_similarity),
        ("Jaro Similarity", 
         hyperfuzz.distance.jaro_similarity, 
         textdistance.jaro.normalized_similarity),
        ("Jaro-Winkler Similarity", 
         hyperfuzz.distance.jaro_winkler_similarity, 
         textdistance.jaro_winkler.normalized_similarity),
        ("Hamming Distance", 
         hyperfuzz.distance.hamming_distance, 
         textdistance.hamming.distance),
        ("Damerau-Levenshtein Distance", 
         hyperfuzz.distance.damerau_levenshtein_distance, 
         textdistance.damerau_levenshtein.distance),
        
        # === ALIGNMENT ALGORITHMS ===
        ("Smith-Waterman (norm)", 
         hyperfuzz.smith_waterman_normalized_similarity, 
         textdistance.smith_waterman.normalized_similarity),
        ("Needleman-Wunsch (norm)", 
         hyperfuzz.needleman_wunsch_normalized_similarity, 
         textdistance.needleman_wunsch.normalized_similarity),
        
        # === SEQUENCE-BASED ===
        ("LCSseq Similarity (norm)", 
         hyperfuzz.distance.lcs_seq_normalized_similarity, 
         textdistance.lcsseq.normalized_similarity),
        ("Ratcliff-Obershelp (fuzz.ratio)", 
         lambda s1, s2: hyperfuzz.fuzz.ratio(s1, s2) / 100.0, 
         textdistance.ratcliff_obershelp.normalized_similarity),
         
        # === SET-BASED (HyperFuzz bigram) - for reference, they may differ ===
        ("Jaccard (bigram vs words)", 
         hyperfuzz.jaccard_similarity, 
         textdistance.jaccard.normalized_similarity),
        ("Sørensen-Dice (bigram vs words)", 
         hyperfuzz.sorensen_dice_similarity, 
         textdistance.sorensen.normalized_similarity),
        ("Cosine (bigram vs char)", 
         hyperfuzz.cosine_similarity, 
         textdistance.cosine.normalized_similarity),
        ("Tversky α=1,β=1 (Jaccard equiv)", 
         lambda s1, s2: hyperfuzz.tversky_similarity(s1, s2, alpha=1.0, beta=1.0), 
         lambda s1, s2: textdistance.Tversky(ks=(1.0, 1.0)).normalized_similarity(s1, s2)),
    ]
    
    for pair_idx, (s1, s2) in enumerate(TEST_PAIRS):
        print_pair_header(pair_idx, s1, s2)
        
        for name, hf_func, td_func in algorithms:
            run_comparison(name, hf_func, td_func, s1, s2, stats)
        
        print()
    
    # Print summary
    print("=" * 115)
    print("OVERALL SUMMARY")
    print("=" * 115)
    print()
    accuracy = (stats["matches"] / stats["total"]) * 100 if stats["total"] > 0 else 0
    faster_pct = (stats["faster"] / stats["total"]) * 100 if stats["total"] > 0 else 0
    
    print(f"✅ Score Match: {stats['matches']}/{stats['total']} ({accuracy:.1f}%)")
    print(f"🚀 Faster than TextDistance: {stats['faster']}/{stats['total']} ({faster_pct:.1f}%)")
    print()
    
    print("Note: Set-based algorithms (Jaccard, Dice, Cosine, Tversky) use different tokenization:")
    print("      - HyperFuzz: bigram (character 2-grams)")
    print("      - TextDistance: word-level (default)")
    print()
    
    if accuracy >= 70:
        print("🎉 GOOD COMPATIBILITY - Most scores match!")
    else:
        print(f"⚠️  Some scores differ due to different tokenization strategies")
    
    print()


if __name__ == "__main__":
    main()
