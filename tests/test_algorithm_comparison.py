#!/usr/bin/env python3
"""
Comprehensive Algorithm Comparison Test: HyperFuzz vs RapidFuzz

Shows results organized by test pair, with all algorithms compared for each pair.
"""

import time
import sys
from typing import Callable, Any

# Import libraries
try:
    import hyperfuzz
    from hyperfuzz import distance as hf_distance
    from hyperfuzz import fuzz as hf_fuzz
except ImportError as e:
    print(f"Error importing HyperFuzz: {e}")
    sys.exit(1)

try:
    import rapidfuzz.distance as rf_distance
    import rapidfuzz.fuzz as rf_fuzz
except ImportError:
    print("Please install rapidfuzz: pip install rapidfuzz")
    sys.exit(1)


# Test pairs
TEST_PAIRS = [
    ("hello", "hello"),
    ("hello", "hallo"),
    ("hello", "world"),
    ("the quick brown fox", "the quick brown fox"),
    ("the quick brown fox", "the quik brown fox"),
    ("the quick brown fox", "fox brown quick the"),
    ("the quick brown fox jumps over the lazy dog", "the quick brown fox jumped over the lazy dog"),
    ("the quick brown fox jumps over the lazy dog", "pack my box with five dozen liquor jugs"),
    ("Lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor",
     "Lorem ipsum dolor sit amet consectetur adipisicing elit sed do eiusmod tempo"),
    ("café résumé", "cafe resume"),
]

# Algorithm definitions: (name, hf_func, rf_func)
ALGORITHMS = [
    # Distance algorithms
    ("Levenshtein", 
     lambda s1, s2: hf_distance.levenshtein_distance(s1, s2), 
     lambda s1, s2: rf_distance.Levenshtein.distance(s1, s2)),
    ("Levenshtein Norm", 
     lambda s1, s2: hf_distance.levenshtein_normalized_similarity(s1, s2), 
     lambda s1, s2: rf_distance.Levenshtein.normalized_similarity(s1, s2)),
    ("Jaro", 
     lambda s1, s2: hf_distance.jaro_similarity(s1, s2), 
     lambda s1, s2: rf_distance.Jaro.similarity(s1, s2)),
    ("Jaro-Winkler", 
     lambda s1, s2: hf_distance.jaro_winkler_similarity(s1, s2), 
     lambda s1, s2: rf_distance.JaroWinkler.similarity(s1, s2)),
    ("Indel", 
     lambda s1, s2: hf_distance.indel_distance(s1, s2), 
     lambda s1, s2: rf_distance.Indel.distance(s1, s2)),
    ("LCSseq", 
     lambda s1, s2: hf_distance.lcs_seq_distance(s1, s2), 
     lambda s1, s2: rf_distance.LCSseq.distance(s1, s2)),
    ("OSA", 
     lambda s1, s2: hf_distance.osa_distance(s1, s2), 
     lambda s1, s2: rf_distance.OSA.distance(s1, s2)),
    ("Prefix", 
     lambda s1, s2: hf_distance.prefix_similarity(s1, s2), 
     lambda s1, s2: rf_distance.Prefix.similarity(s1, s2)),
    ("Postfix", 
     lambda s1, s2: hf_distance.postfix_similarity(s1, s2), 
     lambda s1, s2: rf_distance.Postfix.similarity(s1, s2)),
    # Fuzz algorithms
    ("fuzz.ratio", 
     lambda s1, s2: hf_fuzz.ratio(s1, s2), 
     lambda s1, s2: rf_fuzz.ratio(s1, s2)),
    ("fuzz.partial_ratio", 
     lambda s1, s2: hf_fuzz.partial_ratio(s1, s2), 
     lambda s1, s2: rf_fuzz.partial_ratio(s1, s2)),
    ("fuzz.token_sort", 
     lambda s1, s2: hf_fuzz.token_sort_ratio(s1, s2), 
     lambda s1, s2: rf_fuzz.token_sort_ratio(s1, s2)),
    ("fuzz.token_set", 
     lambda s1, s2: hf_fuzz.token_set_ratio(s1, s2), 
     lambda s1, s2: rf_fuzz.token_set_ratio(s1, s2)),
    ("fuzz.token_ratio", 
     lambda s1, s2: hf_fuzz.token_ratio(s1, s2), 
     lambda s1, s2: rf_fuzz.token_ratio(s1, s2)),
    ("fuzz.partial_token_sort", 
     lambda s1, s2: hf_fuzz.partial_token_sort_ratio(s1, s2), 
     lambda s1, s2: rf_fuzz.partial_token_sort_ratio(s1, s2)),
    ("fuzz.partial_token_set", 
     lambda s1, s2: hf_fuzz.partial_token_set_ratio(s1, s2), 
     lambda s1, s2: rf_fuzz.partial_token_set_ratio(s1, s2)),
    ("fuzz.WRatio", 
     lambda s1, s2: hf_fuzz.w_ratio(s1, s2), 
     lambda s1, s2: rf_fuzz.WRatio(s1, s2)),
    ("fuzz.QRatio", 
     lambda s1, s2: hf_fuzz.q_ratio(s1, s2), 
     lambda s1, s2: rf_fuzz.QRatio(s1, s2)),
]


def measure_time(func: Callable, s1: str, s2: str) -> tuple[Any, float]:
    """Execute function and measure time in microseconds."""
    start = time.perf_counter()
    result = func(s1, s2)
    elapsed = (time.perf_counter() - start) * 1_000_000
    return result, elapsed


def compare_values(hf_val: float, rf_val: float, tolerance: float = 0.01) -> bool:
    """Compare two float values with tolerance."""
    if hf_val is None or rf_val is None:
        return hf_val == rf_val
    return abs(hf_val - rf_val) < tolerance


def main():
    print("=" * 110)
    print("🔥 HYPERFUZZ vs RAPIDFUZZ - TEST PAIR BASED COMPARISON 🔥")
    print("=" * 110)
    
    # Track overall statistics
    total_matches = 0
    total_faster = 0
    total_tests = 0
    
    # Algorithm-level stats
    algo_stats = {name: {"matches": 0, "faster": 0} for name, _, _ in ALGORITHMS}
    
    # Run tests for each pair
    for pair_idx, (s1, s2) in enumerate(TEST_PAIRS, 1):
        # Display test pair header
        print(f"\n{'='*110}")
        print(f"📋 TEST {pair_idx}: '{s1[:40]}{'...' if len(s1) > 40 else ''}' vs '{s2[:40]}{'...' if len(s2) > 40 else ''}'")
        print("=" * 110)
        
        # Table header
        print(f"\n{'Algorithm':<22} {'Match':<6} {'HF Score':<12} {'RF Score':<12} {'HF Time':<10} {'RF Time':<10} {'Speedup':<12}")
        print("-" * 110)
        
        for algo_name, hf_func, rf_func in ALGORITHMS:
            try:
                hf_score, hf_time = measure_time(hf_func, s1, s2)
                rf_score, rf_time = measure_time(rf_func, s1, s2)
                
                match = compare_values(hf_score, rf_score)
                speedup = rf_time / hf_time if hf_time > 0 else float('inf')
                faster = hf_time < rf_time
                
                # Update stats
                if match:
                    total_matches += 1
                    algo_stats[algo_name]["matches"] += 1
                if faster:
                    total_faster += 1
                    algo_stats[algo_name]["faster"] += 1
                total_tests += 1
                
                # Format output
                match_icon = "✅" if match else "❌"
                speed_icon = "🚀" if speedup > 1 else "🐢"
                
                # Format scores based on type
                if isinstance(hf_score, int):
                    hf_score_str = f"{hf_score}"
                    rf_score_str = f"{rf_score}"
                else:
                    hf_score_str = f"{hf_score:.4f}"
                    rf_score_str = f"{rf_score:.4f}"
                
                print(f"{algo_name:<22} {match_icon:<6} {hf_score_str:<12} {rf_score_str:<12} {hf_time:<10.3f} {rf_time:<10.3f} {speedup:>5.2f}x {speed_icon}")
                
            except Exception as e:
                print(f"{algo_name:<22} ⚠️  ERROR: {e}")
        
        print("-" * 110)
    
    # Final Summary
    print("\n")
    print("=" * 110)
    print("📊 FINAL SUMMARY")
    print("=" * 110)
    
    # Per-algorithm summary
    print(f"\n{'Algorithm':<22} {'Score Match':<20} {'HyperFuzz Faster':<20}")
    print("-" * 70)
    
    for algo_name, _, _ in ALGORITHMS:
        matches = algo_stats[algo_name]["matches"]
        faster = algo_stats[algo_name]["faster"]
        match_pct = (matches / len(TEST_PAIRS)) * 100
        faster_pct = (faster / len(TEST_PAIRS)) * 100
        
        match_str = f"{matches}/{len(TEST_PAIRS)} ({match_pct:.0f}%)"
        faster_str = f"{faster}/{len(TEST_PAIRS)} ({faster_pct:.0f}%)"
        
        print(f"{algo_name:<22} {match_str:<20} {faster_str:<20}")
    
    print("-" * 70)
    
    # Overall totals
    match_pct = (total_matches / total_tests) * 100
    faster_pct = (total_faster / total_tests) * 100
    
    print(f"{'TOTAL':<22} {total_matches}/{total_tests} ({match_pct:.1f}%)      {total_faster}/{total_tests} ({faster_pct:.1f}%)")
    
    print("\n" + "=" * 110)
    print("✅ Score Match = HyperFuzz and RapidFuzz return the same value")
    print("🚀 HyperFuzz Faster = HyperFuzz executes faster than RapidFuzz")
    print("🐢 RapidFuzz Faster = RapidFuzz executes faster than HyperFuzz")
    print("=" * 110)


if __name__ == "__main__":
    main()
