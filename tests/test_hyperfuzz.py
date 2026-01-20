#!/usr/bin/env python3
"""
HyperFuzz Test Script

Compares HyperFuzz algorithms against RapidFuzz for:
1. Correctness (results should match)
2. Performance (HyperFuzz should be faster)
"""

import time
from typing import Callable

import rapidfuzz.distance as rf_distance
import rapidfuzz.fuzz as rf_fuzz

from hyperfuzz import distance as hf_distance
from hyperfuzz import fuzz as hf_fuzz


# ============ Test Data ============

TEST_PAIRS = [
    ("kitten", "sitting"),
    ("hello", "hallo"),
    ("fuzzy wuzzy was a bear", "wuzzy fuzzy was a bear"),
    ("this is a test", "this is a test!"),
    ("", "abc"),
    ("abc", ""),
    ("abc", "abc"),
    ("Python", "Pthon"),
    ("California", "Calafornia"),
    ("intention", "execution"),
    ("The quick brown fox", "The quik brown fox"),
    ("", ""),
]

BATCH_SIZE = 1000
TIMING_ITERATIONS = 100


def benchmark(func: Callable, iterations: int = TIMING_ITERATIONS) -> float:
    """Run a function multiple times and return average time in microseconds."""
    start = time.perf_counter()
    for _ in range(iterations):
        func()
    elapsed = time.perf_counter() - start
    return (elapsed / iterations) * 1_000_000  # Convert to microseconds


def compare_results(name: str, hf_result, rf_result, tolerance: float = 0.01):
    """Compare HyperFuzz and RapidFuzz results."""
    if isinstance(hf_result, float):
        match = abs(hf_result - rf_result) < tolerance
    else:
        match = hf_result == rf_result
    
    status = "✅" if match else "❌"
    print(f"  {status} {name}: HyperFuzz={hf_result:.4f}, RapidFuzz={rf_result:.4f}" 
          if isinstance(hf_result, float) 
          else f"  {status} {name}: HyperFuzz={hf_result}, RapidFuzz={rf_result}")
    return match


def test_levenshtein():
    """Test Levenshtein distance algorithm."""
    print("\n" + "=" * 60)
    print("LEVENSHTEIN DISTANCE")
    print("=" * 60)
    
    all_passed = True
    
    for s1, s2 in TEST_PAIRS:
        print(f"\n  Testing: '{s1}' vs '{s2}'")
        
        # Distance
        hf_dist = hf_distance.levenshtein_distance(s1, s2)
        rf_dist = rf_distance.Levenshtein.distance(s1, s2)
        if not compare_results("distance", hf_dist, rf_dist):
            all_passed = False
        
        # Normalized similarity
        hf_sim = hf_distance.levenshtein_normalized_similarity(s1, s2)
        rf_sim = rf_distance.Levenshtein.normalized_similarity(s1, s2)
        if not compare_results("normalized_similarity", hf_sim, rf_sim):
            all_passed = False
    
    # Performance comparison
    print("\n  ⏱️  Performance Comparison:")
    
    def hf_test():
        for s1, s2 in TEST_PAIRS:
            hf_distance.levenshtein_distance(s1, s2)
    
    def rf_test():
        for s1, s2 in TEST_PAIRS:
            rf_distance.Levenshtein.distance(s1, s2)
    
    hf_time = benchmark(hf_test)
    rf_time = benchmark(rf_test)
    speedup = rf_time / hf_time if hf_time > 0 else 0
    
    print(f"      HyperFuzz: {hf_time:.2f} μs")
    print(f"      RapidFuzz: {rf_time:.2f} μs")
    print(f"      Speedup:   {speedup:.2f}x {'🚀' if speedup > 1 else '🐢'}")
    
    return all_passed


def test_ratio():
    """Test ratio (normalized Levenshtein as percentage) algorithm."""
    print("\n" + "=" * 60)
    print("RATIO (fuzz.ratio)")
    print("=" * 60)
    
    all_passed = True
    
    for s1, s2 in TEST_PAIRS:
        print(f"\n  Testing: '{s1}' vs '{s2}'")
        
        hf_ratio = hf_fuzz.ratio(s1, s2)
        rf_ratio = rf_fuzz.ratio(s1, s2)
        if not compare_results("ratio", hf_ratio, rf_ratio):
            all_passed = False
    
    # Performance comparison
    print("\n  ⏱️  Performance Comparison:")
    
    def hf_test():
        for s1, s2 in TEST_PAIRS:
            hf_fuzz.ratio(s1, s2)
    
    def rf_test():
        for s1, s2 in TEST_PAIRS:
            rf_fuzz.ratio(s1, s2)
    
    hf_time = benchmark(hf_test)
    rf_time = benchmark(rf_test)
    speedup = rf_time / hf_time if hf_time > 0 else 0
    
    print(f"      HyperFuzz: {hf_time:.2f} μs")
    print(f"      RapidFuzz: {rf_time:.2f} μs")
    print(f"      Speedup:   {speedup:.2f}x {'🚀' if speedup > 1 else '🐢'}")
    
    return all_passed


def test_batch_operations():
    """Test batch operations performance."""
    print("\n" + "=" * 60)
    print("BATCH OPERATIONS")
    print("=" * 60)
    
    # Generate batch test data
    pairs = [(f"test_string_{i}", f"test_strng_{i}") for i in range(BATCH_SIZE)]
    
    print(f"\n  Testing batch of {BATCH_SIZE} pairs...")
    
    # HyperFuzz batch
    start = time.perf_counter()
    hf_results = hf_distance.levenshtein_distance_batch(pairs)
    hf_time = (time.perf_counter() - start) * 1000
    
    # RapidFuzz sequential (no batch API)
    start = time.perf_counter()
    rf_results = [rf_distance.Levenshtein.distance(s1, s2) for s1, s2 in pairs]
    rf_time = (time.perf_counter() - start) * 1000
    
    # Verify results match
    matches = sum(1 for h, r in zip(hf_results, rf_results) if h == r)
    print(f"  ✅ Results match: {matches}/{BATCH_SIZE}")
    
    speedup = rf_time / hf_time if hf_time > 0 else 0
    print(f"\n  ⏱️  Performance:")
    print(f"      HyperFuzz (parallel): {hf_time:.2f} ms")
    print(f"      RapidFuzz (sequential): {rf_time:.2f} ms")
    print(f"      Speedup: {speedup:.2f}x {'🚀' if speedup > 1 else '🐢'}")
    
    return matches == BATCH_SIZE


def main():
    """Run all tests."""
    print("\n" + "🔥" * 30)
    print("  HYPERFUZZ TEST SUITE")
    print("🔥" * 30)
    
    results = {
        "Levenshtein": test_levenshtein(),
        "Ratio": test_ratio(),
        "Batch": test_batch_operations(),
    }
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False
    
    print("\n" + ("✅ All tests passed!" if all_passed else "❌ Some tests failed!"))
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
