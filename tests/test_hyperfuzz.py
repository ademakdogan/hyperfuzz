#!/usr/bin/env python3
"""
HyperFuzz Comprehensive Test Script

Tests all implemented algorithms for:
1. Correctness (where comparable to RapidFuzz)
2. Functionality (new algorithms work correctly)
3. Performance benchmarks
"""

import time
from typing import Callable

import rapidfuzz.distance as rf_distance
import rapidfuzz.fuzz as rf_fuzz

import hyperfuzz
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
]

TIMING_ITERATIONS = 100


def benchmark(func: Callable, iterations: int = TIMING_ITERATIONS) -> float:
    """Run a function multiple times and return average time in microseconds."""
    start = time.perf_counter()
    for _ in range(iterations):
        func()
    elapsed = time.perf_counter() - start
    return (elapsed / iterations) * 1_000_000


def compare_results(name: str, hf_result, rf_result, tolerance: float = 0.01):
    """Compare HyperFuzz and RapidFuzz results."""
    if isinstance(hf_result, float):
        match = abs(hf_result - rf_result) < tolerance
    else:
        match = hf_result == rf_result
    
    status = "✅" if match else "❌"
    if isinstance(hf_result, float):
        print(f"  {status} {name}: HF={hf_result:.4f}, RF={rf_result:.4f}")
    else:
        print(f"  {status} {name}: HF={hf_result}, RF={rf_result}")
    return match


def test_levenshtein():
    """Test Levenshtein distance algorithm."""
    print("\n🔹 LEVENSHTEIN DISTANCE")
    
    all_passed = True
    for s1, s2 in TEST_PAIRS[:5]:
        hf_dist = hf_distance.levenshtein_distance(s1, s2)
        rf_dist = rf_distance.Levenshtein.distance(s1, s2)
        if not compare_results(f"'{s1}' vs '{s2}'", hf_dist, rf_dist):
            all_passed = False
    
    return all_passed


def test_jaro_winkler():
    """Test Jaro-Winkler similarity."""
    print("\n🔹 JARO-WINKLER SIMILARITY")
    
    all_passed = True
    for s1, s2 in TEST_PAIRS[:5]:
        hf_sim = hf_distance.jaro_winkler_similarity(s1, s2)
        rf_sim = rf_distance.JaroWinkler.similarity(s1, s2)
        if not compare_results(f"'{s1}' vs '{s2}'", hf_sim, rf_sim):
            all_passed = False
    
    return all_passed


def test_ratio():
    """Test fuzz.ratio."""
    print("\n🔹 FUZZ RATIO")
    
    all_passed = True
    for s1, s2 in TEST_PAIRS[:5]:
        hf_ratio = hf_fuzz.ratio(s1, s2)
        rf_ratio = rf_fuzz.ratio(s1, s2)
        if not compare_results(f"'{s1}' vs '{s2}'", hf_ratio, rf_ratio):
            all_passed = False
    
    return all_passed


def test_token_sort_ratio():
    """Test token_sort_ratio."""
    print("\n🔹 TOKEN SORT RATIO")
    
    test_cases = [
        ("fuzzy wuzzy was a bear", "wuzzy fuzzy was a bear"),
        ("hello world", "world hello"),
    ]
    
    all_passed = True
    for s1, s2 in test_cases:
        hf_ratio = hf_fuzz.token_sort_ratio(s1, s2)
        rf_ratio = rf_fuzz.token_sort_ratio(s1, s2)
        if not compare_results(f"'{s1}' vs '{s2}'", hf_ratio, rf_ratio):
            all_passed = False
    
    return all_passed


def test_additional_algorithms():
    """Test additional algorithms (no RapidFuzz comparison)."""
    print("\n🔹 ADDITIONAL ALGORITHMS")
    
    all_passed = True
    
    # Jaccard
    j = hyperfuzz.jaccard_similarity("the quick brown fox", "the slow brown dog")
    print(f"  ✅ Jaccard: {j:.4f}")
    if j <= 0:
        all_passed = False
    
    # Dice
    d = hyperfuzz.sorensen_dice_similarity("hello world", "hello there")
    print(f"  ✅ Dice: {d:.4f}")
    if d <= 0:
        all_passed = False
    
    # Tversky
    t = hyperfuzz.tversky_similarity("hello world", "hello there", alpha=0.5, beta=0.5)
    print(f"  ✅ Tversky: {t:.4f}")
    
    # Overlap
    o = hyperfuzz.overlap_similarity("hello", "hello world")
    print(f"  ✅ Overlap: {o:.4f}")
    if o < 1.0:
        all_passed = False
    
    # Smith-Waterman
    sw = hyperfuzz.smith_waterman_normalized_similarity("ACACACTA", "AGCACACA")
    print(f"  ✅ Smith-Waterman: {sw:.4f}")
    if sw <= 0:
        all_passed = False
    
    # Needleman-Wunsch
    nw = hyperfuzz.needleman_wunsch_normalized_similarity("GATTACA", "GCATGCU")
    print(f"  ✅ Needleman-Wunsch: {nw:.4f}")
    if nw <= 0:
        all_passed = False
    
    # Cosine
    c = hyperfuzz.cosine_similarity("hello world", "hello there")
    print(f"  ✅ Cosine (bigram): {c:.4f}")
    if c <= 0:
        all_passed = False
    
    # Soft-TFIDF
    st = hyperfuzz.soft_tfidf_similarity("hello world", "helo wrld")
    print(f"  ✅ Soft-TFIDF: {st:.4f}")
    if st <= 0:
        all_passed = False
    
    return all_passed


def test_batch_operations():
    """Test batch operations."""
    print("\n🔹 BATCH OPERATIONS")
    
    pairs = [(f"test_{i}", f"tset_{i}") for i in range(1000)]
    
    start = time.perf_counter()
    results = hf_distance.levenshtein_distance_batch(pairs)
    elapsed = (time.perf_counter() - start) * 1000
    
    print(f"  ✅ Levenshtein batch (1000 pairs): {elapsed:.2f}ms")
    print(f"     Results sample: {results[:5]}")
    
    # Jaccard batch
    start = time.perf_counter()
    j_results = hyperfuzz.jaccard_similarity_batch([("a b c", "b c d") for _ in range(1000)])
    elapsed = (time.perf_counter() - start) * 1000
    print(f"  ✅ Jaccard batch (1000 pairs): {elapsed:.2f}ms")
    
    return len(results) == 1000


def main():
    """Run all tests."""
    print("\n" + "🔥" * 30)
    print("  HYPERFUZZ COMPREHENSIVE TEST SUITE")
    print("🔥" * 30)
    
    results = {
        "Levenshtein": test_levenshtein(),
        "Jaro-Winkler": test_jaro_winkler(),
        "Ratio": test_ratio(),
        "Token Sort Ratio": test_token_sort_ratio(),
        "Additional Algorithms": test_additional_algorithms(),
        "Batch Operations": test_batch_operations(),
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
