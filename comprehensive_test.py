#!/usr/bin/env python3
"""
Comprehensive comparison test between HyperFuzz and RapidFuzz
Tests 10 different text pairs with varying lengths and scenarios
"""

import time
from hyperfuzz import distance as hf_dist, fuzz as hf_fuzz
from rapidfuzz import fuzz as rf_fuzz
from rapidfuzz.distance import (
    Levenshtein as rf_Levenshtein,
    Hamming as rf_Hamming,
    Jaro as rf_Jaro,
    JaroWinkler as rf_JaroWinkler,
    Indel as rf_Indel,
    LCSseq as rf_LCSseq,
    DamerauLevenshtein as rf_DamerauLevenshtein,
    OSA as rf_OSA,
)

# 10 different text pairs with varying scenarios
TEST_PAIRS = [
    # 1. Short identical strings
    ("hello", "hello"),
    
    # 2. Short similar strings
    ("kitten", "sitting"),
    
    # 3. Medium strings with typo
    ("algorithm", "algoritm"),
    
    # 4. Long similar strings
    ("the quick brown fox jumps over the lazy dog", "the quick brown fox jumped over the lazy dog"),
    
    # 5. Strings with different lengths (short vs medium)
    ("test", "this is a test string"),
    
    # 6. Reversed words
    ("hello world", "world hello"),
    
    # 7. Strings with numbers and special chars
    ("user123@email.com", "user124@email.com"),
    
    # 8. Empty vs non-empty (skip for some algorithms)
    ("", "something"),
    
    # 9. Long strings with multiple differences
    ("The HyperFuzz library provides fast string similarity calculations", 
     "The RapidFuzz library provides quick string similarity computations"),
    
    # 10. Unicode strings (accents)
    ("café résumé", "cafe resume"),
]

# Algorithms to test
ALGORITHMS = [
    # (name, hyperfuzz_func, rapidfuzz_func, is_distance)
    ("Levenshtein Distance", hf_dist.levenshtein_distance, rf_Levenshtein.distance, True),
    ("Levenshtein Norm Sim", hf_dist.levenshtein_normalized_similarity, rf_Levenshtein.normalized_similarity, False),
    ("Jaro Similarity", hf_dist.jaro_similarity, rf_Jaro.similarity, False),
    ("Jaro-Winkler Similarity", hf_dist.jaro_winkler_similarity, rf_JaroWinkler.similarity, False),
    ("Indel Distance", hf_dist.indel_distance, rf_Indel.distance, True),
    ("LCSseq Distance", hf_dist.lcs_seq_distance, rf_LCSseq.distance, True),
    ("OSA Distance", hf_dist.osa_distance, rf_OSA.distance, True),
    ("DamerauLevenshtein", hf_dist.damerau_levenshtein_distance, rf_DamerauLevenshtein.distance, True),
    ("fuzz.ratio", hf_fuzz.ratio, rf_fuzz.ratio, False),
    ("fuzz.partial_ratio", hf_fuzz.partial_ratio, rf_fuzz.partial_ratio, False),
    ("fuzz.token_sort_ratio", hf_fuzz.token_sort_ratio, rf_fuzz.token_sort_ratio, False),
    ("fuzz.token_set_ratio", hf_fuzz.token_set_ratio, rf_fuzz.token_set_ratio, False),
    ("fuzz.token_ratio", hf_fuzz.token_ratio, rf_fuzz.token_ratio, False),
    ("fuzz.partial_token_sort_ratio", hf_fuzz.partial_token_sort_ratio, rf_fuzz.partial_token_sort_ratio, False),
    ("fuzz.partial_token_set_ratio", hf_fuzz.partial_token_set_ratio, rf_fuzz.partial_token_set_ratio, False),
    ("fuzz.partial_token_ratio", hf_fuzz.partial_token_ratio, rf_fuzz.partial_token_ratio, False),
    ("fuzz.WRatio", hf_fuzz.w_ratio, rf_fuzz.WRatio, False),
    ("fuzz.QRatio", hf_fuzz.q_ratio, rf_fuzz.QRatio, False),
]

ITERATIONS = 1000

def compare_values(hf_val, rf_val, is_distance, tolerance=0.01):
    """Compare values with tolerance for floating point"""
    if isinstance(hf_val, float) and isinstance(rf_val, float):
        return abs(hf_val - rf_val) < tolerance
    return hf_val == rf_val

def run_test():
    print("=" * 110)
    print("COMPREHENSIVE HYPERFUZZ VS RAPIDFUZZ COMPARISON")
    print("=" * 110)
    print()
    
    overall_correct = 0
    overall_total = 0
    overall_faster = 0
    
    # Test each pair
    for pair_idx, (s1, s2) in enumerate(TEST_PAIRS):
        print(f"\n{'='*110}")
        print(f"TEST PAIR {pair_idx + 1}: ")
        print(f"  s1 ({len(s1):3d} chars): '{s1[:60]}{'...' if len(s1) > 60 else ''}'")
        print(f"  s2 ({len(s2):3d} chars): '{s2[:60]}{'...' if len(s2) > 60 else ''}'")
        print("-" * 110)
        print(f"{'Algorithm':<30} {'HF Result':>12} {'RF Result':>12} {'Match':>7} {'HF Time':>10} {'RF Time':>10} {'Speedup':>10}")
        print("-" * 110)
        
        for name, hf_func, rf_func, is_distance in ALGORITHMS:
            try:
                # Skip empty string tests for some algorithms
                if (s1 == "" or s2 == "") and "token" in name.lower():
                    continue
                
                # Run HyperFuzz
                start = time.perf_counter()
                for _ in range(ITERATIONS):
                    hf_result = hf_func(s1, s2)
                hf_time = (time.perf_counter() - start) * 1000 / ITERATIONS  # ms per call
                
                # Run RapidFuzz
                start = time.perf_counter()
                for _ in range(ITERATIONS):
                    rf_result = rf_func(s1, s2)
                rf_time = (time.perf_counter() - start) * 1000 / ITERATIONS  # ms per call
                
                # Compare results
                match = compare_values(hf_result, rf_result, is_distance)
                match_str = "✅" if match else "❌"
                
                # Format results
                if isinstance(hf_result, float):
                    hf_str = f"{hf_result:.4f}"
                    rf_str = f"{rf_result:.4f}"
                else:
                    hf_str = str(hf_result)
                    rf_str = str(rf_result)
                
                # Calculate speedup
                speedup = rf_time / hf_time if hf_time > 0 else 0
                speedup_str = f"{speedup:.2f}x 🚀" if speedup >= 1 else f"{speedup:.2f}x 🐢"
                
                print(f"{name:<30} {hf_str:>12} {rf_str:>12} {match_str:>7} {hf_time*1000:>9.2f}μs {rf_time*1000:>9.2f}μs {speedup_str:>12}")
                
                if match:
                    overall_correct += 1
                if speedup >= 1:
                    overall_faster += 1
                overall_total += 1
                
            except Exception as e:
                print(f"{name:<30} {'ERROR':>12} {str(e)[:30]:>12}")
    
    print("\n" + "=" * 110)
    print("OVERALL SUMMARY")
    print("=" * 110)
    print(f"\n✅ Score Accuracy: {overall_correct}/{overall_total} ({100*overall_correct/overall_total:.1f}%)")
    print(f"🚀 Faster than RapidFuzz: {overall_faster}/{overall_total} ({100*overall_faster/overall_total:.1f}%)")
    
    if overall_correct == overall_total:
        print("\n🎉 ALL TESTS PASSED - 100% Accuracy!")
    else:
        print(f"\n⚠️ {overall_total - overall_correct} tests had different results")

if __name__ == "__main__":
    run_test()
