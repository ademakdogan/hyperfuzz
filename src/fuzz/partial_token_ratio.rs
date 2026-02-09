//! Partial token ratio algorithms - Ultra optimized v2
//!
//! Uses bit-parallel partial_ratio from partial_ratio.rs
//! Key optimizations:
//! 1. Early exit when intersection exists (return 100)
//! 2. Bit-parallel LCS with character filtering
//! 3. Avoid duplicate calculations

use pyo3::prelude::*;
use rayon::prelude::*;
use smallvec::SmallVec;
use ahash::AHashSet;

use crate::fuzz::partial_ratio::partial_ratio_internal;

type TokenVec<'a> = SmallVec<[&'a str; 16]>;

#[inline(always)]
fn tokenize<'a>(s: &'a str) -> TokenVec<'a> {
    s.split_whitespace().collect()
}

#[inline(always)]
fn join_tokens(tokens: &[&str]) -> String {
    if tokens.is_empty() { return String::new(); }
    if tokens.len() == 1 { return tokens[0].to_string(); }
    
    let capacity: usize = tokens.iter().map(|t| t.len()).sum::<usize>() + tokens.len() - 1;
    let mut result = String::with_capacity(capacity);
    result.push_str(tokens[0]);
    for t in &tokens[1..] {
        result.push(' ');
        result.push_str(t);
    }
    result
}

/// Partial token sort ratio - uses optimized partial_ratio
#[pyfunction]
#[pyo3(signature = (s1, s2, *, score_cutoff=None))]
pub fn partial_token_sort_ratio(s1: &str, s2: &str, score_cutoff: Option<f64>) -> f64 {
    if s1 == s2 { return 100.0; }
    
    let mut tokens1 = tokenize(s1);
    let mut tokens2 = tokenize(s2);
    
    tokens1.sort_unstable();
    tokens2.sort_unstable();
    
    if tokens1 == tokens2 { return 100.0; }
    
    let sorted1 = join_tokens(&tokens1);
    let sorted2 = join_tokens(&tokens2);
    
    // Use bit-parallel partial_ratio
    let result = partial_ratio_internal(&sorted1, &sorted2);
    
    match score_cutoff {
        Some(cutoff) if result < cutoff => 0.0,
        _ => result,
    }
}

/// Partial token set ratio - KEY OPTIMIZATION: return 100 if intersection exists
#[pyfunction]
#[pyo3(signature = (s1, s2, *, score_cutoff=None))]
pub fn partial_token_set_ratio(s1: &str, s2: &str, score_cutoff: Option<f64>) -> f64 {
    if s1 == s2 { return 100.0; }
    
    let tokens1: AHashSet<&str> = tokenize(s1).into_iter().collect();
    let tokens2: AHashSet<&str> = tokenize(s2).into_iter().collect();
    
    if tokens1 == tokens2 { return 100.0; }
    if tokens1.is_empty() || tokens2.is_empty() { return 0.0; }
    
    // KEY OPTIMIZATION from RapidFuzz: If there's any common word, return 100
    if tokens1.intersection(&tokens2).next().is_some() {
        return 100.0;
    }
    
    // No intersection - compare sorted differences
    let mut diff_ab: TokenVec = tokens1.difference(&tokens2).copied().collect();
    let mut diff_ba: TokenVec = tokens2.difference(&tokens1).copied().collect();
    
    diff_ab.sort_unstable();
    diff_ba.sort_unstable();
    
    let diff_ab_joined = join_tokens(&diff_ab);
    let diff_ba_joined = join_tokens(&diff_ba);
    
    let result = partial_ratio_internal(&diff_ab_joined, &diff_ba_joined);
    
    match score_cutoff {
        Some(cutoff) if result < cutoff => 0.0,
        _ => result,
    }
}

/// Partial token ratio - RapidFuzz algorithm
#[pyfunction]
#[pyo3(signature = (s1, s2, *, score_cutoff=None))]
pub fn partial_token_ratio(s1: &str, s2: &str, score_cutoff: Option<f64>) -> f64 {
    if s1 == s2 { return 100.0; }
    
    let tokens_split_a = tokenize(s1);
    let tokens_split_b = tokenize(s2);
    let tokens_a: AHashSet<&str> = tokens_split_a.iter().copied().collect();
    let tokens_b: AHashSet<&str> = tokens_split_b.iter().copied().collect();
    
    // KEY OPTIMIZATION: If there's any common word, return 100
    if tokens_a.intersection(&tokens_b).next().is_some() {
        return 100.0;
    }
    
    let diff_ab: TokenVec = tokens_a.difference(&tokens_b).copied().collect();
    let diff_ba: TokenVec = tokens_b.difference(&tokens_a).copied().collect();
    
    // Calculate partial_token_sort_ratio
    let mut sorted_a: TokenVec = tokens_split_a.clone();
    let mut sorted_b: TokenVec = tokens_split_b.clone();
    sorted_a.sort_unstable();
    sorted_b.sort_unstable();
    
    let sorted_a_str = join_tokens(&sorted_a);
    let sorted_b_str = join_tokens(&sorted_b);
    
    let mut result = partial_ratio_internal(&sorted_a_str, &sorted_b_str);
    
    // Avoid duplicate calculation if sets are same as original tokens
    if tokens_split_a.len() == diff_ab.len() && tokens_split_b.len() == diff_ba.len() {
        return match score_cutoff {
            Some(cutoff) if result < cutoff => 0.0,
            _ => result,
        };
    }
    
    // Calculate partial ratio for differences
    let mut diff_ab_sorted: TokenVec = diff_ab;
    let mut diff_ba_sorted: TokenVec = diff_ba;
    diff_ab_sorted.sort_unstable();
    diff_ba_sorted.sort_unstable();
    
    let diff_ab_str = join_tokens(&diff_ab_sorted);
    let diff_ba_str = join_tokens(&diff_ba_sorted);
    
    let diff_result = partial_ratio_internal(&diff_ab_str, &diff_ba_str);
    result = result.max(diff_result);
    
    match score_cutoff {
        Some(cutoff) if result < cutoff => 0.0,
        _ => result,
    }
}

// ============ Batch Operations ============

#[pyfunction]
#[pyo3(signature = (pairs, *, score_cutoff=None))]
pub fn partial_token_sort_ratio_batch(pairs: Vec<(String, String)>, score_cutoff: Option<f64>) -> Vec<f64> {
    pairs.par_iter().map(|(s1, s2)| partial_token_sort_ratio(s1, s2, score_cutoff)).collect()
}

#[pyfunction]
#[pyo3(signature = (pairs, *, score_cutoff=None))]
pub fn partial_token_set_ratio_batch(pairs: Vec<(String, String)>, score_cutoff: Option<f64>) -> Vec<f64> {
    pairs.par_iter().map(|(s1, s2)| partial_token_set_ratio(s1, s2, score_cutoff)).collect()
}

#[pyfunction]
#[pyo3(signature = (pairs, *, score_cutoff=None))]
pub fn partial_token_ratio_batch(pairs: Vec<(String, String)>, score_cutoff: Option<f64>) -> Vec<f64> {
    pairs.par_iter().map(|(s1, s2)| partial_token_ratio(s1, s2, score_cutoff)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_partial_token_sort() {
        let r = partial_token_sort_ratio("fuzzy wuzzy was a bear", "wuzzy fuzzy", None);
        assert!(r > 90.0);
    }
    
    #[test]
    fn test_partial_token_set_ratio_intersection() {
        let r = partial_token_set_ratio("hello world", "world peace", None);
        assert!((r - 100.0).abs() < 0.01);
    }
    
    #[test]
    fn test_partial_token_set_ratio_no_intersection() {
        let r = partial_token_set_ratio("kitten", "sitting", None);
        assert!((r - 66.67).abs() < 0.1);
    }
}
