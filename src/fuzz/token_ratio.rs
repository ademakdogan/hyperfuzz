//! Token-based ratio algorithms - Ultra Optimized V2
//!
//! Uses the optimized lcs_core for thread-local buffer reuse
//! and aggressive caching strategies.

use pyo3::prelude::*;
use rayon::prelude::*;
use smallvec::SmallVec;
use ahash::AHashSet;

use crate::lcs_core::{lcs_fast, ratio_from_lcs};

// Token collection with small stack allocation
type TokenVec<'a> = SmallVec<[&'a str; 16]>;

/// Tokenize with inline optimization
#[inline(always)]
fn tokenize<'a>(s: &'a str) -> TokenVec<'a> {
    s.split_whitespace().collect()
}

/// Efficient string join with pre-allocated capacity
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

/// Fast ratio using lcs_core
#[inline(always)]
fn ratio_internal(s1: &str, s2: &str) -> f64 {
    if s1 == s2 { return 100.0; }
    if s1.is_empty() && s2.is_empty() { return 100.0; }
    if s1.is_empty() || s2.is_empty() { return 0.0; }
    
    let len1 = if s1.is_ascii() { s1.len() } else { s1.chars().count() };
    let len2 = if s2.is_ascii() { s2.len() } else { s2.chars().count() };
    let lcs = lcs_fast(s1, s2);
    
    ratio_from_lcs(len1, len2, lcs)
}

/// Token sort ratio
#[pyfunction]
#[pyo3(signature = (s1, s2, *, score_cutoff=None))]
pub fn token_sort_ratio(s1: &str, s2: &str, score_cutoff: Option<f64>) -> f64 {
    if s1 == s2 { return 100.0; }
    
    let mut tokens1 = tokenize(s1);
    let mut tokens2 = tokenize(s2);
    
    tokens1.sort_unstable();
    tokens2.sort_unstable();
    
    // Fast path - identical after sort
    if tokens1 == tokens2 { return 100.0; }
    
    let sorted1 = join_tokens(&tokens1);
    let sorted2 = join_tokens(&tokens2);
    
    let result = ratio_internal(&sorted1, &sorted2);
    
    match score_cutoff {
        Some(cutoff) if result < cutoff => 0.0,
        _ => result,
    }
}

/// Token set ratio with optimized set operations
#[pyfunction]
#[pyo3(signature = (s1, s2, *, score_cutoff=None))]
pub fn token_set_ratio(s1: &str, s2: &str, score_cutoff: Option<f64>) -> f64 {
    if s1 == s2 { return 100.0; }
    
    let tokens1: AHashSet<&str> = tokenize(s1).into_iter().collect();
    let tokens2: AHashSet<&str> = tokenize(s2).into_iter().collect();
    
    // Fast paths
    if tokens1 == tokens2 { return 100.0; }
    if tokens1.is_empty() && tokens2.is_empty() { return 100.0; }
    if tokens1.is_empty() || tokens2.is_empty() { return 0.0; }
    
    // Compute intersections and differences
    let mut inter: TokenVec = tokens1.intersection(&tokens2).copied().collect();
    let mut diff1: TokenVec = tokens1.difference(&tokens2).copied().collect();
    let mut diff2: TokenVec = tokens2.difference(&tokens1).copied().collect();
    
    inter.sort_unstable();
    diff1.sort_unstable();
    diff2.sort_unstable();
    
    let inter_str = join_tokens(&inter);
    
    // No intersection - direct comparison
    if inter_str.is_empty() {
        let c1 = join_tokens(&diff1);
        let c2 = join_tokens(&diff2);
        let result = ratio_internal(&c1, &c2);
        return match score_cutoff {
            Some(cutoff) if result < cutoff => 0.0,
            _ => result,
        };
    }
    
    // Build combined strings
    let combined1 = if diff1.is_empty() {
        inter_str.clone()
    } else {
        format!("{} {}", inter_str, join_tokens(&diff1))
    };
    
    let combined2 = if diff2.is_empty() {
        inter_str.clone()
    } else {
        format!("{} {}", inter_str, join_tokens(&diff2))
    };
    
    // Calculate all ratios
    let r1 = ratio_internal(&inter_str, &combined1);
    let r2 = ratio_internal(&inter_str, &combined2);
    let r3 = ratio_internal(&combined1, &combined2);
    
    let result = r1.max(r2).max(r3);
    
    match score_cutoff {
        Some(cutoff) if result < cutoff => 0.0,
        _ => result,
    }
}

/// Token ratio - max of sort and set
#[pyfunction]
#[pyo3(signature = (s1, s2, *, score_cutoff=None))]
pub fn token_ratio(s1: &str, s2: &str, score_cutoff: Option<f64>) -> f64 {
    if s1 == s2 { return 100.0; }
    
    let sort_result = token_sort_ratio(s1, s2, None);
    
    // Early exit if already maxed
    if sort_result >= 100.0 {
        return match score_cutoff {
            Some(cutoff) if sort_result < cutoff => 0.0,
            _ => sort_result,
        };
    }
    
    let set_result = token_set_ratio(s1, s2, None);
    let result = sort_result.max(set_result);
    
    match score_cutoff {
        Some(cutoff) if result < cutoff => 0.0,
        _ => result,
    }
}

// ============ Batch Operations ============

#[pyfunction]
#[pyo3(signature = (pairs, *, score_cutoff=None))]
pub fn token_sort_ratio_batch(pairs: Vec<(String, String)>, score_cutoff: Option<f64>) -> Vec<f64> {
    pairs.par_iter().map(|(s1, s2)| token_sort_ratio(s1, s2, score_cutoff)).collect()
}

#[pyfunction]
#[pyo3(signature = (pairs, *, score_cutoff=None))]
pub fn token_set_ratio_batch(pairs: Vec<(String, String)>, score_cutoff: Option<f64>) -> Vec<f64> {
    pairs.par_iter().map(|(s1, s2)| token_set_ratio(s1, s2, score_cutoff)).collect()
}

#[pyfunction]
#[pyo3(signature = (pairs, *, score_cutoff=None))]
pub fn token_ratio_batch(pairs: Vec<(String, String)>, score_cutoff: Option<f64>) -> Vec<f64> {
    pairs.par_iter().map(|(s1, s2)| token_ratio(s1, s2, score_cutoff)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_sort_ratio() {
        let r = token_sort_ratio("fuzzy wuzzy was a bear", "wuzzy fuzzy was a bear", None);
        assert!((r - 100.0).abs() < 0.01);
    }

    #[test]
    fn test_token_set_ratio() {
        let r = token_set_ratio("fuzzy was a bear", "fuzzy fuzzy was a bear", None);
        assert!((r - 100.0).abs() < 0.01);
    }
    
    #[test]
    fn test_token_set_ratio_single_words() {
        let r = token_set_ratio("kitten", "sitting", None);
        assert!((r - 61.54).abs() < 0.1);
    }
}
