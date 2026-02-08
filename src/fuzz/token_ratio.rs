//! Token-based ratio algorithms - RapidFuzz-inspired optimizations
//!
//! Key optimizations from RapidFuzz:
//! 1. Early exit when intersection exists (for set ratios)
//! 2. Mathematical distance calculation without full LCS
//! 3. Indel distance instead of LCS where applicable

use pyo3::prelude::*;
use rayon::prelude::*;
use smallvec::SmallVec;
use ahash::AHashSet;

use crate::lcs_core::{lcs_fast, ratio_from_lcs};

type TokenVec<'a> = SmallVec<[&'a str; 16]>;

/// Tokenize
#[inline(always)]
fn tokenize<'a>(s: &'a str) -> TokenVec<'a> {
    s.split_whitespace().collect()
}

/// Efficient join with pre-allocated capacity
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

/// Fast ratio
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

/// Calculate normalized distance from indel distance
#[inline(always)]
fn norm_distance(dist: usize, lensum: usize, score_cutoff: f64) -> f64 {
    let score = if lensum > 0 { 100.0 - 100.0 * dist as f64 / lensum as f64 } else { 100.0 };
    if score >= score_cutoff { score } else { 0.0 }
}

/// Indel distance (len1 + len2 - 2*LCS)
#[inline(always)]
fn indel_dist(s1: &str, s2: &str) -> usize {
    if s1 == s2 { return 0; }
    let len1 = if s1.is_ascii() { s1.len() } else { s1.chars().count() };
    let len2 = if s2.is_ascii() { s2.len() } else { s2.chars().count() };
    let lcs = lcs_fast(s1, s2);
    len1 + len2 - 2 * lcs
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
    
    if tokens1 == tokens2 { return 100.0; }
    
    let sorted1 = join_tokens(&tokens1);
    let sorted2 = join_tokens(&tokens2);
    
    let result = ratio_internal(&sorted1, &sorted2);
    
    match score_cutoff {
        Some(cutoff) if result < cutoff => 0.0,
        _ => result,
    }
}

/// Token set ratio - RapidFuzz algorithm with mathematical shortcuts
#[pyfunction]
#[pyo3(signature = (s1, s2, *, score_cutoff=None))]
pub fn token_set_ratio(s1: &str, s2: &str, score_cutoff: Option<f64>) -> f64 {
    if s1 == s2 { return 100.0; }
    
    let tokens1: AHashSet<&str> = tokenize(s1).into_iter().collect();
    let tokens2: AHashSet<&str> = tokenize(s2).into_iter().collect();
    
    if tokens1 == tokens2 { return 100.0; }
    if tokens1.is_empty() || tokens2.is_empty() { return 0.0; }
    
    let intersect: AHashSet<&str> = tokens1.intersection(&tokens2).copied().collect();
    let diff_ab: TokenVec = tokens1.difference(&tokens2).copied().collect();
    let diff_ba: TokenVec = tokens2.difference(&tokens1).copied().collect();
    
    // KEY OPTIMIZATION: One sentence is part of the other
    if !intersect.is_empty() && (diff_ab.is_empty() || diff_ba.is_empty()) {
        return 100.0;
    }
    
    let score_cutoff = score_cutoff.unwrap_or(0.0);
    
    // Sort differences
    let mut diff_ab_sorted: TokenVec = diff_ab.clone();
    let mut diff_ba_sorted: TokenVec = diff_ba.clone();
    diff_ab_sorted.sort_unstable();
    diff_ba_sorted.sort_unstable();
    
    let diff_ab_joined = join_tokens(&diff_ab_sorted);
    let diff_ba_joined = join_tokens(&diff_ba_sorted);
    
    let ab_len = diff_ab_joined.chars().count();
    let ba_len = diff_ba_joined.chars().count();
    
    // Calculate sect_len (intersection string length)
    let sect_len: usize = if intersect.is_empty() {
        0
    } else {
        let inter_vec: TokenVec = intersect.iter().copied().collect();
        join_tokens(&inter_vec).chars().count()
    };
    
    // String lengths: sect+ab <-> sect and sect+ba <-> sect
    let sect_ab_len = sect_len + if sect_len != 0 { 1 } else { 0 } + ab_len;
    let sect_ba_len = sect_len + if sect_len != 0 { 1 } else { 0 } + ba_len;
    
    // Calculate distance between diff strings
    let dist = indel_dist(&diff_ab_joined, &diff_ba_joined);
    let mut result = norm_distance(dist, sect_ab_len + sect_ba_len, score_cutoff);
    
    // Exit early if no intersection
    if sect_len == 0 {
        return result;
    }
    
    // Mathematical shortcuts for sect+ab <-> sect and sect+ba <-> sect
    let sect_ab_dist = if sect_len != 0 { 1 } else { 0 } + ab_len;
    let sect_ab_ratio = norm_distance(sect_ab_dist, sect_len + sect_ab_len, score_cutoff);
    
    let sect_ba_dist = if sect_len != 0 { 1 } else { 0 } + ba_len;
    let sect_ba_ratio = norm_distance(sect_ba_dist, sect_len + sect_ba_len, score_cutoff);
    
    result = result.max(sect_ab_ratio).max(sect_ba_ratio);
    result
}

/// Token ratio - max of sort and set
#[pyfunction]
#[pyo3(signature = (s1, s2, *, score_cutoff=None))]
pub fn token_ratio(s1: &str, s2: &str, score_cutoff: Option<f64>) -> f64 {
    if s1 == s2 { return 100.0; }
    
    // token_set_ratio often returns 100 early, check it first
    let set_result = token_set_ratio(s1, s2, None);
    if set_result >= 100.0 {
        return match score_cutoff {
            Some(cutoff) if set_result < cutoff => 0.0,
            _ => set_result,
        };
    }
    
    let sort_result = token_sort_ratio(s1, s2, None);
    let result = set_result.max(sort_result);
    
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
    fn test_token_set_ratio_subset() {
        // One is subset of other - should return 100
        let r = token_set_ratio("fuzzy was a bear", "fuzzy fuzzy was a bear", None);
        assert!((r - 100.0).abs() < 0.01);
    }
    
    #[test]
    fn test_token_set_ratio_single_words() {
        let r = token_set_ratio("kitten", "sitting", None);
        assert!((r - 61.54).abs() < 0.1);
    }
}
