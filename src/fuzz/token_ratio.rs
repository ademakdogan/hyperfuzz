//! Token-based ratio algorithms
//!
//! Provides token_sort_ratio, token_set_ratio, and token_ratio.

use pyo3::prelude::*;
use rayon::prelude::*;
use std::cmp::max;
use ahash::AHashSet;

/// Tokenize a string (split by whitespace).
#[inline(always)]
fn tokenize(s: &str) -> Vec<&str> {
    s.split_whitespace().collect()
}

/// Calculate LCS length between two strings.
#[inline(always)]
fn lcs_length(s1: &str, s2: &str) -> usize {
    let s1_chars: Vec<char> = s1.chars().collect();
    let s2_chars: Vec<char> = s2.chars().collect();

    let m = s1_chars.len();
    let n = s2_chars.len();

    if m == 0 || n == 0 {
        return 0;
    }

    let mut prev: Vec<usize> = vec![0; n + 1];
    let mut curr: Vec<usize> = vec![0; n + 1];

    for i in 1..=m {
        for j in 1..=n {
            if s1_chars[i - 1] == s2_chars[j - 1] {
                curr[j] = prev[j - 1] + 1;
            } else {
                curr[j] = max(prev[j], curr[j - 1]);
            }
        }
        std::mem::swap(&mut prev, &mut curr);
        curr.fill(0);
    }

    prev[n]
}

/// Calculate ratio for two strings.
#[inline(always)]
fn ratio_internal(s1: &str, s2: &str) -> f64 {
    let len1 = s1.chars().count();
    let len2 = s2.chars().count();
    let total = len1 + len2;

    if total == 0 {
        return 100.0;
    }
    if s1 == s2 {
        return 100.0;
    }

    let lcs = lcs_length(s1, s2);
    100.0 * (2.0 * lcs as f64) / (total as f64)
}

/// Token sort ratio - sort tokens before comparison.
#[pyfunction]
#[pyo3(signature = (s1, s2, *, score_cutoff=None))]
pub fn token_sort_ratio(s1: &str, s2: &str, score_cutoff: Option<f64>) -> f64 {
    let mut tokens1: Vec<&str> = tokenize(s1);
    let mut tokens2: Vec<&str> = tokenize(s2);
    
    tokens1.sort_unstable();
    tokens2.sort_unstable();
    
    let sorted1 = tokens1.join(" ");
    let sorted2 = tokens2.join(" ");
    
    let result = ratio_internal(&sorted1, &sorted2);
    
    match score_cutoff {
        Some(cutoff) if result < cutoff => 0.0,
        _ => result,
    }
}

/// Token set ratio - compare token sets.
/// 
/// RapidFuzz algorithm:
/// 1. Tokenize both strings
/// 2. Calculate intersection, difference1 (s1-s2), difference2 (s2-s1)
/// 3. Create 3 strings: sorted_intersection, sorted_intersection + sorted_diff1, etc.
/// 4. Return max of comparing these strings using ratio
#[pyfunction]
#[pyo3(signature = (s1, s2, *, score_cutoff=None))]
pub fn token_set_ratio(s1: &str, s2: &str, score_cutoff: Option<f64>) -> f64 {
    let tokens1: AHashSet<&str> = tokenize(s1).into_iter().collect();
    let tokens2: AHashSet<&str> = tokenize(s2).into_iter().collect();
    
    if tokens1.is_empty() && tokens2.is_empty() {
        return 100.0;
    }
    if tokens1.is_empty() || tokens2.is_empty() {
        return 0.0;
    }
    
    let intersection: Vec<&str> = tokens1.intersection(&tokens2).copied().collect();
    let diff1: Vec<&str> = tokens1.difference(&tokens2).copied().collect();
    let diff2: Vec<&str> = tokens2.difference(&tokens1).copied().collect();
    
    // Sort all token lists
    let mut sorted_inter = intersection.clone();
    let mut sorted_diff1 = diff1.clone();
    let mut sorted_diff2 = diff2.clone();
    sorted_inter.sort_unstable();
    sorted_diff1.sort_unstable();
    sorted_diff2.sort_unstable();
    
    let inter_str = sorted_inter.join(" ");
    
    // Build combined strings
    let combined1 = if sorted_diff1.is_empty() {
        inter_str.clone()
    } else if inter_str.is_empty() {
        sorted_diff1.join(" ")
    } else {
        format!("{} {}", inter_str, sorted_diff1.join(" "))
    };
    
    let combined2 = if sorted_diff2.is_empty() {
        inter_str.clone()
    } else if inter_str.is_empty() {
        sorted_diff2.join(" ")
    } else {
        format!("{} {}", inter_str, sorted_diff2.join(" "))
    };
    
    // RapidFuzz compares:
    // 1. intersection vs combined1
    // 2. intersection vs combined2
    // 3. combined1 vs combined2
    // But if intersection is empty, it just falls back to regular ratio behavior
    
    let result = if inter_str.is_empty() {
        // No common tokens, compare the full sorted token strings
        ratio_internal(&combined1, &combined2)
    } else {
        let r1 = ratio_internal(&inter_str, &combined1);
        let r2 = ratio_internal(&inter_str, &combined2);
        let r3 = ratio_internal(&combined1, &combined2);
        r1.max(r2).max(r3)
    };
    
    match score_cutoff {
        Some(cutoff) if result < cutoff => 0.0,
        _ => result,
    }
}

/// Token ratio - max of token_sort_ratio and token_set_ratio.
#[pyfunction]
#[pyo3(signature = (s1, s2, *, score_cutoff=None))]
pub fn token_ratio(s1: &str, s2: &str, score_cutoff: Option<f64>) -> f64 {
    let sort_result = token_sort_ratio(s1, s2, None);
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
    pairs
        .par_iter()
        .map(|(s1, s2)| token_sort_ratio(s1, s2, score_cutoff))
        .collect()
}

#[pyfunction]
#[pyo3(signature = (pairs, *, score_cutoff=None))]
pub fn token_set_ratio_batch(pairs: Vec<(String, String)>, score_cutoff: Option<f64>) -> Vec<f64> {
    pairs
        .par_iter()
        .map(|(s1, s2)| token_set_ratio(s1, s2, score_cutoff))
        .collect()
}

#[pyfunction]
#[pyo3(signature = (pairs, *, score_cutoff=None))]
pub fn token_ratio_batch(pairs: Vec<(String, String)>, score_cutoff: Option<f64>) -> Vec<f64> {
    pairs
        .par_iter()
        .map(|(s1, s2)| token_ratio(s1, s2, score_cutoff))
        .collect()
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
        // For single different words, should behave like ratio
        let r = token_set_ratio("kitten", "sitting", None);
        // 61.538... like ratio
        assert!((r - 61.54).abs() < 0.1);
    }
}
