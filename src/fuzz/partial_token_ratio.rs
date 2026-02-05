//! Partial token ratio algorithms
//!
//! Combines partial matching with token-based approaches.

use pyo3::prelude::*;
use rayon::prelude::*;
use std::cmp::max;
use ahash::AHashSet;

/// Tokenize a string.
#[inline(always)]
fn tokenize(s: &str) -> Vec<&str> {
    s.split_whitespace().collect()
}

/// Calculate LCS length between two char slices.
#[inline(always)]
fn lcs_length_chars(s1: &[char], s2: &[char]) -> usize {
    let m = s1.len();
    let n = s2.len();

    if m == 0 || n == 0 {
        return 0;
    }

    let mut prev: Vec<usize> = vec![0; n + 1];
    let mut curr: Vec<usize> = vec![0; n + 1];

    for i in 1..=m {
        for j in 1..=n {
            if s1[i - 1] == s2[j - 1] {
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

/// Calculate LCS length between two strings.
#[inline(always)]
fn lcs_length(s1: &str, s2: &str) -> usize {
    let s1_chars: Vec<char> = s1.chars().collect();
    let s2_chars: Vec<char> = s2.chars().collect();
    lcs_length_chars(&s1_chars, &s2_chars)
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

/// Calculate partial ratio between two strings.
#[inline(always)]
fn partial_ratio_internal(s1: &str, s2: &str) -> f64 {
    let s1_chars: Vec<char> = s1.chars().collect();
    let s2_chars: Vec<char> = s2.chars().collect();

    let len1 = s1_chars.len();
    let len2 = s2_chars.len();

    if len1 == 0 && len2 == 0 {
        return 100.0;
    }
    if len1 == 0 || len2 == 0 {
        return 0.0;
    }
    if s1 == s2 {
        return 100.0;
    }

    let (shorter, longer) = if len1 <= len2 {
        (&s1_chars, &s2_chars)
    } else {
        (&s2_chars, &s1_chars)
    };

    let short_len = shorter.len();
    let long_len = longer.len();

    let mut best_score = 0.0f64;

    for i in 0..=(long_len - short_len) {
        let window = &longer[i..i + short_len];
        let lcs = lcs_length_chars(shorter, window);
        let score = 100.0 * (2.0 * lcs as f64) / ((short_len + short_len) as f64);
        if score > best_score {
            best_score = score;
        }
        if best_score >= 100.0 {
            break;
        }
    }

    best_score
}

/// Partial token sort ratio.
#[pyfunction]
#[pyo3(signature = (s1, s2, *, score_cutoff=None))]
pub fn partial_token_sort_ratio(s1: &str, s2: &str, score_cutoff: Option<f64>) -> f64 {
    let mut tokens1: Vec<&str> = tokenize(s1);
    let mut tokens2: Vec<&str> = tokenize(s2);
    
    tokens1.sort_unstable();
    tokens2.sort_unstable();
    
    let sorted1 = tokens1.join(" ");
    let sorted2 = tokens2.join(" ");
    
    let result = partial_ratio_internal(&sorted1, &sorted2);
    
    match score_cutoff {
        Some(cutoff) if result < cutoff => 0.0,
        _ => result,
    }
}

/// Partial token set ratio.
#[pyfunction]
#[pyo3(signature = (s1, s2, *, score_cutoff=None))]
pub fn partial_token_set_ratio(s1: &str, s2: &str, score_cutoff: Option<f64>) -> f64 {
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
    
    let mut sorted_inter = intersection.clone();
    let mut sorted_diff1 = diff1.clone();
    let mut sorted_diff2 = diff2.clone();
    sorted_inter.sort_unstable();
    sorted_diff1.sort_unstable();
    sorted_diff2.sort_unstable();
    
    let inter_str = sorted_inter.join(" ");
    
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
    
    let result = if inter_str.is_empty() {
        partial_ratio_internal(&combined1, &combined2)
    } else {
        let r1 = partial_ratio_internal(&inter_str, &combined1);
        let r2 = partial_ratio_internal(&inter_str, &combined2);
        let r3 = partial_ratio_internal(&combined1, &combined2);
        r1.max(r2).max(r3)
    };
    
    match score_cutoff {
        Some(cutoff) if result < cutoff => 0.0,
        _ => result,
    }
}

/// Partial token ratio - max of partial_token_sort and partial_token_set.
#[pyfunction]
#[pyo3(signature = (s1, s2, *, score_cutoff=None))]
pub fn partial_token_ratio(s1: &str, s2: &str, score_cutoff: Option<f64>) -> f64 {
    let sort_result = partial_token_sort_ratio(s1, s2, None);
    let set_result = partial_token_set_ratio(s1, s2, None);
    let result = sort_result.max(set_result);
    
    match score_cutoff {
        Some(cutoff) if result < cutoff => 0.0,
        _ => result,
    }
}

// ============ Batch Operations ============

#[pyfunction]
#[pyo3(signature = (pairs, *, score_cutoff=None))]
pub fn partial_token_sort_ratio_batch(pairs: Vec<(String, String)>, score_cutoff: Option<f64>) -> Vec<f64> {
    pairs
        .par_iter()
        .map(|(s1, s2)| partial_token_sort_ratio(s1, s2, score_cutoff))
        .collect()
}

#[pyfunction]
#[pyo3(signature = (pairs, *, score_cutoff=None))]
pub fn partial_token_set_ratio_batch(pairs: Vec<(String, String)>, score_cutoff: Option<f64>) -> Vec<f64> {
    pairs
        .par_iter()
        .map(|(s1, s2)| partial_token_set_ratio(s1, s2, score_cutoff))
        .collect()
}

#[pyfunction]
#[pyo3(signature = (pairs, *, score_cutoff=None))]
pub fn partial_token_ratio_batch(pairs: Vec<(String, String)>, score_cutoff: Option<f64>) -> Vec<f64> {
    pairs
        .par_iter()
        .map(|(s1, s2)| partial_token_ratio(s1, s2, score_cutoff))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_partial_token_sort() {
        let r = partial_token_sort_ratio("fuzzy wuzzy was a bear", "wuzzy fuzzy", None);
        assert!(r > 90.0);
    }
}
