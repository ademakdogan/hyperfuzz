//! Token-based ratio algorithms - Optimized
//!
//! Provides token_sort_ratio, token_set_ratio, and token_ratio.

use pyo3::prelude::*;
use rayon::prelude::*;
use smallvec::SmallVec;
use ahash::AHashSet;
use std::cmp::max;

// Stack allocation for typical cases
type CharVec = SmallVec<[char; 64]>;
type RowVec = SmallVec<[usize; 64]>;

/// Tokenize a string (split by whitespace).
#[inline(always)]
fn tokenize(s: &str) -> SmallVec<[&str; 16]> {
    s.split_whitespace().collect()
}

/// Calculate LCS length for ASCII bytes
#[inline(always)]
fn lcs_length_ascii(s1: &[u8], s2: &[u8]) -> usize {
    let m = s1.len();
    let n = s2.len();

    if m == 0 || n == 0 {
        return 0;
    }

    let (s1, s2, m, n) = if m > n {
        (s2, s1, n, m)
    } else {
        (s1, s2, m, n)
    };

    let mut prev: RowVec = SmallVec::from_elem(0, m + 1);
    let mut curr: RowVec = SmallVec::from_elem(0, m + 1);

    for j in 1..=n {
        for i in 1..=m {
            curr[i] = if s1[i - 1] == s2[j - 1] {
                prev[i - 1] + 1
            } else {
                max(prev[i], curr[i - 1])
            };
        }
        std::mem::swap(&mut prev, &mut curr);
        curr.fill(0);
    }

    prev[m]
}

/// Calculate LCS length for Unicode chars
#[inline(always)]
fn lcs_length_chars(s1: &[char], s2: &[char]) -> usize {
    let m = s1.len();
    let n = s2.len();

    if m == 0 || n == 0 {
        return 0;
    }

    let (s1, s2, m, n) = if m > n {
        (s2, s1, n, m)
    } else {
        (s1, s2, m, n)
    };

    let mut prev: RowVec = SmallVec::from_elem(0, m + 1);
    let mut curr: RowVec = SmallVec::from_elem(0, m + 1);

    for j in 1..=n {
        for i in 1..=m {
            curr[i] = if s1[i - 1] == s2[j - 1] {
                prev[i - 1] + 1
            } else {
                max(prev[i], curr[i - 1])
            };
        }
        std::mem::swap(&mut prev, &mut curr);
        curr.fill(0);
    }

    prev[m]
}

/// Calculate ratio for two strings.
#[inline(always)]
fn ratio_internal(s1: &str, s2: &str) -> f64 {
    if s1 == s2 {
        return 100.0;
    }
    
    let (len1, len2, lcs) = if s1.is_ascii() && s2.is_ascii() {
        let b1 = s1.as_bytes();
        let b2 = s2.as_bytes();
        (b1.len(), b2.len(), lcs_length_ascii(b1, b2))
    } else {
        let c1: CharVec = s1.chars().collect();
        let c2: CharVec = s2.chars().collect();
        let lcs = lcs_length_chars(&c1, &c2);
        (c1.len(), c2.len(), lcs)
    };
    
    let total = len1 + len2;

    if total == 0 {
        return 100.0;
    }

    100.0 * (2.0 * lcs as f64) / (total as f64)
}

/// Token sort ratio - sort tokens before comparison.
#[pyfunction]
#[pyo3(signature = (s1, s2, *, score_cutoff=None))]
pub fn token_sort_ratio(s1: &str, s2: &str, score_cutoff: Option<f64>) -> f64 {
    // Fast path
    if s1 == s2 {
        return 100.0;
    }
    
    let mut tokens1 = tokenize(s1);
    let mut tokens2 = tokenize(s2);
    
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
#[pyfunction]
#[pyo3(signature = (s1, s2, *, score_cutoff=None))]
pub fn token_set_ratio(s1: &str, s2: &str, score_cutoff: Option<f64>) -> f64 {
    // Fast path
    if s1 == s2 {
        return 100.0;
    }
    
    let tokens1: AHashSet<&str> = tokenize(s1).into_iter().collect();
    let tokens2: AHashSet<&str> = tokenize(s2).into_iter().collect();
    
    if tokens1.is_empty() && tokens2.is_empty() {
        return 100.0;
    }
    if tokens1.is_empty() || tokens2.is_empty() {
        return 0.0;
    }
    
    let intersection: SmallVec<[&str; 16]> = tokens1.intersection(&tokens2).copied().collect();
    let diff1: SmallVec<[&str; 16]> = tokens1.difference(&tokens2).copied().collect();
    let diff2: SmallVec<[&str; 16]> = tokens2.difference(&tokens1).copied().collect();
    
    let mut sorted_inter = intersection;
    let mut sorted_diff1 = diff1;
    let mut sorted_diff2 = diff2;
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
    if s1 == s2 {
        return 100.0;
    }
    
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
        let r = token_set_ratio("kitten", "sitting", None);
        assert!((r - 61.54).abs() < 0.1);
    }
}
