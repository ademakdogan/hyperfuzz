//! LCSseq - Longest Common Subsequence algorithm
//!
//! Measures the length of the longest subsequence common to both strings.

use pyo3::prelude::*;
use rayon::prelude::*;
use std::cmp::max;

/// Calculate LCS length between two strings.
#[inline(always)]
fn lcs_length_internal(s1: &str, s2: &str) -> usize {
    let s1_chars: Vec<char> = s1.chars().collect();
    let s2_chars: Vec<char> = s2.chars().collect();

    let m = s1_chars.len();
    let n = s2_chars.len();

    if m == 0 || n == 0 {
        return 0;
    }

    // Fast path for identical strings
    if s1 == s2 {
        return m;
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

/// Calculate LCSseq distance.
/// distance = max(len1, len2) - LCS_length
#[pyfunction]
#[pyo3(signature = (s1, s2, *, score_cutoff=None))]
pub fn lcs_seq_distance(s1: &str, s2: &str, score_cutoff: Option<usize>) -> usize {
    let len1 = s1.chars().count();
    let len2 = s2.chars().count();
    let lcs = lcs_length_internal(s1, s2);
    let dist = max(len1, len2) - lcs;

    match score_cutoff {
        Some(cutoff) if dist > cutoff => cutoff + 1,
        _ => dist,
    }
}

/// Calculate LCSseq similarity (LCS length).
#[pyfunction]
#[pyo3(signature = (s1, s2, *, score_cutoff=None))]
pub fn lcs_seq_similarity(s1: &str, s2: &str, score_cutoff: Option<usize>) -> usize {
    let sim = lcs_length_internal(s1, s2);

    match score_cutoff {
        Some(cutoff) if sim < cutoff => 0,
        _ => sim,
    }
}

/// Calculate normalized LCSseq distance (0.0 to 1.0).
#[pyfunction]
#[pyo3(signature = (s1, s2, *, score_cutoff=None))]
pub fn lcs_seq_normalized_distance(s1: &str, s2: &str, score_cutoff: Option<f64>) -> f64 {
    let len1 = s1.chars().count();
    let len2 = s2.chars().count();
    let max_len = max(len1, len2);

    if max_len == 0 {
        return 0.0;
    }

    let lcs = lcs_length_internal(s1, s2);
    let dist = (max_len - lcs) as f64 / max_len as f64;

    match score_cutoff {
        Some(cutoff) if dist > cutoff => 1.0,
        _ => dist,
    }
}

/// Calculate normalized LCSseq similarity (0.0 to 1.0).
#[pyfunction]
#[pyo3(signature = (s1, s2, *, score_cutoff=None))]
pub fn lcs_seq_normalized_similarity(s1: &str, s2: &str, score_cutoff: Option<f64>) -> f64 {
    let norm_dist = lcs_seq_normalized_distance(s1, s2, None);
    let norm_sim = 1.0 - norm_dist;

    match score_cutoff {
        Some(cutoff) if norm_sim < cutoff => 0.0,
        _ => norm_sim,
    }
}

// ============ Batch Operations ============

#[pyfunction]
#[pyo3(signature = (pairs, *, score_cutoff=None))]
pub fn lcs_seq_distance_batch(
    pairs: Vec<(String, String)>,
    score_cutoff: Option<usize>,
) -> Vec<usize> {
    pairs
        .par_iter()
        .map(|(s1, s2)| lcs_seq_distance(s1, s2, score_cutoff))
        .collect()
}

#[pyfunction]
#[pyo3(signature = (pairs, *, score_cutoff=None))]
pub fn lcs_seq_normalized_similarity_batch(
    pairs: Vec<(String, String)>,
    score_cutoff: Option<f64>,
) -> Vec<f64> {
    pairs
        .par_iter()
        .map(|(s1, s2)| lcs_seq_normalized_similarity(s1, s2, score_cutoff))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lcs_length() {
        assert_eq!(lcs_length_internal("abcde", "ace"), 3);
        assert_eq!(lcs_length_internal("abc", "abc"), 3);
        assert_eq!(lcs_length_internal("abc", "def"), 0);
    }
}
