//! LCSseq (Longest Common Subsequence) algorithm - Optimized
//!
//! Calculates the longest common subsequence between two strings.

use pyo3::prelude::*;
use rayon::prelude::*;
use smallvec::SmallVec;
use std::cmp::max;

type CharVec = SmallVec<[char; 64]>;
type RowVec = SmallVec<[usize; 64]>;

/// Calculate LCS length for ASCII strings
#[inline(always)]
fn lcs_length_ascii(s1: &[u8], s2: &[u8]) -> usize {
    let m = s1.len();
    let n = s2.len();

    if m == 0 || n == 0 {
        return 0;
    }

    // Make s1 the shorter string
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

/// Calculate LCS length.
#[inline(always)]
fn lcs_length(s1: &str, s2: &str) -> usize {
    if s1 == s2 {
        return if s1.is_ascii() { s1.len() } else { s1.chars().count() };
    }
    
    if s1.is_ascii() && s2.is_ascii() {
        return lcs_length_ascii(s1.as_bytes(), s2.as_bytes());
    }
    
    let s1_chars: CharVec = s1.chars().collect();
    let s2_chars: CharVec = s2.chars().collect();
    lcs_length_chars(&s1_chars, &s2_chars)
}

/// Get string length efficiently
#[inline(always)]
fn str_len(s: &str) -> usize {
    if s.is_ascii() { s.len() } else { s.chars().count() }
}

/// LCS distance: max(len1, len2) - LCS
#[pyfunction]
#[pyo3(signature = (s1, s2, *, score_cutoff=None))]
pub fn lcs_seq_distance(s1: &str, s2: &str, score_cutoff: Option<usize>) -> usize {
    if s1 == s2 {
        return 0;
    }
    
    let len1 = str_len(s1);
    let len2 = str_len(s2);
    let lcs = lcs_length(s1, s2);
    let dist = len1.max(len2) - lcs;  // Correct formula

    match score_cutoff {
        Some(cutoff) if dist > cutoff => cutoff + 1,
        _ => dist,
    }
}

/// LCS similarity: LCS length
#[pyfunction]
#[pyo3(signature = (s1, s2, *, score_cutoff=None))]
pub fn lcs_seq_similarity(s1: &str, s2: &str, score_cutoff: Option<usize>) -> usize {
    let lcs = lcs_length(s1, s2);

    match score_cutoff {
        Some(cutoff) if lcs < cutoff => 0,
        _ => lcs,
    }
}

/// Normalized LCS distance
#[pyfunction]
#[pyo3(signature = (s1, s2, *, score_cutoff=None))]
pub fn lcs_seq_normalized_distance(s1: &str, s2: &str, score_cutoff: Option<f64>) -> f64 {
    if s1 == s2 {
        return 0.0;
    }
    
    let len1 = str_len(s1);
    let len2 = str_len(s2);
    let max_len = len1.max(len2);

    if max_len == 0 {
        return 0.0;
    }

    let lcs = lcs_length(s1, s2);
    let dist = (max_len - lcs) as f64 / max_len as f64;

    match score_cutoff {
        Some(cutoff) if dist > cutoff => 1.0,
        _ => dist,
    }
}

/// Normalized LCS similarity
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
    fn test_lcs_distance() {
        assert_eq!(lcs_seq_distance("abc", "abc", None), 0);
        assert_eq!(lcs_seq_distance("abc", "def", None), 6);
    }
}
