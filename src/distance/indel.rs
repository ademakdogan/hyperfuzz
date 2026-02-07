//! Indel distance algorithm (insertion/deletion only) - Optimized
//!
//! Based on Longest Common Subsequence. Only insertions and deletions,
//! no substitutions allowed.

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

/// Calculate LCS length between two strings.
#[inline(always)]
fn lcs_length(s1: &str, s2: &str) -> usize {
    if s1.is_ascii() && s2.is_ascii() {
        return lcs_length_ascii(s1.as_bytes(), s2.as_bytes());
    }
    
    let s1_chars: CharVec = s1.chars().collect();
    let s2_chars: CharVec = s2.chars().collect();
    lcs_length_chars(&s1_chars, &s2_chars)
}

/// Get string length (fast for ASCII)
#[inline(always)]
fn str_len(s: &str) -> usize {
    if s.is_ascii() { s.len() } else { s.chars().count() }
}

/// Calculate the Indel distance (insertions + deletions needed).
/// Formula: len(s1) + len(s2) - 2 * LCS(s1, s2)
#[pyfunction]
#[pyo3(signature = (s1, s2, *, score_cutoff=None))]
pub fn indel_distance(s1: &str, s2: &str, score_cutoff: Option<usize>) -> usize {
    if s1 == s2 {
        return 0;
    }
    
    let len1 = str_len(s1);
    let len2 = str_len(s2);
    let lcs = lcs_length(s1, s2);
    let dist = len1 + len2 - 2 * lcs;

    match score_cutoff {
        Some(cutoff) if dist > cutoff => cutoff + 1,
        _ => dist,
    }
}

/// Calculate Indel similarity.
/// similarity = len1 + len2 - distance = 2 * LCS
#[pyfunction]
#[pyo3(signature = (s1, s2, *, score_cutoff=None))]
pub fn indel_similarity(s1: &str, s2: &str, score_cutoff: Option<usize>) -> usize {
    if s1 == s2 {
        return str_len(s1) * 2;
    }
    
    let lcs = lcs_length(s1, s2);
    let sim = 2 * lcs;

    match score_cutoff {
        Some(cutoff) if sim < cutoff => 0,
        _ => sim,
    }
}

/// Calculate normalized Indel distance (0.0 to 1.0).
#[pyfunction]
#[pyo3(signature = (s1, s2, *, score_cutoff=None))]
pub fn indel_normalized_distance(s1: &str, s2: &str, score_cutoff: Option<f64>) -> f64 {
    if s1 == s2 {
        return 0.0;
    }
    
    let len1 = str_len(s1);
    let len2 = str_len(s2);
    let total = len1 + len2;

    if total == 0 {
        return 0.0;
    }

    let lcs = lcs_length(s1, s2);
    let dist = (len1 + len2 - 2 * lcs) as f64 / total as f64;

    match score_cutoff {
        Some(cutoff) if dist > cutoff => 1.0,
        _ => dist,
    }
}

/// Calculate normalized Indel similarity (0.0 to 1.0).
/// This is what fuzz.ratio uses: 2 * LCS / (len1 + len2)
#[pyfunction]
#[pyo3(signature = (s1, s2, *, score_cutoff=None))]
pub fn indel_normalized_similarity(s1: &str, s2: &str, score_cutoff: Option<f64>) -> f64 {
    let norm_dist = indel_normalized_distance(s1, s2, None);
    let norm_sim = 1.0 - norm_dist;

    match score_cutoff {
        Some(cutoff) if norm_sim < cutoff => 0.0,
        _ => norm_sim,
    }
}

// ============ Batch Operations ============

#[pyfunction]
#[pyo3(signature = (pairs, *, score_cutoff=None))]
pub fn indel_distance_batch(
    pairs: Vec<(String, String)>,
    score_cutoff: Option<usize>,
) -> Vec<usize> {
    pairs
        .par_iter()
        .map(|(s1, s2)| indel_distance(s1, s2, score_cutoff))
        .collect()
}

#[pyfunction]
#[pyo3(signature = (pairs, *, score_cutoff=None))]
pub fn indel_normalized_similarity_batch(
    pairs: Vec<(String, String)>,
    score_cutoff: Option<f64>,
) -> Vec<f64> {
    pairs
        .par_iter()
        .map(|(s1, s2)| indel_normalized_similarity(s1, s2, score_cutoff))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lcs_length() {
        assert_eq!(lcs_length("abcde", "ace"), 3);
        assert_eq!(lcs_length("abc", "abc"), 3);
        assert_eq!(lcs_length("abc", "def"), 0);
    }

    #[test]
    fn test_indel_distance() {
        assert_eq!(indel_distance("abcde", "ace", None), 2);
    }
}
