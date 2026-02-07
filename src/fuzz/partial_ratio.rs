//! Partial ratio algorithm - Optimized
//!
//! Finds the best partial match between two strings by sliding the shorter
//! string across the longer one.

use pyo3::prelude::*;
use rayon::prelude::*;
use smallvec::SmallVec;
use std::cmp::max;

type CharVec = SmallVec<[char; 64]>;
type RowVec = SmallVec<[usize; 64]>;

/// Calculate LCS length for byte slices
#[inline(always)]
fn lcs_length_bytes(s1: &[u8], s2: &[u8]) -> usize {
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

/// Calculate ratio for byte slices.
#[inline(always)]
fn ratio_bytes(s1: &[u8], s2: &[u8]) -> f64 {
    let total = s1.len() + s2.len();
    if total == 0 {
        return 100.0;
    }
    let lcs = lcs_length_bytes(s1, s2);
    100.0 * (2.0 * lcs as f64) / (total as f64)
}

/// Calculate LCS length for char slices.
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

/// Calculate ratio for char slices.
#[inline(always)]
fn ratio_chars(s1: &[char], s2: &[char]) -> f64 {
    let total = s1.len() + s2.len();
    if total == 0 {
        return 100.0;
    }
    let lcs = lcs_length_chars(s1, s2);
    100.0 * (2.0 * lcs as f64) / (total as f64)
}

/// Find the best partial match ratio for ASCII strings.
#[inline(always)]
fn partial_ratio_ascii(s1: &[u8], s2: &[u8]) -> f64 {
    let len1 = s1.len();
    let len2 = s2.len();

    if len1 == 0 && len2 == 0 {
        return 100.0;
    }
    if len1 == 0 || len2 == 0 {
        return 0.0;
    }

    let (shorter, longer) = if len1 <= len2 {
        (s1, s2)
    } else {
        (s2, s1)
    };

    let short_len = shorter.len();
    let long_len = longer.len();

    let mut best_score = 0.0f64;

    for i in 0..=(long_len - short_len) {
        let window = &longer[i..i + short_len];
        let score = ratio_bytes(shorter, window);
        if score > best_score {
            best_score = score;
        }
        if best_score >= 100.0 {
            break;
        }
    }

    best_score
}

/// Find the best partial match ratio for Unicode strings.
#[inline(always)]
fn partial_ratio_chars(s1: &[char], s2: &[char]) -> f64 {
    let len1 = s1.len();
    let len2 = s2.len();

    if len1 == 0 && len2 == 0 {
        return 100.0;
    }
    if len1 == 0 || len2 == 0 {
        return 0.0;
    }

    let (shorter, longer) = if len1 <= len2 {
        (s1, s2)
    } else {
        (s2, s1)
    };

    let short_len = shorter.len();
    let long_len = longer.len();

    let mut best_score = 0.0f64;

    for i in 0..=(long_len - short_len) {
        let window = &longer[i..i + short_len];
        let score = ratio_chars(shorter, window);
        if score > best_score {
            best_score = score;
        }
        if best_score >= 100.0 {
            break;
        }
    }

    best_score
}

/// Find the best partial match ratio.
#[inline(always)]
pub fn partial_ratio_internal(s1: &str, s2: &str) -> f64 {
    // Fast path for identical strings
    if s1 == s2 {
        return 100.0;
    }

    // ASCII fast path
    if s1.is_ascii() && s2.is_ascii() {
        return partial_ratio_ascii(s1.as_bytes(), s2.as_bytes());
    }

    // Unicode path
    let s1_chars: CharVec = s1.chars().collect();
    let s2_chars: CharVec = s2.chars().collect();
    partial_ratio_chars(&s1_chars, &s2_chars)
}

/// Calculate partial ratio - best partial match.
#[pyfunction]
#[pyo3(signature = (s1, s2, *, score_cutoff=None))]
pub fn partial_ratio(s1: &str, s2: &str, score_cutoff: Option<f64>) -> f64 {
    let result = partial_ratio_internal(s1, s2);
    
    match score_cutoff {
        Some(cutoff) if result < cutoff => 0.0,
        _ => result,
    }
}

/// Calculate partial ratio for batch of string pairs.
#[pyfunction]
#[pyo3(signature = (pairs, *, score_cutoff=None))]
pub fn partial_ratio_batch(pairs: Vec<(String, String)>, score_cutoff: Option<f64>) -> Vec<f64> {
    pairs
        .par_iter()
        .map(|(s1, s2)| partial_ratio(s1, s2, score_cutoff))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_partial_ratio() {
        let r = partial_ratio_internal("this is a test", "this is a test!");
        assert!((r - 100.0).abs() < 0.01);
    }
}
