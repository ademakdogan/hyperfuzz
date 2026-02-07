//! Ratio algorithm - normalized Indel similarity percentage (Optimized)
//!
//! This is equivalent to RapidFuzz's fuzz.ratio function.
//! Uses LCS-based similarity: 2 * matches / (len1 + len2)

use pyo3::prelude::*;
use rayon::prelude::*;
use smallvec::SmallVec;
use std::cmp::max;

// Stack allocation for small strings
type CharVec = SmallVec<[char; 64]>;
type RowVec = SmallVec<[usize; 64]>;

/// Calculate the Longest Common Subsequence length for ASCII bytes
#[inline(always)]
fn lcs_length_ascii(s1: &[u8], s2: &[u8]) -> usize {
    let m = s1.len();
    let n = s2.len();

    if m == 0 || n == 0 {
        return 0;
    }

    // Ensure s1 is shorter for space efficiency
    let (s1, s2, m, n) = if m > n {
        (s2, s1, n, m)
    } else {
        (s1, s2, m, n)
    };

    let mut prev: RowVec = SmallVec::from_elem(0, m + 1);
    let mut curr: RowVec = SmallVec::from_elem(0, m + 1);

    for j in 1..=n {
        for i in 1..=m {
            if s1[i - 1] == s2[j - 1] {
                curr[i] = prev[i - 1] + 1;
            } else {
                curr[i] = max(prev[i], curr[i - 1]);
            }
        }
        std::mem::swap(&mut prev, &mut curr);
        curr.fill(0);
    }

    prev[m]
}

/// Calculate the Longest Common Subsequence length for Unicode chars
#[inline(always)]
fn lcs_length_chars(s1: &[char], s2: &[char]) -> usize {
    let m = s1.len();
    let n = s2.len();

    if m == 0 || n == 0 {
        return 0;
    }

    // Ensure s1 is shorter
    let (s1, s2, m, n) = if m > n {
        (s2, s1, n, m)
    } else {
        (s1, s2, m, n)
    };

    let mut prev: RowVec = SmallVec::from_elem(0, m + 1);
    let mut curr: RowVec = SmallVec::from_elem(0, m + 1);

    for j in 1..=n {
        for i in 1..=m {
            if s1[i - 1] == s2[j - 1] {
                curr[i] = prev[i - 1] + 1;
            } else {
                curr[i] = max(prev[i], curr[i - 1]);
            }
        }
        std::mem::swap(&mut prev, &mut curr);
        curr.fill(0);
    }

    prev[m]
}

/// Calculate the Longest Common Subsequence length between two strings.
#[inline(always)]
pub fn lcs_length(s1: &str, s2: &str) -> usize {
    // Fast path for ASCII strings
    if s1.is_ascii() && s2.is_ascii() {
        return lcs_length_ascii(s1.as_bytes(), s2.as_bytes());
    }
    
    // Unicode path
    let s1_chars: CharVec = s1.chars().collect();
    let s2_chars: CharVec = s2.chars().collect();
    lcs_length_chars(&s1_chars, &s2_chars)
}

/// Calculate the ratio (normalized Indel similarity) as a percentage (0-100).
///
/// This is equivalent to RapidFuzz's fuzz.ratio function.
/// Formula: 100 * 2 * LCS_length / (len1 + len2)
#[pyfunction]
#[pyo3(signature = (s1, s2, *, score_cutoff=None))]
pub fn ratio(s1: &str, s2: &str, score_cutoff: Option<f64>) -> f64 {
    // Fast path for identical strings
    if s1 == s2 {
        return if s1.is_empty() { 100.0 } else { 100.0 };
    }
    
    let (len1, len2) = if s1.is_ascii() && s2.is_ascii() {
        (s1.len(), s2.len())
    } else {
        (s1.chars().count(), s2.chars().count())
    };
    
    let total_len = len1 + len2;

    if total_len == 0 {
        return 100.0;
    }

    let lcs_len = lcs_length(s1, s2);
    let similarity = 100.0 * (2.0 * lcs_len as f64) / (total_len as f64);

    match score_cutoff {
        Some(cutoff) if similarity < cutoff => 0.0,
        _ => similarity,
    }
}

/// Calculate ratio for batch of string pairs.
#[pyfunction]
#[pyo3(signature = (pairs, *, score_cutoff=None))]
pub fn ratio_batch(pairs: Vec<(String, String)>, score_cutoff: Option<f64>) -> Vec<f64> {
    pairs
        .par_iter()
        .map(|(s1, s2)| ratio(s1, s2, score_cutoff))
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
    fn test_ratio() {
        // Test against known RapidFuzz values
        let r = ratio("this is a test", "this is a test!", None);
        assert!((r - 96.55).abs() < 0.1);
    }

    #[test]
    fn test_ratio_identical() {
        assert!((ratio("test", "test", None) - 100.0).abs() < 0.001);
    }

    #[test]
    fn test_ratio_empty() {
        assert!((ratio("", "", None) - 100.0).abs() < 0.001);
    }
}
