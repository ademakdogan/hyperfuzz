//! Ratio algorithm - normalized Indel similarity percentage
//!
//! This is equivalent to RapidFuzz's fuzz.ratio function.
//! Uses LCS-based similarity: 2 * matches / (len1 + len2)

use pyo3::prelude::*;
use rayon::prelude::*;
use std::cmp::max;

/// Calculate the Longest Common Subsequence length between two strings.
#[inline(always)]
fn lcs_length(s1: &str, s2: &str) -> usize {
    let s1_chars: Vec<char> = s1.chars().collect();
    let s2_chars: Vec<char> = s2.chars().collect();

    let m = s1_chars.len();
    let n = s2_chars.len();

    if m == 0 || n == 0 {
        return 0;
    }

    // Use two rows for space optimization
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

/// Calculate the ratio (normalized Indel similarity) as a percentage (0-100).
///
/// This is equivalent to RapidFuzz's fuzz.ratio function.
/// Formula: 100 * 2 * LCS_length / (len1 + len2)
#[pyfunction]
#[pyo3(signature = (s1, s2, *, score_cutoff=None))]
pub fn ratio(s1: &str, s2: &str, score_cutoff: Option<f64>) -> f64 {
    let len1 = s1.chars().count();
    let len2 = s2.chars().count();
    let total_len = len1 + len2;

    if total_len == 0 {
        return 100.0;
    }

    // Fast path for identical strings
    if s1 == s2 {
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
