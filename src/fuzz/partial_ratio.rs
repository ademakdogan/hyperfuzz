//! Partial ratio algorithm
//!
//! Finds the best partial match between two strings by sliding the shorter
//! string across the longer one.

use pyo3::prelude::*;
use rayon::prelude::*;
use std::cmp::max;

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

/// Find the best partial match ratio.
/// Slides the shorter string across the longer one to find the best match.
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

    // Ensure s1 is the shorter string
    let (shorter, longer) = if len1 <= len2 {
        (&s1_chars, &s2_chars)
    } else {
        (&s2_chars, &s1_chars)
    };

    let short_len = shorter.len();
    let long_len = longer.len();

    let mut best_score = 0.0f64;

    // Slide shorter string across longer string
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
        // "this is a test" is fully contained in "this is a test!"
        let r = partial_ratio_internal("this is a test", "this is a test!");
        assert!((r - 100.0).abs() < 0.01);
    }
}
