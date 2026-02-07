//! Ratio algorithm - Ultra Optimized
//!
//! Uses Indel similarity (LCS-based): 100 * 2 * LCS / (len1 + len2)

use pyo3::prelude::*;
use rayon::prelude::*;
use smallvec::SmallVec;
use std::cmp::max;

type RowVec = SmallVec<[usize; 128]>;

/// Ultra-fast LCS for ASCII bytes with cache-friendly access
#[inline(always)]
fn lcs_ascii(s1: &[u8], s2: &[u8]) -> usize {
    let m = s1.len();
    let n = s2.len();
    
    if m == 0 || n == 0 { return 0; }
    
    // Make s1 shorter for better cache behavior
    let (s1, s2, m, n) = if m > n {
        (s2, s1, n, m)
    } else {
        (s1, s2, m, n)
    };

    // Optimized single-row DP with manual loop unrolling potential
    let mut prev: RowVec = SmallVec::from_elem(0, m + 1);
    let mut curr: RowVec = SmallVec::from_elem(0, m + 1);

    for c2 in s2.iter() {
        // Process in chunks for better cache utilization
        for i in 0..m {
            curr[i + 1] = if s1[i] == *c2 {
                prev[i] + 1
            } else {
                max(prev[i + 1], curr[i])
            };
        }
        std::mem::swap(&mut prev, &mut curr);
        // Only fill if there's another iteration
        if n > 1 {
            curr.fill(0);
        }
    }

    prev[m]
}

/// LCS for Unicode chars
#[inline(always)]
fn lcs_unicode(s1: &[char], s2: &[char]) -> usize {
    let m = s1.len();
    let n = s2.len();
    
    if m == 0 || n == 0 { return 0; }
    
    let (s1, s2, m, _n) = if m > n {
        (s2, s1, n, m)
    } else {
        (s1, s2, m, n)
    };

    let mut prev: RowVec = SmallVec::from_elem(0, m + 1);
    let mut curr: RowVec = SmallVec::from_elem(0, m + 1);

    for c2 in s2.iter() {
        for i in 0..m {
            curr[i + 1] = if s1[i] == *c2 {
                prev[i] + 1
            } else {
                max(prev[i + 1], curr[i])
            };
        }
        std::mem::swap(&mut prev, &mut curr);
        curr.fill(0);
    }

    prev[m]
}

/// Calculate LCS length
#[inline(always)]
pub fn lcs_length(s1: &str, s2: &str) -> usize {
    if s1.is_ascii() && s2.is_ascii() {
        return lcs_ascii(s1.as_bytes(), s2.as_bytes());
    }
    
    let c1: SmallVec<[char; 64]> = s1.chars().collect();
    let c2: SmallVec<[char; 64]> = s2.chars().collect();
    lcs_unicode(&c1, &c2)
}

/// Calculate ratio - Ultra optimized
#[pyfunction]
#[pyo3(signature = (s1, s2, *, score_cutoff=None))]
pub fn ratio(s1: &str, s2: &str, score_cutoff: Option<f64>) -> f64 {
    // Fast path for identical strings
    if s1 == s2 {
        return 100.0;
    }
    
    // Fast path for empty strings
    if s1.is_empty() && s2.is_empty() {
        return 100.0;
    }
    if s1.is_empty() || s2.is_empty() {
        return 0.0;
    }
    
    // Get lengths - ASCII fast path
    let (len1, len2) = if s1.is_ascii() && s2.is_ascii() {
        (s1.len(), s2.len())
    } else {
        (s1.chars().count(), s2.chars().count())
    };
    
    let total = len1 + len2;
    
    // Quick rejection based on length ratio
    if let Some(cutoff) = score_cutoff {
        // Maximum possible score
        let max_possible = 100.0 * (2.0 * len1.min(len2) as f64) / (total as f64);
        if max_possible < cutoff {
            return 0.0;
        }
    }

    let lcs = lcs_length(s1, s2);
    let similarity = 100.0 * (2.0 * lcs as f64) / (total as f64);

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
