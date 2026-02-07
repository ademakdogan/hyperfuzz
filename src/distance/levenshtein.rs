//! Levenshtein distance algorithm - Optimized
//!
//! Calculates the minimum number of single-character edits (insertions, deletions,
//! or substitutions) required to change one string into the other.

use pyo3::prelude::*;
use rayon::prelude::*;
use smallvec::SmallVec;

// Stack-allocate arrays up to 64 elements to avoid heap allocation for short strings
type CharVec = SmallVec<[char; 64]>;
type RowVec = SmallVec<[usize; 64]>;

/// Calculate the Levenshtein distance between two strings.
///
/// Optimized Wagner-Fischer algorithm with:
/// - SmallVec for stack allocation of small strings
/// - Early termination checks
/// - Reduced branching in hot loop
/// - Common prefix/suffix elimination
#[inline(always)]
fn levenshtein_distance_internal(s1: &str, s2: &str) -> usize {
    // Fast path for identical strings
    if s1 == s2 {
        return 0;
    }
    
    let s1_bytes = s1.as_bytes();
    let s2_bytes = s2.as_bytes();
    
    // For ASCII strings, work with bytes directly (much faster)
    if s1.is_ascii() && s2.is_ascii() {
        return levenshtein_ascii(s1_bytes, s2_bytes);
    }
    
    // Unicode path
    let s1_chars: CharVec = s1.chars().collect();
    let s2_chars: CharVec = s2.chars().collect();

    levenshtein_chars(&s1_chars, &s2_chars)
}

/// Levenshtein for ASCII strings (byte-level, faster)
#[inline(always)]
fn levenshtein_ascii(s1: &[u8], s2: &[u8]) -> usize {
    let mut m = s1.len();
    let mut n = s2.len();

    // Early exits
    if m == 0 { return n; }
    if n == 0 { return m; }

    // Make s1 the shorter string
    let (s1, s2) = if m > n {
        std::mem::swap(&mut m, &mut n);
        (s2, s1)
    } else {
        (s1, s2)
    };

    // Skip common prefix
    let prefix_len = s1.iter().zip(s2.iter()).take_while(|(a, b)| a == b).count();
    if prefix_len == m {
        return n - m;
    }
    
    let s1 = &s1[prefix_len..];
    let s2 = &s2[prefix_len..];
    let m = s1.len();
    let n = s2.len();

    // Skip common suffix
    let suffix_len = s1.iter().rev().zip(s2.iter().rev()).take_while(|(a, b)| a == b).count();
    let s1 = &s1[..m - suffix_len];
    let s2 = &s2[..n - suffix_len];
    let m = s1.len();
    let n = s2.len();

    if m == 0 { return n; }
    if n == 0 { return m; }

    // Use SmallVec for stack allocation
    let mut prev_row: RowVec = (0..=m).collect();
    let mut curr_row: RowVec = SmallVec::from_elem(0, m + 1);

    for (j, c2) in s2.iter().enumerate() {
        curr_row[0] = j + 1;

        for (i, c1) in s1.iter().enumerate() {
            let sub_cost = prev_row[i] + (*c1 != *c2) as usize;
            let del_cost = prev_row[i + 1] + 1;
            let ins_cost = curr_row[i] + 1;

            curr_row[i + 1] = sub_cost.min(del_cost).min(ins_cost);
        }

        std::mem::swap(&mut prev_row, &mut curr_row);
    }

    prev_row[m]
}

/// Levenshtein for Unicode chars
#[inline(always)]
fn levenshtein_chars(s1: &[char], s2: &[char]) -> usize {
    let mut m = s1.len();
    let mut n = s2.len();

    // Early exits
    if m == 0 { return n; }
    if n == 0 { return m; }

    // Make s1 the shorter string
    let (s1, s2) = if m > n {
        std::mem::swap(&mut m, &mut n);
        (s2, s1)
    } else {
        (s1, s2)
    };

    // Skip common prefix
    let prefix_len = s1.iter().zip(s2.iter()).take_while(|(a, b)| a == b).count();
    if prefix_len == m {
        return n - m;
    }
    
    let s1 = &s1[prefix_len..];
    let s2 = &s2[prefix_len..];
    let m = s1.len();
    let n = s2.len();

    // Skip common suffix
    let suffix_len = s1.iter().rev().zip(s2.iter().rev()).take_while(|(a, b)| a == b).count();
    let s1 = &s1[..m - suffix_len];
    let s2 = &s2[..n - suffix_len];
    let m = s1.len();
    let n = s2.len();

    if m == 0 { return n; }
    if n == 0 { return m; }

    let mut prev_row: RowVec = (0..=m).collect();
    let mut curr_row: RowVec = SmallVec::from_elem(0, m + 1);

    for (j, c2) in s2.iter().enumerate() {
        curr_row[0] = j + 1;

        for (i, c1) in s1.iter().enumerate() {
            let sub_cost = prev_row[i] + (*c1 != *c2) as usize;
            let del_cost = prev_row[i + 1] + 1;
            let ins_cost = curr_row[i] + 1;

            curr_row[i + 1] = sub_cost.min(del_cost).min(ins_cost);
        }

        std::mem::swap(&mut prev_row, &mut curr_row);
    }

    prev_row[m]
}

/// Calculate the Levenshtein distance between two strings.
#[pyfunction]
#[pyo3(signature = (s1, s2, *, score_cutoff=None))]
pub fn levenshtein_distance(s1: &str, s2: &str, score_cutoff: Option<usize>) -> usize {
    let dist = levenshtein_distance_internal(s1, s2);
    match score_cutoff {
        Some(cutoff) if dist > cutoff => cutoff + 1,
        _ => dist,
    }
}

/// Calculate the Levenshtein similarity between two strings.
/// similarity = max(len(s1), len(s2)) - distance
#[pyfunction]
#[pyo3(signature = (s1, s2, *, score_cutoff=None))]
pub fn levenshtein_similarity(s1: &str, s2: &str, score_cutoff: Option<usize>) -> usize {
    let len1 = if s1.is_ascii() { s1.len() } else { s1.chars().count() };
    let len2 = if s2.is_ascii() { s2.len() } else { s2.chars().count() };
    let max_len = len1.max(len2);
    let dist = levenshtein_distance_internal(s1, s2);
    let sim = max_len.saturating_sub(dist);

    match score_cutoff {
        Some(cutoff) if sim < cutoff => 0,
        _ => sim,
    }
}

/// Calculate the normalized Levenshtein distance (0.0 to 1.0).
#[pyfunction]
#[pyo3(signature = (s1, s2, *, score_cutoff=None))]
pub fn levenshtein_normalized_distance(s1: &str, s2: &str, score_cutoff: Option<f64>) -> f64 {
    let len1 = if s1.is_ascii() { s1.len() } else { s1.chars().count() };
    let len2 = if s2.is_ascii() { s2.len() } else { s2.chars().count() };
    let max_len = len1.max(len2);

    if max_len == 0 {
        return 0.0;
    }

    let dist = levenshtein_distance_internal(s1, s2);
    let norm_dist = dist as f64 / max_len as f64;

    match score_cutoff {
        Some(cutoff) if norm_dist > cutoff => 1.0,
        _ => norm_dist,
    }
}

/// Calculate the normalized Levenshtein similarity (0.0 to 1.0).
#[pyfunction]
#[pyo3(signature = (s1, s2, *, score_cutoff=None))]
pub fn levenshtein_normalized_similarity(s1: &str, s2: &str, score_cutoff: Option<f64>) -> f64 {
    let norm_dist = levenshtein_normalized_distance(s1, s2, None);
    let norm_sim = 1.0 - norm_dist;

    match score_cutoff {
        Some(cutoff) if norm_sim < cutoff => 0.0,
        _ => norm_sim,
    }
}

// ============ Batch Operations ============

/// Calculate Levenshtein distance for batch of string pairs.
#[pyfunction]
#[pyo3(signature = (pairs, *, score_cutoff=None))]
pub fn levenshtein_distance_batch(
    pairs: Vec<(String, String)>,
    score_cutoff: Option<usize>,
) -> Vec<usize> {
    pairs
        .par_iter()
        .map(|(s1, s2)| levenshtein_distance(s1, s2, score_cutoff))
        .collect()
}

/// Calculate Levenshtein similarity for batch of string pairs.
#[pyfunction]
#[pyo3(signature = (pairs, *, score_cutoff=None))]
pub fn levenshtein_similarity_batch(
    pairs: Vec<(String, String)>,
    score_cutoff: Option<usize>,
) -> Vec<usize> {
    pairs
        .par_iter()
        .map(|(s1, s2)| levenshtein_similarity(s1, s2, score_cutoff))
        .collect()
}

/// Calculate normalized Levenshtein distance for batch of string pairs.
#[pyfunction]
#[pyo3(signature = (pairs, *, score_cutoff=None))]
pub fn levenshtein_normalized_distance_batch(
    pairs: Vec<(String, String)>,
    score_cutoff: Option<f64>,
) -> Vec<f64> {
    pairs
        .par_iter()
        .map(|(s1, s2)| levenshtein_normalized_distance(s1, s2, score_cutoff))
        .collect()
}

/// Calculate normalized Levenshtein similarity for batch of string pairs.
#[pyfunction]
#[pyo3(signature = (pairs, *, score_cutoff=None))]
pub fn levenshtein_normalized_similarity_batch(
    pairs: Vec<(String, String)>,
    score_cutoff: Option<f64>,
) -> Vec<f64> {
    pairs
        .par_iter()
        .map(|(s1, s2)| levenshtein_normalized_similarity(s1, s2, score_cutoff))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_levenshtein_distance() {
        assert_eq!(levenshtein_distance_internal("kitten", "sitting"), 3);
        assert_eq!(levenshtein_distance_internal("", "abc"), 3);
        assert_eq!(levenshtein_distance_internal("abc", ""), 3);
        assert_eq!(levenshtein_distance_internal("abc", "abc"), 0);
        assert_eq!(levenshtein_distance_internal("abc", "abd"), 1);
    }

    #[test]
    fn test_ascii_optimization() {
        // ASCII path
        assert_eq!(levenshtein_distance_internal("hello", "hallo"), 1);
        // With common prefix
        assert_eq!(levenshtein_distance_internal("prefix_abc", "prefix_abd"), 1);
        // With common suffix
        assert_eq!(levenshtein_distance_internal("abc_suffix", "abd_suffix"), 1);
    }

    #[test]
    fn test_normalized_similarity() {
        let sim = 1.0 - (levenshtein_distance_internal("abc", "abc") as f64 / 3.0);
        assert!((sim - 1.0).abs() < 0.001);
    }
}
