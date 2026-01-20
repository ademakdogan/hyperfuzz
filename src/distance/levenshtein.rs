//! Levenshtein distance algorithm
//!
//! Calculates the minimum number of single-character edits (insertions, deletions,
//! or substitutions) required to change one string into the other.

use pyo3::prelude::*;
use rayon::prelude::*;
use std::cmp::min;

/// Calculate the Levenshtein distance between two strings.
///
/// Uses Wagner-Fischer algorithm with O(min(m,n)) space complexity.
#[inline(always)]
fn levenshtein_distance_internal(s1: &str, s2: &str) -> usize {
    let s1_chars: Vec<char> = s1.chars().collect();
    let s2_chars: Vec<char> = s2.chars().collect();

    let m = s1_chars.len();
    let n = s2_chars.len();

    // Early exits
    if m == 0 {
        return n;
    }
    if n == 0 {
        return m;
    }
    if s1 == s2 {
        return 0;
    }

    // Ensure s1 is the shorter string for space optimization
    let (s1_chars, s2_chars, m, n) = if m > n {
        (s2_chars, s1_chars, n, m)
    } else {
        (s1_chars, s2_chars, m, n)
    };

    // Use two rows instead of full matrix
    let mut prev_row: Vec<usize> = (0..=m).collect();
    let mut curr_row: Vec<usize> = vec![0; m + 1];

    for j in 1..=n {
        curr_row[0] = j;

        for i in 1..=m {
            let cost = if s1_chars[i - 1] == s2_chars[j - 1] {
                0
            } else {
                1
            };

            curr_row[i] = min(
                min(
                    prev_row[i] + 1,     // deletion
                    curr_row[i - 1] + 1, // insertion
                ),
                prev_row[i - 1] + cost, // substitution
            );
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
    let max_len = s1.chars().count().max(s2.chars().count());
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
    let m = s1.chars().count();
    let n = s2.chars().count();
    let max_len = m.max(n);

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
    fn test_normalized_similarity() {
        let sim = 1.0 - (levenshtein_distance_internal("abc", "abc") as f64 / 3.0);
        assert!((sim - 1.0).abs() < 0.001);
    }
}
