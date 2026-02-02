//! OSA - Optimal String Alignment distance
//!
//! Similar to Damerau-Levenshtein but with restriction that 
//! each substring can only be edited once.

use pyo3::prelude::*;
use rayon::prelude::*;
use std::cmp::min;

/// Calculate OSA distance.
#[inline(always)]
fn osa_internal(s1: &str, s2: &str) -> usize {
    let s1_chars: Vec<char> = s1.chars().collect();
    let s2_chars: Vec<char> = s2.chars().collect();

    let len1 = s1_chars.len();
    let len2 = s2_chars.len();

    if len1 == 0 {
        return len2;
    }
    if len2 == 0 {
        return len1;
    }
    if s1 == s2 {
        return 0;
    }

    // Need 3 rows for transposition check
    let mut d0: Vec<usize> = (0..=len2).collect();
    let mut d1: Vec<usize> = vec![0; len2 + 1];
    let mut d2: Vec<usize> = vec![0; len2 + 1];

    for i in 1..=len1 {
        d1[0] = i;

        for j in 1..=len2 {
            let cost = if s1_chars[i - 1] == s2_chars[j - 1] {
                0
            } else {
                1
            };

            d1[j] = min(
                min(
                    d0[j] + 1,     // deletion
                    d1[j - 1] + 1, // insertion
                ),
                d0[j - 1] + cost, // substitution
            );

            // Check for transposition
            if i > 1
                && j > 1
                && s1_chars[i - 1] == s2_chars[j - 2]
                && s1_chars[i - 2] == s2_chars[j - 1]
            {
                d1[j] = min(d1[j], d2[j - 2] + cost);
            }
        }

        std::mem::swap(&mut d2, &mut d0);
        std::mem::swap(&mut d0, &mut d1);
        d1.fill(0);
    }

    d0[len2]
}

/// Calculate OSA distance.
#[pyfunction]
#[pyo3(signature = (s1, s2, *, score_cutoff=None))]
pub fn osa_distance(s1: &str, s2: &str, score_cutoff: Option<usize>) -> usize {
    let dist = osa_internal(s1, s2);
    
    match score_cutoff {
        Some(cutoff) if dist > cutoff => cutoff + 1,
        _ => dist,
    }
}

/// Calculate OSA similarity.
#[pyfunction]
#[pyo3(signature = (s1, s2, *, score_cutoff=None))]
pub fn osa_similarity(s1: &str, s2: &str, score_cutoff: Option<usize>) -> usize {
    let max_len = s1.chars().count().max(s2.chars().count());
    let dist = osa_internal(s1, s2);
    let sim = max_len.saturating_sub(dist);

    match score_cutoff {
        Some(cutoff) if sim < cutoff => 0,
        _ => sim,
    }
}

/// Calculate normalized OSA distance.
#[pyfunction]
#[pyo3(signature = (s1, s2, *, score_cutoff=None))]
pub fn osa_normalized_distance(s1: &str, s2: &str, score_cutoff: Option<f64>) -> f64 {
    let m = s1.chars().count();
    let n = s2.chars().count();
    let max_len = m.max(n);

    if max_len == 0 {
        return 0.0;
    }

    let dist = osa_internal(s1, s2);
    let norm_dist = dist as f64 / max_len as f64;

    match score_cutoff {
        Some(cutoff) if norm_dist > cutoff => 1.0,
        _ => norm_dist,
    }
}

/// Calculate normalized OSA similarity.
#[pyfunction]
#[pyo3(signature = (s1, s2, *, score_cutoff=None))]
pub fn osa_normalized_similarity(s1: &str, s2: &str, score_cutoff: Option<f64>) -> f64 {
    let norm_dist = osa_normalized_distance(s1, s2, None);
    let norm_sim = 1.0 - norm_dist;

    match score_cutoff {
        Some(cutoff) if norm_sim < cutoff => 0.0,
        _ => norm_sim,
    }
}

// ============ Batch Operations ============

#[pyfunction]
#[pyo3(signature = (pairs, *, score_cutoff=None))]
pub fn osa_distance_batch(
    pairs: Vec<(String, String)>,
    score_cutoff: Option<usize>,
) -> Vec<usize> {
    pairs
        .par_iter()
        .map(|(s1, s2)| osa_distance(s1, s2, score_cutoff))
        .collect()
}

#[pyfunction]
#[pyo3(signature = (pairs, *, score_cutoff=None))]
pub fn osa_normalized_similarity_batch(
    pairs: Vec<(String, String)>,
    score_cutoff: Option<f64>,
) -> Vec<f64> {
    pairs
        .par_iter()
        .map(|(s1, s2)| osa_normalized_similarity(s1, s2, score_cutoff))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_osa() {
        assert_eq!(osa_internal("ab", "ba"), 1); // transposition
        assert_eq!(osa_internal("abc", "abc"), 0);
    }
}
