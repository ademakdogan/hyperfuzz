//! Damerau-Levenshtein distance algorithm
//!
//! Extension of Levenshtein that also allows transpositions
//! (swapping two adjacent characters).

use pyo3::prelude::*;
use rayon::prelude::*;
use std::cmp::min;
use std::collections::HashMap;

/// Calculate true Damerau-Levenshtein distance.
/// Allows insertions, deletions, substitutions, and transpositions.
#[inline(always)]
fn damerau_levenshtein_internal(s1: &str, s2: &str) -> usize {
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

    let max_dist = len1 + len2;

    // Create matrix with extra row/col for boundary conditions
    let mut d: Vec<Vec<usize>> = vec![vec![0; len2 + 2]; len1 + 2];
    
    d[0][0] = max_dist;
    for i in 0..=len1 {
        d[i + 1][0] = max_dist;
        d[i + 1][1] = i;
    }
    for j in 0..=len2 {
        d[0][j + 1] = max_dist;
        d[1][j + 1] = j;
    }

    // Last occurrence of each character
    let mut da: HashMap<char, usize> = HashMap::new();

    for i in 1..=len1 {
        let mut db = 0usize;
        
        for j in 1..=len2 {
            let k = *da.get(&s2_chars[j - 1]).unwrap_or(&0);
            let l = db;

            let cost = if s1_chars[i - 1] == s2_chars[j - 1] {
                db = j;
                0
            } else {
                1
            };

            d[i + 1][j + 1] = min(
                min(
                    d[i][j + 1] + 1,     // deletion
                    d[i + 1][j] + 1,     // insertion
                ),
                min(
                    d[i][j] + cost,      // substitution
                    d[k][l] + (i - k - 1) + 1 + (j - l - 1), // transposition
                ),
            );
        }

        da.insert(s1_chars[i - 1], i);
    }

    d[len1 + 1][len2 + 1]
}

/// Calculate Damerau-Levenshtein distance.
#[pyfunction]
#[pyo3(signature = (s1, s2, *, score_cutoff=None))]
pub fn damerau_levenshtein_distance(s1: &str, s2: &str, score_cutoff: Option<usize>) -> usize {
    let dist = damerau_levenshtein_internal(s1, s2);
    
    match score_cutoff {
        Some(cutoff) if dist > cutoff => cutoff + 1,
        _ => dist,
    }
}

/// Calculate Damerau-Levenshtein similarity.
#[pyfunction]
#[pyo3(signature = (s1, s2, *, score_cutoff=None))]
pub fn damerau_levenshtein_similarity(s1: &str, s2: &str, score_cutoff: Option<usize>) -> usize {
    let max_len = s1.chars().count().max(s2.chars().count());
    let dist = damerau_levenshtein_internal(s1, s2);
    let sim = max_len.saturating_sub(dist);

    match score_cutoff {
        Some(cutoff) if sim < cutoff => 0,
        _ => sim,
    }
}

/// Calculate normalized Damerau-Levenshtein distance.
#[pyfunction]
#[pyo3(signature = (s1, s2, *, score_cutoff=None))]
pub fn damerau_levenshtein_normalized_distance(
    s1: &str,
    s2: &str,
    score_cutoff: Option<f64>,
) -> f64 {
    let m = s1.chars().count();
    let n = s2.chars().count();
    let max_len = m.max(n);

    if max_len == 0 {
        return 0.0;
    }

    let dist = damerau_levenshtein_internal(s1, s2);
    let norm_dist = dist as f64 / max_len as f64;

    match score_cutoff {
        Some(cutoff) if norm_dist > cutoff => 1.0,
        _ => norm_dist,
    }
}

/// Calculate normalized Damerau-Levenshtein similarity.
#[pyfunction]
#[pyo3(signature = (s1, s2, *, score_cutoff=None))]
pub fn damerau_levenshtein_normalized_similarity(
    s1: &str,
    s2: &str,
    score_cutoff: Option<f64>,
) -> f64 {
    let norm_dist = damerau_levenshtein_normalized_distance(s1, s2, None);
    let norm_sim = 1.0 - norm_dist;

    match score_cutoff {
        Some(cutoff) if norm_sim < cutoff => 0.0,
        _ => norm_sim,
    }
}

// ============ Batch Operations ============

#[pyfunction]
#[pyo3(signature = (pairs, *, score_cutoff=None))]
pub fn damerau_levenshtein_distance_batch(
    pairs: Vec<(String, String)>,
    score_cutoff: Option<usize>,
) -> Vec<usize> {
    pairs
        .par_iter()
        .map(|(s1, s2)| damerau_levenshtein_distance(s1, s2, score_cutoff))
        .collect()
}

#[pyfunction]
#[pyo3(signature = (pairs, *, score_cutoff=None))]
pub fn damerau_levenshtein_normalized_similarity_batch(
    pairs: Vec<(String, String)>,
    score_cutoff: Option<f64>,
) -> Vec<f64> {
    pairs
        .par_iter()
        .map(|(s1, s2)| damerau_levenshtein_normalized_similarity(s1, s2, score_cutoff))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_damerau_levenshtein() {
        // "ca" -> "abc": needs 2 ops (not 3 like Levenshtein)
        assert_eq!(damerau_levenshtein_internal("ca", "abc"), 2);
        assert_eq!(damerau_levenshtein_internal("abc", "abc"), 0);
    }

    #[test]
    fn test_transposition() {
        // "ab" -> "ba": 1 transposition
        assert_eq!(damerau_levenshtein_internal("ab", "ba"), 1);
    }
}
