//! Jaro similarity algorithm - Optimized
//!
//! Measures similarity between two strings based on matching characters
//! and transpositions.

use pyo3::prelude::*;
use rayon::prelude::*;
use smallvec::SmallVec;
use std::cmp::{max, min};

// Stack allocation for typical string lengths
type CharVec = SmallVec<[char; 64]>;
type BoolVec = SmallVec<[bool; 64]>;

/// Calculate Jaro similarity for ASCII strings (byte-level)
#[inline(always)]
fn jaro_similarity_ascii(s1: &[u8], s2: &[u8]) -> f64 {
    let len1 = s1.len();
    let len2 = s2.len();

    if len1 == 0 && len2 == 0 {
        return 1.0;
    }
    if len1 == 0 || len2 == 0 {
        return 0.0;
    }

    // Match window size
    let match_distance = max(len1, len2) / 2;
    let match_distance = if match_distance > 0 { match_distance - 1 } else { 0 };

    let mut s1_matches: BoolVec = SmallVec::from_elem(false, len1);
    let mut s2_matches: BoolVec = SmallVec::from_elem(false, len2);
    let mut matches = 0usize;
    let mut transpositions = 0usize;

    // Find matching characters
    for i in 0..len1 {
        let start = i.saturating_sub(match_distance);
        let end = min(i + match_distance + 1, len2);

        for j in start..end {
            if s2_matches[j] || s1[i] != s2[j] {
                continue;
            }
            s1_matches[i] = true;
            s2_matches[j] = true;
            matches += 1;
            break;
        }
    }

    if matches == 0 {
        return 0.0;
    }

    // Count transpositions
    let mut k = 0;
    for i in 0..len1 {
        if !s1_matches[i] {
            continue;
        }
        while !s2_matches[k] {
            k += 1;
        }
        if s1[i] != s2[k] {
            transpositions += 1;
        }
        k += 1;
    }

    let m = matches as f64;
    let t = (transpositions / 2) as f64;

    (m / len1 as f64 + m / len2 as f64 + (m - t) / m) / 3.0
}

/// Calculate Jaro similarity for Unicode strings
#[inline(always)]
fn jaro_similarity_chars(s1_chars: &[char], s2_chars: &[char]) -> f64 {
    let len1 = s1_chars.len();
    let len2 = s2_chars.len();

    if len1 == 0 && len2 == 0 {
        return 1.0;
    }
    if len1 == 0 || len2 == 0 {
        return 0.0;
    }

    let match_distance = max(len1, len2) / 2;
    let match_distance = if match_distance > 0 { match_distance - 1 } else { 0 };

    let mut s1_matches: BoolVec = SmallVec::from_elem(false, len1);
    let mut s2_matches: BoolVec = SmallVec::from_elem(false, len2);
    let mut matches = 0usize;
    let mut transpositions = 0usize;

    for i in 0..len1 {
        let start = i.saturating_sub(match_distance);
        let end = min(i + match_distance + 1, len2);

        for j in start..end {
            if s2_matches[j] || s1_chars[i] != s2_chars[j] {
                continue;
            }
            s1_matches[i] = true;
            s2_matches[j] = true;
            matches += 1;
            break;
        }
    }

    if matches == 0 {
        return 0.0;
    }

    let mut k = 0;
    for i in 0..len1 {
        if !s1_matches[i] {
            continue;
        }
        while !s2_matches[k] {
            k += 1;
        }
        if s1_chars[i] != s2_chars[k] {
            transpositions += 1;
        }
        k += 1;
    }

    let m = matches as f64;
    let t = (transpositions / 2) as f64;

    (m / len1 as f64 + m / len2 as f64 + (m - t) / m) / 3.0
}

/// Calculate Jaro similarity between two strings.
#[inline(always)]
fn jaro_similarity_internal(s1: &str, s2: &str) -> f64 {
    // Fast path for identical strings
    if s1 == s2 {
        return if s1.is_empty() && s2.is_empty() { 1.0 } else if s1.is_empty() { 0.0 } else { 1.0 };
    }

    // ASCII fast path
    if s1.is_ascii() && s2.is_ascii() {
        return jaro_similarity_ascii(s1.as_bytes(), s2.as_bytes());
    }

    // Unicode path
    let s1_chars: CharVec = s1.chars().collect();
    let s2_chars: CharVec = s2.chars().collect();
    jaro_similarity_chars(&s1_chars, &s2_chars)
}

/// Calculate Jaro similarity between two strings.
#[pyfunction]
#[pyo3(signature = (s1, s2, *, score_cutoff=None))]
pub fn jaro_similarity(s1: &str, s2: &str, score_cutoff: Option<f64>) -> f64 {
    let sim = jaro_similarity_internal(s1, s2);
    
    match score_cutoff {
        Some(cutoff) if sim < cutoff => 0.0,
        _ => sim,
    }
}

/// Calculate Jaro distance (1 - similarity).
#[pyfunction]
#[pyo3(signature = (s1, s2, *, score_cutoff=None))]
pub fn jaro_distance(s1: &str, s2: &str, score_cutoff: Option<f64>) -> f64 {
    let dist = 1.0 - jaro_similarity_internal(s1, s2);
    
    match score_cutoff {
        Some(cutoff) if dist > cutoff => 1.0,
        _ => dist,
    }
}

/// Alias for jaro_similarity (normalized)
#[pyfunction]
#[pyo3(signature = (s1, s2, *, score_cutoff=None))]
pub fn jaro_normalized_similarity(s1: &str, s2: &str, score_cutoff: Option<f64>) -> f64 {
    jaro_similarity(s1, s2, score_cutoff)
}

/// Alias for jaro_distance (normalized)
#[pyfunction]
#[pyo3(signature = (s1, s2, *, score_cutoff=None))]
pub fn jaro_normalized_distance(s1: &str, s2: &str, score_cutoff: Option<f64>) -> f64 {
    jaro_distance(s1, s2, score_cutoff)
}

// ============ Batch Operations ============

#[pyfunction]
#[pyo3(signature = (pairs, *, score_cutoff=None))]
pub fn jaro_similarity_batch(
    pairs: Vec<(String, String)>,
    score_cutoff: Option<f64>,
) -> Vec<f64> {
    pairs
        .par_iter()
        .map(|(s1, s2)| jaro_similarity(s1, s2, score_cutoff))
        .collect()
}

#[pyfunction]
#[pyo3(signature = (pairs, *, score_cutoff=None))]
pub fn jaro_distance_batch(
    pairs: Vec<(String, String)>,
    score_cutoff: Option<f64>,
) -> Vec<f64> {
    pairs
        .par_iter()
        .map(|(s1, s2)| jaro_distance(s1, s2, score_cutoff))
        .collect()
}

// Re-export the internal function for use by Jaro-Winkler
pub fn jaro_similarity_raw(s1: &str, s2: &str) -> f64 {
    jaro_similarity_internal(s1, s2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jaro_similarity() {
        let sim = jaro_similarity_internal("MARTHA", "MARHTA");
        assert!((sim - 0.944).abs() < 0.01);
    }

    #[test]
    fn test_jaro_identical() {
        assert!((jaro_similarity_internal("test", "test") - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_jaro_empty() {
        assert!((jaro_similarity_internal("", "") - 1.0).abs() < 0.001);
        assert!((jaro_similarity_internal("abc", "") - 0.0).abs() < 0.001);
    }
}
