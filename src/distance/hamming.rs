//! Hamming distance algorithm
//!
//! Calculates the number of positions at which corresponding characters differ.
//! Only works for strings of equal length.

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use rayon::prelude::*;

/// Calculate Hamming distance between two strings.
/// Requires strings to be of equal length.
#[inline(always)]
fn hamming_distance_internal(s1: &str, s2: &str) -> Result<usize, &'static str> {
    let s1_chars: Vec<char> = s1.chars().collect();
    let s2_chars: Vec<char> = s2.chars().collect();

    if s1_chars.len() != s2_chars.len() {
        return Err("Strings must have equal length for Hamming distance");
    }

    let distance = s1_chars
        .iter()
        .zip(s2_chars.iter())
        .filter(|(a, b)| a != b)
        .count();

    Ok(distance)
}

/// Calculate the Hamming distance between two strings.
#[pyfunction]
#[pyo3(signature = (s1, s2, *, score_cutoff=None))]
pub fn hamming_distance(s1: &str, s2: &str, score_cutoff: Option<usize>) -> PyResult<usize> {
    let dist = hamming_distance_internal(s1, s2)
        .map_err(|e| PyValueError::new_err(e))?;
    
    Ok(match score_cutoff {
        Some(cutoff) if dist > cutoff => cutoff + 1,
        _ => dist,
    })
}

/// Calculate the Hamming similarity between two strings.
#[pyfunction]
#[pyo3(signature = (s1, s2, *, score_cutoff=None))]
pub fn hamming_similarity(s1: &str, s2: &str, score_cutoff: Option<usize>) -> PyResult<usize> {
    let len = s1.chars().count();
    let dist = hamming_distance_internal(s1, s2)
        .map_err(|e| PyValueError::new_err(e))?;
    let sim = len.saturating_sub(dist);

    Ok(match score_cutoff {
        Some(cutoff) if sim < cutoff => 0,
        _ => sim,
    })
}

/// Calculate the normalized Hamming distance (0.0 to 1.0).
#[pyfunction]
#[pyo3(signature = (s1, s2, *, score_cutoff=None))]
pub fn hamming_normalized_distance(s1: &str, s2: &str, score_cutoff: Option<f64>) -> PyResult<f64> {
    let len = s1.chars().count();
    
    if len == 0 {
        return Ok(0.0);
    }

    let dist = hamming_distance_internal(s1, s2)
        .map_err(|e| PyValueError::new_err(e))?;
    let norm_dist = dist as f64 / len as f64;

    Ok(match score_cutoff {
        Some(cutoff) if norm_dist > cutoff => 1.0,
        _ => norm_dist,
    })
}

/// Calculate the normalized Hamming similarity (0.0 to 1.0).
#[pyfunction]
#[pyo3(signature = (s1, s2, *, score_cutoff=None))]
pub fn hamming_normalized_similarity(s1: &str, s2: &str, score_cutoff: Option<f64>) -> PyResult<f64> {
    let norm_dist = hamming_normalized_distance(s1, s2, None)?;
    let norm_sim = 1.0 - norm_dist;

    Ok(match score_cutoff {
        Some(cutoff) if norm_sim < cutoff => 0.0,
        _ => norm_sim,
    })
}

// ============ Batch Operations ============

/// Calculate Hamming distance for batch of string pairs.
#[pyfunction]
#[pyo3(signature = (pairs, *, score_cutoff=None))]
pub fn hamming_distance_batch(
    pairs: Vec<(String, String)>,
    score_cutoff: Option<usize>,
) -> PyResult<Vec<usize>> {
    pairs
        .par_iter()
        .map(|(s1, s2)| {
            hamming_distance_internal(s1, s2)
                .map(|d| match score_cutoff {
                    Some(cutoff) if d > cutoff => cutoff + 1,
                    _ => d,
                })
                .map_err(|e| PyValueError::new_err(e))
        })
        .collect::<Result<Vec<_>, _>>()
}

/// Calculate normalized Hamming similarity for batch of string pairs.
#[pyfunction]
#[pyo3(signature = (pairs, *, score_cutoff=None))]
pub fn hamming_normalized_similarity_batch(
    pairs: Vec<(String, String)>,
    score_cutoff: Option<f64>,
) -> PyResult<Vec<f64>> {
    pairs
        .par_iter()
        .map(|(s1, s2)| hamming_normalized_similarity(s1, s2, score_cutoff))
        .collect::<Result<Vec<_>, _>>()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hamming_distance() {
        assert_eq!(hamming_distance_internal("karolin", "kathrin"), Ok(3));
        assert_eq!(hamming_distance_internal("abc", "abc"), Ok(0));
        assert!(hamming_distance_internal("abc", "ab").is_err());
    }
}
