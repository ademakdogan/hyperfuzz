//! Hamming distance algorithm - Optimized
//!
//! Calculates the number of positions at which corresponding characters differ.
//! Only works for strings of equal length.

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use rayon::prelude::*;
use smallvec::SmallVec;

type CharVec = SmallVec<[char; 64]>;

/// Calculate Hamming distance for ASCII strings (byte-level, faster)
#[inline(always)]
fn hamming_ascii(s1: &[u8], s2: &[u8]) -> usize {
    s1.iter().zip(s2.iter()).filter(|(a, b)| a != b).count()
}

/// Calculate Hamming distance for Unicode strings
#[inline(always)]
fn hamming_chars(s1: &[char], s2: &[char]) -> usize {
    s1.iter().zip(s2.iter()).filter(|(a, b)| a != b).count()
}

/// Calculate Hamming distance internal
#[inline(always)]
fn hamming_distance_internal(s1: &str, s2: &str) -> Result<usize, &'static str> {
    // Fast path for identical strings
    if s1 == s2 {
        return Ok(0);
    }
    
    // ASCII fast path
    if s1.is_ascii() && s2.is_ascii() {
        let b1 = s1.as_bytes();
        let b2 = s2.as_bytes();
        
        if b1.len() != b2.len() {
            return Err("Strings must have equal length for Hamming distance");
        }
        
        return Ok(hamming_ascii(b1, b2));
    }
    
    // Unicode path
    let s1_chars: CharVec = s1.chars().collect();
    let s2_chars: CharVec = s2.chars().collect();
    
    if s1_chars.len() != s2_chars.len() {
        return Err("Strings must have equal length for Hamming distance");
    }
    
    Ok(hamming_chars(&s1_chars, &s2_chars))
}

/// Calculate Hamming distance.
#[pyfunction]
#[pyo3(signature = (s1, s2, *, score_cutoff=None))]
pub fn hamming_distance(s1: &str, s2: &str, score_cutoff: Option<usize>) -> PyResult<usize> {
    let dist = hamming_distance_internal(s1, s2)
        .map_err(|e| PyValueError::new_err(e))?;
    
    match score_cutoff {
        Some(cutoff) if dist > cutoff => Ok(cutoff + 1),
        _ => Ok(dist),
    }
}

/// Calculate Hamming similarity.
#[pyfunction]
#[pyo3(signature = (s1, s2, *, score_cutoff=None))]
pub fn hamming_similarity(s1: &str, s2: &str, score_cutoff: Option<usize>) -> PyResult<usize> {
    let len = if s1.is_ascii() { s1.len() } else { s1.chars().count() };
    let dist = hamming_distance_internal(s1, s2)
        .map_err(|e| PyValueError::new_err(e))?;
    let sim = len.saturating_sub(dist);

    match score_cutoff {
        Some(cutoff) if sim < cutoff => Ok(0),
        _ => Ok(sim),
    }
}

/// Calculate normalized Hamming distance.
#[pyfunction]
#[pyo3(signature = (s1, s2, *, score_cutoff=None))]
pub fn hamming_normalized_distance(s1: &str, s2: &str, score_cutoff: Option<f64>) -> PyResult<f64> {
    let len = if s1.is_ascii() { s1.len() } else { s1.chars().count() };
    
    if len == 0 {
        return Ok(0.0);
    }

    let dist = hamming_distance_internal(s1, s2)
        .map_err(|e| PyValueError::new_err(e))?;
    let norm_dist = dist as f64 / len as f64;

    match score_cutoff {
        Some(cutoff) if norm_dist > cutoff => Ok(1.0),
        _ => Ok(norm_dist),
    }
}

/// Calculate normalized Hamming similarity.
#[pyfunction]
#[pyo3(signature = (s1, s2, *, score_cutoff=None))]
pub fn hamming_normalized_similarity(s1: &str, s2: &str, score_cutoff: Option<f64>) -> PyResult<f64> {
    let norm_dist = hamming_normalized_distance(s1, s2, None)?;
    let norm_sim = 1.0 - norm_dist;

    match score_cutoff {
        Some(cutoff) if norm_sim < cutoff => Ok(0.0),
        _ => Ok(norm_sim),
    }
}

// ============ Batch Operations ============

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
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| PyValueError::new_err(e))
}

#[pyfunction]
#[pyo3(signature = (pairs, *, score_cutoff=None))]
pub fn hamming_normalized_similarity_batch(
    pairs: Vec<(String, String)>,
    score_cutoff: Option<f64>,
) -> PyResult<Vec<f64>> {
    pairs
        .par_iter()
        .map(|(s1, s2)| {
            let len = if s1.is_ascii() { s1.len() } else { s1.chars().count() };
            if len == 0 {
                return Ok(1.0);
            }
            hamming_distance_internal(s1, s2)
                .map(|d| {
                    let norm_sim = 1.0 - (d as f64 / len as f64);
                    match score_cutoff {
                        Some(cutoff) if norm_sim < cutoff => 0.0,
                        _ => norm_sim,
                    }
                })
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| PyValueError::new_err(e))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hamming_distance() {
        assert_eq!(hamming_distance_internal("karolin", "kathrin").unwrap(), 3);
        assert_eq!(hamming_distance_internal("abc", "abc").unwrap(), 0);
    }
}
