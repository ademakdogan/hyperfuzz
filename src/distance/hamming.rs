//! Hamming distance algorithm - Ultra optimized with SIMD XOR + popcount
//!
//! Hamming distance counts the number of positions where corresponding
//! characters differ. Uses SIMD for byte-level XOR operations.

use pyo3::prelude::*;
use rayon::prelude::*;
use smallvec::SmallVec;

/// SIMD-accelerated Hamming distance for ASCII using XOR + popcount
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn hamming_simd_ascii(s1: &[u8], s2: &[u8]) -> usize {
    use std::arch::aarch64::*;
    
    let len = s1.len();
    let mut diff_count = 0usize;
    let mut i = 0;
    
    // Process 16 bytes at a time with NEON
    if len >= 16 {
        unsafe {
            while i + 16 <= len {
                let v1 = vld1q_u8(s1.as_ptr().add(i));
                let v2 = vld1q_u8(s2.as_ptr().add(i));
                
                // XOR to find differences
                let diff = veorq_u8(v1, v2);
                
                // Check which bytes are non-zero (different)
                let ne = vcgtq_u8(diff, vdupq_n_u8(0));
                
                // Count non-zero bytes
                let count = vaddvq_u8(vandq_u8(ne, vdupq_n_u8(1)));
                diff_count += count as usize;
                
                i += 16;
            }
        }
    }
    
    // Process remaining bytes
    while i < len {
        if s1[i] != s2[i] {
            diff_count += 1;
        }
        i += 1;
    }
    
    diff_count
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn hamming_simd_ascii(s1: &[u8], s2: &[u8]) -> usize {
    use std::arch::x86_64::*;
    
    let len = s1.len();
    let mut diff_count = 0usize;
    let mut i = 0;
    
    // Process 16 bytes at a time with SSE2
    if len >= 16 {
        unsafe {
            while i + 16 <= len {
                let v1 = _mm_loadu_si128(s1.as_ptr().add(i) as *const __m128i);
                let v2 = _mm_loadu_si128(s2.as_ptr().add(i) as *const __m128i);
                
                // XOR to find differences
                let diff = _mm_xor_si128(v1, v2);
                
                // Compare with zero, inverted mask gives us differences
                let zero = _mm_setzero_si128();
                let eq = _mm_cmpeq_epi8(diff, zero);
                let mask = _mm_movemask_epi8(eq) as u32;
                
                // Count differing bytes (bits that are 0 in mask)
                diff_count += (16 - mask.count_ones()) as usize;
                
                i += 16;
            }
        }
    }
    
    // Process remaining bytes
    while i < len {
        if s1[i] != s2[i] {
            diff_count += 1;
        }
        i += 1;
    }
    
    diff_count
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
#[inline(always)]
fn hamming_simd_ascii(s1: &[u8], s2: &[u8]) -> usize {
    s1.iter().zip(s2.iter()).filter(|(a, b)| a != b).count()
}

/// Hamming distance for Unicode
#[inline(always)]
fn hamming_unicode(s1: &str, s2: &str) -> Option<usize> {
    let c1: SmallVec<[char; 64]> = s1.chars().collect();
    let c2: SmallVec<[char; 64]> = s2.chars().collect();
    
    if c1.len() != c2.len() {
        return None;
    }
    
    Some(c1.iter().zip(c2.iter()).filter(|(a, b)| a != b).count())
}

/// Main Hamming distance
#[inline(always)]
fn hamming_distance_internal(s1: &str, s2: &str) -> Option<usize> {
    // Fast path for identical strings
    if s1 == s2 {
        return Some(0);
    }
    
    // ASCII fast path with SIMD
    if s1.is_ascii() && s2.is_ascii() {
        let b1 = s1.as_bytes();
        let b2 = s2.as_bytes();
        
        if b1.len() != b2.len() {
            return None;
        }
        
        return Some(hamming_simd_ascii(b1, b2));
    }
    
    hamming_unicode(s1, s2)
}

/// Calculate Hamming distance
#[pyfunction]
#[pyo3(signature = (s1, s2, *, score_cutoff=None))]
pub fn hamming_distance(s1: &str, s2: &str, score_cutoff: Option<usize>) -> Option<usize> {
    let dist = hamming_distance_internal(s1, s2)?;
    
    match score_cutoff {
        Some(cutoff) if dist > cutoff => Some(cutoff + 1),
        _ => Some(dist),
    }
}

/// Calculate Hamming similarity
#[pyfunction]
#[pyo3(signature = (s1, s2, *, score_cutoff=None))]
pub fn hamming_similarity(s1: &str, s2: &str, score_cutoff: Option<usize>) -> Option<usize> {
    let len1 = if s1.is_ascii() { s1.len() } else { s1.chars().count() };
    let dist = hamming_distance_internal(s1, s2)?;
    let sim = len1.saturating_sub(dist);

    match score_cutoff {
        Some(cutoff) if sim < cutoff => Some(0),
        _ => Some(sim),
    }
}

/// Calculate normalized Hamming distance
#[pyfunction]
#[pyo3(signature = (s1, s2, *, score_cutoff=None))]
pub fn hamming_normalized_distance(s1: &str, s2: &str, score_cutoff: Option<f64>) -> Option<f64> {
    let len1 = if s1.is_ascii() { s1.len() } else { s1.chars().count() };
    
    if len1 == 0 {
        return Some(0.0);
    }

    let dist = hamming_distance_internal(s1, s2)?;
    let norm_dist = dist as f64 / len1 as f64;

    match score_cutoff {
        Some(cutoff) if norm_dist > cutoff => Some(1.0),
        _ => Some(norm_dist),
    }
}

/// Calculate normalized Hamming similarity
#[pyfunction]
#[pyo3(signature = (s1, s2, *, score_cutoff=None))]
pub fn hamming_normalized_similarity(s1: &str, s2: &str, score_cutoff: Option<f64>) -> Option<f64> {
    let norm_dist = hamming_normalized_distance(s1, s2, None)?;
    let norm_sim = 1.0 - norm_dist;

    match score_cutoff {
        Some(cutoff) if norm_sim < cutoff => Some(0.0),
        _ => Some(norm_sim),
    }
}

// ============ Batch Operations ============

#[pyfunction]
#[pyo3(signature = (pairs, *, score_cutoff=None))]
pub fn hamming_distance_batch(pairs: Vec<(String, String)>, score_cutoff: Option<usize>) -> Vec<Option<usize>> {
    pairs.par_iter().map(|(s1, s2)| hamming_distance(s1, s2, score_cutoff)).collect()
}

#[pyfunction]
#[pyo3(signature = (pairs, *, score_cutoff=None))]
pub fn hamming_normalized_similarity_batch(pairs: Vec<(String, String)>, score_cutoff: Option<f64>) -> Vec<Option<f64>> {
    pairs.par_iter().map(|(s1, s2)| hamming_normalized_similarity(s1, s2, score_cutoff)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hamming_distance() {
        assert_eq!(hamming_distance_internal("karolin", "kathrin"), Some(3));
        assert_eq!(hamming_distance_internal("abc", "abc"), Some(0));
        assert_eq!(hamming_distance_internal("abc", "ab"), None);
    }
    
    #[test]
    fn test_hamming_simd() {
        // Test with 16+ byte strings
        let s1 = "this is a test string";
        let s2 = "this is a tast strong";
        assert_eq!(hamming_distance_internal(s1, s2), Some(2));
    }
}
