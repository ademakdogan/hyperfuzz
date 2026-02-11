//! Levenshtein distance - Ultra optimized with Myers' bit-parallel algorithm
//!
//! Implements Myers' bit-parallel algorithm for O(n*m/64) complexity
//! Reference: Gene Myers - "A fast bit-vector algorithm for approximate 
//!            string matching based on dynamic programming" (1999)

use pyo3::prelude::*;
use rayon::prelude::*;
use smallvec::SmallVec;
use std::cell::RefCell;

use crate::lcs_core::simd_str_equal;

// Thread-local buffer for DP fallback
thread_local! {
    static LEV_BUF: RefCell<Vec<usize>> = RefCell::new(Vec::with_capacity(256));
}

/// Myers' bit-parallel Levenshtein for ASCII strings <= 64 chars
/// O(n) for fixed pattern length, O(n*m/64) overall
#[inline(always)]
fn levenshtein_myers_64(s1: &[u8], s2: &[u8]) -> usize {
    let len1 = s1.len();
    let len2 = s2.len();
    
    if len1 == 0 { return len2; }
    if len2 == 0 { return len1; }
    
    // Build block map for s1
    let mut block = [0u64; 256];
    for (i, &c) in s1.iter().enumerate() {
        block[c as usize] |= 1u64 << i;
    }
    
    // Initialize bit vectors
    let mut vp: u64 = !0u64;  // Vertical positive
    let mut vn: u64 = 0u64;   // Vertical negative
    let mut curr_dist = len1;
    let mask = 1u64 << (len1 - 1);
    
    // Process each character of s2
    for &c2 in s2.iter() {
        let pm_j = block[c2 as usize];
        
        // Step 1: Computing D0
        let x = pm_j;
        let d0 = (((x & vp).wrapping_add(vp)) ^ vp) | x | vn;
        
        // Step 2: Computing HP and HN
        let hp = vn | !(d0 | vp);
        let hn = d0 & vp;
        
        // Step 3: Computing the value D[m,j]
        if (hp & mask) != 0 {
            curr_dist += 1;
        }
        if (hn & mask) != 0 {
            curr_dist -= 1;
        }
        
        // Step 4: Computing VP and VN
        let hp_shifted = (hp << 1) | 1;
        let hn_shifted = hn << 1;
        vp = hn_shifted | !(d0 | hp_shifted);
        vn = hp_shifted & d0;
    }
    
    curr_dist
}

/// Block-based Myers for longer strings (>64 chars)
#[inline(always)]
fn levenshtein_myers_block(s1: &[u8], s2: &[u8]) -> usize {
    // For now, fall back to DP for strings > 64 chars
    // Full block implementation would be complex
    levenshtein_dp(s1, s2)
}

/// Optimized DP with prefix/suffix elimination
#[inline(always)]
fn levenshtein_dp(s1: &[u8], s2: &[u8]) -> usize {
    let mut m = s1.len();
    let mut n = s2.len();

    if m == 0 { return n; }
    if n == 0 { return m; }
    
    // Swap to ensure m <= n
    let (s1, s2) = if m > n {
        std::mem::swap(&mut m, &mut n);
        (s2, s1)
    } else {
        (s1, s2)
    };

    // Skip common prefix
    let prefix_len = s1.iter().zip(s2.iter()).take_while(|(a, b)| a == b).count();
    if prefix_len == m { return n - m; }
    
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
    
    // Use short strings with Myers
    if m <= 64 {
        return levenshtein_myers_64(s1, s2);
    }

    // DP fallback
    LEV_BUF.with(|buf| {
        let mut row = buf.borrow_mut();
        row.clear();
        row.extend(0..=m);
        
        for c2 in s2.iter() {
            let mut prev_diag = row[0];
            row[0] += 1;
            
            // Unrolled loop
            let mut i = 0;
            while i + 4 <= m {
                let old0 = row[i + 1];
                let old1 = row[i + 2];
                let old2 = row[i + 3];
                let old3 = row[i + 4];
                
                row[i + 1] = if s1[i] == *c2 { prev_diag } else { prev_diag.min(row[i + 1]).min(row[i]) + 1 };
                row[i + 2] = if s1[i + 1] == *c2 { old0 } else { old0.min(row[i + 2]).min(row[i + 1]) + 1 };
                row[i + 3] = if s1[i + 2] == *c2 { old1 } else { old1.min(row[i + 3]).min(row[i + 2]) + 1 };
                row[i + 4] = if s1[i + 3] == *c2 { old2 } else { old2.min(row[i + 4]).min(row[i + 3]) + 1 };
                
                prev_diag = old3;
                i += 4;
            }
            
            while i < m {
                let old_diag = row[i + 1];
                row[i + 1] = if s1[i] == *c2 { prev_diag } else { prev_diag.min(row[i + 1]).min(row[i]) + 1 };
                prev_diag = old_diag;
                i += 1;
            }
        }
        
        row[m]
    })
}

/// Levenshtein for Unicode
#[inline(always)]
fn levenshtein_unicode(s1: &str, s2: &str) -> usize {
    let s1_chars: SmallVec<[char; 64]> = s1.chars().collect();
    let s2_chars: SmallVec<[char; 64]> = s2.chars().collect();
    
    let mut m = s1_chars.len();
    let mut n = s2_chars.len();

    if m == 0 { return n; }
    if n == 0 { return m; }
    if s1_chars == s2_chars { return 0; }

    let (s1_chars, s2_chars) = if m > n {
        std::mem::swap(&mut m, &mut n);
        (s2_chars, s1_chars)
    } else {
        (s1_chars, s2_chars)
    };

    // Skip common prefix
    let prefix_len = s1_chars.iter().zip(s2_chars.iter()).take_while(|(a, b)| a == b).count();
    if prefix_len == m { return n - m; }
    
    let s1 = &s1_chars[prefix_len..];
    let s2 = &s2_chars[prefix_len..];
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

    let mut row: SmallVec<[usize; 128]> = (0..=m).collect();
    
    for c2 in s2.iter() {
        let mut prev_diag = row[0];
        row[0] += 1;
        
        for (i, c1) in s1.iter().enumerate() {
            let old_diag = row[i + 1];
            row[i + 1] = if *c1 == *c2 { prev_diag } else { prev_diag.min(row[i + 1]).min(row[i]) + 1 };
            prev_diag = old_diag;
        }
    }

    row[m]
}

/// Main entry point
#[inline(always)]
fn levenshtein_distance_internal(s1: &str, s2: &str) -> usize {
    if s1 == s2 { return 0; }
    
    if s1.is_ascii() && s2.is_ascii() {
        let b1 = s1.as_bytes();
        let b2 = s2.as_bytes();
        
        // SIMD equality check
        if b1.len() == b2.len() && simd_str_equal(b1, b2) {
            return 0;
        }
        
        return levenshtein_dp(b1, b2);
    }
    
    levenshtein_unicode(s1, s2)
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

    if max_len == 0 { return 0.0; }

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

#[pyfunction]
#[pyo3(signature = (pairs, *, score_cutoff=None))]
pub fn levenshtein_distance_batch(pairs: Vec<(String, String)>, score_cutoff: Option<usize>) -> Vec<usize> {
    pairs.par_iter().map(|(s1, s2)| levenshtein_distance(s1, s2, score_cutoff)).collect()
}

#[pyfunction]
#[pyo3(signature = (pairs, *, score_cutoff=None))]
pub fn levenshtein_similarity_batch(pairs: Vec<(String, String)>, score_cutoff: Option<usize>) -> Vec<usize> {
    pairs.par_iter().map(|(s1, s2)| levenshtein_similarity(s1, s2, score_cutoff)).collect()
}

#[pyfunction]
#[pyo3(signature = (pairs, *, score_cutoff=None))]
pub fn levenshtein_normalized_distance_batch(pairs: Vec<(String, String)>, score_cutoff: Option<f64>) -> Vec<f64> {
    pairs.par_iter().map(|(s1, s2)| levenshtein_normalized_distance(s1, s2, score_cutoff)).collect()
}

#[pyfunction]
#[pyo3(signature = (pairs, *, score_cutoff=None))]
pub fn levenshtein_normalized_similarity_batch(pairs: Vec<(String, String)>, score_cutoff: Option<f64>) -> Vec<f64> {
    pairs.par_iter().map(|(s1, s2)| levenshtein_normalized_similarity(s1, s2, score_cutoff)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_levenshtein_myers() {
        assert_eq!(levenshtein_myers_64(b"kitten", b"sitting"), 3);
        assert_eq!(levenshtein_myers_64(b"abc", b"abc"), 0);
        assert_eq!(levenshtein_myers_64(b"abc", b"abd"), 1);
    }

    #[test]
    fn test_levenshtein_distance() {
        assert_eq!(levenshtein_distance_internal("kitten", "sitting"), 3);
        assert_eq!(levenshtein_distance_internal("", "abc"), 3);
        assert_eq!(levenshtein_distance_internal("abc", ""), 3);
        assert_eq!(levenshtein_distance_internal("abc", "abc"), 0);
    }

    #[test]
    fn test_prefix_suffix() {
        assert_eq!(levenshtein_distance_internal("prefix_abc", "prefix_abd"), 1);
        assert_eq!(levenshtein_distance_internal("abc_suffix", "abd_suffix"), 1);
    }
}
