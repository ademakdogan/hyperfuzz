//! Levenshtein distance - Ultra optimized with Myers' bit-parallel algorithm
//!
//! Implements Myers' bit-parallel algorithm for O(n*m/64) complexity
//! Reference: Gene Myers - "A fast bit-vector algorithm for approximate 
//!            string matching based on dynamic programming" (1999)

use pyo3::prelude::*;
use rayon::prelude::*;
use smallvec::SmallVec;

use crate::lcs_core::simd_str_equal;

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

/// Multi-block Myers bit-parallel for longer strings (>64 chars)
/// Based on RapidFuzz's levenshtein_hyrroe2003_block algorithm
/// O(n * m / 64) complexity
#[inline(always)]
fn levenshtein_myers_block(s1: &[u8], s2: &[u8]) -> usize {
    let len1 = s1.len();
    let len2 = s2.len();
    
    if len1 == 0 { return len2; }
    if len2 == 0 { return len1; }
    
    // Ensure s1 is the shorter string for block map
    let (s1, s2) = if len1 > len2 { (s2, s1) } else { (s1, s2) };
    let len1 = s1.len();
    let _len2 = s2.len();
    
    const WORD_SIZE: usize = 64;
    let words = (len1 + WORD_SIZE - 1) / WORD_SIZE;
    
    // Build block pattern match vectors for s1
    // blocks[word][char] = bitmask where char appears in that word of s1
    let mut blocks: Vec<[u64; 256]> = vec![[0u64; 256]; words];
    
    for (i, &c) in s1.iter().enumerate() {
        let word = i / WORD_SIZE;
        let bit_pos = i % WORD_SIZE;
        blocks[word][c as usize] |= 1u64 << bit_pos;
    }
    
    // Initialize VP/VN vectors for each word
    let mut vp: SmallVec<[u64; 8]> = smallvec::smallvec![!0u64; words];
    let mut vn: SmallVec<[u64; 8]> = smallvec::smallvec![0u64; words];
    
    // Initial scores for each block
    let mut scores: SmallVec<[usize; 8]> = (0..words)
        .map(|i| if i < words - 1 { (i + 1) * WORD_SIZE } else { len1 })
        .collect();
    
    // Last bit mask for final word
    let last_bit = 1u64 << ((len1 - 1) % WORD_SIZE);
    
    // Process each character of s2
    for &c2 in s2.iter() {
        let mut hp_carry: u64 = 1;
        let mut hn_carry: u64 = 0;
        
        for word in 0..words {
            // Step 1: Computing D0
            let pm_j = blocks[word][c2 as usize];
            let old_vp = vp[word];
            let old_vn = vn[word];
            
            let x = pm_j | hn_carry;
            let d0 = (((x & old_vp).wrapping_add(old_vp)) ^ old_vp) | x | old_vn;
            
            // Step 2: Computing HP and HN
            let hp = old_vn | !(d0 | old_vp);
            let hn = d0 & old_vp;
            
            // Compute carry for next block
            let hp_carry_temp = hp_carry;
            let hn_carry_temp = hn_carry;
            
            if word < words - 1 {
                hp_carry = hp >> 63;
                hn_carry = hn >> 63;
            } else {
                hp_carry = if (hp & last_bit) != 0 { 1 } else { 0 };
                hn_carry = if (hn & last_bit) != 0 { 1 } else { 0 };
            }
            
            // Step 3: Update score
            scores[word] = (scores[word] as isize + (hp_carry as isize) - (hn_carry as isize)) as usize;
            
            // Step 4: Computing VP and VN
            let hp_shifted = (hp << 1) | hp_carry_temp;
            let hn_shifted = (hn << 1) | hn_carry_temp;
            
            vp[word] = hn_shifted | !(d0 | hp_shifted);
            vn[word] = hp_shifted & d0;
        }
    }
    
    scores[words - 1]
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
    
    // Use Myers bit-parallel for all sizes
    if m <= 64 {
        return levenshtein_myers_64(s1, s2);
    }
    
    // Use multi-block Myers for longer strings
    levenshtein_myers_block(s1, s2)
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
    
    // Check if all chars fit in Latin-1 (0-255) for fast bit-parallel path
    let all_latin1 = s1.iter().chain(s2.iter()).all(|&c| c as u32 <= 255);
    
    if all_latin1 {
        // Convert to bytes and use fast bit-parallel
        let s1_bytes: SmallVec<[u8; 256]> = s1.iter().map(|&c| c as u8).collect();
        let s2_bytes: SmallVec<[u8; 256]> = s2.iter().map(|&c| c as u8).collect();
        
        if m <= 64 {
            return levenshtein_myers_64(&s1_bytes, &s2_bytes);
        }
        return levenshtein_myers_block(&s1_bytes, &s2_bytes);
    }

    // Full Unicode DP fallback
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
