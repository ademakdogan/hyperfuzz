//! OSA - Optimal String Alignment distance with Hyyrö's bit-parallel algorithm
//!
//! Uses bit-parallel algorithm from:
//! Hyyrö (2003) - "A Bit-Vector Algorithm for Computing Levenshtein and 
//! Damerau Edit Distances"

use pyo3::prelude::*;
use rayon::prelude::*;
use smallvec::SmallVec;

/// Bit-parallel OSA for ASCII strings <= 64 chars (Hyyrö 2003)
#[inline(always)]
fn osa_bitparallel_ascii(s1: &[u8], s2: &[u8]) -> usize {
    let len1 = s1.len();
    let len2 = s2.len();
    
    if len1 == 0 { return len2; }
    if len2 == 0 { return len1; }
    
    // Build character block map
    let mut block = [0u64; 256];
    for (i, &c) in s1.iter().enumerate() {
        block[c as usize] |= 1u64 << i;
    }
    
    // Initialize bit vectors
    let mut vp: u64 = (1u64 << len1) - 1;  // All 1s
    let mut vn: u64 = 0;
    let mut d0: u64 = 0;
    let mut pm_j_old: u64 = 0;
    let mut curr_dist = len1;
    let mask = 1u64 << (len1 - 1);
    
    for &c2 in s2.iter() {
        let pm_j = block[c2 as usize];
        
        // Step 1: Computing D0 with transposition
        let tr = (((!d0) & pm_j) << 1) & pm_j_old;
        d0 = (((pm_j & vp).wrapping_add(vp)) ^ vp) | pm_j | vn;
        d0 |= tr;
        
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
        pm_j_old = pm_j;
    }
    
    curr_dist
}

/// Standard DP OSA (fallback for >64 chars or Unicode)
#[inline(always)]
fn osa_dp(s1_chars: &[char], s2_chars: &[char]) -> usize {
    let len1 = s1_chars.len();
    let len2 = s2_chars.len();

    if len1 == 0 { return len2; }
    if len2 == 0 { return len1; }

    // Need 3 rows for transposition check
    let mut d0: SmallVec<[usize; 128]> = (0..=len2).collect();
    let mut d1: SmallVec<[usize; 128]> = SmallVec::from_elem(0, len2 + 1);
    let mut d2: SmallVec<[usize; 128]> = SmallVec::from_elem(0, len2 + 1);

    for i in 1..=len1 {
        d1[0] = i;

        for j in 1..=len2 {
            let cost = if s1_chars[i - 1] == s2_chars[j - 1] { 0 } else { 1 };

            d1[j] = (d0[j] + 1).min(d1[j - 1] + 1).min(d0[j - 1] + cost);

            // Check for transposition
            if i > 1 && j > 1
                && s1_chars[i - 1] == s2_chars[j - 2]
                && s1_chars[i - 2] == s2_chars[j - 1]
            {
                d1[j] = d1[j].min(d2[j - 2] + cost);
            }
        }

        std::mem::swap(&mut d2, &mut d0);
        std::mem::swap(&mut d0, &mut d1);
        d1.fill(0);
    }

    d0[len2]
}

/// Main OSA distance implementation
#[inline(always)]
fn osa_internal(s1: &str, s2: &str) -> usize {
    if s1 == s2 { return 0; }
    
    // ASCII + short strings: use bit-parallel
    if s1.is_ascii() && s2.is_ascii() {
        let b1 = s1.as_bytes();
        let b2 = s2.as_bytes();
        
        if b1.len() <= 64 {
            return osa_bitparallel_ascii(b1, b2);
        }
    }
    
    // Unicode/long string path
    let s1_chars: SmallVec<[char; 64]> = s1.chars().collect();
    let s2_chars: SmallVec<[char; 64]> = s2.chars().collect();
    osa_dp(&s1_chars, &s2_chars)
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
    let len1 = if s1.is_ascii() { s1.len() } else { s1.chars().count() };
    let len2 = if s2.is_ascii() { s2.len() } else { s2.chars().count() };
    let max_len = len1.max(len2);
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
    let len1 = if s1.is_ascii() { s1.len() } else { s1.chars().count() };
    let len2 = if s2.is_ascii() { s2.len() } else { s2.chars().count() };
    let max_len = len1.max(len2);

    if max_len == 0 { return 0.0; }

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
    pairs.par_iter().map(|(s1, s2)| osa_distance(s1, s2, score_cutoff)).collect()
}

#[pyfunction]
#[pyo3(signature = (pairs, *, score_cutoff=None))]
pub fn osa_normalized_similarity_batch(
    pairs: Vec<(String, String)>,
    score_cutoff: Option<f64>,
) -> Vec<f64> {
    pairs.par_iter().map(|(s1, s2)| osa_normalized_similarity(s1, s2, score_cutoff)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_osa_bitparallel() {
        assert_eq!(osa_bitparallel_ascii(b"ab", b"ba"), 1); // transposition
        assert_eq!(osa_bitparallel_ascii(b"abc", b"abc"), 0);
        assert_eq!(osa_bitparallel_ascii(b"CA", b"AC"), 2);
    }
    
    #[test]
    fn test_osa_internal() {
        assert_eq!(osa_internal("ab", "ba"), 1);
        assert_eq!(osa_internal("abc", "abc"), 0);
    }
}
