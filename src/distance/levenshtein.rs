//! Levenshtein distance algorithm - Ultra Optimized
//!
//! Uses multiple optimization techniques:
//! 1. ASCII byte-level operations
//! 2. SmallVec for stack allocation
//! 3. Common prefix/suffix elimination
//! 4. Diagonal banding for score_cutoff
//! 5. Cache-friendly row-major access pattern

use pyo3::prelude::*;
use rayon::prelude::*;
use smallvec::SmallVec;

// Stack-allocate arrays up to 128 elements
type RowVec = SmallVec<[usize; 128]>;

/// Ultra-optimized Levenshtein for ASCII bytes with banding
#[inline(always)]
fn levenshtein_ascii_banded(s1: &[u8], s2: &[u8], max_dist: usize) -> usize {
    let mut m = s1.len();
    let mut n = s2.len();
    
    // Length difference alone exceeds max_dist
    if m.abs_diff(n) > max_dist {
        return max_dist + 1;
    }

    // Make s1 the shorter string
    let (s1, s2) = if m > n {
        std::mem::swap(&mut m, &mut n);
        (s2, s1)
    } else {
        (s1, s2)
    };

    // Empty string case
    if m == 0 { return n; }

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

    // Use banded computation - only compute cells within max_dist of diagonal
    let band_width = max_dist + 1;
    
    // Single row with sentinel values
    let mut row: RowVec = (0..=m).collect();
    
    for (j, c2) in s2.iter().enumerate() {
        let j1 = j + 1;
        
        // Band boundaries
        let start = if j1 > band_width { j1 - band_width } else { 0 };
        let end = (j1 + band_width).min(m);
        
        let mut prev_diag = row[start];
        row[start] = if start == 0 { j1 } else { max_dist + 1 };
        
        for i in start..end {
            let i1 = i + 1;
            let old_diag = row[i1];
            
            let sub_cost = prev_diag + (s1[i] != *c2) as usize;
            let del_cost = row[i1] + 1;
            let ins_cost = row[i] + 1;
            
            row[i1] = sub_cost.min(del_cost).min(ins_cost);
            prev_diag = old_diag;
        }
        
        // Early termination: if minimum in row exceeds max_dist, abort
        if j1 < n && row[start..=end.min(m)].iter().all(|&x| x > max_dist) {
            return max_dist + 1;
        }
    }

    row[m]
}

/// Ultra-optimized Levenshtein for ASCII bytes (no banding)
#[inline(always)]
fn levenshtein_ascii_full(s1: &[u8], s2: &[u8]) -> usize {
    let mut m = s1.len();
    let mut n = s2.len();

    if m == 0 { return n; }
    if n == 0 { return m; }

    // Make s1 shorter
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

    // Optimized single-row computation
    let mut row: RowVec = (0..=m).collect();
    
    for c2 in s2.iter() {
        let mut prev_diag = row[0];
        row[0] += 1;
        
        for (i, c1) in s1.iter().enumerate() {
            let old_diag = row[i + 1];
            row[i + 1] = if *c1 == *c2 {
                prev_diag
            } else {
                prev_diag.min(row[i + 1]).min(row[i]) + 1
            };
            prev_diag = old_diag;
        }
    }

    row[m]
}

/// Levenshtein for Unicode with prefix/suffix optimization
#[inline(always)]
fn levenshtein_unicode(s1: &str, s2: &str) -> usize {
    let s1_chars: SmallVec<[char; 64]> = s1.chars().collect();
    let s2_chars: SmallVec<[char; 64]> = s2.chars().collect();
    
    let mut m = s1_chars.len();
    let mut n = s2_chars.len();

    if m == 0 { return n; }
    if n == 0 { return m; }

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

    let mut row: RowVec = (0..=m).collect();
    
    for c2 in s2.iter() {
        let mut prev_diag = row[0];
        row[0] += 1;
        
        for (i, c1) in s1.iter().enumerate() {
            let old_diag = row[i + 1];
            row[i + 1] = if *c1 == *c2 {
                prev_diag
            } else {
                prev_diag.min(row[i + 1]).min(row[i]) + 1
            };
            prev_diag = old_diag;
        }
    }

    row[m]
}

/// Calculate the Levenshtein distance between two strings.
#[inline(always)]
fn levenshtein_distance_internal(s1: &str, s2: &str) -> usize {
    // Fast path for identical strings
    if s1 == s2 { return 0; }
    
    // ASCII fast path
    if s1.is_ascii() && s2.is_ascii() {
        return levenshtein_ascii_full(s1.as_bytes(), s2.as_bytes());
    }
    
    levenshtein_unicode(s1, s2)
}

/// Calculate the Levenshtein distance with optional score_cutoff for banding
#[inline(always)]
fn levenshtein_distance_with_cutoff(s1: &str, s2: &str, max_dist: usize) -> usize {
    if s1 == s2 { return 0; }
    
    if s1.is_ascii() && s2.is_ascii() {
        return levenshtein_ascii_banded(s1.as_bytes(), s2.as_bytes(), max_dist);
    }
    
    // For Unicode, fall back to full computation
    levenshtein_unicode(s1, s2)
}

/// Calculate the Levenshtein distance between two strings.
#[pyfunction]
#[pyo3(signature = (s1, s2, *, score_cutoff=None))]
pub fn levenshtein_distance(s1: &str, s2: &str, score_cutoff: Option<usize>) -> usize {
    match score_cutoff {
        Some(cutoff) => {
            let dist = levenshtein_distance_with_cutoff(s1, s2, cutoff);
            if dist > cutoff { cutoff + 1 } else { dist }
        }
        None => levenshtein_distance_internal(s1, s2),
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
pub fn levenshtein_distance_batch(
    pairs: Vec<(String, String)>,
    score_cutoff: Option<usize>,
) -> Vec<usize> {
    pairs
        .par_iter()
        .map(|(s1, s2)| levenshtein_distance(s1, s2, score_cutoff))
        .collect()
}

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
    fn test_banded() {
        assert_eq!(levenshtein_distance_with_cutoff("kitten", "sitting", 5), 3);
        assert_eq!(levenshtein_distance_with_cutoff("kitten", "sitting", 2), 3); // exceeds
    }

    #[test]
    fn test_prefix_suffix() {
        // Common prefix
        assert_eq!(levenshtein_distance_internal("prefix_abc", "prefix_abd"), 1);
        // Common suffix  
        assert_eq!(levenshtein_distance_internal("abc_suffix", "abd_suffix"), 1);
        // Both
        assert_eq!(levenshtein_distance_internal("pre_abc_suf", "pre_abd_suf"), 1);
    }
}
