//! Jaro similarity algorithm - Optimized with bit-vector matching
//!
//! Uses bit-vectors for faster matching window computation
//! when dealing with ASCII strings <= 64 characters.

use pyo3::prelude::*;
use rayon::prelude::*;
use smallvec::SmallVec;
use std::cmp::{max, min};

type CharVec = SmallVec<[char; 64]>;
type BoolVec = SmallVec<[bool; 64]>;

/// Bit-vector based Jaro for ASCII strings <= 64 chars
#[inline(always)]
fn jaro_bitparallel_ascii(s1: &[u8], s2: &[u8]) -> f64 {
    let len1 = s1.len();
    let len2 = s2.len();

    if len1 == 0 && len2 == 0 { return 1.0; }
    if len1 == 0 || len2 == 0 { return 0.0; }

    let match_distance = max(len1, len2) / 2;
    let match_distance = if match_distance > 0 { match_distance - 1 } else { 0 };

    // Use u64 bit vectors for match flags (works for strings <= 64 chars)
    let mut s1_matches: u64 = 0;
    let mut s2_matches: u64 = 0;
    let mut matches = 0usize;

    // Find matching characters using bit operations
    for i in 0..len1 {
        let start = i.saturating_sub(match_distance);
        let end = min(i + match_distance + 1, len2);

        for j in start..end {
            // Check if s2[j] already matched
            if (s2_matches >> j) & 1 != 0 {
                continue;
            }
            if s1[i] != s2[j] {
                continue;
            }
            s1_matches |= 1u64 << i;
            s2_matches |= 1u64 << j;
            matches += 1;
            break;
        }
    }

    if matches == 0 {
        return 0.0;
    }

    // Count transpositions using bit scanning
    let mut transpositions = 0usize;
    let mut k = 0;
    for i in 0..len1 {
        if (s1_matches >> i) & 1 == 0 {
            continue;
        }
        // Find next matched position in s2
        while (s2_matches >> k) & 1 == 0 {
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

/// 128-bit (2×u64) bit-parallel Jaro with RapidFuzz-style pattern matching
/// Uses bit-vectors for O(1) character matching instead of O(k) linear search
#[inline(always)]
fn jaro_bitparallel_128(s1: &[u8], s2: &[u8]) -> f64 {
    let len1 = s1.len();
    let len2 = s2.len();

    if len1 == 0 && len2 == 0 { return 1.0; }
    if len1 == 0 || len2 == 0 { return 0.0; }

    // Match distance (Jaro bound)
    let bound = max(len1, len2) / 2;
    let bound = if bound > 0 { bound - 1 } else { 0 };

    // Build pattern match vector for s1 (character → bit positions)
    // PM[char] = bit mask of positions where char appears in s1
    let mut pm: [u128; 256] = [0u128; 256];
    for (i, &c) in s1.iter().enumerate() {
        pm[c as usize] |= 1u128 << i;
    }

    // Flagged matches
    let mut p_flag: u128 = 0;  // Which positions in s1 are matched
    let mut t_flag: u128 = 0;  // Which positions in s2 are matched
    
    // Sliding bound mask - starts covering positions [0, bound+1)
    // and slides right as we process s2
    let initial_bound_size = min(bound + 1, len1);
    let mut bound_mask: u128 = (1u128 << initial_bound_size) - 1;
    
    // Process s2 and find matching chars in s1
    for (j, &c2) in s2.iter().enumerate() {
        // Get positions in s1 where c2 appears, within current bound window
        let pm_j = pm[c2 as usize] & bound_mask & (!p_flag);
        
        if pm_j != 0 {
            // Extract lowest set bit (first available match)
            let match_bit = pm_j & pm_j.wrapping_neg();
            p_flag |= match_bit;
            t_flag |= 1u128 << j;
        }
        
        // Slide the bound mask for next position
        if j < bound {
            // Still in the initial expansion phase
            bound_mask = (bound_mask << 1) | 1;
        } else {
            // Normal sliding
            bound_mask <<= 1;
        }
    }
    
    let common_chars = p_flag.count_ones() as usize;
    if common_chars == 0 { return 0.0; }

    // Count transpositions
    let mut transpositions = 0usize;
    let mut k = 0usize;
    for i in 0..len1 {
        if (p_flag >> i) & 1 == 0 { continue; }
        while (t_flag >> k) & 1 == 0 { k += 1; }
        if s1[i] != s2[k] { transpositions += 1; }
        k += 1;
    }

    let m = common_chars as f64;
    let t = (transpositions / 2) as f64;

    (m / len1 as f64 + m / len2 as f64 + (m - t) / m) / 3.0
}

/// Standard Jaro for ASCII with bounds trimming (no char indexing - simpler and faster for ~100 char strings)
#[inline(always)]
fn jaro_standard_ascii(s1: &[u8], s2: &[u8]) -> f64 {
    let orig_len1 = s1.len();
    let orig_len2 = s2.len();

    if orig_len1 == 0 && orig_len2 == 0 { return 1.0; }
    if orig_len1 == 0 || orig_len2 == 0 { return 0.0; }

    // RapidFuzz bounds trimming: skip unreachable characters
    let (s1, s2, len1, len2) = if orig_len1 <= orig_len2 {
        let bound = if orig_len2 > 1 { orig_len2 / 2 - 1 } else { 0 };
        let max_text_len = min(orig_len2, orig_len1 + bound);
        (&s1[..], &s2[..max_text_len], orig_len1, max_text_len)
    } else {
        let bound = if orig_len1 > 1 { orig_len1 / 2 - 1 } else { 0 };
        let max_text_len = min(orig_len1, orig_len2 + bound);
        (&s1[..max_text_len], &s2[..], max_text_len, orig_len2)
    };

    let match_distance = max(len1, len2) / 2;
    let match_distance = if match_distance > 0 { match_distance - 1 } else { 0 };

    let mut s1_matches: SmallVec<[bool; 128]> = SmallVec::from_elem(false, len1);
    let mut s2_matches: SmallVec<[bool; 128]> = SmallVec::from_elem(false, len2);
    let mut matches = 0usize;

    for i in 0..len1 {
        let start = i.saturating_sub(match_distance);
        let end = min(i + match_distance + 1, len2);

        for j in start..end {
            if s2_matches[j] || s1[i] != s2[j] { continue; }
            s1_matches[i] = true;
            s2_matches[j] = true;
            matches += 1;
            break;
        }
    }

    if matches == 0 { return 0.0; }

    // Count transpositions
    let mut transpositions = 0usize;
    let mut k = 0;
    for i in 0..len1 {
        if !s1_matches[i] { continue; }
        while !s2_matches[k] { k += 1; }
        if s1[i] != s2[k] { transpositions += 1; }
        k += 1;
    }

    let m = matches as f64;
    let t = (transpositions / 2) as f64;

    (m / orig_len1 as f64 + m / orig_len2 as f64 + (m - t) / m) / 3.0
}

/// Calculate Jaro similarity for ASCII strings
#[inline(always)]
fn jaro_similarity_ascii(s1: &[u8], s2: &[u8]) -> f64 {
    let len1 = s1.len();
    let len2 = s2.len();
    
    // Use bit-parallel for short strings (<= 64 chars)
    if len1 <= 64 && len2 <= 64 {
        return jaro_bitparallel_ascii(s1, s2);
    }
    
    // Use 128-bit bit-parallel for medium strings (65-128 chars)
    if len1 <= 128 && len2 <= 128 {
        return jaro_bitparallel_128(s1, s2);
    }
    
    // Standard algorithm with optimizations for very long strings
    jaro_standard_ascii(s1, s2)
}

/// Calculate Jaro similarity for Unicode strings
#[inline(always)]
fn jaro_similarity_chars(s1_chars: &[char], s2_chars: &[char]) -> f64 {
    let len1 = s1_chars.len();
    let len2 = s2_chars.len();

    if len1 == 0 && len2 == 0 { return 1.0; }
    if len1 == 0 || len2 == 0 { return 0.0; }

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
        if !s1_matches[i] { continue; }
        while !s2_matches[k] { k += 1; }
        if s1_chars[i] != s2_chars[k] { transpositions += 1; }
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
        return if s1.is_empty() { 1.0 } else { 1.0 };
    }

    // ASCII fast path with bit-parallel
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
    
    #[test]
    fn test_jaro_bitparallel() {
        // Test bit-parallel gives same result as standard
        let sim1 = jaro_bitparallel_ascii(b"MARTHA", b"MARHTA");
        let sim2 = jaro_standard_ascii(b"MARTHA", b"MARHTA");
        assert!((sim1 - sim2).abs() < 0.001);
    }
}
