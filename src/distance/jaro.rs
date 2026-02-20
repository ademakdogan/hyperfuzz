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

/// Multi-block bit-parallel Jaro for ASCII strings >128 chars
/// Uses multiple 64-bit blocks for pattern matching
#[inline(always)]
fn jaro_multiblock_ascii(s1: &[u8], s2: &[u8]) -> f64 {
    let orig_len1 = s1.len();
    let orig_len2 = s2.len();

    if orig_len1 == 0 && orig_len2 == 0 { return 1.0; }
    if orig_len1 == 0 || orig_len2 == 0 { return 0.0; }

    // Ensure s1 <= s2 for efficiency
    let (s1, s2, len1, len2) = if orig_len1 <= orig_len2 {
        (s1, s2, orig_len1, orig_len2)
    } else {
        (s2, s1, orig_len2, orig_len1)
    };
    
    // Jaro match distance
    let bound = len2 / 2;
    let bound = if bound > 0 { bound - 1 } else { 0 };
    
    // Trim s2 to only include reachable characters
    let max_s2_len = min(len2, len1 + bound);
    let s2 = &s2[..max_s2_len];
    let len2_trimmed = s2.len();
    
    const WORD_SIZE: usize = 64;
    let p_words = (len1 + WORD_SIZE - 1) / WORD_SIZE;
    let t_words = (len2_trimmed + WORD_SIZE - 1) / WORD_SIZE;
    
    // Build pattern match vectors for s1 (one per 64-char block)
    let mut pm: Vec<[u64; 256]> = vec![[0u64; 256]; p_words];
    for (i, &c) in s1.iter().enumerate() {
        let word = i / WORD_SIZE;
        let bit = i % WORD_SIZE;
        pm[word][c as usize] |= 1u64 << bit;
    }
    
    // Flag vectors
    let mut p_flag: SmallVec<[u64; 8]> = smallvec::smallvec![0u64; p_words];
    let mut t_flag: SmallVec<[u64; 8]> = smallvec::smallvec![0u64; t_words];
    
    // Process each character in s2
    for (j, &c2) in s2.iter().enumerate() {
        let t_word = j / WORD_SIZE;
        let t_bit = j % WORD_SIZE;
        
        // Calculate which words of s1 are in range for this position
        let start_pos = j.saturating_sub(bound);
        let end_pos = min(j + bound + 1, len1);
        
        if start_pos >= end_pos { continue; }
        
        let start_word = start_pos / WORD_SIZE;
        let end_word = (end_pos - 1) / WORD_SIZE;
        
        // Try to find a match in each word
        for word in start_word..=end_word {
            if word >= p_words { break; }
            
            // Create mask for valid positions in this word
            let word_start = word * WORD_SIZE;
            let word_end = min((word + 1) * WORD_SIZE, len1);
            
            let range_start = max(start_pos, word_start);
            let range_end = min(end_pos, word_end);
            
            if range_start >= range_end { continue; }
            
            let local_start = range_start - word_start;
            let local_end = range_end - word_start;
            
            // Mask for valid bit positions
            let range_mask = if local_end >= 64 {
                !0u64 << local_start
            } else {
                (!0u64 >> (64 - local_end)) & (!0u64 << local_start)
            };
            
            // Get matching positions, excluding already matched
            let pm_j = pm[word][c2 as usize] & range_mask & !p_flag[word];
            
            if pm_j != 0 {
                // Found a match - take lowest bit
                let match_bit = pm_j & pm_j.wrapping_neg();
                p_flag[word] |= match_bit;
                t_flag[t_word] |= 1u64 << t_bit;
                break;
            }
        }
    }
    
    // Count common characters
    let common_chars: usize = p_flag.iter().map(|w| w.count_ones() as usize).sum();
    if common_chars == 0 { return 0.0; }
    
    // Count transpositions
    let mut transpositions = 0usize;
    let mut p_word = 0usize;
    let mut p_mask = p_flag[0];
    
    for t_word in 0..t_words {
        let mut t_mask = t_flag[t_word];
        while t_mask != 0 {
            let t_bit = t_mask.trailing_zeros() as usize;
            let t_pos = t_word * WORD_SIZE + t_bit;
            
            // Find next matched position in s1
            while p_mask == 0 {
                p_word += 1;
                if p_word >= p_words { break; }
                p_mask = p_flag[p_word];
            }
            
            if p_word >= p_words { break; }
            
            let p_bit = p_mask.trailing_zeros() as usize;
            let p_pos = p_word * WORD_SIZE + p_bit;
            
            if s1[p_pos] != s2[t_pos] {
                transpositions += 1;
            }
            
            p_mask &= p_mask - 1; // Clear lowest bit
            t_mask &= t_mask - 1;
        }
    }

    let m = common_chars as f64;
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
    jaro_multiblock_ascii(s1, s2)
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
        let sim2 = jaro_multiblock_ascii(b"MARTHA", b"MARHTA");
        assert!((sim1 - sim2).abs() < 0.001);
    }
}
