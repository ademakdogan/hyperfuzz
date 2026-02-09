//! Partial ratio algorithm - RapidFuzz-inspired with bit-parallel LCS
//!
//! Key optimizations:
//! 1. Bit-parallel LCS computation (Hyyrö's algorithm)
//! 2. Character filtering - only check windows containing s1 characters
//! 3. Pre-built block map reused across sliding window
//! 4. Early termination when score reaches 100

use pyo3::prelude::*;
use rayon::prelude::*;
use smallvec::SmallVec;
use std::cmp::max;
use std::cell::RefCell;

use crate::lcs_core::{build_block_map, build_char_set, lcs_with_block, ratio_from_lcs};

// Thread-local buffers for Unicode path
thread_local! {
    static PARTIAL_BUF_1: RefCell<Vec<usize>> = RefCell::new(Vec::with_capacity(128));
    static PARTIAL_BUF_2: RefCell<Vec<usize>> = RefCell::new(Vec::with_capacity(128));
}

/// Bit-parallel partial ratio for ASCII strings <= 64 chars
/// Uses Hyyrö's algorithm with character filtering
#[inline(always)]
fn partial_ratio_bitparallel(shorter: &[u8], longer: &[u8]) -> f64 {
    let len1 = shorter.len();
    let len2 = longer.len();
    
    if len1 == 0 && len2 == 0 { return 100.0; }
    if len1 == 0 || len2 == 0 { return 0.0; }
    
    // Build block map and char set once
    let block = build_block_map(shorter);
    let char_set = build_char_set(shorter);
    
    let mut best_score = 0.0f64;
    
    // Phase 1: Windows at the start (partial overlap from left)
    for i in 1..len1 {
        let substr_last = longer[i - 1];
        // Character filtering: skip if last char not in s1
        if !char_set[substr_last as usize] { continue; }
        
        let window = &longer[..i];
        let lcs = lcs_with_block(&block, len1, window);
        let score = ratio_from_lcs(len1, window.len(), lcs);
        if score > best_score { best_score = score; }
        if best_score >= 100.0 { return 100.0; }
    }
    
    // Phase 2: Full windows (sliding through middle)
    for i in 0..=(len2 - len1) {
        let substr_last = longer[i + len1 - 1];
        // Character filtering: skip if last char not in s1
        if !char_set[substr_last as usize] { continue; }
        
        let window = &longer[i..i + len1];
        let lcs = lcs_with_block(&block, len1, window);
        let score = ratio_from_lcs(len1, len1, lcs);
        if score > best_score { best_score = score; }
        if best_score >= 100.0 { return 100.0; }
    }
    
    // Phase 3: Windows at the end (partial overlap from right)
    for i in (len2 - len1 + 1)..len2 {
        let substr_first = longer[i];
        // Character filtering: skip if first char not in s1
        if !char_set[substr_first as usize] { continue; }
        
        let window = &longer[i..];
        let lcs = lcs_with_block(&block, len1, window);
        let score = ratio_from_lcs(len1, window.len(), lcs);
        if score > best_score { best_score = score; }
        if best_score >= 100.0 { return 100.0; }
    }
    
    best_score
}

/// Fallback DP-based partial ratio for long ASCII strings
#[inline(always)]
fn partial_ratio_dp_ascii(shorter: &[u8], longer: &[u8]) -> f64 {
    let short_len = shorter.len();
    let long_len = longer.len();
    
    let mut best = 0.0f64;
    
    PARTIAL_BUF_1.with(|buf1| {
        PARTIAL_BUF_2.with(|buf2| {
            let mut prev = buf1.borrow_mut();
            let mut curr = buf2.borrow_mut();
            
            prev.clear();
            prev.resize(short_len + 1, 0);
            curr.clear();
            curr.resize(short_len + 1, 0);
            
            for i in 0..=(long_len - short_len) {
                let window = &longer[i..i + short_len];
                
                prev.fill(0);
                for c2 in window.iter() {
                    curr.fill(0);
                    for (j, c1) in shorter.iter().enumerate() {
                        curr[j + 1] = if *c1 == *c2 {
                            prev[j] + 1
                        } else {
                            max(prev[j + 1], curr[j])
                        };
                    }
                    std::mem::swap(&mut *prev, &mut *curr);
                }
                
                let lcs = prev[short_len];
                let score = ratio_from_lcs(short_len, short_len, lcs);
                if score > best { best = score; }
                if best >= 100.0 { break; }
            }
        })
    });
    
    best
}

/// Partial ratio for Unicode strings
#[inline(always)]
fn partial_ratio_unicode(s1_chars: &[char], s2_chars: &[char]) -> f64 {
    let len1 = s1_chars.len();
    let len2 = s2_chars.len();

    if len1 == 0 && len2 == 0 { return 100.0; }
    if len1 == 0 || len2 == 0 { return 0.0; }

    let (shorter, longer) = if len1 <= len2 {
        (s1_chars, s2_chars)
    } else {
        (s2_chars, s1_chars)
    };
    
    let short_len = shorter.len();
    let long_len = longer.len();

    let mut best = 0.0f64;
    
    PARTIAL_BUF_1.with(|buf1| {
        PARTIAL_BUF_2.with(|buf2| {
            let mut prev = buf1.borrow_mut();
            let mut curr = buf2.borrow_mut();
            
            prev.clear();
            prev.resize(short_len + 1, 0);
            curr.clear();
            curr.resize(short_len + 1, 0);
            
            for i in 0..=(long_len - short_len) {
                let window = &longer[i..i + short_len];
                
                prev.fill(0);
                for c2 in window.iter() {
                    curr.fill(0);
                    for (j, c1) in shorter.iter().enumerate() {
                        curr[j + 1] = if *c1 == *c2 {
                            prev[j] + 1
                        } else {
                            max(prev[j + 1], curr[j])
                        };
                    }
                    std::mem::swap(&mut *prev, &mut *curr);
                }
                
                let lcs = prev[short_len];
                let score = ratio_from_lcs(short_len, short_len, lcs);
                if score > best { best = score; }
                if best >= 100.0 { break; }
            }
        })
    });
    
    best
}

/// Main partial ratio implementation
#[inline(always)]
pub fn partial_ratio_internal(s1: &str, s2: &str) -> f64 {
    if s1 == s2 { return 100.0; }
    if s1.is_empty() && s2.is_empty() { return 100.0; }
    if s1.is_empty() || s2.is_empty() { return 0.0; }
    
    // ASCII + short string: use bit-parallel
    if s1.is_ascii() && s2.is_ascii() {
        let b1 = s1.as_bytes();
        let b2 = s2.as_bytes();
        
        let (shorter, longer) = if b1.len() <= b2.len() {
            (b1, b2)
        } else {
            (b2, b1)
        };
        
        // Use bit-parallel for short strings
        if shorter.len() <= 64 {
            return partial_ratio_bitparallel(shorter, longer);
        }
        
        return partial_ratio_dp_ascii(shorter, longer);
    }
    
    // Unicode path
    let s1_chars: SmallVec<[char; 64]> = s1.chars().collect();
    let s2_chars: SmallVec<[char; 64]> = s2.chars().collect();
    partial_ratio_unicode(&s1_chars, &s2_chars)
}

/// Calculate partial ratio
#[pyfunction]
#[pyo3(signature = (s1, s2, *, score_cutoff=None))]
pub fn partial_ratio(s1: &str, s2: &str, score_cutoff: Option<f64>) -> f64 {
    let result = partial_ratio_internal(s1, s2);
    
    match score_cutoff {
        Some(cutoff) if result < cutoff => 0.0,
        _ => result,
    }
}

/// Batch partial ratio
#[pyfunction]
#[pyo3(signature = (pairs, *, score_cutoff=None))]
pub fn partial_ratio_batch(pairs: Vec<(String, String)>, score_cutoff: Option<f64>) -> Vec<f64> {
    pairs
        .par_iter()
        .map(|(s1, s2)| partial_ratio(s1, s2, score_cutoff))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_partial_ratio() {
        let r = partial_ratio_internal("this is a test", "this is a test!");
        assert!((r - 100.0).abs() < 0.01);
    }
    
    #[test]
    fn test_partial_ratio_substring() {
        let r = partial_ratio_internal("test", "this is a test");
        assert!((r - 100.0).abs() < 0.01);
    }
}
