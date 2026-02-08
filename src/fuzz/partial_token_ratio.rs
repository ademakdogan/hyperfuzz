//! Partial token ratio algorithms - RapidFuzz-inspired optimizations
//!
//! Key optimizations from RapidFuzz:
//! 1. Early exit when intersection exists (return 100)
//! 2. Avoid duplicate partial_ratio calculations
//! 3. Reuse token sets

use pyo3::prelude::*;
use rayon::prelude::*;
use smallvec::SmallVec;
use ahash::AHashSet;
use std::cmp::max;
use std::cell::RefCell;

use crate::lcs_core::ratio_from_lcs;

type TokenVec<'a> = SmallVec<[&'a str; 16]>;

// Thread-local buffers for partial ratio sliding window
thread_local! {
    static PARTIAL_BUF_1: RefCell<Vec<usize>> = RefCell::new(Vec::with_capacity(128));
    static PARTIAL_BUF_2: RefCell<Vec<usize>> = RefCell::new(Vec::with_capacity(128));
}

#[inline(always)]
fn tokenize<'a>(s: &'a str) -> TokenVec<'a> {
    s.split_whitespace().collect()
}

#[inline(always)]
fn join_tokens(tokens: &[&str]) -> String {
    if tokens.is_empty() { return String::new(); }
    if tokens.len() == 1 { return tokens[0].to_string(); }
    
    let capacity: usize = tokens.iter().map(|t| t.len()).sum::<usize>() + tokens.len() - 1;
    let mut result = String::with_capacity(capacity);
    result.push_str(tokens[0]);
    for t in &tokens[1..] {
        result.push(' ');
        result.push_str(t);
    }
    result
}

/// Optimized partial ratio for ASCII with thread-local buffers 
#[inline(always)]
fn partial_ratio_ascii_optimized(shorter: &[u8], longer: &[u8]) -> f64 {
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

/// Optimized partial ratio for strings
#[inline(always)]
pub fn partial_ratio_internal(s1: &str, s2: &str) -> f64 {
    if s1 == s2 { return 100.0; }
    if s1.is_empty() && s2.is_empty() { return 100.0; }
    if s1.is_empty() || s2.is_empty() { return 0.0; }
    
    let (len1, len2) = if s1.is_ascii() && s2.is_ascii() {
        (s1.len(), s2.len())
    } else {
        (s1.chars().count(), s2.chars().count())
    };
    
    // ASCII fast path
    if s1.is_ascii() && s2.is_ascii() {
        let (shorter, longer) = if len1 <= len2 {
            (s1.as_bytes(), s2.as_bytes())
        } else {
            (s2.as_bytes(), s1.as_bytes())
        };
        return partial_ratio_ascii_optimized(shorter, longer);
    }
    
    // Unicode path
    let s1_chars: SmallVec<[char; 64]> = s1.chars().collect();
    let s2_chars: SmallVec<[char; 64]> = s2.chars().collect();
    
    let (shorter, longer) = if s1_chars.len() <= s2_chars.len() {
        (&s1_chars[..], &s2_chars[..])
    } else {
        (&s2_chars[..], &s1_chars[..])
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

/// Partial token sort ratio
#[pyfunction]
#[pyo3(signature = (s1, s2, *, score_cutoff=None))]
pub fn partial_token_sort_ratio(s1: &str, s2: &str, score_cutoff: Option<f64>) -> f64 {
    if s1 == s2 { return 100.0; }
    
    let mut tokens1 = tokenize(s1);
    let mut tokens2 = tokenize(s2);
    
    tokens1.sort_unstable();
    tokens2.sort_unstable();
    
    if tokens1 == tokens2 { return 100.0; }
    
    let sorted1 = join_tokens(&tokens1);
    let sorted2 = join_tokens(&tokens2);
    
    let result = partial_ratio_internal(&sorted1, &sorted2);
    
    match score_cutoff {
        Some(cutoff) if result < cutoff => 0.0,
        _ => result,
    }
}

/// Partial token set ratio - KEY OPTIMIZATION: return 100 if intersection exists
#[pyfunction]
#[pyo3(signature = (s1, s2, *, score_cutoff=None))]
pub fn partial_token_set_ratio(s1: &str, s2: &str, score_cutoff: Option<f64>) -> f64 {
    if s1 == s2 { return 100.0; }
    
    let tokens1: AHashSet<&str> = tokenize(s1).into_iter().collect();
    let tokens2: AHashSet<&str> = tokenize(s2).into_iter().collect();
    
    if tokens1 == tokens2 { return 100.0; }
    if tokens1.is_empty() || tokens2.is_empty() { return 0.0; }
    
    // KEY OPTIMIZATION from RapidFuzz: If there's any common word, return 100
    if tokens1.intersection(&tokens2).next().is_some() {
        return 100.0;
    }
    
    // No intersection - compare sorted differences
    let mut diff_ab: TokenVec = tokens1.difference(&tokens2).copied().collect();
    let mut diff_ba: TokenVec = tokens2.difference(&tokens1).copied().collect();
    
    diff_ab.sort_unstable();
    diff_ba.sort_unstable();
    
    let diff_ab_joined = join_tokens(&diff_ab);
    let diff_ba_joined = join_tokens(&diff_ba);
    
    let result = partial_ratio_internal(&diff_ab_joined, &diff_ba_joined);
    
    match score_cutoff {
        Some(cutoff) if result < cutoff => 0.0,
        _ => result,
    }
}

/// Partial token ratio - RapidFuzz algorithm
#[pyfunction]
#[pyo3(signature = (s1, s2, *, score_cutoff=None))]
pub fn partial_token_ratio(s1: &str, s2: &str, score_cutoff: Option<f64>) -> f64 {
    if s1 == s2 { return 100.0; }
    
    let score_cutoff_val = score_cutoff.unwrap_or(0.0);
    
    let tokens_split_a = tokenize(s1);
    let tokens_split_b = tokenize(s2);
    let tokens_a: AHashSet<&str> = tokens_split_a.iter().copied().collect();
    let tokens_b: AHashSet<&str> = tokens_split_b.iter().copied().collect();
    
    // KEY OPTIMIZATION: If there's any common word, return 100
    if tokens_a.intersection(&tokens_b).next().is_some() {
        return 100.0;
    }
    
    let diff_ab: TokenVec = tokens_a.difference(&tokens_b).copied().collect();
    let diff_ba: TokenVec = tokens_b.difference(&tokens_a).copied().collect();
    
    // Calculate partial_token_sort_ratio
    let mut sorted_a: TokenVec = tokens_split_a.clone();
    let mut sorted_b: TokenVec = tokens_split_b.clone();
    sorted_a.sort_unstable();
    sorted_b.sort_unstable();
    
    let sorted_a_str = join_tokens(&sorted_a);
    let sorted_b_str = join_tokens(&sorted_b);
    
    let mut result = partial_ratio_internal(&sorted_a_str, &sorted_b_str);
    
    // Avoid duplicate calculation if sets are same as original tokens
    if tokens_split_a.len() == diff_ab.len() && tokens_split_b.len() == diff_ba.len() {
        return match score_cutoff {
            Some(cutoff) if result < cutoff => 0.0,
            _ => result,
        };
    }
    
    // Calculate partial ratio for differences
    let mut diff_ab_sorted: TokenVec = diff_ab;
    let mut diff_ba_sorted: TokenVec = diff_ba;
    diff_ab_sorted.sort_unstable();
    diff_ba_sorted.sort_unstable();
    
    let diff_ab_str = join_tokens(&diff_ab_sorted);
    let diff_ba_str = join_tokens(&diff_ba_sorted);
    
    let diff_result = partial_ratio_internal(&diff_ab_str, &diff_ba_str);
    result = result.max(diff_result);
    
    match score_cutoff {
        Some(cutoff) if result < cutoff => 0.0,
        _ => result,
    }
}

// ============ Batch Operations ============

#[pyfunction]
#[pyo3(signature = (pairs, *, score_cutoff=None))]
pub fn partial_token_sort_ratio_batch(pairs: Vec<(String, String)>, score_cutoff: Option<f64>) -> Vec<f64> {
    pairs.par_iter().map(|(s1, s2)| partial_token_sort_ratio(s1, s2, score_cutoff)).collect()
}

#[pyfunction]
#[pyo3(signature = (pairs, *, score_cutoff=None))]
pub fn partial_token_set_ratio_batch(pairs: Vec<(String, String)>, score_cutoff: Option<f64>) -> Vec<f64> {
    pairs.par_iter().map(|(s1, s2)| partial_token_set_ratio(s1, s2, score_cutoff)).collect()
}

#[pyfunction]
#[pyo3(signature = (pairs, *, score_cutoff=None))]
pub fn partial_token_ratio_batch(pairs: Vec<(String, String)>, score_cutoff: Option<f64>) -> Vec<f64> {
    pairs.par_iter().map(|(s1, s2)| partial_token_ratio(s1, s2, score_cutoff)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_partial_token_sort() {
        let r = partial_token_sort_ratio("fuzzy wuzzy was a bear", "wuzzy fuzzy", None);
        assert!(r > 90.0);
    }
    
    #[test]
    fn test_partial_token_set_ratio_intersection() {
        // Any common word = 100
        let r = partial_token_set_ratio("hello world", "world peace", None);
        assert!((r - 100.0).abs() < 0.01);
    }
    
    #[test]
    fn test_partial_token_set_ratio_no_intersection() {
        let r = partial_token_set_ratio("kitten", "sitting", None);
        assert!((r - 66.67).abs() < 0.1);
    }
}
