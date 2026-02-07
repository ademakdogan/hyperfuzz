//! Partial token ratio algorithms - Ultra Optimized
//!
//! Optimizations:
//! 1. Shared partial_ratio computation
//! 2. Lazy string building
//! 3. Early termination paths
//! 4. ASCII-optimized sliding window

use pyo3::prelude::*;
use rayon::prelude::*;
use smallvec::SmallVec;
use ahash::AHashSet;
use std::cmp::max;

type TokenVec<'a> = SmallVec<[&'a str; 16]>;
type RowVec = SmallVec<[usize; 128]>;

/// Tokenize a string
#[inline(always)]
fn tokenize(s: &str) -> TokenVec {
    s.split_whitespace().collect()
}

/// Ultra-fast LCS for ASCII bytes
#[inline(always)]
fn lcs_ascii(s1: &[u8], s2: &[u8]) -> usize {
    let m = s1.len();
    let n = s2.len();
    if m == 0 || n == 0 { return 0; }
    
    let (s1, s2, m, _n) = if m > n { (s2, s1, n, m) } else { (s1, s2, m, n) };

    let mut prev: RowVec = SmallVec::from_elem(0, m + 1);
    let mut curr: RowVec = SmallVec::from_elem(0, m + 1);

    for c2 in s2.iter() {
        for (i, c1) in s1.iter().enumerate() {
            curr[i + 1] = if *c1 == *c2 { prev[i] + 1 } else { max(prev[i + 1], curr[i]) };
        }
        std::mem::swap(&mut prev, &mut curr);
        curr.fill(0);
    }
    prev[m]
}

/// LCS for strings
#[inline(always)]
fn lcs_str(s1: &str, s2: &str) -> usize {
    if s1.is_ascii() && s2.is_ascii() {
        return lcs_ascii(s1.as_bytes(), s2.as_bytes());
    }
    
    let s1: SmallVec<[char; 64]> = s1.chars().collect();
    let s2: SmallVec<[char; 64]> = s2.chars().collect();
    
    let m = s1.len();
    let n = s2.len();
    if m == 0 || n == 0 { return 0; }
    
    let (s1, s2, m, _n) = if m > n { (&s2[..], &s1[..], n, m) } else { (&s1[..], &s2[..], m, n) };

    let mut prev: RowVec = SmallVec::from_elem(0, m + 1);
    let mut curr: RowVec = SmallVec::from_elem(0, m + 1);

    for c2 in s2.iter() {
        for (i, c1) in s1.iter().enumerate() {
            curr[i + 1] = if *c1 == *c2 { prev[i] + 1 } else { max(prev[i + 1], curr[i]) };
        }
        std::mem::swap(&mut prev, &mut curr);
        curr.fill(0);
    }
    prev[m]
}

/// Calculate ratio from lengths and LCS
#[inline(always)]
fn ratio_from_parts(len1: usize, len2: usize, lcs: usize) -> f64 {
    let total = len1 + len2;
    if total == 0 { return 100.0; }
    100.0 * (2.0 * lcs as f64) / (total as f64)
}

/// Optimized partial ratio for ASCII bytes
#[inline(always)]
fn partial_ratio_ascii(s1: &[u8], s2: &[u8]) -> f64 {
    let len1 = s1.len();
    let len2 = s2.len();

    if len1 == 0 && len2 == 0 { return 100.0; }
    if len1 == 0 || len2 == 0 { return 0.0; }

    let (shorter, longer) = if len1 <= len2 { (s1, s2) } else { (s2, s1) };
    let short_len = shorter.len();
    let long_len = longer.len();

    let mut best = 0.0f64;

    for i in 0..=(long_len - short_len) {
        let window = &longer[i..i + short_len];
        let lcs = lcs_ascii(shorter, window);
        let score = ratio_from_parts(short_len, short_len, lcs);
        if score > best { best = score; }
        if best >= 100.0 { break; }
    }
    best
}

/// Optimized partial ratio for strings
#[inline(always)]
pub fn partial_ratio_internal(s1: &str, s2: &str) -> f64 {
    if s1 == s2 { return 100.0; }
    
    // ASCII fast path
    if s1.is_ascii() && s2.is_ascii() {
        return partial_ratio_ascii(s1.as_bytes(), s2.as_bytes());
    }

    // Unicode path
    let s1: SmallVec<[char; 64]> = s1.chars().collect();
    let s2: SmallVec<[char; 64]> = s2.chars().collect();
    
    let len1 = s1.len();
    let len2 = s2.len();

    if len1 == 0 && len2 == 0 { return 100.0; }
    if len1 == 0 || len2 == 0 { return 0.0; }

    let (shorter, longer) = if len1 <= len2 { (&s1[..], &s2[..]) } else { (&s2[..], &s1[..]) };
    let short_len = shorter.len();
    let long_len = longer.len();

    let mut best = 0.0f64;
    let mut prev: RowVec = SmallVec::from_elem(0, short_len + 1);
    let mut curr: RowVec = SmallVec::from_elem(0, short_len + 1);

    for i in 0..=(long_len - short_len) {
        let window = &longer[i..i + short_len];
        
        // Inline LCS computation
        prev.fill(0);
        for c2 in window.iter() {
            curr.fill(0);
            for (j, c1) in shorter.iter().enumerate() {
                curr[j + 1] = if *c1 == *c2 { prev[j] + 1 } else { max(prev[j + 1], curr[j]) };
            }
            std::mem::swap(&mut prev, &mut curr);
        }
        
        let lcs = prev[short_len];
        let score = ratio_from_parts(short_len, short_len, lcs);
        if score > best { best = score; }
        if best >= 100.0 { break; }
    }
    best
}

/// Join tokens efficiently
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

/// Partial token set ratio - Heavily optimized
#[pyfunction]
#[pyo3(signature = (s1, s2, *, score_cutoff=None))]
pub fn partial_token_set_ratio(s1: &str, s2: &str, score_cutoff: Option<f64>) -> f64 {
    if s1 == s2 { return 100.0; }
    
    let tokens1: AHashSet<&str> = tokenize(s1).into_iter().collect();
    let tokens2: AHashSet<&str> = tokenize(s2).into_iter().collect();
    
    if tokens1 == tokens2 { return 100.0; }
    if tokens1.is_empty() && tokens2.is_empty() { return 100.0; }
    if tokens1.is_empty() || tokens2.is_empty() { return 0.0; }
    
    let mut inter: TokenVec = tokens1.intersection(&tokens2).copied().collect();
    let mut diff1: TokenVec = tokens1.difference(&tokens2).copied().collect();
    let mut diff2: TokenVec = tokens2.difference(&tokens1).copied().collect();
    
    inter.sort_unstable();
    diff1.sort_unstable();
    diff2.sort_unstable();
    
    let inter_str = join_tokens(&inter);
    
    if inter_str.is_empty() {
        let c1 = join_tokens(&diff1);
        let c2 = join_tokens(&diff2);
        let result = partial_ratio_internal(&c1, &c2);
        return match score_cutoff {
            Some(cutoff) if result < cutoff => 0.0,
            _ => result,
        };
    }
    
    let combined1 = if diff1.is_empty() {
        inter_str.clone()
    } else {
        format!("{} {}", inter_str, join_tokens(&diff1))
    };
    
    let combined2 = if diff2.is_empty() {
        inter_str.clone()
    } else {
        format!("{} {}", inter_str, join_tokens(&diff2))
    };
    
    // Compute all three and take max
    let r1 = partial_ratio_internal(&inter_str, &combined1);
    let r2 = partial_ratio_internal(&inter_str, &combined2);
    let r3 = partial_ratio_internal(&combined1, &combined2);
    
    let result = r1.max(r2).max(r3);
    
    match score_cutoff {
        Some(cutoff) if result < cutoff => 0.0,
        _ => result,
    }
}

/// Partial token ratio - max of sort and set variants
#[pyfunction]
#[pyo3(signature = (s1, s2, *, score_cutoff=None))]
pub fn partial_token_ratio(s1: &str, s2: &str, score_cutoff: Option<f64>) -> f64 {
    if s1 == s2 { return 100.0; }
    
    // Compute sort first (often cheaper)
    let sort_result = partial_token_sort_ratio(s1, s2, None);
    
    // Early exit if already at max
    if sort_result >= 100.0 {
        return match score_cutoff {
            Some(cutoff) if sort_result < cutoff => 0.0,
            _ => sort_result,
        };
    }
    
    let set_result = partial_token_set_ratio(s1, s2, None);
    let result = sort_result.max(set_result);
    
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
    fn test_partial_token_set_ratio() {
        let r = partial_token_set_ratio("kitten", "sitting", None);
        assert!((r - 66.67).abs() < 0.1);
    }
}
