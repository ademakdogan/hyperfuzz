//! Token-based ratio algorithms - Ultra Optimized
//!
//! Optimizations:
//! 1. Cached tokenization
//! 2. Pre-sorted token collections
//! 3. Lazy string building
//! 4. ASCII-optimized LCS
//! 5. Early termination

use pyo3::prelude::*;
use rayon::prelude::*;
use smallvec::SmallVec;
use ahash::AHashSet;
use std::cmp::max;

// Stack allocation limits
type TokenVec<'a> = SmallVec<[&'a str; 16]>;
type RowVec = SmallVec<[usize; 128]>;

/// Tokenize a string (split by whitespace).
#[inline(always)]
fn tokenize(s: &str) -> TokenVec {
    s.split_whitespace().collect()
}

/// Ultra-fast LCS length for ASCII byte slices
#[inline(always)]
fn lcs_ascii(s1: &[u8], s2: &[u8]) -> usize {
    let m = s1.len();
    let n = s2.len();
    
    if m == 0 || n == 0 { return 0; }
    
    // Make s1 shorter for cache efficiency
    let (s1, s2, m, n) = if m > n {
        (s2, s1, n, m)
    } else {
        (s1, s2, m, n)
    };

    let mut prev: RowVec = SmallVec::from_elem(0, m + 1);
    let mut curr: RowVec = SmallVec::from_elem(0, m + 1);

    for c2 in s2.iter() {
        for (i, c1) in s1.iter().enumerate() {
            curr[i + 1] = if *c1 == *c2 {
                prev[i] + 1
            } else {
                max(prev[i + 1], curr[i])
            };
        }
        std::mem::swap(&mut prev, &mut curr);
        curr.fill(0);
    }

    prev[m]
}

/// LCS length for string slices - uses ASCII fast path
#[inline(always)]
fn lcs_str(s1: &str, s2: &str) -> usize {
    if s1.is_ascii() && s2.is_ascii() {
        return lcs_ascii(s1.as_bytes(), s2.as_bytes());
    }
    
    // Unicode fallback
    let s1_chars: SmallVec<[char; 64]> = s1.chars().collect();
    let s2_chars: SmallVec<[char; 64]> = s2.chars().collect();
    
    let m = s1_chars.len();
    let n = s2_chars.len();
    
    if m == 0 || n == 0 { return 0; }
    
    let (s1, s2, m, _n) = if m > n {
        (&s2_chars[..], &s1_chars[..], n, m)
    } else {
        (&s1_chars[..], &s2_chars[..], m, n)
    };

    let mut prev: RowVec = SmallVec::from_elem(0, m + 1);
    let mut curr: RowVec = SmallVec::from_elem(0, m + 1);

    for c2 in s2.iter() {
        for (i, c1) in s1.iter().enumerate() {
            curr[i + 1] = if *c1 == *c2 {
                prev[i] + 1
            } else {
                max(prev[i + 1], curr[i])
            };
        }
        std::mem::swap(&mut prev, &mut curr);
        curr.fill(0);
    }

    prev[m]
}

/// Calculate ratio from LCS - inlined computation
#[inline(always)]
fn ratio_from_lcs(len1: usize, len2: usize, lcs: usize) -> f64 {
    let total = len1 + len2;
    if total == 0 { return 100.0; }
    100.0 * (2.0 * lcs as f64) / (total as f64)
}

/// Calculate ratio for two strings - optimized
#[inline(always)]
fn ratio_internal(s1: &str, s2: &str) -> f64 {
    // Fast paths
    if s1 == s2 { return 100.0; }
    if s1.is_empty() && s2.is_empty() { return 100.0; }
    if s1.is_empty() || s2.is_empty() { return 0.0; }
    
    let len1 = if s1.is_ascii() { s1.len() } else { s1.chars().count() };
    let len2 = if s2.is_ascii() { s2.len() } else { s2.chars().count() };
    let lcs = lcs_str(s1, s2);
    
    ratio_from_lcs(len1, len2, lcs)
}

/// Join tokens with space - optimized for common case
#[inline(always)]
fn join_tokens(tokens: &[&str]) -> String {
    if tokens.is_empty() { return String::new(); }
    if tokens.len() == 1 { return tokens[0].to_string(); }
    
    // Pre-calculate capacity
    let capacity: usize = tokens.iter().map(|t| t.len()).sum::<usize>() + tokens.len() - 1;
    let mut result = String::with_capacity(capacity);
    result.push_str(tokens[0]);
    for t in &tokens[1..] {
        result.push(' ');
        result.push_str(t);
    }
    result
}

/// Token sort ratio - sort tokens before comparison
#[pyfunction]
#[pyo3(signature = (s1, s2, *, score_cutoff=None))]
pub fn token_sort_ratio(s1: &str, s2: &str, score_cutoff: Option<f64>) -> f64 {
    // Fast path for identical strings
    if s1 == s2 { return 100.0; }
    
    let mut tokens1 = tokenize(s1);
    let mut tokens2 = tokenize(s2);
    
    // Fast path - same tokens count and content after sort
    tokens1.sort_unstable();
    tokens2.sort_unstable();
    
    if tokens1 == tokens2 { return 100.0; }
    
    let sorted1 = join_tokens(&tokens1);
    let sorted2 = join_tokens(&tokens2);
    
    let result = ratio_internal(&sorted1, &sorted2);
    
    match score_cutoff {
        Some(cutoff) if result < cutoff => 0.0,
        _ => result,
    }
}

/// Token set ratio - compare token sets
#[pyfunction]
#[pyo3(signature = (s1, s2, *, score_cutoff=None))]
pub fn token_set_ratio(s1: &str, s2: &str, score_cutoff: Option<f64>) -> f64 {
    if s1 == s2 { return 100.0; }
    
    let tokens1: AHashSet<&str> = tokenize(s1).into_iter().collect();
    let tokens2: AHashSet<&str> = tokenize(s2).into_iter().collect();
    
    // Fast path - identical token sets
    if tokens1 == tokens2 { return 100.0; }
    
    if tokens1.is_empty() && tokens2.is_empty() { return 100.0; }
    if tokens1.is_empty() || tokens2.is_empty() { return 0.0; }
    
    // Collect and sort
    let mut inter: TokenVec = tokens1.intersection(&tokens2).copied().collect();
    let mut diff1: TokenVec = tokens1.difference(&tokens2).copied().collect();
    let mut diff2: TokenVec = tokens2.difference(&tokens1).copied().collect();
    
    inter.sort_unstable();
    diff1.sort_unstable();
    diff2.sort_unstable();
    
    // Build strings lazily
    let inter_str = join_tokens(&inter);
    
    // Optimize: avoid building strings when possible
    if inter_str.is_empty() {
        // No intersection - compare sorted diffs directly
        let c1 = join_tokens(&diff1);
        let c2 = join_tokens(&diff2);
        let result = ratio_internal(&c1, &c2);
        return match score_cutoff {
            Some(cutoff) if result < cutoff => 0.0,
            _ => result,
        };
    }
    
    // Build combined strings
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
    
    // Calculate all ratios
    let r1 = ratio_internal(&inter_str, &combined1);
    let r2 = ratio_internal(&inter_str, &combined2);
    let r3 = ratio_internal(&combined1, &combined2);
    
    let result = r1.max(r2).max(r3);
    
    match score_cutoff {
        Some(cutoff) if result < cutoff => 0.0,
        _ => result,
    }
}

/// Token ratio - max of token_sort_ratio and token_set_ratio
#[pyfunction]
#[pyo3(signature = (s1, s2, *, score_cutoff=None))]
pub fn token_ratio(s1: &str, s2: &str, score_cutoff: Option<f64>) -> f64 {
    // Fast path
    if s1 == s2 { return 100.0; }
    
    // Compute token_sort first (usually cheaper)
    let sort_result = token_sort_ratio(s1, s2, None);
    
    // Early termination if already at max
    if sort_result >= 100.0 {
        return match score_cutoff {
            Some(cutoff) if sort_result < cutoff => 0.0,
            _ => sort_result,
        };
    }
    
    let set_result = token_set_ratio(s1, s2, None);
    let result = sort_result.max(set_result);
    
    match score_cutoff {
        Some(cutoff) if result < cutoff => 0.0,
        _ => result,
    }
}

// ============ Batch Operations ============

#[pyfunction]
#[pyo3(signature = (pairs, *, score_cutoff=None))]
pub fn token_sort_ratio_batch(pairs: Vec<(String, String)>, score_cutoff: Option<f64>) -> Vec<f64> {
    pairs
        .par_iter()
        .map(|(s1, s2)| token_sort_ratio(s1, s2, score_cutoff))
        .collect()
}

#[pyfunction]
#[pyo3(signature = (pairs, *, score_cutoff=None))]
pub fn token_set_ratio_batch(pairs: Vec<(String, String)>, score_cutoff: Option<f64>) -> Vec<f64> {
    pairs
        .par_iter()
        .map(|(s1, s2)| token_set_ratio(s1, s2, score_cutoff))
        .collect()
}

#[pyfunction]
#[pyo3(signature = (pairs, *, score_cutoff=None))]
pub fn token_ratio_batch(pairs: Vec<(String, String)>, score_cutoff: Option<f64>) -> Vec<f64> {
    pairs
        .par_iter()
        .map(|(s1, s2)| token_ratio(s1, s2, score_cutoff))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_sort_ratio() {
        let r = token_sort_ratio("fuzzy wuzzy was a bear", "wuzzy fuzzy was a bear", None);
        assert!((r - 100.0).abs() < 0.01);
    }

    #[test]
    fn test_token_set_ratio() {
        let r = token_set_ratio("fuzzy was a bear", "fuzzy fuzzy was a bear", None);
        assert!((r - 100.0).abs() < 0.01);
    }
    
    #[test]
    fn test_token_set_ratio_single_words() {
        let r = token_set_ratio("kitten", "sitting", None);
        assert!((r - 61.54).abs() < 0.1);
    }
    
    #[test]
    fn test_token_ratio() {
        let r = token_ratio("kitten", "sitting", None);
        assert!((r - 61.54).abs() < 0.1);
    }
}
