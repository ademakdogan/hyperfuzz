//! Token-based ratio algorithms - RapidFuzz-inspired optimizations
//!
//! Key optimizations from RapidFuzz:
//! 1. Early exit when intersection exists (for set ratios)
//! 2. Mathematical distance calculation without full LCS
//! 3. Indel distance instead of LCS where applicable

use pyo3::prelude::*;
use rayon::prelude::*;
use smallvec::SmallVec;
use ahash::AHashSet;

use crate::lcs_core::{lcs_fast, ratio_from_lcs};

type TokenVec<'a> = SmallVec<[&'a str; 16]>;

/// Tokenize
#[inline(always)]
fn tokenize<'a>(s: &'a str) -> TokenVec<'a> {
    s.split_whitespace().collect()
}

/// Ultra-fast join with unsafe memory operations
#[inline(always)]
fn join_tokens(tokens: &[&str]) -> String {
    if tokens.is_empty() { return String::new(); }
    if tokens.len() == 1 { return tokens[0].to_string(); }
    
    // Pre-calculate exact capacity
    let capacity: usize = tokens.iter().map(|t| t.len()).sum::<usize>() + tokens.len() - 1;
    
    // Allocate and fill using unsafe
    let mut result = String::with_capacity(capacity);
    
    unsafe {
        let buf = result.as_mut_vec();
        buf.set_len(capacity);
        let ptr = buf.as_mut_ptr();
        let mut offset = 0;
        
        // Copy first token
        std::ptr::copy_nonoverlapping(tokens[0].as_ptr(), ptr, tokens[0].len());
        offset += tokens[0].len();
        
        // Copy remaining with spaces
        for t in &tokens[1..] {
            *ptr.add(offset) = b' ';
            offset += 1;
            std::ptr::copy_nonoverlapping(t.as_ptr(), ptr.add(offset), t.len());
            offset += t.len();
        }
    }
    
    result
}

/// Fast ratio using unsafe byte operations
#[inline(always)]
fn ratio_internal(s1: &str, s2: &str) -> f64 {
    if s1.is_empty() && s2.is_empty() { return 100.0; }
    if s1.is_empty() || s2.is_empty() { return 0.0; }
    if s1 == s2 { return 100.0; }
    
    // For ASCII strings, use direct byte comparison
    if s1.is_ascii() && s2.is_ascii() {
        let b1 = s1.as_bytes();
        let b2 = s2.as_bytes();
        let len1 = b1.len();
        let len2 = b2.len();
        let lensum = len1 + len2;
        
        // Direct call to optimized LCS
        let lcs = crate::lcs_core::lcs_bitparallel_multiblock(b1, b2);
        return 100.0 * (2 * lcs) as f64 / lensum as f64;
    }
    
    // Non-ASCII fallback
    let len1 = s1.chars().count();
    let len2 = s2.chars().count();
    let lcs = lcs_fast(s1, s2);
    
    ratio_from_lcs(len1, len2, lcs)
}

/// Calculate normalized distance from indel distance
#[inline(always)]
fn norm_distance(dist: usize, lensum: usize, score_cutoff: f64) -> f64 {
    let score = if lensum > 0 { 100.0 - 100.0 * dist as f64 / lensum as f64 } else { 100.0 };
    if score >= score_cutoff { score } else { 0.0 }
}

/// Indel distance (len1 + len2 - 2*LCS)
#[inline(always)]
fn indel_dist(s1: &str, s2: &str) -> usize {
    if s1 == s2 { return 0; }
    let len1 = if s1.is_ascii() { s1.len() } else { s1.chars().count() };
    let len2 = if s2.is_ascii() { s2.len() } else { s2.chars().count() };
    let lcs = lcs_fast(s1, s2);
    len1 + len2 - 2 * lcs
}

/// Token sort ratio
#[pyfunction]
#[pyo3(signature = (s1, s2, *, score_cutoff=None))]
pub fn token_sort_ratio(s1: &str, s2: &str, score_cutoff: Option<f64>) -> f64 {
    if s1 == s2 { return 100.0; }
    
    let mut tokens1 = tokenize(s1);
    let mut tokens2 = tokenize(s2);
    
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

/// Token set ratio - Ultra optimized with inline set operations and direct byte LCS
#[pyfunction]
#[pyo3(signature = (s1, s2, *, score_cutoff=None))]
pub fn token_set_ratio(s1: &str, s2: &str, score_cutoff: Option<f64>) -> f64 {
    if s1 == s2 { return 100.0; }
    if s1.is_empty() && s2.is_empty() { return 100.0; }
    if s1.is_empty() || s2.is_empty() { return 0.0; }
    
    // Tokenize once
    let tokens1 = tokenize(s1);
    let tokens2 = tokenize(s2);
    
    if tokens1.is_empty() && tokens2.is_empty() { return 100.0; }
    if tokens1.is_empty() || tokens2.is_empty() { return 0.0; }
    
    // Build sets inline without allocation
    let set1: AHashSet<&str> = tokens1.iter().copied().collect();
    let set2: AHashSet<&str> = tokens2.iter().copied().collect();
    
    if set1 == set2 { return 100.0; }
    
    // Calculate differences inline - no clone, direct iteration
    let mut diff_ab: SmallVec<[&str; 16]> = SmallVec::new();
    let mut diff_ba: SmallVec<[&str; 16]> = SmallVec::new();
    let mut intersect: SmallVec<[&str; 16]> = SmallVec::new();
    
    for &t in set1.iter() {
        if set2.contains(t) {
            intersect.push(t);
        } else {
            diff_ab.push(t);
        }
    }
    for &t in set2.iter() {
        if !set1.contains(t) {
            diff_ba.push(t);
        }
    }
    
    // KEY OPTIMIZATION: One sentence is part of the other
    if !intersect.is_empty() && (diff_ab.is_empty() || diff_ba.is_empty()) {
        return 100.0;
    }
    
    let score_cutoff = score_cutoff.unwrap_or(0.0);
    
    // Sort differences inline
    diff_ab.sort_unstable();
    diff_ba.sort_unstable();
    
    // Calculate lengths without string allocation - use chars().count() for Unicode correctness
    let ab_len: usize = diff_ab.iter().map(|t| t.chars().count()).sum::<usize>() + diff_ab.len().saturating_sub(1);
    let ba_len: usize = diff_ba.iter().map(|t| t.chars().count()).sum::<usize>() + diff_ba.len().saturating_sub(1);
    let sect_len: usize = intersect.iter().map(|t| t.chars().count()).sum::<usize>() + intersect.len().saturating_sub(1);
    
    // String lengths for ratio calculations
    let sect_ab_len = sect_len + if sect_len != 0 && ab_len != 0 { 1 } else { 0 } + ab_len;
    let sect_ba_len = sect_len + if sect_len != 0 && ba_len != 0 { 1 } else { 0 } + ba_len;
    
    // Calculate distance between diff strings - now using direct byte comparison
    let diff_ab_joined = join_tokens(&diff_ab);
    let diff_ba_joined = join_tokens(&diff_ba);
    
    // Direct byte-level LCS for ASCII
    let dist = if diff_ab_joined.is_ascii() && diff_ba_joined.is_ascii() {
        let lcs = crate::lcs_core::lcs_bitparallel_multiblock(diff_ab_joined.as_bytes(), diff_ba_joined.as_bytes());
        ab_len + ba_len - 2 * lcs
    } else {
        indel_dist(&diff_ab_joined, &diff_ba_joined)
    };
    
    let mut result = norm_distance(dist, sect_ab_len + sect_ba_len, score_cutoff);
    
    // Exit early if no intersection
    if sect_len == 0 {
        return result;
    }
    
    // Mathematical shortcuts - avoid extra string operations
    let sect_ab_dist = if sect_len != 0 && ab_len != 0 { 1 } else { 0 } + ab_len;
    let sect_ab_ratio = norm_distance(sect_ab_dist, sect_len + sect_ab_len, score_cutoff);
    
    let sect_ba_dist = if sect_len != 0 && ba_len != 0 { 1 } else { 0 } + ba_len;
    let sect_ba_ratio = norm_distance(sect_ba_dist, sect_len + sect_ba_len, score_cutoff);
    
    result = result.max(sect_ab_ratio).max(sect_ba_ratio);
    result
}

/// Token ratio - Ultra optimized with fully inlined set calculation
#[pyfunction]
#[pyo3(signature = (s1, s2, *, score_cutoff=None))]
pub fn token_ratio(s1: &str, s2: &str, score_cutoff: Option<f64>) -> f64 {
    if s1 == s2 { return 100.0; }
    if s1.is_empty() && s2.is_empty() { return 100.0; }
    if s1.is_empty() || s2.is_empty() { return 0.0; }
    
    // Tokenize once and reuse for both set and sort
    let mut tokens1 = tokenize(s1);
    let mut tokens2 = tokenize(s2);
    
    if tokens1.is_empty() && tokens2.is_empty() { return 100.0; }
    if tokens1.is_empty() || tokens2.is_empty() { return 0.0; }
    
    // Quick set equality check
    let set1: AHashSet<&str> = tokens1.iter().copied().collect();
    let set2: AHashSet<&str> = tokens2.iter().copied().collect();
    if set1 == set2 { return 100.0; }
    
    // Calculate differences inline (reuse for set_ratio)
    let mut diff_ab: SmallVec<[&str; 16]> = SmallVec::new();
    let mut diff_ba: SmallVec<[&str; 16]> = SmallVec::new();
    let mut intersect: SmallVec<[&str; 16]> = SmallVec::new();
    
    for &t in set1.iter() {
        if set2.contains(t) {
            intersect.push(t);
        } else {
            diff_ab.push(t);
        }
    }
    for &t in set2.iter() {
        if !set1.contains(t) {
            diff_ba.push(t);
        }
    }
    
    // Early exit: one is subset of other
    if !intersect.is_empty() && (diff_ab.is_empty() || diff_ba.is_empty()) {
        return 100.0;
    }
    
    // Calculate token_sort_ratio inline
    tokens1.sort_unstable();
    tokens2.sort_unstable();
    
    let sort_result = if tokens1 == tokens2 {
        100.0
    } else {
        let sorted1 = join_tokens(&tokens1);
        let sorted2 = join_tokens(&tokens2);
        ratio_internal(&sorted1, &sorted2)
    };
    
    // Calculate token_set_ratio inline (using already computed differences)
    diff_ab.sort_unstable();
    diff_ba.sort_unstable();
    
    let ab_len: usize = diff_ab.iter().map(|t| t.chars().count()).sum::<usize>() + diff_ab.len().saturating_sub(1);
    let ba_len: usize = diff_ba.iter().map(|t| t.chars().count()).sum::<usize>() + diff_ba.len().saturating_sub(1);
    let sect_len: usize = intersect.iter().map(|t| t.chars().count()).sum::<usize>() + intersect.len().saturating_sub(1);
    
    let sect_ab_len = sect_len + if sect_len != 0 && ab_len != 0 { 1 } else { 0 } + ab_len;
    let sect_ba_len = sect_len + if sect_len != 0 && ba_len != 0 { 1 } else { 0 } + ba_len;
    
    let set_result = if sect_len == 0 {
        // Only compare diff strings
        let diff_ab_joined = join_tokens(&diff_ab);
        let diff_ba_joined = join_tokens(&diff_ba);
        let dist = if diff_ab_joined.is_ascii() && diff_ba_joined.is_ascii() {
            let lcs = crate::lcs_core::lcs_bitparallel_multiblock(diff_ab_joined.as_bytes(), diff_ba_joined.as_bytes());
            ab_len + ba_len - 2 * lcs
        } else {
            indel_dist(&diff_ab_joined, &diff_ba_joined)
        };
        norm_distance(dist, sect_ab_len + sect_ba_len, 0.0)
    } else {
        // Calculate full set ratio with mathematical shortcuts
        let diff_ab_joined = join_tokens(&diff_ab);
        let diff_ba_joined = join_tokens(&diff_ba);
        let dist = if diff_ab_joined.is_ascii() && diff_ba_joined.is_ascii() {
            let lcs = crate::lcs_core::lcs_bitparallel_multiblock(diff_ab_joined.as_bytes(), diff_ba_joined.as_bytes());
            ab_len + ba_len - 2 * lcs
        } else {
            indel_dist(&diff_ab_joined, &diff_ba_joined)
        };
        
        let r1 = norm_distance(dist, sect_ab_len + sect_ba_len, 0.0);
        let sect_ab_dist = if sect_len != 0 && ab_len != 0 { 1 } else { 0 } + ab_len;
        let r2 = norm_distance(sect_ab_dist, sect_len + sect_ab_len, 0.0);
        let sect_ba_dist = if sect_len != 0 && ba_len != 0 { 1 } else { 0 } + ba_len;
        let r3 = norm_distance(sect_ba_dist, sect_len + sect_ba_len, 0.0);
        r1.max(r2).max(r3)
    };
    
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
    pairs.par_iter().map(|(s1, s2)| token_sort_ratio(s1, s2, score_cutoff)).collect()
}

#[pyfunction]
#[pyo3(signature = (pairs, *, score_cutoff=None))]
pub fn token_set_ratio_batch(pairs: Vec<(String, String)>, score_cutoff: Option<f64>) -> Vec<f64> {
    pairs.par_iter().map(|(s1, s2)| token_set_ratio(s1, s2, score_cutoff)).collect()
}

#[pyfunction]
#[pyo3(signature = (pairs, *, score_cutoff=None))]
pub fn token_ratio_batch(pairs: Vec<(String, String)>, score_cutoff: Option<f64>) -> Vec<f64> {
    pairs.par_iter().map(|(s1, s2)| token_ratio(s1, s2, score_cutoff)).collect()
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
    fn test_token_set_ratio_subset() {
        // One is subset of other - should return 100
        let r = token_set_ratio("fuzzy was a bear", "fuzzy fuzzy was a bear", None);
        assert!((r - 100.0).abs() < 0.01);
    }
    
    #[test]
    fn test_token_set_ratio_single_words() {
        let r = token_set_ratio("kitten", "sitting", None);
        assert!((r - 61.54).abs() < 0.1);
    }
}
