//! Partial token ratio algorithms - Optimized
//!
//! Combines partial matching with token-based approaches.

use pyo3::prelude::*;
use rayon::prelude::*;
use smallvec::SmallVec;
use ahash::AHashSet;

use super::partial_ratio::partial_ratio_internal;

/// Tokenize a string.
#[inline(always)]
fn tokenize(s: &str) -> SmallVec<[&str; 16]> {
    s.split_whitespace().collect()
}

/// Partial token sort ratio.
#[pyfunction]
#[pyo3(signature = (s1, s2, *, score_cutoff=None))]
pub fn partial_token_sort_ratio(s1: &str, s2: &str, score_cutoff: Option<f64>) -> f64 {
    if s1 == s2 {
        return 100.0;
    }
    
    let mut tokens1 = tokenize(s1);
    let mut tokens2 = tokenize(s2);
    
    tokens1.sort_unstable();
    tokens2.sort_unstable();
    
    let sorted1 = tokens1.join(" ");
    let sorted2 = tokens2.join(" ");
    
    let result = partial_ratio_internal(&sorted1, &sorted2);
    
    match score_cutoff {
        Some(cutoff) if result < cutoff => 0.0,
        _ => result,
    }
}

/// Partial token set ratio.
#[pyfunction]
#[pyo3(signature = (s1, s2, *, score_cutoff=None))]
pub fn partial_token_set_ratio(s1: &str, s2: &str, score_cutoff: Option<f64>) -> f64 {
    if s1 == s2 {
        return 100.0;
    }
    
    let tokens1: AHashSet<&str> = tokenize(s1).into_iter().collect();
    let tokens2: AHashSet<&str> = tokenize(s2).into_iter().collect();
    
    if tokens1.is_empty() && tokens2.is_empty() {
        return 100.0;
    }
    if tokens1.is_empty() || tokens2.is_empty() {
        return 0.0;
    }
    
    let intersection: SmallVec<[&str; 16]> = tokens1.intersection(&tokens2).copied().collect();
    let diff1: SmallVec<[&str; 16]> = tokens1.difference(&tokens2).copied().collect();
    let diff2: SmallVec<[&str; 16]> = tokens2.difference(&tokens1).copied().collect();
    
    let mut sorted_inter = intersection;
    let mut sorted_diff1 = diff1;
    let mut sorted_diff2 = diff2;
    sorted_inter.sort_unstable();
    sorted_diff1.sort_unstable();
    sorted_diff2.sort_unstable();
    
    let inter_str = sorted_inter.join(" ");
    
    let combined1 = if sorted_diff1.is_empty() {
        inter_str.clone()
    } else if inter_str.is_empty() {
        sorted_diff1.join(" ")
    } else {
        format!("{} {}", inter_str, sorted_diff1.join(" "))
    };
    
    let combined2 = if sorted_diff2.is_empty() {
        inter_str.clone()
    } else if inter_str.is_empty() {
        sorted_diff2.join(" ")
    } else {
        format!("{} {}", inter_str, sorted_diff2.join(" "))
    };
    
    let result = if inter_str.is_empty() {
        partial_ratio_internal(&combined1, &combined2)
    } else {
        let r1 = partial_ratio_internal(&inter_str, &combined1);
        let r2 = partial_ratio_internal(&inter_str, &combined2);
        let r3 = partial_ratio_internal(&combined1, &combined2);
        r1.max(r2).max(r3)
    };
    
    match score_cutoff {
        Some(cutoff) if result < cutoff => 0.0,
        _ => result,
    }
}

/// Partial token ratio - max of partial_token_sort and partial_token_set.
#[pyfunction]
#[pyo3(signature = (s1, s2, *, score_cutoff=None))]
pub fn partial_token_ratio(s1: &str, s2: &str, score_cutoff: Option<f64>) -> f64 {
    if s1 == s2 {
        return 100.0;
    }
    
    let sort_result = partial_token_sort_ratio(s1, s2, None);
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
    pairs
        .par_iter()
        .map(|(s1, s2)| partial_token_sort_ratio(s1, s2, score_cutoff))
        .collect()
}

#[pyfunction]
#[pyo3(signature = (pairs, *, score_cutoff=None))]
pub fn partial_token_set_ratio_batch(pairs: Vec<(String, String)>, score_cutoff: Option<f64>) -> Vec<f64> {
    pairs
        .par_iter()
        .map(|(s1, s2)| partial_token_set_ratio(s1, s2, score_cutoff))
        .collect()
}

#[pyfunction]
#[pyo3(signature = (pairs, *, score_cutoff=None))]
pub fn partial_token_ratio_batch(pairs: Vec<(String, String)>, score_cutoff: Option<f64>) -> Vec<f64> {
    pairs
        .par_iter()
        .map(|(s1, s2)| partial_token_ratio(s1, s2, score_cutoff))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_partial_token_sort() {
        let r = partial_token_sort_ratio("fuzzy wuzzy was a bear", "wuzzy fuzzy", None);
        assert!(r > 90.0);
    }
}
