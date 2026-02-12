//! Damerau-Levenshtein distance - Optimized with Zhao's algorithm
//!
//! Uses linear space Zhao's algorithm with AHashMap for fast char lookups.
//! Reference: "A Novel Two-Row Algorithm for Optimal String Alignment" (Zhao, 2022)

use pyo3::prelude::*;
use rayon::prelude::*;
use smallvec::SmallVec;
use ahash::AHashMap;
use std::cell::RefCell;

// Thread-local buffers to avoid allocation
thread_local! {
    static DL_ROW_1: RefCell<Vec<usize>> = RefCell::new(Vec::with_capacity(128));
    static DL_ROW_2: RefCell<Vec<usize>> = RefCell::new(Vec::with_capacity(128));
    static DL_FR: RefCell<Vec<usize>> = RefCell::new(Vec::with_capacity(128));
}

/// Zhao's algorithm for Damerau-Levenshtein with thread-local buffers
#[inline(always)]
fn damerau_levenshtein_zhao(s1: &[u8], s2: &[u8]) -> usize {
    let len1 = s1.len();
    let len2 = s2.len();
    
    if len1 == 0 { return len2; }
    if len2 == 0 { return len1; }
    
    let max_val = len1.max(len2) + 1;
    
    DL_ROW_1.with(|row1_cell| {
        DL_ROW_2.with(|row2_cell| {
            DL_FR.with(|fr_cell| {
                let mut r = row1_cell.borrow_mut();
                let mut r1 = row2_cell.borrow_mut();
                let mut fr = fr_cell.borrow_mut();
                
                let size = len2 + 2;
                
                r.clear();
                r.resize(size, max_val);
                r1.clear();
                r1.resize(size, max_val);
                fr.clear();
                fr.resize(size, max_val);
                
                // Initialize R: [0, 1, 2, ..., len2, max_val]
                for j in 0..=len2 {
                    r[j] = j;
                }
                r[len2 + 1] = max_val;
                
                let mut last_row_id: AHashMap<u8, usize> = AHashMap::with_capacity(64);
                
                for i in 1..=len1 {
                    // Swap R and R1
                    std::mem::swap(&mut *r, &mut *r1);
                    
                    let mut last_col_id: usize = 0;  // Use 0 as sentinel (1-indexed internally)
                    let mut last_i2l1 = r[0];
                    r[0] = i;
                    let mut t = max_val;
                    
                    for j in 1..=len2 {
                        let c1 = s1[i - 1];
                        let c2 = s2[j - 1];
                        
                        let diag = r1[j - 1] + if c1 != c2 { 1 } else { 0 };
                        let left = r[j - 1] + 1;
                        let up = r1[j] + 1;
                        let mut temp = diag.min(left).min(up);
                        
                        if c1 == c2 {
                            last_col_id = j;
                            if j >= 2 {
                                fr[j] = r1[j - 2];
                            }
                            t = last_i2l1;
                        } else {
                            let k = *last_row_id.get(&c2).unwrap_or(&0);
                            let l = last_col_id;
                            
                            if l > 0 && (j - l) == 1 && k > 0 {
                                let transpose = fr[j] + (i - k);
                                temp = temp.min(transpose);
                            } else if k > 0 && (i - k) == 1 && l > 0 {
                                let transpose = t + (j - l);
                                temp = temp.min(transpose);
                            }
                        }
                        
                        last_i2l1 = r[j];
                        r[j] = temp;
                    }
                    
                    last_row_id.insert(s1[i - 1], i);
                }
                
                r[len2]
            })
        })
    })
}

/// Unicode version of Damerau-Levenshtein
#[inline(always)]
fn damerau_levenshtein_unicode(s1: &[char], s2: &[char]) -> usize {
    let len1 = s1.len();
    let len2 = s2.len();
    
    if len1 == 0 { return len2; }
    if len2 == 0 { return len1; }
    
    let max_val = len1.max(len2) + 1;
    let size = len2 + 2;
    
    let mut r: SmallVec<[usize; 128]> = SmallVec::from_elem(max_val, size);
    let mut r1: SmallVec<[usize; 128]> = SmallVec::from_elem(max_val, size);
    let mut fr: SmallVec<[usize; 128]> = SmallVec::from_elem(max_val, size);
    
    for j in 0..=len2 {
        r[j] = j;
    }
    
    let mut last_row_id: AHashMap<char, usize> = AHashMap::with_capacity(64);
    
    for i in 1..=len1 {
        std::mem::swap(&mut r, &mut r1);
        
        let mut last_col_id: usize = 0;
        let mut last_i2l1 = r[0];
        r[0] = i;
        let mut t = max_val;
        
        for j in 1..=len2 {
            let c1 = s1[i - 1];
            let c2 = s2[j - 1];
            
            let diag = r1[j - 1] + if c1 != c2 { 1 } else { 0 };
            let left = r[j - 1] + 1;
            let up = r1[j] + 1;
            let mut temp = diag.min(left).min(up);
            
            if c1 == c2 {
                last_col_id = j;
                if j >= 2 {
                    fr[j] = r1[j - 2];
                }
                t = last_i2l1;
            } else {
                let k = *last_row_id.get(&c2).unwrap_or(&0);
                let l = last_col_id;
                
                if l > 0 && (j - l) == 1 && k > 0 {
                    let transpose = fr[j] + (i - k);
                    temp = temp.min(transpose);
                } else if k > 0 && (i - k) == 1 && l > 0 {
                    let transpose = t + (j - l);
                    temp = temp.min(transpose);
                }
            }
            
            last_i2l1 = r[j];
            r[j] = temp;
        }
        
        last_row_id.insert(s1[i - 1], i);
    }
    
    r[len2]
}

/// Main Damerau-Levenshtein implementation
#[inline(always)]
fn damerau_levenshtein_internal(s1: &str, s2: &str) -> usize {
    if s1 == s2 { return 0; }
    
    // ASCII fast path
    if s1.is_ascii() && s2.is_ascii() {
        return damerau_levenshtein_zhao(s1.as_bytes(), s2.as_bytes());
    }
    
    // Unicode path
    let s1_chars: SmallVec<[char; 64]> = s1.chars().collect();
    let s2_chars: SmallVec<[char; 64]> = s2.chars().collect();
    damerau_levenshtein_unicode(&s1_chars, &s2_chars)
}

/// Calculate Damerau-Levenshtein distance.
#[pyfunction]
#[pyo3(signature = (s1, s2, *, score_cutoff=None))]
pub fn damerau_levenshtein_distance(s1: &str, s2: &str, score_cutoff: Option<usize>) -> usize {
    let dist = damerau_levenshtein_internal(s1, s2);
    
    match score_cutoff {
        Some(cutoff) if dist > cutoff => cutoff + 1,
        _ => dist,
    }
}

/// Calculate Damerau-Levenshtein similarity.
#[pyfunction]
#[pyo3(signature = (s1, s2, *, score_cutoff=None))]
pub fn damerau_levenshtein_similarity(s1: &str, s2: &str, score_cutoff: Option<usize>) -> usize {
    let len1 = if s1.is_ascii() { s1.len() } else { s1.chars().count() };
    let len2 = if s2.is_ascii() { s2.len() } else { s2.chars().count() };
    let max_len = len1.max(len2);
    let dist = damerau_levenshtein_internal(s1, s2);
    let sim = max_len.saturating_sub(dist);

    match score_cutoff {
        Some(cutoff) if sim < cutoff => 0,
        _ => sim,
    }
}

/// Calculate normalized Damerau-Levenshtein distance.
#[pyfunction]
#[pyo3(signature = (s1, s2, *, score_cutoff=None))]
pub fn damerau_levenshtein_normalized_distance(s1: &str, s2: &str, score_cutoff: Option<f64>) -> f64 {
    let len1 = if s1.is_ascii() { s1.len() } else { s1.chars().count() };
    let len2 = if s2.is_ascii() { s2.len() } else { s2.chars().count() };
    let max_len = len1.max(len2);

    if max_len == 0 { return 0.0; }

    let dist = damerau_levenshtein_internal(s1, s2);
    let norm_dist = dist as f64 / max_len as f64;

    match score_cutoff {
        Some(cutoff) if norm_dist > cutoff => 1.0,
        _ => norm_dist,
    }
}

/// Calculate normalized Damerau-Levenshtein similarity.
#[pyfunction]
#[pyo3(signature = (s1, s2, *, score_cutoff=None))]
pub fn damerau_levenshtein_normalized_similarity(s1: &str, s2: &str, score_cutoff: Option<f64>) -> f64 {
    let norm_dist = damerau_levenshtein_normalized_distance(s1, s2, None);
    let norm_sim = 1.0 - norm_dist;

    match score_cutoff {
        Some(cutoff) if norm_sim < cutoff => 0.0,
        _ => norm_sim,
    }
}

// ============ Batch Operations ============

#[pyfunction]
#[pyo3(signature = (pairs, *, score_cutoff=None))]
pub fn damerau_levenshtein_distance_batch(pairs: Vec<(String, String)>, score_cutoff: Option<usize>) -> Vec<usize> {
    pairs.par_iter().map(|(s1, s2)| damerau_levenshtein_distance(s1, s2, score_cutoff)).collect()
}

#[pyfunction]
#[pyo3(signature = (pairs, *, score_cutoff=None))]
pub fn damerau_levenshtein_normalized_similarity_batch(pairs: Vec<(String, String)>, score_cutoff: Option<f64>) -> Vec<f64> {
    pairs.par_iter().map(|(s1, s2)| damerau_levenshtein_normalized_similarity(s1, s2, score_cutoff)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_damerau_levenshtein() {
        assert_eq!(damerau_levenshtein_internal("ca", "abc"), 2);
        assert_eq!(damerau_levenshtein_internal("abc", "abc"), 0);
    }

    #[test]
    fn test_transposition() {
        assert_eq!(damerau_levenshtein_internal("ab", "ba"), 1);
    }
}
