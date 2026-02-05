//! Sequence alignment algorithms
//!
//! Provides biological-inspired sequence alignment metrics:
//! - Smith-Waterman (local alignment)
//! - Needleman-Wunsch (global alignment)

use pyo3::prelude::*;
use rayon::prelude::*;
use std::cmp::max;

// ============ Smith-Waterman (Local Alignment) ============

/// Calculate Smith-Waterman local alignment score.
/// 
/// Finds the best local alignment between two strings.
/// Uses default scoring: match=2, mismatch=-1, gap=-1
#[inline(always)]
fn smith_waterman_internal(s1: &str, s2: &str, match_score: i32, mismatch_penalty: i32, gap_penalty: i32) -> i32 {
    let s1_chars: Vec<char> = s1.chars().collect();
    let s2_chars: Vec<char> = s2.chars().collect();

    let m = s1_chars.len();
    let n = s2_chars.len();

    if m == 0 || n == 0 {
        return 0;
    }

    let mut matrix: Vec<Vec<i32>> = vec![vec![0; n + 1]; m + 1];
    let mut max_score = 0;

    for i in 1..=m {
        for j in 1..=n {
            let match_val = if s1_chars[i - 1] == s2_chars[j - 1] {
                match_score
            } else {
                mismatch_penalty
            };

            let diag = matrix[i - 1][j - 1] + match_val;
            let up = matrix[i - 1][j] + gap_penalty;
            let left = matrix[i][j - 1] + gap_penalty;

            matrix[i][j] = max(0, max(diag, max(up, left)));
            max_score = max(max_score, matrix[i][j]);
        }
    }

    max_score
}

/// Calculate Smith-Waterman similarity score.
#[pyfunction]
#[pyo3(signature = (s1, s2, *, match_score=2, mismatch_penalty=-1, gap_penalty=-1, score_cutoff=None))]
pub fn smith_waterman_score(
    s1: &str,
    s2: &str,
    match_score: i32,
    mismatch_penalty: i32,
    gap_penalty: i32,
    score_cutoff: Option<i32>,
) -> i32 {
    let score = smith_waterman_internal(s1, s2, match_score, mismatch_penalty, gap_penalty);
    
    match score_cutoff {
        Some(cutoff) if score < cutoff => 0,
        _ => score,
    }
}

/// Calculate normalized Smith-Waterman similarity (0.0 to 1.0).
#[pyfunction]
#[pyo3(signature = (s1, s2, *, match_score=2, mismatch_penalty=-1, gap_penalty=-1, score_cutoff=None))]
pub fn smith_waterman_normalized_similarity(
    s1: &str,
    s2: &str,
    match_score: i32,
    mismatch_penalty: i32,
    gap_penalty: i32,
    score_cutoff: Option<f64>,
) -> f64 {
    let len1 = s1.chars().count();
    let len2 = s2.chars().count();
    let min_len = len1.min(len2);
    
    if min_len == 0 {
        return if len1 == len2 { 1.0 } else { 0.0 };
    }
    
    let score = smith_waterman_internal(s1, s2, match_score, mismatch_penalty, gap_penalty);
    let max_possible = (min_len as i32) * match_score;
    
    let result = if max_possible <= 0 {
        0.0
    } else {
        (score as f64) / (max_possible as f64)
    };
    
    let result = result.clamp(0.0, 1.0);
    
    match score_cutoff {
        Some(cutoff) if result < cutoff => 0.0,
        _ => result,
    }
}

#[pyfunction]
#[pyo3(signature = (pairs, *, match_score=2, mismatch_penalty=-1, gap_penalty=-1, score_cutoff=None))]
pub fn smith_waterman_score_batch(
    pairs: Vec<(String, String)>,
    match_score: i32,
    mismatch_penalty: i32,
    gap_penalty: i32,
    score_cutoff: Option<i32>,
) -> Vec<i32> {
    pairs
        .par_iter()
        .map(|(s1, s2)| smith_waterman_score(s1, s2, match_score, mismatch_penalty, gap_penalty, score_cutoff))
        .collect()
}

// ============ Needleman-Wunsch (Global Alignment) ============

/// Calculate Needleman-Wunsch global alignment score.
#[inline(always)]
fn needleman_wunsch_internal(s1: &str, s2: &str, match_score: i32, mismatch_penalty: i32, gap_penalty: i32) -> i32 {
    let s1_chars: Vec<char> = s1.chars().collect();
    let s2_chars: Vec<char> = s2.chars().collect();

    let m = s1_chars.len();
    let n = s2_chars.len();

    if m == 0 {
        return (n as i32) * gap_penalty;
    }
    if n == 0 {
        return (m as i32) * gap_penalty;
    }

    let mut matrix: Vec<Vec<i32>> = vec![vec![0; n + 1]; m + 1];

    // Initialize first row and column
    for i in 0..=m {
        matrix[i][0] = (i as i32) * gap_penalty;
    }
    for j in 0..=n {
        matrix[0][j] = (j as i32) * gap_penalty;
    }

    for i in 1..=m {
        for j in 1..=n {
            let match_val = if s1_chars[i - 1] == s2_chars[j - 1] {
                match_score
            } else {
                mismatch_penalty
            };

            let diag = matrix[i - 1][j - 1] + match_val;
            let up = matrix[i - 1][j] + gap_penalty;
            let left = matrix[i][j - 1] + gap_penalty;

            matrix[i][j] = max(diag, max(up, left));
        }
    }

    matrix[m][n]
}

/// Calculate Needleman-Wunsch global alignment score.
#[pyfunction]
#[pyo3(signature = (s1, s2, *, match_score=2, mismatch_penalty=-1, gap_penalty=-1, score_cutoff=None))]
pub fn needleman_wunsch_score(
    s1: &str,
    s2: &str,
    match_score: i32,
    mismatch_penalty: i32,
    gap_penalty: i32,
    score_cutoff: Option<i32>,
) -> i32 {
    let score = needleman_wunsch_internal(s1, s2, match_score, mismatch_penalty, gap_penalty);
    
    match score_cutoff {
        Some(cutoff) if score < cutoff => i32::MIN,
        _ => score,
    }
}

/// Calculate normalized Needleman-Wunsch similarity (0.0 to 1.0).
#[pyfunction]
#[pyo3(signature = (s1, s2, *, match_score=2, mismatch_penalty=-1, gap_penalty=-1, score_cutoff=None))]
pub fn needleman_wunsch_normalized_similarity(
    s1: &str,
    s2: &str,
    match_score: i32,
    mismatch_penalty: i32,
    gap_penalty: i32,
    score_cutoff: Option<f64>,
) -> f64 {
    let len1 = s1.chars().count();
    let len2 = s2.chars().count();
    let max_len = len1.max(len2);
    
    if max_len == 0 {
        return 1.0;
    }
    
    let score = needleman_wunsch_internal(s1, s2, match_score, mismatch_penalty, gap_penalty);
    
    // Calculate score range
    let max_possible = (max_len as i32) * match_score;
    let min_possible = (max_len as i32) * gap_penalty.min(mismatch_penalty);
    
    let range = (max_possible - min_possible) as f64;
    let result = if range == 0.0 {
        1.0
    } else {
        ((score - min_possible) as f64) / range
    };
    
    let result = result.clamp(0.0, 1.0);
    
    match score_cutoff {
        Some(cutoff) if result < cutoff => 0.0,
        _ => result,
    }
}

#[pyfunction]
#[pyo3(signature = (pairs, *, match_score=2, mismatch_penalty=-1, gap_penalty=-1, score_cutoff=None))]
pub fn needleman_wunsch_score_batch(
    pairs: Vec<(String, String)>,
    match_score: i32,
    mismatch_penalty: i32,
    gap_penalty: i32,
    score_cutoff: Option<i32>,
) -> Vec<i32> {
    pairs
        .par_iter()
        .map(|(s1, s2)| needleman_wunsch_score(s1, s2, match_score, mismatch_penalty, gap_penalty, score_cutoff))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_smith_waterman() {
        let score = smith_waterman_internal("ACACACTA", "AGCACACA", 2, -1, -1);
        assert!(score > 0);
    }

    #[test]
    fn test_needleman_wunsch() {
        let score = needleman_wunsch_internal("GATTACA", "GCATGCU", 1, -1, -1);
        // Should have a reasonable score
        assert!(score != 0);
    }

    #[test]
    fn test_identical() {
        let sw = smith_waterman_normalized_similarity("hello", "hello", 2, -1, -1, None);
        assert!((sw - 1.0).abs() < 0.01);
    }
}
