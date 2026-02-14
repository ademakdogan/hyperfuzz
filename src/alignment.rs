//! Sequence alignment algorithms
//!
//! Provides biological-inspired sequence alignment metrics:
//! - Smith-Waterman (local alignment)
//! - Needleman-Wunsch (global alignment)
//!
//! Default scoring matches TextDistance:
//! - match=1, mismatch=0 (identity function), gap_cost=1

use pyo3::prelude::*;
use rayon::prelude::*;

// ============ Smith-Waterman (Local Alignment) ============

/// Calculate Smith-Waterman local alignment score.
/// 
/// Finds the best local alignment between two strings.
/// Default scoring: match=1, mismatch=0, gap_cost=1 (TextDistance compatible)
#[inline(always)]
fn smith_waterman_internal(s1: &str, s2: &str, match_score: f64, mismatch_score: f64, gap_cost: f64) -> f64 {
    let s1_chars: Vec<char> = s1.chars().collect();
    let s2_chars: Vec<char> = s2.chars().collect();

    let m = s1_chars.len();
    let n = s2_chars.len();

    if m == 0 || n == 0 {
        return 0.0;
    }

    let mut matrix: Vec<Vec<f64>> = vec![vec![0.0; n + 1]; m + 1];
    let mut max_score = 0.0;

    for i in 1..=m {
        for j in 1..=n {
            let match_val = if s1_chars[i - 1] == s2_chars[j - 1] {
                match_score
            } else {
                mismatch_score
            };

            let diag = matrix[i - 1][j - 1] + match_val;
            let up = matrix[i - 1][j] - gap_cost;
            let left = matrix[i][j - 1] - gap_cost;

            let score = diag.max(up).max(left).max(0.0);
            matrix[i][j] = score;
            if score > max_score {
                max_score = score;
            }
        }
    }

    max_score
}

/// Calculate Smith-Waterman similarity score.
#[pyfunction]
#[pyo3(signature = (s1, s2, *, match_score=1.0, mismatch_score=0.0, gap_cost=1.0, score_cutoff=None))]
pub fn smith_waterman_score(
    s1: &str,
    s2: &str,
    match_score: f64,
    mismatch_score: f64,
    gap_cost: f64,
    score_cutoff: Option<f64>,
) -> f64 {
    let score = smith_waterman_internal(s1, s2, match_score, mismatch_score, gap_cost);
    
    match score_cutoff {
        Some(cutoff) if score < cutoff => 0.0,
        _ => score,
    }
}

/// Calculate normalized Smith-Waterman similarity (0.0 to 1.0).
/// 
/// Normalization: score / min(len(s1), len(s2)) (TextDistance compatible)
#[pyfunction]
#[pyo3(signature = (s1, s2, *, match_score=1.0, mismatch_score=0.0, gap_cost=1.0, score_cutoff=None))]
pub fn smith_waterman_normalized_similarity(
    s1: &str,
    s2: &str,
    match_score: f64,
    mismatch_score: f64,
    gap_cost: f64,
    score_cutoff: Option<f64>,
) -> f64 {
    let len1 = s1.chars().count();
    let len2 = s2.chars().count();
    
    if len1 == 0 && len2 == 0 {
        return 1.0;
    }
    if len1 == 0 || len2 == 0 {
        return 0.0;
    }
    
    let score = smith_waterman_internal(s1, s2, match_score, mismatch_score, gap_cost);
    
    // TextDistance: maximum = min(len(s1), len(s2))
    let maximum = (len1.min(len2) as f64) * match_score;
    
    let result = if maximum <= 0.0 {
        0.0
    } else {
        score / maximum
    };
    
    let result = result.clamp(0.0, 1.0);
    
    match score_cutoff {
        Some(cutoff) if result < cutoff => 0.0,
        _ => result,
    }
}

#[pyfunction]
#[pyo3(signature = (pairs, *, match_score=1.0, mismatch_score=0.0, gap_cost=1.0, score_cutoff=None))]
pub fn smith_waterman_score_batch(
    pairs: Vec<(String, String)>,
    match_score: f64,
    mismatch_score: f64,
    gap_cost: f64,
    score_cutoff: Option<f64>,
) -> Vec<f64> {
    pairs
        .par_iter()
        .map(|(s1, s2)| smith_waterman_score(s1, s2, match_score, mismatch_score, gap_cost, score_cutoff))
        .collect()
}

// ============ Needleman-Wunsch (Global Alignment) ============

/// Calculate Needleman-Wunsch global alignment score.
/// 
/// Default scoring: match=1, mismatch=0, gap_cost=1 (TextDistance compatible)
#[inline(always)]
fn needleman_wunsch_internal(s1: &str, s2: &str, match_score: f64, mismatch_score: f64, gap_cost: f64) -> f64 {
    let s1_chars: Vec<char> = s1.chars().collect();
    let s2_chars: Vec<char> = s2.chars().collect();

    let m = s1_chars.len();
    let n = s2_chars.len();

    if m == 0 {
        return -(n as f64) * gap_cost;
    }
    if n == 0 {
        return -(m as f64) * gap_cost;
    }

    let mut matrix: Vec<Vec<f64>> = vec![vec![0.0; n + 1]; m + 1];

    // Initialize first row and column
    for i in 0..=m {
        matrix[i][0] = -(i as f64) * gap_cost;
    }
    for j in 0..=n {
        matrix[0][j] = -(j as f64) * gap_cost;
    }

    for i in 1..=m {
        for j in 1..=n {
            let match_val = if s1_chars[i - 1] == s2_chars[j - 1] {
                match_score
            } else {
                mismatch_score
            };

            let diag = matrix[i - 1][j - 1] + match_val;
            let up = matrix[i - 1][j] - gap_cost;
            let left = matrix[i][j - 1] - gap_cost;

            matrix[i][j] = diag.max(up).max(left);
        }
    }

    matrix[m][n]
}

/// Calculate Needleman-Wunsch global alignment score.
#[pyfunction]
#[pyo3(signature = (s1, s2, *, match_score=1.0, mismatch_score=0.0, gap_cost=1.0, score_cutoff=None))]
pub fn needleman_wunsch_score(
    s1: &str,
    s2: &str,
    match_score: f64,
    mismatch_score: f64,
    gap_cost: f64,
    score_cutoff: Option<f64>,
) -> f64 {
    let score = needleman_wunsch_internal(s1, s2, match_score, mismatch_score, gap_cost);
    
    match score_cutoff {
        Some(cutoff) if score < cutoff => f64::MIN,
        _ => score,
    }
}

/// Calculate normalized Needleman-Wunsch similarity (0.0 to 1.0).
/// 
/// TextDistance formula: (score - minimum) / (maximum * 2)
/// where minimum = -max(len(s1), len(s2)) * gap_cost
///       maximum = max(len(s1), len(s2)) * match_score
#[pyfunction]
#[pyo3(signature = (s1, s2, *, match_score=1.0, mismatch_score=0.0, gap_cost=1.0, score_cutoff=None))]
pub fn needleman_wunsch_normalized_similarity(
    s1: &str,
    s2: &str,
    match_score: f64,
    mismatch_score: f64,
    gap_cost: f64,
    score_cutoff: Option<f64>,
) -> f64 {
    let len1 = s1.chars().count();
    let len2 = s2.chars().count();
    let max_len = len1.max(len2);
    
    if max_len == 0 {
        return 1.0;
    }
    
    let score = needleman_wunsch_internal(s1, s2, match_score, mismatch_score, gap_cost);
    
    // TextDistance normalization formula
    let minimum = -(max_len as f64) * gap_cost;
    let maximum = (max_len as f64) * match_score;
    
    let result = if maximum == 0.0 {
        1.0
    } else {
        (score - minimum) / (maximum * 2.0)
    };
    
    let result = result.clamp(0.0, 1.0);
    
    match score_cutoff {
        Some(cutoff) if result < cutoff => 0.0,
        _ => result,
    }
}

#[pyfunction]
#[pyo3(signature = (pairs, *, match_score=1.0, mismatch_score=0.0, gap_cost=1.0, score_cutoff=None))]
pub fn needleman_wunsch_score_batch(
    pairs: Vec<(String, String)>,
    match_score: f64,
    mismatch_score: f64,
    gap_cost: f64,
    score_cutoff: Option<f64>,
) -> Vec<f64> {
    pairs
        .par_iter()
        .map(|(s1, s2)| needleman_wunsch_score(s1, s2, match_score, mismatch_score, gap_cost, score_cutoff))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_smith_waterman() {
        let score = smith_waterman_internal("ACACACTA", "AGCACACA", 1.0, 0.0, 1.0);
        assert!(score > 0.0);
    }

    #[test]
    fn test_needleman_wunsch() {
        let score = needleman_wunsch_internal("GATTACA", "GCATGCU", 1.0, 0.0, 1.0);
        // Should have a reasonable score
        assert!(score != 0.0);
    }

    #[test]
    fn test_identical() {
        let sw = smith_waterman_normalized_similarity("hello", "hello", 1.0, 0.0, 1.0, None);
        assert!((sw - 1.0).abs() < 0.01);
        
        let nw = needleman_wunsch_normalized_similarity("hello", "hello", 1.0, 0.0, 1.0, None);
        assert!((nw - 1.0).abs() < 0.01);
    }
    
    #[test]
    fn test_empty() {
        let sw = smith_waterman_normalized_similarity("", "", 1.0, 0.0, 1.0, None);
        assert!((sw - 1.0).abs() < 0.01);
        
        let nw = needleman_wunsch_normalized_similarity("", "", 1.0, 0.0, 1.0, None);
        assert!((nw - 1.0).abs() < 0.01);
    }
}
