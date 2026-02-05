//! Set-based similarity algorithms
//!
//! Provides token/word set similarity metrics:
//! - Jaccard Similarity
//! - Sørensen-Dice Coefficient
//! - Tversky Index
//! - Overlap Coefficient

use pyo3::prelude::*;
use rayon::prelude::*;
use ahash::AHashSet;

/// Tokenize a string into words.
#[inline(always)]
fn tokenize(s: &str) -> AHashSet<&str> {
    s.split_whitespace().collect()
}

// ============ Jaccard Similarity ============
// |A ∩ B| / |A ∪ B|

/// Calculate Jaccard similarity between two strings.
/// Based on word/token sets.
#[pyfunction]
#[pyo3(signature = (s1, s2, *, score_cutoff=None))]
pub fn jaccard_similarity(s1: &str, s2: &str, score_cutoff: Option<f64>) -> f64 {
    let set1 = tokenize(s1);
    let set2 = tokenize(s2);
    
    if set1.is_empty() && set2.is_empty() {
        return 1.0;
    }
    
    let intersection = set1.intersection(&set2).count();
    let union = set1.union(&set2).count();
    
    let result = if union == 0 {
        0.0
    } else {
        intersection as f64 / union as f64
    };
    
    match score_cutoff {
        Some(cutoff) if result < cutoff => 0.0,
        _ => result,
    }
}

/// Calculate Jaccard distance (1 - similarity).
#[pyfunction]
#[pyo3(signature = (s1, s2, *, score_cutoff=None))]
pub fn jaccard_distance(s1: &str, s2: &str, score_cutoff: Option<f64>) -> f64 {
    let sim = jaccard_similarity(s1, s2, None);
    let dist = 1.0 - sim;
    
    match score_cutoff {
        Some(cutoff) if dist > cutoff => 1.0,
        _ => dist,
    }
}

#[pyfunction]
#[pyo3(signature = (pairs, *, score_cutoff=None))]
pub fn jaccard_similarity_batch(pairs: Vec<(String, String)>, score_cutoff: Option<f64>) -> Vec<f64> {
    pairs
        .par_iter()
        .map(|(s1, s2)| jaccard_similarity(s1, s2, score_cutoff))
        .collect()
}

// ============ Sørensen-Dice Coefficient ============
// 2 * |A ∩ B| / (|A| + |B|)

/// Calculate Sørensen-Dice coefficient.
#[pyfunction]
#[pyo3(signature = (s1, s2, *, score_cutoff=None))]
pub fn sorensen_dice_similarity(s1: &str, s2: &str, score_cutoff: Option<f64>) -> f64 {
    let set1 = tokenize(s1);
    let set2 = tokenize(s2);
    
    if set1.is_empty() && set2.is_empty() {
        return 1.0;
    }
    
    let intersection = set1.intersection(&set2).count();
    let total = set1.len() + set2.len();
    
    let result = if total == 0 {
        0.0
    } else {
        (2.0 * intersection as f64) / total as f64
    };
    
    match score_cutoff {
        Some(cutoff) if result < cutoff => 0.0,
        _ => result,
    }
}

/// Calculate Sørensen-Dice distance.
#[pyfunction]
#[pyo3(signature = (s1, s2, *, score_cutoff=None))]
pub fn sorensen_dice_distance(s1: &str, s2: &str, score_cutoff: Option<f64>) -> f64 {
    let sim = sorensen_dice_similarity(s1, s2, None);
    let dist = 1.0 - sim;
    
    match score_cutoff {
        Some(cutoff) if dist > cutoff => 1.0,
        _ => dist,
    }
}

#[pyfunction]
#[pyo3(signature = (pairs, *, score_cutoff=None))]
pub fn sorensen_dice_similarity_batch(pairs: Vec<(String, String)>, score_cutoff: Option<f64>) -> Vec<f64> {
    pairs
        .par_iter()
        .map(|(s1, s2)| sorensen_dice_similarity(s1, s2, score_cutoff))
        .collect()
}

// ============ Tversky Index ============
// |A ∩ B| / (|A ∩ B| + α|A - B| + β|B - A|)

/// Calculate Tversky index.
/// 
/// Generalization of Jaccard and Dice:
/// - α=1, β=1: Jaccard
/// - α=0.5, β=0.5: Dice
#[pyfunction]
#[pyo3(signature = (s1, s2, *, alpha=1.0, beta=1.0, score_cutoff=None))]
pub fn tversky_similarity(
    s1: &str,
    s2: &str,
    alpha: f64,
    beta: f64,
    score_cutoff: Option<f64>,
) -> f64 {
    let set1 = tokenize(s1);
    let set2 = tokenize(s2);
    
    if set1.is_empty() && set2.is_empty() {
        return 1.0;
    }
    
    let intersection = set1.intersection(&set2).count() as f64;
    let diff1 = set1.difference(&set2).count() as f64;
    let diff2 = set2.difference(&set1).count() as f64;
    
    let denominator = intersection + alpha * diff1 + beta * diff2;
    
    let result = if denominator == 0.0 {
        0.0
    } else {
        intersection / denominator
    };
    
    match score_cutoff {
        Some(cutoff) if result < cutoff => 0.0,
        _ => result,
    }
}

/// Calculate Tversky distance.
#[pyfunction]
#[pyo3(signature = (s1, s2, *, alpha=1.0, beta=1.0, score_cutoff=None))]
pub fn tversky_distance(
    s1: &str,
    s2: &str,
    alpha: f64,
    beta: f64,
    score_cutoff: Option<f64>,
) -> f64 {
    let sim = tversky_similarity(s1, s2, alpha, beta, None);
    let dist = 1.0 - sim;
    
    match score_cutoff {
        Some(cutoff) if dist > cutoff => 1.0,
        _ => dist,
    }
}

#[pyfunction]
#[pyo3(signature = (pairs, *, alpha=1.0, beta=1.0, score_cutoff=None))]
pub fn tversky_similarity_batch(
    pairs: Vec<(String, String)>,
    alpha: f64,
    beta: f64,
    score_cutoff: Option<f64>,
) -> Vec<f64> {
    pairs
        .par_iter()
        .map(|(s1, s2)| tversky_similarity(s1, s2, alpha, beta, score_cutoff))
        .collect()
}

// ============ Overlap Coefficient ============
// |A ∩ B| / min(|A|, |B|)

/// Calculate Overlap coefficient.
/// Measures if one set is a subset of another.
#[pyfunction]
#[pyo3(signature = (s1, s2, *, score_cutoff=None))]
pub fn overlap_similarity(s1: &str, s2: &str, score_cutoff: Option<f64>) -> f64 {
    let set1 = tokenize(s1);
    let set2 = tokenize(s2);
    
    if set1.is_empty() && set2.is_empty() {
        return 1.0;
    }
    
    let intersection = set1.intersection(&set2).count();
    let min_size = set1.len().min(set2.len());
    
    let result = if min_size == 0 {
        0.0
    } else {
        intersection as f64 / min_size as f64
    };
    
    match score_cutoff {
        Some(cutoff) if result < cutoff => 0.0,
        _ => result,
    }
}

/// Calculate Overlap distance.
#[pyfunction]
#[pyo3(signature = (s1, s2, *, score_cutoff=None))]
pub fn overlap_distance(s1: &str, s2: &str, score_cutoff: Option<f64>) -> f64 {
    let sim = overlap_similarity(s1, s2, None);
    let dist = 1.0 - sim;
    
    match score_cutoff {
        Some(cutoff) if dist > cutoff => 1.0,
        _ => dist,
    }
}

#[pyfunction]
#[pyo3(signature = (pairs, *, score_cutoff=None))]
pub fn overlap_similarity_batch(pairs: Vec<(String, String)>, score_cutoff: Option<f64>) -> Vec<f64> {
    pairs
        .par_iter()
        .map(|(s1, s2)| overlap_similarity(s1, s2, score_cutoff))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jaccard() {
        // "a b c" vs "b c d" -> intersection=2, union=4 -> 0.5
        let j = jaccard_similarity("a b c", "b c d", None);
        assert!((j - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_dice() {
        // "a b c" vs "b c d" -> intersection=2, total=6 -> 2*2/6 = 0.667
        let d = sorensen_dice_similarity("a b c", "b c d", None);
        assert!((d - 0.667).abs() < 0.01);
    }

    #[test]
    fn test_overlap() {
        // "a b" vs "a b c d" -> intersection=2, min=2 -> 1.0
        let o = overlap_similarity("a b", "a b c d", None);
        assert!((o - 1.0).abs() < 0.01);
    }
}
