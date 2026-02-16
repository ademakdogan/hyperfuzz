//! Set-based similarity algorithms
//!
//! Provides token/word similarity metrics using multiset (bag) semantics:
//! - Jaccard Similarity
//! - Sørensen-Dice Coefficient
//! - Tversky Index
//! - Overlap Coefficient
//!
//! TextDistance compatible: uses Counter (multiset) not Set

use pyo3::prelude::*;
use rayon::prelude::*;
use ahash::AHashMap;

/// Tokenize a string into word counts (multiset/bag of words).
#[inline(always)]
fn tokenize_counter(s: &str) -> AHashMap<&str, usize> {
    let mut counter: AHashMap<&str, usize> = AHashMap::new();
    for word in s.split_whitespace() {
        *counter.entry(word).or_insert(0) += 1;
    }
    counter
}

/// Calculate intersection count (minimum of each element's count)
#[inline(always)]
fn intersection_count(c1: &AHashMap<&str, usize>, c2: &AHashMap<&str, usize>) -> usize {
    let mut count = 0;
    for (key, &val1) in c1 {
        if let Some(&val2) = c2.get(key) {
            count += val1.min(val2);
        }
    }
    count
}

/// Calculate union count (maximum of each element's count)
#[inline(always)]
fn union_count(c1: &AHashMap<&str, usize>, c2: &AHashMap<&str, usize>) -> usize {
    let mut merged = c1.clone();
    for (key, &val2) in c2 {
        merged.entry(key)
            .and_modify(|v| *v = (*v).max(val2))
            .or_insert(val2);
    }
    merged.values().sum()
}

/// Calculate total count of all elements
#[inline(always)]
fn total_count(c: &AHashMap<&str, usize>) -> usize {
    c.values().sum()
}

// ============ Jaccard Similarity ============
// |A ∩ B| / |A ∪ B| using multiset semantics

/// Calculate Jaccard similarity between two strings.
/// Uses multiset (Counter) semantics for TextDistance compatibility.
#[pyfunction]
#[pyo3(signature = (s1, s2, *, score_cutoff=None))]
pub fn jaccard_similarity(s1: &str, s2: &str, score_cutoff: Option<f64>) -> f64 {
    let counter1 = tokenize_counter(s1);
    let counter2 = tokenize_counter(s2);
    
    if counter1.is_empty() && counter2.is_empty() {
        return 1.0;
    }
    
    let intersection = intersection_count(&counter1, &counter2);
    let union = union_count(&counter1, &counter2);
    
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
    let counter1 = tokenize_counter(s1);
    let counter2 = tokenize_counter(s2);
    
    if counter1.is_empty() && counter2.is_empty() {
        return 1.0;
    }
    
    let intersection = intersection_count(&counter1, &counter2);
    let total = total_count(&counter1) + total_count(&counter2);
    
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
    let counter1 = tokenize_counter(s1);
    let counter2 = tokenize_counter(s2);
    
    if counter1.is_empty() && counter2.is_empty() {
        return 1.0;
    }
    
    let intersection = intersection_count(&counter1, &counter2) as f64;
    
    // Calculate differences: |A - B| and |B - A|
    let total1 = total_count(&counter1) as f64;
    let total2 = total_count(&counter2) as f64;
    let diff1 = total1 - intersection;  // |A - B|
    let diff2 = total2 - intersection;  // |B - A|
    
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
    let counter1 = tokenize_counter(s1);
    let counter2 = tokenize_counter(s2);
    
    if counter1.is_empty() && counter2.is_empty() {
        return 1.0;
    }
    
    let intersection = intersection_count(&counter1, &counter2);
    let min_size = total_count(&counter1).min(total_count(&counter2));
    
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
    fn test_jaccard_with_duplicates() {
        // "the the" vs "the the" -> intersection=2, union=2 -> 1.0
        let j = jaccard_similarity("the the", "the the", None);
        assert!((j - 1.0).abs() < 0.01);
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
