//! Vector and statistical similarity methods
//!
//! Provides:
//! - Cosine Similarity (word-based by default, optionally n-gram based)
//! - Soft-TFIDF (TF-IDF with fuzzy token matching)

use pyo3::prelude::*;
use rayon::prelude::*;
use ahash::AHashMap;

// ============ Tokenization ============

/// Tokenize string into words (whitespace-separated)
#[inline(always)]
fn tokenize_words(s: &str) -> Vec<String> {
    s.split_whitespace().map(|w| w.to_string()).collect()
}

/// Generate character n-grams from a string.
#[inline(always)]
fn generate_ngrams(s: &str, n: usize) -> Vec<String> {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() < n {
        return vec![s.to_string()];
    }
    
    chars.windows(n)
        .map(|w| w.iter().collect::<String>())
        .collect()
}

/// Build term frequency vector.
#[inline(always)]
fn build_tf_vector(tokens: &[String]) -> AHashMap<String, f64> {
    let mut tf: AHashMap<String, f64> = AHashMap::new();
    for token in tokens {
        *tf.entry(token.clone()).or_insert(0.0) += 1.0;
    }
    tf
}

// ============ Cosine Similarity ============

/// Calculate cosine similarity between two TF vectors.
#[inline(always)]
fn cosine_similarity_from_vectors(tf1: &AHashMap<String, f64>, tf2: &AHashMap<String, f64>) -> f64 {
    let mut dot_product = 0.0;
    let mut mag1 = 0.0;
    let mut mag2 = 0.0;
    
    for (key, val1) in tf1 {
        mag1 += val1 * val1;
        if let Some(val2) = tf2.get(key) {
            dot_product += val1 * val2;
        }
    }
    
    for val2 in tf2.values() {
        mag2 += val2 * val2;
    }
    
    let magnitude = (mag1.sqrt()) * (mag2.sqrt());
    
    if magnitude == 0.0 {
        0.0
    } else {
        dot_product / magnitude
    }
}

/// Calculate cosine similarity between two strings.
/// 
/// When use_words=true (default): Uses word-based tokenization (TextDistance compatible)
/// When use_words=false: Uses character n-grams
#[pyfunction]
#[pyo3(signature = (s1, s2, *, use_words=true, ngram_size=2, score_cutoff=None))]
pub fn cosine_similarity(
    s1: &str, 
    s2: &str, 
    use_words: bool,
    ngram_size: usize, 
    score_cutoff: Option<f64>
) -> f64 {
    let tokens1: Vec<String>;
    let tokens2: Vec<String>;
    
    if use_words {
        tokens1 = tokenize_words(s1);
        tokens2 = tokenize_words(s2);
    } else {
        tokens1 = generate_ngrams(s1, ngram_size);
        tokens2 = generate_ngrams(s2, ngram_size);
    }
    
    if tokens1.is_empty() && tokens2.is_empty() {
        return 1.0;
    }
    if tokens1.is_empty() || tokens2.is_empty() {
        return 0.0;
    }
    
    let tf1 = build_tf_vector(&tokens1);
    let tf2 = build_tf_vector(&tokens2);
    
    let result = cosine_similarity_from_vectors(&tf1, &tf2);
    
    match score_cutoff {
        Some(cutoff) if result < cutoff => 0.0,
        _ => result,
    }
}

/// Calculate cosine distance (1 - similarity).
#[pyfunction]
#[pyo3(signature = (s1, s2, *, use_words=true, ngram_size=2, score_cutoff=None))]
pub fn cosine_distance(
    s1: &str, 
    s2: &str, 
    use_words: bool,
    ngram_size: usize, 
    score_cutoff: Option<f64>
) -> f64 {
    let sim = cosine_similarity(s1, s2, use_words, ngram_size, None);
    let dist = 1.0 - sim;
    
    match score_cutoff {
        Some(cutoff) if dist > cutoff => 1.0,
        _ => dist,
    }
}

#[pyfunction]
#[pyo3(signature = (pairs, *, use_words=true, ngram_size=2, score_cutoff=None))]
pub fn cosine_similarity_batch(
    pairs: Vec<(String, String)>, 
    use_words: bool,
    ngram_size: usize, 
    score_cutoff: Option<f64>
) -> Vec<f64> {
    pairs
        .par_iter()
        .map(|(s1, s2)| cosine_similarity(s1, s2, use_words, ngram_size, score_cutoff))
        .collect()
}

// ============ Soft-TFIDF ============

/// Calculate Jaro-Winkler similarity for token comparison.
#[inline(always)]
fn jaro_winkler_sim(s1: &str, s2: &str) -> f64 {
    let s1_chars: Vec<char> = s1.chars().collect();
    let s2_chars: Vec<char> = s2.chars().collect();

    let len1 = s1_chars.len();
    let len2 = s2_chars.len();

    if len1 == 0 && len2 == 0 {
        return 1.0;
    }
    if len1 == 0 || len2 == 0 {
        return 0.0;
    }
    if s1 == s2 {
        return 1.0;
    }

    let match_distance = std::cmp::max(len1, len2) / 2;
    let match_distance = if match_distance > 0 { match_distance - 1 } else { 0 };

    let mut s1_matches = vec![false; len1];
    let mut s2_matches = vec![false; len2];
    let mut matches = 0usize;
    let mut transpositions = 0usize;

    for i in 0..len1 {
        let start = if i > match_distance { i - match_distance } else { 0 };
        let end = std::cmp::min(i + match_distance + 1, len2);

        for j in start..end {
            if s2_matches[j] || s1_chars[i] != s2_chars[j] {
                continue;
            }
            s1_matches[i] = true;
            s2_matches[j] = true;
            matches += 1;
            break;
        }
    }

    if matches == 0 {
        return 0.0;
    }

    let mut k = 0;
    for i in 0..len1 {
        if !s1_matches[i] {
            continue;
        }
        while !s2_matches[k] {
            k += 1;
        }
        if s1_chars[i] != s2_chars[k] {
            transpositions += 1;
        }
        k += 1;
    }

    let m = matches as f64;
    let t = (transpositions / 2) as f64;
    let jaro = (m / len1 as f64 + m / len2 as f64 + (m - t) / m) / 3.0;

    // Winkler prefix bonus
    let prefix_len = s1_chars.iter()
        .zip(s2_chars.iter())
        .take(4)
        .take_while(|(a, b)| a == b)
        .count() as f64;

    jaro + (prefix_len * 0.1 * (1.0 - jaro))
}

/// Calculate Soft-TFIDF similarity.
/// 
/// Combines TF-IDF weighting with fuzzy token matching.
/// Tokens are considered similar if their Jaro-Winkler similarity >= threshold.
#[pyfunction]
#[pyo3(signature = (s1, s2, *, threshold=0.8, score_cutoff=None))]
pub fn soft_tfidf_similarity(s1: &str, s2: &str, threshold: f64, score_cutoff: Option<f64>) -> f64 {
    let tokens1: Vec<&str> = s1.split_whitespace().collect();
    let tokens2: Vec<&str> = s2.split_whitespace().collect();
    
    if tokens1.is_empty() && tokens2.is_empty() {
        return 1.0;
    }
    if tokens1.is_empty() || tokens2.is_empty() {
        return 0.0;
    }
    
    // Build TF vectors (using simple term frequency)
    let mut tf1: AHashMap<&str, f64> = AHashMap::new();
    let mut tf2: AHashMap<&str, f64> = AHashMap::new();
    
    for t in &tokens1 {
        *tf1.entry(*t).or_insert(0.0) += 1.0;
    }
    for t in &tokens2 {
        *tf2.entry(*t).or_insert(0.0) += 1.0;
    }
    
    // Normalize TF
    let norm1: f64 = tf1.values().map(|v| v * v).sum::<f64>().sqrt();
    let norm2: f64 = tf2.values().map(|v| v * v).sum::<f64>().sqrt();
    
    if norm1 == 0.0 || norm2 == 0.0 {
        return 0.0;
    }
    
    // Calculate soft similarity
    let mut similarity = 0.0;
    
    for (t1, w1) in &tf1 {
        let mut best_sim = 0.0;
        let mut best_w2 = 0.0;
        
        for (t2, w2) in &tf2 {
            let sim = jaro_winkler_sim(t1, t2);
            if sim >= threshold && sim > best_sim {
                best_sim = sim;
                best_w2 = *w2;
            }
        }
        
        if best_sim > 0.0 {
            similarity += (w1 / norm1) * (best_w2 / norm2) * best_sim;
        }
    }
    
    match score_cutoff {
        Some(cutoff) if similarity < cutoff => 0.0,
        _ => similarity,
    }
}

/// Calculate Soft-TFIDF distance.
#[pyfunction]
#[pyo3(signature = (s1, s2, *, threshold=0.8, score_cutoff=None))]
pub fn soft_tfidf_distance(s1: &str, s2: &str, threshold: f64, score_cutoff: Option<f64>) -> f64 {
    let sim = soft_tfidf_similarity(s1, s2, threshold, None);
    let dist = 1.0 - sim;
    
    match score_cutoff {
        Some(cutoff) if dist > cutoff => 1.0,
        _ => dist,
    }
}

#[pyfunction]
#[pyo3(signature = (pairs, *, threshold=0.8, score_cutoff=None))]
pub fn soft_tfidf_similarity_batch(
    pairs: Vec<(String, String)>,
    threshold: f64,
    score_cutoff: Option<f64>,
) -> Vec<f64> {
    pairs
        .par_iter()
        .map(|(s1, s2)| soft_tfidf_similarity(s1, s2, threshold, score_cutoff))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ngrams() {
        let ngrams = generate_ngrams("hello", 2);
        assert_eq!(ngrams, vec!["he", "el", "ll", "lo"]);
    }

    #[test]
    fn test_cosine() {
        // Test with n-grams (use_words=false)
        let sim = cosine_similarity("hello", "hello", false, 2, None);
        assert!((sim - 1.0).abs() < 0.01);
        
        // Test with words (use_words=true)
        let sim_words = cosine_similarity("hello world", "hello world", true, 2, None);
        assert!((sim_words - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_soft_tfidf() {
        let sim = soft_tfidf_similarity("hello world", "helo wrld", 0.8, None);
        assert!(sim > 0.5);
    }
}
