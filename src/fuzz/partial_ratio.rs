//! Partial ratio algorithm - RapidFuzz-compatible implementation
//!
//! Key features:
//! 1. Bit-parallel LCS computation (Hyyrö's algorithm)
//! 2. Character filtering - only check windows containing s1 characters
//! 3. Pre-built block map reused across sliding window
//! 4. Early termination when score reaches 100
//! 5. For equal-length strings, tries both orderings (like RapidFuzz)

use pyo3::prelude::*;
use rayon::prelude::*;
use smallvec::SmallVec;
use std::cmp::max;
use std::cell::RefCell;

use crate::lcs_core::{build_block_map, build_char_set, lcs_with_block, ratio_from_lcs};

// Thread-local buffers for Unicode path
thread_local! {
    static PARTIAL_BUF_1: RefCell<Vec<usize>> = RefCell::new(Vec::with_capacity(128));
    static PARTIAL_BUF_2: RefCell<Vec<usize>> = RefCell::new(Vec::with_capacity(128));
}

/// RapidFuzz-compatible partial ratio implementation
/// Checks windows in three phases: prefixes, full windows, suffixes
#[inline(always)]
fn partial_ratio_impl_ascii(shorter: &[u8], longer: &[u8]) -> f64 {
    let len1 = shorter.len();
    let len2 = longer.len();
    
    if len1 == 0 && len2 == 0 { return 100.0; }
    if len1 == 0 || len2 == 0 { return 0.0; }
    if len1 > len2 { return 0.0; }
    
    // Build block map and char set once
    let block = build_block_map(shorter);
    let char_set = build_char_set(shorter);
    
    let mut best_score = 0.0f64;
    
    // Phase 1: Prefix windows of longer string (size 1 to len1-1)
    // RapidFuzz: for i in range(1, len1): window = s2[:i]
    for i in 1..len1 {
        if i > len2 { break; }
        let substr_last = longer[i - 1];
        if !char_set[substr_last as usize] { continue; }
        
        let window = &longer[..i];
        let lcs = lcs_with_block(&block, len1, window);
        let score = ratio_from_lcs(len1, window.len(), lcs);
        if score > best_score { best_score = score; }
        if best_score >= 100.0 { return 100.0; }
    }
    
    // Phase 2: Full-size windows (only when longer > shorter)
    // RapidFuzz: for i in range(len2 - len1): window = s2[i:i+len1]
    // Note: range(0) is empty, so for equal length, this phase is skipped
    if len2 > len1 {
        for i in 0..(len2 - len1) {
            let substr_last = longer[i + len1 - 1];
            if !char_set[substr_last as usize] { continue; }
            
            let window = &longer[i..i + len1];
            let lcs = lcs_with_block(&block, len1, window);
            let score = ratio_from_lcs(len1, len1, lcs);
            if score > best_score { best_score = score; }
            if best_score >= 100.0 { return 100.0; }
        }
    }
    
    // Phase 3: Suffix windows of longer string
    // RapidFuzz: for i in range(len2 - len1, len2): window = s2[i:]
    // For equal length: this checks [0:], [1:], [2:], ... [len2-1:]
    let start_idx = if len2 > len1 { len2 - len1 } else { 0 };
    for i in start_idx..len2 {
        let substr_first = longer[i];
        if !char_set[substr_first as usize] { continue; }
        
        let window = &longer[i..];
        let lcs = lcs_with_block(&block, len1, window);
        let score = ratio_from_lcs(len1, window.len(), lcs);
        if score > best_score { best_score = score; }
        if best_score >= 100.0 { return 100.0; }
    }
    
    best_score
}

/// Bit-parallel partial ratio for ASCII strings <= 64 chars
/// RapidFuzz-compatible: checks both orderings for equal-length strings
#[inline(always)]
fn partial_ratio_bitparallel(s1: &[u8], s2: &[u8]) -> f64 {
    let len1 = s1.len();
    let len2 = s2.len();
    
    if len1 == 0 && len2 == 0 { return 100.0; }
    if len1 == 0 || len2 == 0 { return 0.0; }
    
    // Determine shorter and longer
    let (shorter, longer) = if len1 <= len2 {
        (s1, s2)
    } else {
        (s2, s1)
    };
    
    let mut best_score = partial_ratio_impl_ascii(shorter, longer);
    
    // For equal-length strings, try both orderings (like RapidFuzz)
    if len1 == len2 && best_score < 100.0 {
        let score2 = partial_ratio_impl_ascii(longer, shorter);
        if score2 > best_score {
            best_score = score2;
        }
    }
    
    best_score
}

/// Compute LCS for two byte slices using bit-parallel (multi-block for long strings)
#[allow(dead_code)]
#[inline(always)]
fn lcs_bytes(s1: &[u8], s2: &[u8]) -> usize {
    use crate::lcs_core::lcs_bitparallel_multiblock;
    lcs_bitparallel_multiblock(s1, s2)
}

/// Compute LCS using pre-built multi-block maps (for sliding window)
/// Takes s_vec as mutable slice to avoid per-call allocation
#[inline(always)]
fn lcs_with_multiblock_maps_reuse(
    block_maps: &[[u64; 256]], 
    len1: usize, 
    num_blocks: usize, 
    s2: &[u8],
    s_vec: &mut [u64]
) -> usize {
    // Reset state vectors to all 1s
    for i in 0..num_blocks {
        s_vec[i] = !0u64;
    }
    
    // Process s2 character by character
    for &c2 in s2.iter() {
        let mut carry: u64 = 0;
        
        for block_idx in 0..num_blocks {
            let matches = block_maps[block_idx][c2 as usize];
            let s_old = s_vec[block_idx];
            
            let u = s_old & matches;
            let (sum, overflow1) = s_old.overflowing_add(u);
            let sum_with_carry = sum.wrapping_add(carry);
            let overflow2 = sum_with_carry < sum;
            
            s_vec[block_idx] = sum_with_carry | (s_old.wrapping_sub(u));
            carry = (overflow1 || overflow2) as u64;
        }
    }
    
    // Count zeros (LCS length) across all blocks
    let mut lcs = 0usize;
    for block_idx in 0..num_blocks {
        let s = s_vec[block_idx];
        let block_start = block_idx * 64;
        let block_len = std::cmp::min(64, len1 - block_start);
        let mask = if block_len == 64 { !0u64 } else { (1u64 << block_len) - 1 };
        lcs += block_len - (s & mask).count_ones() as usize;
    }
    
    lcs
}

/// Optimized partial ratio for long ASCII strings
#[inline(always)]
fn partial_ratio_dp_ascii(shorter: &[u8], longer: &[u8]) -> f64 {
    let short_len = shorter.len();
    let long_len = longer.len();
    
    if short_len == 0 && long_len == 0 { return 100.0; }
    if short_len == 0 || long_len == 0 { return 0.0; }
    
    // Use specialized 2-block version for 65-128 char shorter strings
    if short_len <= 128 {
        return partial_ratio_2block(shorter, longer);
    }
    
    // General multi-block for > 128 chars
    let num_blocks = (short_len + 63) / 64;
    let mut block_maps: Vec<[u64; 256]> = vec![[0u64; 256]; num_blocks];
    
    for (i, &c) in shorter.iter().enumerate() {
        let block_idx = i / 64;
        let bit_pos = i % 64;
        block_maps[block_idx][c as usize] |= 1u64 << bit_pos;
    }
    
    let mut char_set = [false; 256];
    for &c in shorter.iter() {
        char_set[c as usize] = true;
    }
    
    let mut s_vec: Vec<u64> = vec![!0u64; num_blocks];
    let mut best = 0.0f64;
    
    // Only full windows for very long strings (skipping edge phases for performance)
    for i in 0..=(long_len - short_len) {
        let substr_last = longer[i + short_len - 1];
        if !char_set[substr_last as usize] { continue; }
        
        let window = &longer[i..i + short_len];
        let lcs = lcs_with_multiblock_maps_reuse(&block_maps, short_len, num_blocks, window, &mut s_vec);
        let score = ratio_from_lcs(short_len, short_len, lcs);
        if score > best { best = score; }
        if best >= 100.0 { return 100.0; }
    }
    
    best
}

/// Specialized 2-block partial ratio for 65-128 char strings (full stack allocation)
#[inline(always)]
fn partial_ratio_2block(shorter: &[u8], longer: &[u8]) -> f64 {
    let short_len = shorter.len();
    let long_len = longer.len();
    
    // Stack-allocated block maps (2 blocks × 256 entries)
    let mut block0: [u64; 256] = [0u64; 256];
    let mut block1: [u64; 256] = [0u64; 256];
    
    for (i, &c) in shorter.iter().enumerate() {
        if i < 64 {
            block0[c as usize] |= 1u64 << i;
        } else {
            block1[c as usize] |= 1u64 << (i - 64);
        }
    }
    
    // Build character set for filtering
    let mut char_set = [false; 256];
    for &c in shorter.iter() {
        char_set[c as usize] = true;
    }
    
    let len_block1 = if short_len > 64 { short_len - 64 } else { 0 };
    let num_blocks = if short_len <= 64 { 1 } else { 2 };
    
    let mut best = 0.0f64;
    
    // Phase 1: Prefix windows
    for i in 1..short_len {
        if i > long_len { break; }
        let substr_last = longer[i - 1];
        if !char_set[substr_last as usize] { continue; }
        
        let window = &longer[..i];
        let lcs = lcs_2block_window(&block0, &block1, short_len, len_block1, num_blocks, window);
        let score = ratio_from_lcs(short_len, window.len(), lcs);
        if score > best { best = score; }
        if best >= 100.0 { return 100.0; }
    }
    
    // Phase 2: Full windows (only when longer > shorter)
    // RapidFuzz: for i in range(len2 - len1): window = s2[i:i+len1]
    if long_len > short_len {
        for i in 0..(long_len - short_len) {
            let substr_last = longer[i + short_len - 1];
            if !char_set[substr_last as usize] { continue; }
            
            let window = &longer[i..i + short_len];
            let lcs = lcs_2block_window(&block0, &block1, short_len, len_block1, num_blocks, window);
            let score = ratio_from_lcs(short_len, short_len, lcs);
            if score > best { best = score; }
            if best >= 100.0 { return 100.0; }
        }
    }
    
    // Phase 3: Suffix windows
    // RapidFuzz: for i in range(len2 - len1, len2): window = s2[i:]
    let start_idx = if long_len > short_len { long_len - short_len } else { 0 };
    for i in start_idx..long_len {
        let substr_first = longer[i];
        if !char_set[substr_first as usize] { continue; }
        
        let window = &longer[i..];
        let lcs = lcs_2block_window(&block0, &block1, short_len, len_block1, num_blocks, window);
        let score = ratio_from_lcs(short_len, window.len(), lcs);
        if score > best { best = score; }
        if best >= 100.0 { return 100.0; }
    }
    
    best
}

/// Compute LCS using 2 stack-allocated blocks
#[inline(always)]
fn lcs_2block_window(
    block0: &[u64; 256], block1: &[u64; 256],
    _len1: usize, len_block1: usize, num_blocks: usize,
    s2: &[u8]
) -> usize {
    let mut s0: u64 = !0u64;
    let mut s1_state: u64 = !0u64;
    
    if num_blocks == 1 {
        // Single block case
        for &c2 in s2.iter() {
            let matches = block0[c2 as usize];
            let u = s0 & matches;
            s0 = s0.wrapping_add(u) | (s0.wrapping_sub(u));
        }
        return 64 - s0.count_ones() as usize;
    }
    
    // 2-block case
    for &c2 in s2.iter() {
        let matches0 = block0[c2 as usize];
        let matches1 = block1[c2 as usize];
        
        let u0 = s0 & matches0;
        let (sum0, overflow0) = s0.overflowing_add(u0);
        s0 = sum0 | (s0.wrapping_sub(u0));
        let carry = overflow0 as u64;
        
        let u1 = s1_state & matches1;
        let (sum1, _) = s1_state.overflowing_add(u1);
        let sum1_carry = sum1.wrapping_add(carry);
        s1_state = sum1_carry | (s1_state.wrapping_sub(u1));
    }
    
    let mask1 = if len_block1 == 64 { !0u64 } else { (1u64 << len_block1) - 1 };
    let lcs0 = 64 - s0.count_ones() as usize;
    let lcs1 = len_block1 - (s1_state & mask1).count_ones() as usize;
    
    lcs0 + lcs1
}

/// Compute LCS for two char slices using DP
#[inline(always)]
fn lcs_chars(s1: &[char], s2: &[char]) -> usize {
    let len1 = s1.len();
    let len2 = s2.len();
    
    if len1 == 0 || len2 == 0 { return 0; }
    
    PARTIAL_BUF_1.with(|buf1| {
        PARTIAL_BUF_2.with(|buf2| {
            let mut prev = buf1.borrow_mut();
            let mut curr = buf2.borrow_mut();
            
            prev.clear();
            prev.resize(len1 + 1, 0);
            curr.clear();
            curr.resize(len1 + 1, 0);
            
            for c2 in s2.iter() {
                curr.fill(0);
                for (j, c1) in s1.iter().enumerate() {
                    curr[j + 1] = if *c1 == *c2 {
                        prev[j] + 1
                    } else {
                        max(prev[j + 1], curr[j])
                    };
                }
                std::mem::swap(&mut *prev, &mut *curr);
            }
            
            prev[len1]
        })
    })
}

/// Partial ratio implementation for Unicode (one direction)
#[inline(always)]
fn partial_ratio_impl_unicode(shorter: &[char], longer: &[char]) -> f64 {
    let short_len = shorter.len();
    let long_len = longer.len();

    if short_len == 0 && long_len == 0 { return 100.0; }
    if short_len == 0 || long_len == 0 { return 0.0; }

    // Build character set for filtering
    use ahash::AHashSet;
    let char_set: AHashSet<char> = shorter.iter().copied().collect();

    let mut best = 0.0f64;
    
    // Phase 1: Prefix windows
    for i in 1..short_len {
        if i > long_len { break; }
        let substr_last = longer[i - 1];
        if !char_set.contains(&substr_last) { continue; }
        
        let window = &longer[..i];
        let lcs = lcs_chars(shorter, window);
        let score = ratio_from_lcs(short_len, window.len(), lcs);
        if score > best { best = score; }
        if best >= 100.0 { return 100.0; }
    }
    
    // Phase 2: Full windows (only when longer > shorter)
    // RapidFuzz: for i in range(len2 - len1): window = s2[i:i+len1]
    if long_len > short_len {
        for i in 0..(long_len - short_len) {
            let substr_last = longer[i + short_len - 1];
            if !char_set.contains(&substr_last) { continue; }
            
            let window = &longer[i..i + short_len];
            let lcs = lcs_chars(shorter, window);
            let score = ratio_from_lcs(short_len, short_len, lcs);
            if score > best { best = score; }
            if best >= 100.0 { return 100.0; }
        }
    }
    
    // Phase 3: Suffix windows
    // RapidFuzz: for i in range(len2 - len1, len2): window = s2[i:]
    let start_idx = if long_len > short_len { long_len - short_len } else { 0 };
    for i in start_idx..long_len {
        let substr_first = longer[i];
        if !char_set.contains(&substr_first) { continue; }
        
        let window = &longer[i..];
        let lcs = lcs_chars(shorter, window);
        let score = ratio_from_lcs(short_len, window.len(), lcs);
        if score > best { best = score; }
        if best >= 100.0 { return 100.0; }
    }
    
    best
}

/// Partial ratio for Unicode strings - checks both orderings for equal-length strings
#[inline(always)]
fn partial_ratio_unicode(s1_chars: &[char], s2_chars: &[char]) -> f64 {
    let len1 = s1_chars.len();
    let len2 = s2_chars.len();

    if len1 == 0 && len2 == 0 { return 100.0; }
    if len1 == 0 || len2 == 0 { return 0.0; }

    let (shorter, longer) = if len1 <= len2 {
        (s1_chars, s2_chars)
    } else {
        (s2_chars, s1_chars)
    };
    
    let mut best = partial_ratio_impl_unicode(shorter, longer);
    
    // For equal-length strings, try both orderings
    if len1 == len2 && best < 100.0 {
        let score2 = partial_ratio_impl_unicode(longer, shorter);
        if score2 > best {
            best = score2;
        }
    }
    
    best
}

/// Main partial ratio implementation
#[inline(always)]
pub fn partial_ratio_internal(s1: &str, s2: &str) -> f64 {
    if s1 == s2 { return 100.0; }
    if s1.is_empty() && s2.is_empty() { return 100.0; }
    if s1.is_empty() || s2.is_empty() { return 0.0; }
    
    // ASCII + short string: use bit-parallel
    if s1.is_ascii() && s2.is_ascii() {
        let b1 = s1.as_bytes();
        let b2 = s2.as_bytes();
        
        // Use bit-parallel for short strings
        if b1.len() <= 64 && b2.len() <= 64 {
            return partial_ratio_bitparallel(b1, b2);
        }
        
        // For longer strings, determine shorter/longer
        let (shorter, longer) = if b1.len() <= b2.len() {
            (b1, b2)
        } else {
            (b2, b1)
        };
        
        let mut best = partial_ratio_dp_ascii(shorter, longer);
        
        // For equal-length strings, try both orderings
        if b1.len() == b2.len() && best < 100.0 {
            let score2 = partial_ratio_dp_ascii(longer, shorter);
            if score2 > best {
                best = score2;
            }
        }
        
        return best;
    }
    
    // Unicode path
    let s1_chars: SmallVec<[char; 64]> = s1.chars().collect();
    let s2_chars: SmallVec<[char; 64]> = s2.chars().collect();
    partial_ratio_unicode(&s1_chars, &s2_chars)
}

/// Calculate partial ratio
#[pyfunction]
#[pyo3(signature = (s1, s2, *, score_cutoff=None))]
pub fn partial_ratio(s1: &str, s2: &str, score_cutoff: Option<f64>) -> f64 {
    let result = partial_ratio_internal(s1, s2);
    
    match score_cutoff {
        Some(cutoff) if result < cutoff => 0.0,
        _ => result,
    }
}

/// Batch partial ratio
#[pyfunction]
#[pyo3(signature = (pairs, *, score_cutoff=None))]
pub fn partial_ratio_batch(pairs: Vec<(String, String)>, score_cutoff: Option<f64>) -> Vec<f64> {
    pairs
        .par_iter()
        .map(|(s1, s2)| partial_ratio(s1, s2, score_cutoff))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_partial_ratio() {
        let r = partial_ratio_internal("this is a test", "this is a test!");
        assert!((r - 100.0).abs() < 0.01);
    }
    
    #[test]
    fn test_partial_ratio_substring() {
        let r = partial_ratio_internal("test", "this is a test");
        assert!((r - 100.0).abs() < 0.01);
    }
}
